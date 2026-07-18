"""Background worker: owns the model and executes queued tasks one by one.

Model lifecycle (lazy by default):
  - the GPU stays free until a task arrives; the model loads on demand;
  - after `idle_unload_seconds` without work the model is unloaded, unless
    the user set keep_loaded (do_not_unload_model);
  - "выгрузить" pauses the worker: the model is unloaded and will not
    auto-load until "загрузить" is pressed or a new task is submitted;
  - cancelling a running task aborts generation via
    DotsMOCRParser.abort_event; finished pages keep their results.
"""

import threading
import time
import traceback
from pathlib import Path

from demo import db

# Authors' recommendations (rednote-hilab/dots.mocr demo):
# fitz preprocess for document-style prompts on raw images, per-prompt
# temperature (0.9 for SVG generation, 0.1 otherwise).
PROMPT_TO_FITZ_PREPROCESS = {
    "prompt_layout_all_en": True,
    "prompt_layout_only_en": True,
    "prompt_ocr": True,
    "prompt_grounding_ocr": True,
    "prompt_web_parsing": False,
    "prompt_scene_spotting": False,
    "prompt_image_to_svg": False,
    "prompt_general": False,
}
PROMPT_TO_TEMPERATURE = {
    "prompt_image_to_svg": 0.9,
}
DEFAULT_TEMPERATURE = 0.1

# Rough seconds-per-page priors for the UI progress bar, measured on an
# RTX 4090 (reports/benchmark_2x4090_2026-07-15.md; layout ~44 s/page at
# dpi 150, OCR is lighter, SVG generation is much longer). The UI refines
# these with the measured time of already finished pages of the same task.
PAGE_SECONDS_ESTIMATE = {
    "prompt_layout_all_en": 45,
    "prompt_layout_only_en": 12,
    "prompt_ocr": 30,
    "prompt_grounding_ocr": 10,
    "prompt_web_parsing": 45,
    "prompt_scene_spotting": 25,
    "prompt_general": 15,
    "prompt_image_to_svg": 140,
}


def default_temperature(prompt_mode):
    return PROMPT_TO_TEMPERATURE.get(prompt_mode, DEFAULT_TEMPERATURE)


_default_attn = None


def default_attn_implementation():
    """The parser's own default backend, read from its signature.

    Resolved lazily and cached: importing dots_mocr pulls in torch, which this
    module deliberately avoids at import time. Reading the signature rather than
    hard-coding the string keeps the demo from carrying a second copy of the
    default that could silently drift from the parser's.
    """
    global _default_attn
    if _default_attn is None:
        import inspect

        from dots_mocr.cli import DotsMOCRParser

        parameter = inspect.signature(DotsMOCRParser).parameters["attn_implementation"]
        _default_attn = parameter.default
    return _default_attn


class DemoWorker(threading.Thread):
    """One worker per demo process; the parser is used only by this thread."""

    def __init__(self, ckpt, jobs_dir, device="auto", dpi=150,
                 max_pixels=2_200_000, max_completion_tokens=16384,
                 parser_factory=None, autostart=False, keep_loaded=False,
                 idle_unload_seconds=180, attn_implementation=None,
                 engine="transformers", vllm_url=None, vllm_model=None):
        super().__init__(daemon=True, name="demo-worker")
        self.ckpt = ckpt
        self.jobs_dir = Path(jobs_dir)
        self.device = device
        # None => whatever DotsMOCRParser defaults to, so the demo never carries
        # a second copy of that default that could drift from the parser's.
        self.attn_implementation = attn_implementation
        # "transformers" runs the model in this process; "vllm" drives a server
        # that owns the GPU instead. Everything downstream — artifacts, the queue,
        # the tokens/s readout — is identical, so the two are directly comparable.
        self.engine = engine
        self.vllm_url = vllm_url or "http://127.0.0.1:8000/v1"
        self.vllm_model = vllm_model or "rednote-hilab/dots.mocr"
        self.dpi = dpi
        # Was chosen because the dense sdpa mask OOMed above ~2.2M px/page on 24GB
        # (reports/benchmark_2x4090_2026-07-15.md). The flex_attention backend
        # removed that wall — 2.2M px now fits in 6.2 GiB on a 12GB card, where sdpa
        # cannot run it at all (reports/flexattn/) — so this cap is now about decode
        # time per page, not memory. The authors' 11.3M default still assumes vLLM.
        self.max_pixels = max_pixels
        self.max_completion_tokens = max_completion_tokens
        self._parser_factory = parser_factory or self._default_parser_factory
        self.parser = None
        self.model_state = "stopped"       # stopped | loading | loaded | error
        self.model_error = None
        self.paused = False                # paused => no auto-load on demand
        # Idle-unloading exists to hand the GPU back between tasks. With vLLM the
        # GPU belongs to the server, so unloading would only drop a healthy HTTP
        # client and show a misleading "выгрузка через N s" countdown.
        self.keep_loaded = keep_loaded or engine == "vllm"
        self.idle_unload_seconds = idle_unload_seconds
        self.current_task_id = None
        # GenerationStats of the page being decoded right now. The parser mutates
        # it in place, so /api/state reads live tokens/s straight off this object
        # without any per-token bookkeeping of our own.
        self.live_stats = None
        self.live_page = None
        self.abort_event = threading.Event()
        self._last_used = time.time()
        self._load_now = autostart         # explicit load request pending
        self._reload_requested = False     # device changed -> unload + reload
        self._wakeup = threading.Event()
        self._shutdown = threading.Event()

    # ------------------------------------------------------------ control

    def request_start(self):
        """'загрузить модель': load now and resume auto-loading."""
        self.paused = False
        self._load_now = True
        self._wakeup.set()

    def request_stop(self):
        """'выгрузить модель': abort generation, unload, pause auto-load."""
        self.paused = True
        self._load_now = False
        self.abort_event.set()
        self._wakeup.set()

    def set_keep_loaded(self, value):
        self.keep_loaded = bool(value)
        self._wakeup.set()

    def set_device(self, device):
        """Switch the inference GPU (e.g. 'cuda:0' -> 'cuda:1').

        Aborts any running generation, unloads the model, and reloads it on the
        new device (immediately if it was loaded / keep_loaded, else lazily on
        the next task). No-op if unchanged.
        """
        if not device or device == self.device:
            return
        self.device = device
        self.paused = False
        self._reload_requested = True
        self.abort_event.set()
        self._wakeup.set()

    def notify_new_task(self):
        """A new task resumes a paused worker — the user asked for compute."""
        self.paused = False
        self._wakeup.set()

    def cancel_task(self, task_id):
        status = db.request_cancel(task_id)
        if status == "cancelling" and task_id == self.current_task_id:
            self.abort_event.set()
        return status

    def shutdown(self, join_timeout=10.0):
        self._shutdown.set()
        self.abort_event.set()
        self._wakeup.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=join_timeout)

    def status(self):
        idle_for = time.time() - self._last_used
        unload_in = None
        if (self.model_state == "loaded" and not self.keep_loaded
                and self.current_task_id is None):
            unload_in = max(0, round(self.idle_unload_seconds - idle_for))
        return {
            "model_state": self.model_state,
            "model_error": self.model_error,
            "paused": self.paused,
            "keep_loaded": self.keep_loaded,
            "idle_unload_seconds": self.idle_unload_seconds,
            "unload_in_seconds": unload_in,
            "current_task_id": self.current_task_id,
            "device": getattr(self.parser, "device", None) if self.parser else None,
            "configured_device": self.device,  # user-selected target (even when unloaded)
            "engine": self.engine,
            # once loaded the parser knows the EFFECTIVE backend (it may have
            # demoted flex to sdpa on cpu); before that, show what will be used
            "attn_implementation": (
                None if self.engine == "vllm"
                else (getattr(self.parser, "attn_implementation", None) if self.parser
                      else (self.attn_implementation or self._default_attn()))),
            "live": self.live_generation(),
        }

    @staticmethod
    def _default_attn():
        """Never let a status poll fail because the default could not be read."""
        try:
            return default_attn_implementation()
        except Exception:  # noqa: BLE001 - status must not raise
            return None

    def live_generation(self):
        """Snapshot of the in-flight generation for the UI's tokens/s readout."""
        stats = self.live_stats
        if stats is None or stats.started_at is None:
            return None
        return {
            "task_id": self.current_task_id,
            "page_no": self.live_page,
            "done": stats.finished_at is not None,
            **stats.to_dict(),
        }

    # ------------------------------------------------------------ model

    def _default_parser_factory(self):
        common = dict(
            ckpt=self.ckpt,
            temperature=DEFAULT_TEMPERATURE,
            max_completion_tokens=self.max_completion_tokens,
            dpi=self.dpi,
            max_pixels=self.max_pixels,
            num_thread=1,
        )
        if self.engine == "vllm":
            from dots_mocr.model.vllm_parser import VllmDotsMOCRParser

            return VllmDotsMOCRParser(
                vllm_url=self.vllm_url, vllm_model=self.vllm_model,
                device="vllm", dtype="auto", **common,
            )

        from dots_mocr.cli import DotsMOCRParser

        return DotsMOCRParser(
            device=self.device,
            dtype="bfloat16" if self.device != "cpu" else "float32",
            **({"attn_implementation": self.attn_implementation}
               if self.attn_implementation else {}),
            **common,
        )

    def _on_generation_start(self, stats):
        self.live_stats = stats

    def _load_model(self):
        self.model_state = "loading"
        self.model_error = None
        # A vLLM parser that is merely asleep only needs waking; rebuilding the
        # client would leave the weights offloaded and the card still empty.
        if self.engine == "vllm" and self.parser is not None:
            try:
                self.parser.wake()
                self.model_state = "loaded"
                self._last_used = time.time()
            except Exception as error:  # surfaced in the UI
                self.model_state = "error"
                self.model_error = f"{type(error).__name__}: {error}"
                self.paused = True
                traceback.print_exc()
            return
        try:
            self.parser = self._parser_factory()
            self.parser.abort_event = self.abort_event
            self.parser.generation_listener = self._on_generation_start
            self.model_state = "loaded"
            self._last_used = time.time()
        except Exception as error:  # surfaced in the UI
            self.model_state = "error"
            self.model_error = f"{type(error).__name__}: {error}"
            # do not retry in a loop: wait for an explicit start or a new task
            self.paused = True
            traceback.print_exc()

    def _unload_model(self):
        if self.engine == "vllm" and self.parser is not None:
            # The GPU belongs to the vLLM server. Dropping our HTTP client would
            # free nothing while the UI reported "stopped", so ask the server to
            # offload its weights instead — that is what actually returns memory.
            try:
                self.parser.sleep()
                self.model_state = "stopped"
            except Exception as error:  # noqa: BLE001 - report, do not pretend
                self.model_state = "error"
                self.model_error = f"{type(error).__name__}: {error}"
                traceback.print_exc()
            return

        parser = self.parser
        self.parser = None
        if self.model_state != "error":
            self.model_state = "stopped"
        if parser is not None:
            try:
                del parser.model
            except AttributeError:
                pass
            del parser
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    # ------------------------------------------------------------ main loop

    def run(self):
        while not self._shutdown.is_set():
            if self._reload_requested:
                self._reload_requested = False
                was_loaded = self.model_state == "loaded"
                self._unload_model()
                if was_loaded or self.keep_loaded:
                    self._load_now = True  # reload on the new device

            if self.paused:
                if self.model_state == "loaded":
                    self._unload_model()
                self._wait()
                continue

            demand = self._load_now or db.has_queued_tasks()
            if demand and self.model_state in ("stopped", "error"):
                self._load_model()
            self._load_now = False

            if self.model_state != "loaded":
                self._wait()
                continue

            task = db.claim_next_task()
            if task is not None:
                self._run_task(task)
                self._last_used = time.time()
                continue

            if (not self.keep_loaded
                    and time.time() - self._last_used > self.idle_unload_seconds):
                self._unload_model()
            self._wait()

    def _wait(self):
        self._wakeup.wait(timeout=1.0)
        self._wakeup.clear()

    # ------------------------------------------------------------ task

    def _run_task(self, task):
        self.current_task_id = task["id"]
        self.abort_event.clear()
        if self.paused:
            # request_stop() landed between the loop's paused check and the
            # claim: hand the task back to the queue instead of running it
            self.current_task_id = None
            db.update_task(task["id"], status="queued", started_at=None)
            return
        try:
            self._execute(task)
        except Exception as error:
            traceback.print_exc()
            db.update_task(
                task["id"], status="error",
                error=f"{type(error).__name__}: {error}", finished_at=time.time(),
            )
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # release fragments after OOM
            except ImportError:
                pass
        finally:
            self.current_task_id = None
            self.live_stats = None
            self.live_page = None
            self.abort_event.clear()

    @staticmethod
    def _scale_bbox(bbox, view_size, origin_image):
        """Map a bbox drawn on the viewer render onto the inference render."""
        if bbox is None:
            return None
        if not view_size or not view_size[0]:
            return bbox
        scale_x = origin_image.width / view_size[0]
        scale_y = origin_image.height / view_size[1]
        return [
            int(bbox[0] * scale_x), int(bbox[1] * scale_y),
            int(bbox[2] * scale_x), int(bbox[3] * scale_y),
        ]

    def _input_path(self, job):
        job_dir = self.jobs_dir / job["id"]
        matches = list(job_dir.glob("input.*"))
        if not matches:
            raise FileNotFoundError(f"no input file for job {job['id']}")
        return matches[0]

    def _execute(self, task):
        from dots_mocr.utils.doc_utils import load_pdf_pages
        from dots_mocr.utils.image_utils import fetch_image

        job = db.get_job(task["job_id"])
        if job is None:
            raise ValueError(f"job {task['job_id']} not found")
        input_path = self._input_path(job)
        out_dir = self.jobs_dir / job["id"] / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        params = task["params"]
        prompt_mode = task["prompt_mode"]
        temperature = params.get("temperature")
        if temperature is None:
            temperature = default_temperature(prompt_mode)
        custom_prompt = params.get("custom_prompt") or None
        bbox = params.get("bbox")
        dpi = int(params.get("dpi") or self.dpi)
        max_new_tokens = params.get("max_new_tokens")
        if max_new_tokens:
            # the parser is owned by this thread only, safe to adjust per task
            self.parser.max_completion_tokens = int(max_new_tokens)
        else:
            self.parser.max_completion_tokens = self.max_completion_tokens
        pages = task["pages"] or [0]
        save_name = f"task_{task['id']}"

        results = []
        total = len(pages)
        db.update_task(task["id"], progress={"done": 0, "total": total})

        for index, page_no in enumerate(pages):
            if db.is_cancel_requested(task["id"]) or self.abort_event.is_set():
                db.update_task(
                    task["id"], status="cancelled", result=results,
                    progress={"done": index, "total": total},
                    finished_at=time.time(),
                )
                return
            db.update_task(
                task["id"],
                progress={"done": index, "total": total, "current_page": page_no,
                          "page_started_at": time.time()},
            )

            started = time.time()
            self.live_page = page_no
            if job["kind"] == "pdf":
                rendered = load_pdf_pages(str(input_path), dpi=dpi, page_ids=[page_no])
                if not rendered:
                    raise ValueError(f"page {page_no} did not render")
                origin_image = rendered[0][1]
                page_bbox = self._scale_bbox(bbox, params.get("bbox_view_size"), origin_image)
                page_result = self.parser._parse_single_image(
                    origin_image, prompt_mode, str(out_dir), save_name,
                    source="pdf", page_idx=page_no, bbox=page_bbox,
                    custom_prompt=custom_prompt, temperature=temperature,
                )
            else:
                origin_image = fetch_image(str(input_path))
                fitz_preprocess = PROMPT_TO_FITZ_PREPROCESS.get(prompt_mode, False)
                page_bbox = self._scale_bbox(bbox, params.get("bbox_view_size"), origin_image)
                page_result = self.parser._parse_single_image(
                    origin_image, prompt_mode, str(out_dir), f"{save_name}_page_{page_no}",
                    source="image", page_idx=page_no, bbox=page_bbox,
                    fitz_preprocess=fitz_preprocess,
                    custom_prompt=custom_prompt, temperature=temperature,
                )

            aborted = self.abort_event.is_set() or db.is_cancel_requested(task["id"])
            if aborted:
                # generation was interrupted mid-page: drop the partial page
                db.update_task(
                    task["id"], status="cancelled", result=results,
                    progress={"done": index, "total": total},
                    finished_at=time.time(),
                )
                return

            page_result["seconds"] = round(time.time() - started, 2)
            results.append(page_result)
            db.update_task(
                task["id"], result=results,
                progress={"done": index + 1, "total": total},
            )

        db.update_task(
            task["id"], status="done", result=results,
            progress={"done": total, "total": total}, finished_at=time.time(),
        )

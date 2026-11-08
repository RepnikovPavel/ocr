"""Background worker: owns the model and executes queued tasks one by one.

Model lifecycle (lazy by default):
  - the GPU stays free until a task arrives; the model loads on demand;
  - after `idle_unload_seconds` without work the model is unloaded;
    keep_loaded (do_not_unload_model) only stretches that window to
    `keep_loaded_idle_seconds` — a forgotten toggle must not pin the GPU
    forever;
  - an idle vLLM container left running without any load cycle (fresh
    deploy, host reboot with restart=unless-stopped) is stopped on the
    same idle clock, so the service converges to zero VRAM from any state;
  - "выгрузить" pauses the worker: the model is unloaded and will not
    auto-load until "загрузить" is pressed or a new task is submitted;
  - cancelling a running task aborts generation via
    DotsMOCRParser.abort_event; finished pages keep their results.
"""

import os
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
                 idle_unload_seconds=180, keep_loaded_idle_seconds=3600,
                 attn_implementation=None,
                 engine="transformers", vllm_url=None, vllm_model=None,
                 vllm_urls=None, vllm_containers=None, vllm_startup_timeout=600):
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
        self.vllm_urls = [u.rstrip("/") for u in (vllm_urls or [self.vllm_url])]
        self.vllm_url = self.vllm_urls[0]
        if len(set(self.vllm_urls)) != len(self.vllm_urls):
            raise ValueError("DEMO_VLLM_URLS must contain distinct endpoints")
        if engine != "vllm" and len(self.vllm_urls) > 1:
            raise ValueError("multiple endpoints require DEMO_ENGINE=vllm")
        self.vllm_containers = list(vllm_containers or
                                    [self._VLLM_CONTAINER] + [None] * (len(self.vllm_urls) - 1))
        if len(self.vllm_containers) != len(self.vllm_urls):
            raise ValueError("DEMO_VLLM_CONTAINERS must match DEMO_VLLM_URLS")
        names = [c for c in self.vllm_containers if c]
        if len(names) != len(set(names)):
            raise ValueError("vLLM containers must be distinct")
        self.vllm_startup_timeout = float(vllm_startup_timeout)
        if self.vllm_startup_timeout <= 0:
            raise ValueError("vLLM startup timeout must be positive")
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
        self.parsers = []  # one client per independent GPU process, never shared concurrently
        self._live_slots = {}
        self._live_lock = threading.Lock()
        self.model_state = "stopped"       # stopped | loading | loaded | error
        self.model_error = None
        self.paused = False                # paused => no auto-load on demand
        # Idle-unloading hands the GPU back when no agent is asking for anything.
        # It applies to both engines: the in-process one drops its weights, and
        # the vLLM one is put to sleep, which returns the server's memory too
        # (measured 9.6 -> 2.3 GiB). The next queued task wakes it automatically,
        # so an idle service costs no VRAM and a busy one is unaffected.
        self.keep_loaded = keep_loaded
        self.idle_unload_seconds = idle_unload_seconds
        # keep_loaded is a bounded pin, not "never unload": it stretches the
        # idle window to this many seconds. A toggle switched on once and
        # forgotten used to pin ~12 GB of VRAM indefinitely; the window keeps
        # the "hot between bursts of work" use case while guaranteeing the GPU
        # is eventually handed back. 0 restores the old "never unload".
        self.keep_loaded_idle_seconds = keep_loaded_idle_seconds
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
        print(f"[worker] keep_loaded={self.keep_loaded} "
              f"(idle-unload after {self._idle_limit()}s of no work)", flush=True)
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

    def _idle_limit(self):
        """Seconds without work after which the model is unloaded.

        keep_loaded stretches the window instead of disabling the unload
        outright (see __init__); keep_loaded_idle_seconds=0 opts back into
        the old "never unload while pinned" behaviour.
        """
        if not self.keep_loaded:
            return self.idle_unload_seconds
        if self.keep_loaded_idle_seconds > 0:
            return self.keep_loaded_idle_seconds
        return float("inf")

    def status(self):
        idle_for = time.time() - self._last_used
        unload_in = None
        limit = self._idle_limit()
        if (self.model_state == "loaded" and self.current_task_id is None
                and limit != float("inf")):
            unload_in = max(0, round(limit - idle_for))
        return {
            "model_state": self.model_state,
            "model_error": self.model_error,
            "paused": self.paused,
            "keep_loaded": self.keep_loaded,
            "idle_unload_seconds": self.idle_unload_seconds,
            "keep_loaded_idle_seconds": self.keep_loaded_idle_seconds,
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
            "live_generations": self.live_generations(),
            "page_workers": len(self.vllm_urls) if self.engine == "vllm" else 1,
            "vllm_urls": self.vllm_urls if self.engine == "vllm" else [],
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
        parallel = self.live_generations()
        if parallel:
            return parallel[0]
        stats = self.live_stats
        if stats is None or stats.started_at is None:
            return None
        return {
            "task_id": self.current_task_id,
            "page_no": self.live_page,
            "done": stats.finished_at is not None,
            **stats.to_dict(),
        }

    def live_generations(self):
        with self._live_lock:
            slots = list(self._live_slots.items())
        return [
            {"task_id": self.current_task_id, "worker": slot, "page_no": page,
             "backend_url": self.vllm_urls[slot] if self.engine == "vllm" else None,
             "done": stats.finished_at is not None, **stats.to_dict()}
            for slot, (page, stats) in sorted(slots)
            if stats is not None and stats.started_at is not None
        ]

    # ------------------------------------------------------------ model

    def _default_parser_factory(self, vllm_url=None):
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
                vllm_url=vllm_url or self.vllm_url, vllm_model=self.vllm_model,
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
        self.abort_event.clear()
        try:
            # A running container can still be loading weights / compiling CUDA
            # graphs. Wait for readiness on EVERY load, including after reboot.
            self._start_vllm_container()
            if (len(self.parsers) == len(self.vllm_urls)
                    and all(callable(getattr(p, "wake", None)) for p in self.parsers)):
                with ThreadPoolExecutor(max_workers=len(self.parsers)) as pool:
                    list(pool.map(lambda p: p.wake(), self.parsers))
            else:
                factories = [self._parser_factory]
                if self.engine == "vllm":
                    factories += [lambda url=url: self._default_parser_factory(url)
                                  for url in self.vllm_urls[1:]]
                # Keep references to successfully constructed clients if another
                # endpoint fails, so unloading can still release their weights.
                self.parsers = []
                for factory in factories:
                    parser = factory()
                    parser.abort_event = self.abort_event
                    parser.generation_listener = self._on_generation_start
                    self.parsers.append(parser)
            self.parser = self.parsers[0]
            self.model_state = "loaded"
            self._last_used = time.time()
        except InterruptedError:
            self.model_state = "stopped"
        except Exception as error:  # surfaced in the UI
            self.model_state = "error"
            self.model_error = f"{type(error).__name__}: {error}"
            # do not retry in a loop: wait for an explicit start or a new task
            self.paused = True
            # A model error used to leave every task queued forever (ocrc polls
            # task status, not model status). Give all waiting callers the reason.
            db.fail_queued_tasks(self.model_error)
            traceback.print_exc()

    # ------------------------------------------------------------ vllm container

    _DOCKER_SOCKET = "/var/run/docker.sock"
    _VLLM_CONTAINER = os.environ.get("DEMO_VLLM_CONTAINER")  # e.g. "dots_vllm"

    @classmethod
    def _docker_api(cls, method, path, body=None):
        """Call the Docker Engine API over the unix socket. Returns (status, json).

        Used to start/stop the vLLM container from inside the demo process so
        that idle-unload frees ALL VRAM (stopping the process, not just
        offloading weights). Requires /var/run/docker.sock to be bind-mounted
        into the demo container. When the socket is absent (local dev without
        Docker), every call no-ops and returns (0, None).
        """
        import socket as _socket
        sock_path = cls._DOCKER_SOCKET
        if not os.path.exists(sock_path):
            return 0, None
        try:
            s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            s.settimeout(120)
            s.connect(sock_path)
            body_bytes = body.encode() if body else b""
            headers = (
                f"{method} {path} HTTP/1.0\r\n"
                f"Host: docker\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
                "\r\n"
            ).encode()
            s.sendall(headers + body_bytes)
            raw = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                raw += chunk
            s.close()
            status_line = raw.split(b"\r\n", 1)[0].decode(errors="replace")
            status_code = int(status_line.split()[1]) if " " in status_line else 0
            # body is after the blank line separating headers from body
            parts = raw.split(b"\r\n\r\n", 1)
            resp_body = parts[1] if len(parts) > 1 else b""
            return status_code, resp_body
        except Exception as error:  # noqa: BLE001
            print(f"[worker] docker API {method} {path} failed: {error}", flush=True)
            return 0, None

    def _vllm_container_running(self, container=None):
        """True/False from the Docker API, None when unknown (no socket etc.)."""
        status, body = self._docker_api(
            "GET", f"/containers/{container or self._VLLM_CONTAINER}/json")
        if status != 200:
            return None
        import json as _json
        try:
            return bool(_json.loads(body).get("State", {}).get("Running", False))
        except Exception:  # noqa: BLE001
            return None

    def _stop_vllm_container(self):
        """Stop the vLLM Docker container to free ALL VRAM (not just weights).

        Quiet no-op when the container is already stopped — the idle reaper
        calls this on every worker-loop pass once the idle clock has run out.
        """
        for container in self.vllm_containers:
            if container:
                self._stop_one_vllm_container(container)

    def _stop_one_vllm_container(self, container):
        if self._vllm_container_running(container) is False:
            return
        status, _ = self._docker_api("POST", f"/containers/{container}/stop?t=5")
        if status in (204, 304):
            print(f"[worker] stopped vLLM container '{container}' — "
                  f"VRAM fully freed", flush=True)
        elif status == 0:
            pass  # no Docker socket — silent no-op
        else:
            print(f"[worker] stop vLLM container returned {status}", flush=True)

    def _start_vllm_container(self):
        """Start independent GPU processes together and wait for all of them."""
        if self.engine != "vllm":
            return
        with ThreadPoolExecutor(max_workers=len(self.vllm_urls)) as pool:
            list(pool.map(lambda target: self._start_one_vllm_container(*target),
                          zip(self.vllm_containers, self.vllm_urls)))

    def _start_one_vllm_container(self, container, url):
        if not container:
            return  # external server; the parser checks /v1/models
        running = self._vllm_container_running(container)
        if running is None:
            return  # no Docker socket; the parser will report connectivity errors
        if not running:
            print(f"[worker] starting vLLM container '{container}'...", flush=True)
            status, body = self._docker_api("POST", f"/containers/{container}/start")
            if status not in (204, 304):
                raise RuntimeError(f"cannot start vLLM container {container}: HTTP {status}: {body!r}")
        import urllib.request as _url
        base = url.removesuffix("/v1").rstrip("/")
        health_url = base + "/health"
        deadline = time.monotonic() + self.vllm_startup_timeout
        last_error = "not ready"
        while time.monotonic() < deadline:
            if self._shutdown.is_set() or self.paused:
                raise InterruptedError("model startup stopped")
            try:
                with _url.urlopen(health_url, timeout=3) as r:
                    if r.status == 200:
                        print(f"[worker] vLLM container '{container}' ready", flush=True)
                        return
                    last_error = f"HTTP {r.status}"
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
            if self._vllm_container_running(container) is False:
                raise RuntimeError(f"vLLM container {container} exited during startup; "
                                   f"check docker logs {container}; {last_error}")
            self._shutdown.wait(min(2, max(0, deadline - time.monotonic())))
        raise TimeoutError(f"vLLM container {container} not ready at {health_url} "
                           f"after {self.vllm_startup_timeout:g}s: {last_error}")

    def _unload_model(self):
        # Gate on the capability, not on the engine name: a parser that can offload
        # its own weights is the thing that makes "unload" mean something here.
        # For vLLM the GPU belongs to the server, so dropping our HTTP client would
        # free nothing while the UI reported "stopped".
        parsers = self.parsers or ([self.parser] if self.parser else [])
        if parsers and all(callable(getattr(p, "sleep", None)) for p in parsers):
            def offload(parser):
                try:
                    parser.sleep()
                except Exception:
                    pass  # best-effort; container stop below is the real unload
            with ThreadPoolExecutor(max_workers=len(parsers)) as pool:
                list(pool.map(offload, parsers))
            # If we know the vLLM Docker container name, stop it outright.
            # Sleep-mode alone leaves ~700 MiB of CUDA context / runtime
            # resident; stopping the container frees ALL VRAM back to the OS.
            self._stop_vllm_container()
            self.model_state = "stopped"
            return

        parser = self.parser
        self.parser = None
        self.parsers = []
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
        # Self-healing outer loop. The inner `while` does the real work; any
        # Exception that escapes it (most painfully sqlite3.OperationalError
        # from db.has_queued_tasks() / db.claim_next_task() when the state
        # volume briefly refuses a write — a recurring incident in production)
        # used to kill this daemon thread silently. There is no other consumer
        # for tasks, so the queue froze until someone noticed and restarted the
        # container. We instead log, drop model state if it looks corrupt, and
        # keep looping. The persisted queue lives in SQLite, so nothing in
        # flight is lost; the next iteration will simply re-claim it.
        while not self._shutdown.is_set():
            try:
                self._run_loop_once()
            except Exception as error:  # noqa: BLE001 — never let the worker die
                traceback.print_exc()
                print(f"[worker] main loop caught {type(error).__name__}: "
                      f"{error}; backing off 10s and continuing", flush=True)
                # An OperationalError often means the SQLite file/handle is in a
                # bad state for this process right now; force the model state to
                # be re-evaluated from scratch on the next pass.
                try:
                    if self.model_state == "loaded":
                        self._unload_model()
                except Exception:  # noqa: BLE001
                    pass
                # Sleep without holding the wakeup lock so we don't busy-loop
                # on a persistent error (e.g. full disk).
                time.sleep(10.0)

    def _reap_idle_vllm_container(self):
        """Stop an idle vLLM container left running outside any load cycle.

        vLLM owns the GPU in a separate container with restart=unless-stopped,
        so a fresh deploy or a host reboot starts it with the weights resident
        even when nobody has queued work — and this worker, holding no parser
        in the `stopped` state, would never touch it. Reaping it on the same
        idle clock as a loaded model makes the service converge to zero VRAM
        from any state. No-op unless the engine is vLLM and the idle clock
        has run out; cheap otherwise (one GET over the docker socket, and
        _stop_vllm_container stays quiet once the container is down).
        """
        if (self.engine == "vllm"
                and time.time() - self._last_used > self._idle_limit()):
            self._stop_vllm_container()

    def _run_loop_once(self):
        """One pass of the worker loop. See run() for the recovery wrapper."""
        if self._reload_requested:
            self._reload_requested = False
            was_loaded = self.model_state == "loaded"
            self._unload_model()
            if was_loaded or self.keep_loaded:
                self._load_now = True  # reload on the new device

        if self.paused:
            if self.model_state == "loaded":
                self._unload_model()
            self._reap_idle_vllm_container()
            self._wait()
            return

        demand = self._load_now or db.has_queued_tasks()
        if demand and self.model_state in ("stopped", "error"):
            self._load_model()
        self._load_now = False

        if self.model_state != "loaded":
            if not demand:
                self._reap_idle_vllm_container()
            self._wait()
            return

        task = db.claim_next_task()
        if task is not None:
            self._run_task(task)
            self._last_used = time.time()
            return

        if time.time() - self._last_used > self._idle_limit():
            self._unload_model()
        self._wait()

    def _wait(self):
        self._wakeup.wait(timeout=1.0)
        self._wakeup.clear()

    # ------------------------------------------------------------ task

    def _ensure_vllm_awake(self):
        parsers = self.parsers or ([self.parser] if self.parser else [])
        if parsers:
            with ThreadPoolExecutor(max_workers=len(parsers)) as pool:
                list(pool.map(self._ensure_one_vllm_awake, parsers))

    @staticmethod
    def _ensure_one_vllm_awake(parser):
        """Wake vLLM if it's sleeping, blocking until the weights are resident.

        Cheap when vLLM is already awake (one short GET /is_sleeping → false,
        return immediately). Expensive but unavoidable when it's asleep:
        POST /wake_up blocks until vLLM has reloaded the weights into VRAM,
        which is exactly the latency we'd otherwise impose on the first page
        of the task as a confusing hang.

        All failures are swallowed: this helper must NEVER crash the task or
        push the worker into the `error` state. If the wake-up endpoint is
        unreachable, the in-flight request will fail on its own with a much
        more informative error from httpx.
        """
        is_sleeping = getattr(parser, "is_sleeping", None)
        wake = getattr(parser, "wake", None)
        if is_sleeping is None or wake is None:
            return  # in-process transformers engine — no concept of sleep
        try:
            sleeping = is_sleeping()
        except Exception as error:  # noqa: BLE001 — vLLM might be mid-load
            print(f"[worker] is_sleeping probe failed "
                  f"({type(error).__name__}: {error}); assuming awake",
                  flush=True)
            return
        if not sleeping:
            return
        try:
            print("[worker] vLLM is sleeping — waking it up before the task",
                  flush=True)
            wake()
            print("[worker] vLLM wake_up returned; weights should be resident",
                  flush=True)
        except Exception as error:  # noqa: BLE001 — log, don't crash the task
            traceback.print_exc()
            print(f"[worker] wake_up failed: {type(error).__name__}: {error}",
                  flush=True)

    def _run_task(self, task):
        self.current_task_id = task["id"]
        self.abort_event.clear()
        if self.paused:
            # request_stop() landed between the loop's paused check and the
            # claim: hand the task back to the queue instead of running it
            self.current_task_id = None
            db.update_task(task["id"], status="queued", started_at=None)
            return
        # Belt-and-suspenders wake-up. The run loop normally re-loads the
        # model (which calls parser.wake()) when it sees `model_state in
        # ("stopped","error")`. But two edge cases bypass that:
        #   1) an external script / docker restart put vLLM to sleep while
        #      the demo kept `model_state="loaded"`;
        #   2) the previous idle-unload's `sleep()` call returned before the
        #      offload was complete, and the demo's `model_state="stopped"`
        #      was set optimistically.
        # In both, the first request would hang for tens of seconds waiting
        # for vLLM to reload the weights on demand. Asking `is_sleeping()`
        # and explicitly waking here makes the latency the demo's problem to
        # report, not the user's to wonder about.
        self._ensure_vllm_awake()
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
            with self._live_lock:
                self._live_slots.clear()
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

    def _input_path(self, job, page=0):
        job_dir = self.jobs_dir / job["id"]
        if job["kind"] == "image":
            # multi-image jobs store one file per page (input_000.png, ...);
            # jobs created before that scheme have a single legacy input.png
            indexed = sorted(job_dir.glob(f"input_{page:03d}.*"))
            if indexed:
                return indexed[0]
        matches = sorted(job_dir.glob("input.*"))
        if not matches:
            raise FileNotFoundError(f"no input file for job {job['id']}")
        return matches[0]

    def _parse_page(self, parser, slot, task, job, page_no, out_dir):
        from dots_mocr.utils.doc_utils import load_pdf_pages
        from dots_mocr.utils.image_utils import fetch_image

        if self.abort_event.is_set():
            return None
        started = time.time()
        params = task["params"]
        prompt_mode = task["prompt_mode"]
        temperature = params.get("temperature")
        if temperature is None:
            temperature = default_temperature(prompt_mode)
        dpi = int(params.get("dpi") or self.dpi)
        save_name = f"task_{task['id']}"
        parser.max_completion_tokens = int(params.get("max_new_tokens") or self.max_completion_tokens)

        def on_generation(stats):
            with self._live_lock:
                self._live_slots[slot] = (page_no, stats)

        parser.generation_listener = on_generation
        backend = self.vllm_urls[slot] if self.engine == "vllm" else self.device
        print(f"[worker] task={task['id']} page={page_no} backend={backend} started", flush=True)
        if job["kind"] == "pdf":
            rendered = load_pdf_pages(str(self._input_path(job)), dpi=dpi, page_ids=[page_no])
            if not rendered:
                raise ValueError(f"page {page_no} did not render")
            origin_image = rendered[0][1]
        else:
            origin_image = fetch_image(str(self._input_path(job, page_no)))
        try:
            if self.abort_event.is_set():
                return None
            page_bbox = self._scale_bbox(params.get("bbox"), params.get("bbox_view_size"), origin_image)
            page_result = parser._parse_single_image(
                origin_image, prompt_mode, str(out_dir),
                save_name if job["kind"] == "pdf" else f"{save_name}_page_{page_no}",
                source=job["kind"], page_idx=page_no, bbox=page_bbox,
                fitz_preprocess=(job["kind"] == "image" and
                                 PROMPT_TO_FITZ_PREPROCESS.get(prompt_mode, False)),
                custom_prompt=params.get("custom_prompt") or None, temperature=temperature,
            )
            if self.abort_event.is_set() or (page_result.get("generation") or {}).get("aborted"):
                return None
            page_result["seconds"] = round(time.time() - started, 2)
            if self.engine == "vllm":
                page_result["backend_url"] = backend
            print(f"[worker] task={task['id']} page={page_no} backend={backend} finished", flush=True)
            return page_result
        finally:
            origin_image.close()

    def _execute(self, task):
        """Bounded page scheduler: one in-flight page per independent parser.

        The next free GPU takes the next page; slow pages do not hold up the
        other GPU. Only this coordinator writes task progress / results to DB.
        Futures are drained before returning, including on cancellation/error,
        so no page can keep writing artifacts after the task becomes terminal.
        """
        job = db.get_job(task["job_id"])
        if job is None:
            raise ValueError(f"job {task['job_id']} not found")
        out_dir = self.jobs_dir / job["id"] / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        pages = task["pages"] or [0]
        total = len(pages)
        parsers = self.parsers or [self.parser]
        completed = {}
        active = {}
        next_index = 0
        failure = None
        cancelled = False

        def ordered_results():
            return [completed[i] for i in sorted(completed)]

        def persist():
            progress = {"done": len(completed), "total": total}
            if active:
                current = sorted(active.values(), key=lambda item: item[1])
                progress.update(current_page=current[0][2], page_started_at=current[0][3])
                if len(parsers) > 1:
                    progress["active_pages"] = [item[2] for item in current]
            db.update_task(task["id"], result=ordered_results(), progress=progress)

        persist()
        with ThreadPoolExecutor(max_workers=len(parsers), thread_name_prefix="ocr-page") as pool:
            def dispatch(slot):
                nonlocal next_index
                if next_index >= total or self.abort_event.is_set():
                    return
                index = next_index
                next_index += 1
                page_no = pages[index]
                with self._live_lock:
                    self._live_slots[slot] = (page_no, None)
                future = pool.submit(self._parse_page, parsers[slot], slot,
                                     task, job, page_no, out_dir)
                active[future] = (slot, index, page_no, time.time())

            try:
                if db.is_cancel_requested(task["id"]) or self.abort_event.is_set():
                    cancelled = True
                    self.abort_event.set()
                else:
                    for slot in range(min(len(parsers), total)):
                        dispatch(slot)
                    persist()
                while active:
                    if db.is_cancel_requested(task["id"]) or self.abort_event.is_set():
                        cancelled = True
                        self.abort_event.set()
                    done, _ = wait(active, timeout=0.1, return_when=FIRST_COMPLETED)
                    free_slots = []
                    for future in done:
                        slot, index, page_no, _ = active.pop(future)
                        with self._live_lock:
                            self._live_slots.pop(slot, None)
                        try:
                            result = future.result()
                            if result is not None:
                                completed[index] = result
                            else:
                                cancelled = True
                                self.abort_event.set()
                        except Exception as error:
                            if failure is None:
                                failure = RuntimeError(f"page {page_no}, worker {slot}: {type(error).__name__}: {error}")
                            self.abort_event.set()
                        free_slots.append(slot)
                    if failure is None and not cancelled:
                        for slot in free_slots:
                            dispatch(slot)
                    if done:
                        persist()
            except BaseException:
                self.abort_event.set()
                raise
        if failure is not None:
            raise failure
        results = ordered_results()
        if cancelled or self.abort_event.is_set() or db.is_cancel_requested(task["id"]):
            db.update_task(task["id"], status="cancelled", result=results,
                           progress={"done": len(results), "total": total}, finished_at=time.time())
            return
        self._record_in_docstore(task, job, results)
        db.update_task(task["id"], status="done", result=results,
                       progress={"done": total, "total": total}, finished_at=time.time())

    @staticmethod
    def _record_in_docstore(task, job, results):
        """Cache the finished parse so an identical resubmission is a lookup.

        Only tasks that came in through the agent API carry a sha256, and only a
        complete result is worth caching — a cancelled or partial run would
        otherwise be served forever as the answer.
        """
        sha256 = (task.get("params") or {}).get("sha256")
        if not sha256 or not results:
            return
        from demo import docstore

        pieces = []
        tokens = 0
        seconds = 0.0
        for page in results:
            path = page.get("md_content_path")
            if path:
                try:
                    pieces.append(Path(path).read_text(encoding="utf-8"))
                except OSError:
                    continue
            tokens += (page.get("generation") or {}).get("generated_tokens") or 0
            seconds += page.get("seconds") or 0
        if not pieces:
            return
        try:
            docstore.store_result(
                sha256=sha256, prompt_mode=task["prompt_mode"], pages=task["pages"],
                task_id=task["id"], job_id=job["id"], markdown="\n\n".join(pieces),
                pages_done=len(results), generated_tokens=tokens,
                seconds=round(seconds, 2), filename=job.get("filename") or "")
        except Exception:  # noqa: BLE001 - caching must never fail a finished task
            traceback.print_exc()

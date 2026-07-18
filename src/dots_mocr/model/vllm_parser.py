"""Drive a vLLM server through the same interface as the local parser.

Everything that turns a model response into artifacts — layout JSON, markdown,
picture crops, the bbox overlay — lives in DotsMOCRParser._parse_single_image and
has nothing to do with how the tokens were produced. So the vLLM engine is a
subclass that replaces exactly two things: loading (there is no local model, only
a server to reach) and generation (an HTTP call instead of model.generate).

That keeps the demo, the artifact layout and the task queue identical between
engines, so switching DEMO_ENGINE changes the inference path and nothing else.

Streaming is used rather than a single response because the demo shows live
tokens/s: each streamed chunk ticks the same GenerationStats the local path fills
from its stopping criteria, so the UI cannot tell which engine it is watching.
"""

import base64
import io
import json

from dots_mocr.cli import DotsMOCRParser
from dots_mocr.utils.generation_stats import GenerationStats


class VllmUnavailable(RuntimeError):
    """The server is not reachable — surfaced in the demo as a model error."""


class VllmDotsMOCRParser(DotsMOCRParser):
    """DotsMOCRParser that generates through a vLLM OpenAI-compatible endpoint."""

    def __init__(self, *args, vllm_url="http://127.0.0.1:8000/v1",
                 vllm_model="rednote-hilab/dots.mocr", request_timeout=900.0, **kwargs):
        self.vllm_url = vllm_url.rstrip("/")
        self.vllm_model = vllm_model
        self.request_timeout = request_timeout
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------ loading

    def _load_model(self, ckpt):
        """No weights here — just prove the server is up and serving this model.

        Checking at load time rather than on the first page means a misconfigured
        URL shows up as a model error in the UI, next to the load button, instead
        of as a failed task minutes later.
        """
        import httpx

        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.get(f"{self.vllm_url}/models")
                response.raise_for_status()
                models = response.json().get("data", [])
        except Exception as error:  # noqa: BLE001 - any failure means unusable
            raise VllmUnavailable(
                f"vLLM is not reachable at {self.vllm_url}: {type(error).__name__}: {error}"
            ) from error

        served = [m["id"] for m in models]
        if self.vllm_model not in served:
            raise VllmUnavailable(
                f"vLLM at {self.vllm_url} serves {served}, not {self.vllm_model!r}. "
                "Start it with --served-model-name, or set DEMO_VLLM_MODEL.")

        # The context window bounds prompt + output together, so the per-request
        # budget is computed per image in _output_budget rather than clamped once.
        self.max_model_len = next(
            (m.get("max_model_len") for m in models if m["id"] == self.vllm_model), None)

        self.model = None
        self.processor = None
        self.served_models = served
        print(f"vLLM engine ready: {self.vllm_url}, model {self.vllm_model}, "
              f"max_model_len={self.max_model_len}")

    def _resolve_device(self, device):
        # The GPU belongs to the vLLM process; nothing is placed here.
        return "vllm"

    def _resolve_dtype(self, dtype):
        return None

    # ------------------------------------------------------------ generation

    # Text of the prompt plus the chat template; measured at ~216 tokens, rounded
    # up so a longer custom prompt still fits.
    _PROMPT_TEXT_RESERVE = 512
    _MIN_OUTPUT_TOKENS = 256

    def _output_budget(self, image):
        """How many output tokens may be requested for this image.

        vLLM rejects a request whose prompt + output exceeds the context window,
        so the image has to be paid for before asking for output. One language
        model token covers a 28x28 px patch (patch_size 14, merged 2x2), which
        makes the prompt size exactly computable from the resized image rather
        than guessed.
        """
        budget = self.max_completion_tokens
        max_model_len = getattr(self, "max_model_len", None)
        if not max_model_len:
            return budget

        from dots_mocr.utils.image_utils import smart_resize

        height, width = smart_resize(image.height, image.width)
        pixels_per_token = 14 * 2
        image_tokens = (height // pixels_per_token) * (width // pixels_per_token)
        room = max_model_len - image_tokens - self._PROMPT_TEXT_RESERVE
        if room < budget:
            budget = max(room, self._MIN_OUTPUT_TOKENS)
        return budget

    @staticmethod
    def _image_data_url(image):
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def _inference(self, image, prompt, temperature=None, stats=None):
        import httpx

        temperature = self.temperature if temperature is None else temperature
        max_tokens = self._output_budget(image)
        body = {
            "model": self.vllm_model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": self._image_data_url(image)}},
                # the explicit image tags are what upstream's vLLM example sends;
                # without them vLLM v1 substitutes a newline in their place
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"},
            ]}],
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature > 0:
            body["top_p"] = self.top_p

        if stats is not None:
            stats.start()
        chunks = []
        usage = None
        aborted = False

        with httpx.Client(timeout=self.request_timeout) as client:
            with client.stream("POST", f"{self.vllm_url}/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    # the body carries the only useful part (which parameter, what
                    # limit); a bare "400 Bad Request" in the UI is undiagnosable
                    response.read()
                    detail = response.text.strip()
                    try:
                        detail = json.loads(detail)["error"]["message"]
                    except (ValueError, KeyError, TypeError):
                        pass
                    raise RuntimeError(f"vLLM {response.status_code}: {detail[:400]}")
                for line in response.iter_lines():
                    if self.abort_event is not None and self.abort_event.is_set():
                        # closing the stream cancels the request server-side, which
                        # is what makes the demo's stop button work against vLLM too
                        aborted = True
                        break
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    event = json.loads(payload)
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            chunks.append(piece)
                            if stats is not None:
                                stats.record_token()

        if stats is not None:
            generated = usage.get("completion_tokens") if usage else None
            stats.finish(generated_tokens=generated, aborted=aborted)
        return "".join(chunks)

    # ------------------------------------------------------------ demo surface

    def warm_up(self):
        """Nothing to warm: the server owns the weights and its own compilation."""
        return True

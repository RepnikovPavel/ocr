"""Drive a GLM-OCR vLLM server through the same demo interface as dots.mocr.

GLM-OCR (`zai-org/GLM-OCR`, 0.9B) is served by a *different* vLLM build than
dots.mocr: it is a native `glm_ocr` architecture in transformers 5.x, so it
needs a current vLLM (>=0.19), whereas dots.mocr's `dots_ocr`+auto_map loads
only on the older 0.17 build the repo already ships. The two cannot share one
server, so on a 2x4090 box we keep them hot on separate GPUs and the demo
switches between them with a model selector (see demo/models.py).

Only three things differ from `VllmDotsMOCRParser`, and exactly those are
overridden here:

  * loading  — same `/v1/models` handshake, different served-model default
  * generation — the request body. dots.mocr wants the upstream quirk of
    explicit `<|img|><|imgpad|><|endofimg|>` tokens; GLM-OCR's chat template
    injects its own `<|begin_of_image|>` from the `image_url`, so the standard
    OpenAI `image_url` content part is all that is needed.
  * token budget — patch geometry is the same (14px, merged 2x2 => 28px/token),
    but the context window is far larger, so the cap rarely bites.

Everything that turns a response into artifacts is inherited unchanged from
`DotsMOCRParser._parse_single_image` — the GLM trigger modes route through its
`else` branch (raw markdown), reusing the same save/preview plumbing.
"""

from dots_mocr.model.vllm_parser import VllmDotsMOCRParser


class GlmOcrVllmParser(VllmDotsMOCRParser):
    """VllmDotsMOCRParser tuned for GLM-OCR's native chat template."""

    def __init__(self, *args, vllm_model="glm-ocr", **kwargs):
        # The served-model name the GLM-OCR vLLM is started with
        # (--served-model-name=glm-ocr). Overridable per-instance for tests.
        super().__init__(*args, vllm_model=vllm_model, **kwargs)

    def _inference(self, image, prompt, temperature=None, stats=None):
        """Same streaming chat-completion as the parent, but with GLM's content.

        The only material change vs VllmDotsMOCRParser._inference is the user
        message: no hand-written image tokens. GLM-OCR's chat template wraps an
        `image_url` content part in `<|begin_of_image|><|image|><|end_of_image|>`
        itself, and vLLM replaces that with the right number of image patches.
        Sending dots.mocr's `<|img|><|imgpad|>` here would be literal junk text
        to GLM-OCR.
        """
        import httpx
        import json

        temperature = self.temperature if temperature is None else temperature
        max_tokens = self._output_budget(image)
        body = {
            "model": self.vllm_model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": self._image_data_url(image)}},
                {"type": "text", "text": prompt},
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
                    response.read()
                    detail = response.text.strip()
                    try:
                        detail = json.loads(detail)["error"]["message"]
                    except (ValueError, KeyError, TypeError):
                        pass
                    raise RuntimeError(f"vLLM {response.status_code}: {detail[:400]}")
                for line in response.iter_lines():
                    if self.abort_event is not None and self.abort_event.is_set():
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

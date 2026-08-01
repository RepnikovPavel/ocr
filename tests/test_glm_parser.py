"""GLM-OCR vLLM parser: the request shape differs from dots.mocr's.

Each case here pins one difference that would silently break the A/B comparison:
  * the served-model default is glm-ocr (not rednote-hilab/dots.mocr)
  * the user content is a plain image_url + text — NO dots.mocr image tokens
    (<|img|><|imgpad|>), which GLM's chat template would read as junk text
  * the streaming/abort/usage path is inherited unchanged from the parent

All without a vLLM server, the way test_vllm_parser.py fakes httpx.
"""

import json

import pytest
from PIL import Image

from dots_mocr.model.glm_parser import GlmOcrVllmParser
from dots_mocr.model.vllm_parser import VllmUnavailable


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text else (json.dumps(payload) if payload is not None else "")

    def json(self):
        return self._payload

    def read(self):
        return self.text.encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeClient:
    def __init__(self, get=None, post=None):
        self._get, self._post = get, post
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self._get

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self._post


def install_client(monkeypatch, client):
    import httpx
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: client)
    return client


def make_parser(monkeypatch, max_model_len=32768, max_completion_tokens=16384):
    """A GLM parser whose load-time handshake is answered by a fake server."""
    models = {"data": [{"id": "glm-ocr", "max_model_len": max_model_len}]}
    install_client(monkeypatch, FakeClient(get=FakeResponse(payload=models)))
    return GlmOcrVllmParser(
        ckpt="/nonexistent", max_completion_tokens=max_completion_tokens,
        device="vllm", dtype="auto")


# ---------------------------------------------------------------- handshake

def test_default_served_model_is_glm_ocr(monkeypatch):
    install_client(monkeypatch, FakeClient(
        get=FakeResponse(payload={"data": [{"id": "glm-ocr"}]})))
    parser = GlmOcrVllmParser(ckpt="/nonexistent", device="vllm", dtype="auto")
    assert parser.vllm_model == "glm-ocr"


def test_load_rejects_a_server_not_serving_glm_ocr(monkeypatch):
    install_client(monkeypatch, FakeClient(
        get=FakeResponse(payload={"data": [{"id": "rednote-hilab/dots.mocr"}]})))
    with pytest.raises(VllmUnavailable, match="not 'glm-ocr'"):
        GlmOcrVllmParser(ckpt="/nonexistent", device="vllm", dtype="auto")


def test_records_glm_context_window(monkeypatch):
    # GLM-OCR's 131072 max_position_embeddings dwarfs dots.mocr's; the budget
    # math must read whatever the server actually reports, not a hard-coded cap.
    parser = make_parser(monkeypatch, max_model_len=131072)
    assert parser.max_model_len == 131072


# ---------------------------------------------------------------- request shape

class StreamingClient(FakeClient):
    """Replays a streamed chat completion the way vLLM sends it (SSE lines)."""

    def __init__(self, chunks, status_code=200):
        self._chunks = chunks
        self._status = status_code
        super().__init__()

    def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        client = self
        lines = []
        for piece, usage in client._chunks:
            delta = {"choices": [{"delta": {"content": piece}}]}
            if usage is not None:
                delta["usage"] = usage
            lines.append("data: " + json.dumps(delta))
        lines.append("data: [DONE]")
        body = "\n".join(lines)

        class Ctx:
            status_code = client._status
            text = body

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def iter_lines(self):
                return iter(body.split("\n"))

        return Ctx()


def test_inference_sends_plain_image_url_not_dots_tokens(monkeypatch):
    """GLM's chat template injects its own image tokens from image_url.

    dots.mocr's parser prepends `<|img|><|imgpad|><|endofimg|>` to the text part;
    sending those to GLM-OCR would be literal junk. This pins that the GLM body
    carries only the standard image_url + text content parts.
    """
    parser = make_parser(monkeypatch)
    client = install_client(monkeypatch, StreamingClient([("hello", None),
                                                          (" world",
                                                           {"completion_tokens": 2})]))
    parser._inference(Image.new("RGB", (560, 560)), "Text Recognition:")

    assert len(client.calls) == 1
    _, _, kwargs = client.calls[0]
    # httpx is called with json=<dict>, so kwargs["json"] is already the body
    body = kwargs["json"]
    content = body["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image_url", "text"]
    # the whole point: no hand-written image tokens in the text part
    assert content[1]["text"] == "Text Recognition:"
    assert "<|img" not in json.dumps(body)
    assert body["model"] == "glm-ocr"


def test_inference_streams_tokens_and_finishes_stats(monkeypatch):
    parser = make_parser(monkeypatch)
    install_client(monkeypatch, StreamingClient(
        [("foo ", None), ("bar", {"completion_tokens": 2})]))

    from dots_mocr.utils.generation_stats import GenerationStats
    stats = GenerationStats()
    out = parser._inference(Image.new("RGB", (560, 560)), "Text Recognition:",
                            stats=stats)
    assert out == "foo bar"
    assert stats.generated_tokens == 2     # authoritative count from usage
    assert stats.aborted is False


def test_http_error_surfaces_server_message(monkeypatch):
    parser = make_parser(monkeypatch)
    install_client(monkeypatch, _ErrorClient())
    with pytest.raises(RuntimeError, match="vLLM 400"):
        parser._inference(Image.new("RGB", (560, 560)), "Text Recognition:")


class _ErrorClient(FakeClient):
    def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))

        class Ctx:
            status_code = 400
            text = json.dumps({"error": {"message": "bad image"}})

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):  # parser calls this to surface the error body
                return self.text.encode()

        return Ctx()

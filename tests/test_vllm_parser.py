"""The vLLM engine, without a vLLM server.

Everything here is the logic that stands between the demo and the server, and
each case is one that actually went wrong while wiring it up: a token budget that
ignored the image and got a 400, an error that arrived as a bare status code with
the useful part discarded, and a /v1 suffix that has to come off before the
dev endpoints are reachable.
"""

import json

import pytest
from PIL import Image

from dots_mocr.model.vllm_parser import VllmDotsMOCRParser, VllmUnavailable


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
    """Records requests and replays canned responses."""

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


def make_parser(monkeypatch, max_model_len=16384, max_completion_tokens=16384):
    """A parser whose load-time handshake is answered by a fake server."""
    models = {"data": [{"id": "rednote-hilab/dots.mocr", "max_model_len": max_model_len}]}
    install_client(monkeypatch, FakeClient(get=FakeResponse(payload=models)))
    return VllmDotsMOCRParser(
        ckpt="/nonexistent", max_completion_tokens=max_completion_tokens,
        device="vllm", dtype="auto")


# ---------------------------------------------------------------- handshake

def test_load_rejects_a_server_that_does_not_serve_this_model(monkeypatch):
    install_client(monkeypatch, FakeClient(get=FakeResponse(payload={"data": [{"id": "other"}]})))
    with pytest.raises(VllmUnavailable, match="not 'rednote-hilab/dots.mocr'"):
        VllmDotsMOCRParser(ckpt="/nonexistent", device="vllm", dtype="auto")


def test_load_reports_an_unreachable_server_with_its_address(monkeypatch):
    import httpx

    def explode(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(httpx, "Client", explode)
    with pytest.raises(VllmUnavailable, match="127.0.0.1:8000"):
        VllmDotsMOCRParser(ckpt="/nonexistent", device="vllm", dtype="auto")


def test_load_records_the_context_window(monkeypatch):
    parser = make_parser(monkeypatch, max_model_len=12288)
    assert parser.max_model_len == 12288
    assert parser.model is None, "no weights are held in this process"


# ---------------------------------------------------------------- token budget

def test_output_budget_pays_for_the_image_first(monkeypatch):
    """vLLM bounds prompt + output together, so the image has to be subtracted.

    A 1288x1652 page is 2.13 Mpx; at one token per 28x28 px that is 2714 tokens,
    which must come out of a 16384 window before any output is requested.
    """
    parser = make_parser(monkeypatch, max_model_len=16384)
    budget = parser._output_budget(Image.new("RGB", (1288, 1652)))
    assert budget < 16384 - 2700, "the image was not paid for"
    assert budget > 12000, f"budget {budget} is implausibly small for this page"


def test_output_budget_shrinks_as_the_page_grows(monkeypatch):
    parser = make_parser(monkeypatch, max_model_len=16384)
    small = parser._output_budget(Image.new("RGB", (616, 784)))
    large = parser._output_budget(Image.new("RGB", (1288, 1652)))
    assert small > large


def test_output_budget_never_goes_below_the_floor(monkeypatch):
    """A page that fills the window must still ask for something, not zero or a
    negative count, which the server rejects outright."""
    parser = make_parser(monkeypatch, max_model_len=2048)
    budget = parser._output_budget(Image.new("RGB", (1288, 1652)))
    assert budget == VllmDotsMOCRParser._MIN_OUTPUT_TOKENS


def test_output_budget_is_the_request_when_the_server_reports_no_window(monkeypatch):
    parser = make_parser(monkeypatch, max_completion_tokens=4096)
    parser.max_model_len = None
    assert parser._output_budget(Image.new("RGB", (616, 784))) == 4096


# ---------------------------------------------------------------- error surface

def test_http_errors_carry_the_server_message(monkeypatch):
    """A bare '400 Bad Request' in the UI is undiagnosable; the body names the
    parameter and the limit."""
    parser = make_parser(monkeypatch)
    body = json.dumps({"error": {"message": "max_completion_tokens=16384 cannot be greater "
                                            "than max_model_len=12288"}})

    class StreamingClient(FakeClient):
        def stream(self, method, url, **kwargs):
            client = self

            class Ctx:
                def __enter__(self):
                    client.calls.append((method, url, kwargs))
                    return FakeResponse(status_code=400, text=body)

                def __exit__(self, *exc):
                    return False

            return Ctx()

    install_client(monkeypatch, StreamingClient())
    with pytest.raises(RuntimeError, match="max_model_len=12288"):
        parser._inference(Image.new("RGB", (560, 560)), "prompt")


# ---------------------------------------------------------------- sleep / wake

def test_sleep_and_wake_target_the_root_not_the_v1_path(monkeypatch):
    """The dev endpoints live beside /v1, not under it."""
    parser = make_parser(monkeypatch)
    client = install_client(monkeypatch, FakeClient(post=FakeResponse()))
    parser.sleep()
    parser.wake()
    urls = [url for _, url, _ in client.calls]
    assert urls == ["http://127.0.0.1:8000/sleep", "http://127.0.0.1:8000/wake_up"]
    assert client.calls[0][2]["params"] == {"level": 1}


def test_missing_dev_endpoints_name_the_flags_that_enable_them(monkeypatch):
    """Without --enable-sleep-mode the route 404s; the failure has to say so
    rather than let the UI claim the GPU was freed."""
    parser = make_parser(monkeypatch)
    install_client(monkeypatch, FakeClient(post=FakeResponse(status_code=404)))
    with pytest.raises(VllmUnavailable, match="enable-sleep-mode"):
        parser.sleep()


# ---------------------------------------------------------------- request shape

def test_image_is_sent_as_a_png_data_url():
    url = VllmDotsMOCRParser._image_data_url(Image.new("RGB", (8, 8), "white"))
    assert url.startswith("data:image/png;base64,")
    assert len(url) > 40

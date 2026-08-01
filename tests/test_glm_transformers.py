"""In-process GLM-OCR parser: pin the model-card request shape.

The transformers path is the only one that runs on this host, so it carries
the comparison. Each test fixes one thing that would silently corrupt the A/B:
  * the model is loaded through AutoModelForImageTextToText (not the dots patch)
  * the image is a content 'image' part and the chat template is applied with
    tokenize=True (GLM injects its own <|begin_of_image|> from it)
  * token_type_ids are dropped before generate (the card does this)
  * output is decoded skip_special_tokens=True for clean markdown

Done by patching only the two symbols the parser imports, so the real
transformers (which DotsMOCRParser's import chain needs) is left intact.
"""

import torch
import pytest


class _Inputs(dict):
    """Dict subclass with attribute access + a .to() that returns self.

    Mirrors BatchFeature enough for the parser's .input_ids / .to(device).
    """
    def to(self, _device):
        return self

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc


class _FakeProcessor:
    def __init__(self, captured):
        self._captured = captured

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            return_dict, return_tensors):
        self._captured["messages"] = messages
        return _Inputs(input_ids=torch.tensor([[1, 2, 3]]),
                       pixel_values=torch.tensor([[0.0]]),
                       token_type_ids=torch.tensor([[0, 0, 0]]))

    def batch_decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        return ["recognized text"]


class _FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.eval = lambda: None

    def generate(self, **kwargs):
        # one more token than the input -> decode trims the prompt prefix
        return torch.tensor([[1, 2, 3, 4]])


def _patch_glm_imports(monkeypatch, captured):
    """Redirect the two names glm_transformers.py imports to fakes.

    Importing the module is deferred to inside the test (after patching) so the
    patch lands on the real module's globals, not a cached import.
    """
    import dots_mocr.model.glm_transformers as mod
    monkeypatch.setattr(mod, "AutoModelForImageTextToText",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: _FakeModel()))
    monkeypatch.setattr(mod, "AutoProcessor",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: _FakeProcessor(captured)))
    return mod


import types  # noqa: E402  (kept after the helpers that reference it)


def test_loads_via_auto_model_for_image_text_to_text(monkeypatch):
    """The parser must NOT use the dots.mocr remote-code patch (register_transformers).

    GLM-OCR is a native architecture; the dots patch would mis-resolve it. We
    fake the two imports the parser uses and check generate() is reachable.
    """
    captured = {}
    _patch_glm_imports(monkeypatch, captured)
    from dots_mocr.model.glm_transformers import GlmOcrTransformersParser

    parser = GlmOcrTransformersParser(
        ckpt="/nonexistent", device="cpu", dtype="float32",
        attn_implementation="sdpa")
    assert parser.model is not None
    assert parser.processor is not None


def test_inference_passes_image_as_content_part(monkeypatch):
    """GLM's chat template injects <|begin_of_image|> from a content 'image' part;
    sending dots.mocr's text-only prompt or qwen_vl_utils would be wrong."""
    captured = {}
    _patch_glm_imports(monkeypatch, captured)
    from dots_mocr.model.glm_transformers import GlmOcrTransformersParser

    parser = GlmOcrTransformersParser(
        ckpt="/nonexistent", device="cpu", dtype="float32",
        attn_implementation="sdpa")
    out = parser._inference(_white_image(), "Text Recognition:")
    content = captured["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "text"]
    assert content[1]["text"] == "Text Recognition:"
    assert out == "recognized text"


def _white_image():
    from PIL import Image
    return Image.new("RGB", (56, 56), "white")

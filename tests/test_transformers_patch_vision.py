import pytest
import torch

from dots_mocr.transformers_patch import DotsOCRConfig, DotsOCRForCausalLM

from test_transformers_patch import multimodal_inputs, tiny_config


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return DotsOCRForCausalLM(tiny_config()).eval()


def test_pixel_values_change_logits(tiny_model):
    """The vision tower must be wired into the language model embeddings."""
    inputs = multimodal_inputs(4)
    with torch.inference_mode():
        base = tiny_model(**inputs).logits
        torch.manual_seed(123)
        inputs_changed = dict(inputs)
        inputs_changed["pixel_values"] = torch.randn_like(inputs["pixel_values"]) * 3
        changed = tiny_model(**inputs_changed).logits
    assert not torch.allclose(base, changed)


def test_text_only_forward_without_pixel_values(tiny_model):
    input_ids = torch.tensor([[1, 4, 5, 6]])
    with torch.inference_mode():
        out = tiny_model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    assert out.logits.shape == (1, 4, tiny_model.config.vocab_size)


def test_greedy_generate_is_deterministic(tiny_model):
    inputs = multimodal_inputs(4)
    with torch.inference_mode():
        first = tiny_model.generate(**inputs, do_sample=False, max_new_tokens=4)
        second = tiny_model.generate(**inputs, do_sample=False, max_new_tokens=4)
    assert torch.equal(first, second)


def test_image_embeddings_are_scattered_at_image_positions(tiny_model):
    inputs = multimodal_inputs(4)
    img_mask = inputs["input_ids"] == tiny_model.config.image_token_id
    with torch.inference_mode():
        embeds = tiny_model.prepare_inputs_embeds(
            inputs["input_ids"], inputs["pixel_values"], inputs["image_grid_thw"], img_mask,
        )
        vision = tiny_model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])
    torch.testing.assert_close(embeds[img_mask], vision.to(embeds.dtype))
    text_embeds = tiny_model.get_input_embeddings()(inputs["input_ids"])
    torch.testing.assert_close(embeds[~img_mask], text_embeds[~img_mask])


def test_prepare_inputs_for_generation_drops_pixels_after_first_step(tiny_model):
    inputs = multimodal_inputs(4)
    first = tiny_model.prepare_inputs_for_generation(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        is_first_iteration=True,
    )
    assert first.get("pixel_values") is not None
    later = tiny_model.prepare_inputs_for_generation(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        is_first_iteration=False,
    )
    assert "pixel_values" not in later
    assert "image_grid_thw" not in later


def test_vision_tower_output_dtype_and_shape(tiny_model):
    inputs = multimodal_inputs(4)
    with torch.inference_mode():
        vision = tiny_model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])
    # 4x4 grid with spatial_merge_size=2 -> 4 merged tokens
    assert vision.shape == (4, tiny_model.config.vision_config.hidden_size)
    assert torch.isfinite(vision).all()


def test_config_roundtrip_preserves_vision_config():
    config = tiny_config()
    data = config.to_dict()
    rebuilt = DotsOCRConfig(**{k: v for k, v in data.items() if k != "transformers_version"})
    assert rebuilt.vision_config.embed_dim == config.vision_config.embed_dim
    assert rebuilt.image_token_id == config.image_token_id

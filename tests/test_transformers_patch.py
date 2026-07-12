import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING

from dots_mocr.transformers_patch import (
    DotsOCRConfig,
    DotsOCRForCausalLM,
    DotsVLProcessor,
    register_transformers,
)


def tiny_config():
    return DotsOCRConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        image_token_id=63,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        vision_config={
            "embed_dim": 16,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "attn_implementation": "eager",
        },
    )


def multimodal_inputs(image_token_count):
    side = 2 if image_token_count == 1 else 4
    input_ids = torch.tensor([[1, *([63] * image_token_count), 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.tensor([[0, *([1] * image_token_count), 0, 0]])
    image_grid_thw = torch.tensor([[1, side, side]])
    pixel_values = torch.randn(side * side, 12)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "image_grid_thw": image_grid_thw,
        "pixel_values": pixel_values,
    }


def test_transformers_registries():
    register_transformers()
    register_transformers()
    config = AutoConfig.for_model(
        "dots_ocr",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vision_config=tiny_config().vision_config.to_dict(),
    )
    model = AutoModelForCausalLM.from_config(config)
    assert isinstance(config, DotsOCRConfig)
    assert isinstance(model, DotsOCRForCausalLM)
    assert PROCESSOR_MAPPING[type(config)] is DotsVLProcessor


@pytest.mark.parametrize("image_token_count", [1, 4])
def test_multimodal_forward_and_generate(image_token_count):
    torch.manual_seed(0)
    model = DotsOCRForCausalLM(tiny_config()).eval()
    inputs = multimodal_inputs(image_token_count)
    with torch.inference_mode():
        output = model(**inputs)
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=2,
            use_cache=True,
        )
    assert output.logits.shape == (*inputs["input_ids"].shape, model.config.vocab_size)
    assert generated.shape == (1, inputs["input_ids"].shape[1] + 2)


def test_save_pretrained_roundtrip_preserves_linear_weights_and_rotary(tmp_path):
    register_transformers()
    model = DotsOCRForCausalLM(tiny_config()).eval()
    names = (
        "vision_tower.blocks.0.attn.qkv.weight",
        "vision_tower.merger.mlp.2.weight",
        "model.layers.0.mlp.down_proj.weight",
    )
    parameters = dict(model.named_parameters())
    expected = {}
    with torch.no_grad():
        for index, name in enumerate(names, start=1):
            value = torch.linspace(
                -index / 3,
                index / 5,
                parameters[name].numel(),
                dtype=parameters[name].dtype,
            ).reshape_as(parameters[name])
            parameters[name].copy_(value)
            expected[name] = value.clone()
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True)
    loaded_parameters = dict(loaded.named_parameters())
    for name, value in expected.items():
        assert torch.equal(loaded_parameters[name], value)
    rotary = loaded.vision_tower.rotary_pos_emb
    assert not rotary.inv_freq.is_meta
    expected_inv_freq = 1.0 / (
        rotary.theta
        ** (torch.arange(0, rotary.dim, 2, dtype=rotary.inv_freq.dtype) / rotary.dim)
    )
    torch.testing.assert_close(rotary.inv_freq.cpu(), expected_inv_freq)
    assert torch.isfinite(rotary(8)).all()

"""Regression tests for the flex_attention vision backend.

The contract is that switching the vision tower to flex_attention does not change
the answer. sdpa is kept only as the reference these tests compare against — it is
no longer a runtime fallback, so an unsupported configuration must raise rather
than quietly produce sdpa numbers. Several tests below assert exactly that, because
a silent demotion is what would make the equivalence tests pass for the wrong
reason (comparing sdpa against sdpa).

Three levels:
  * the mask     — _flex_segment_ids must induce exactly the block-diagonal
                   predicate the sdpa bool mask paints (exact, integer);
  * the tower    — flex vs sdpa output, in float32 where the two agree to ~1e-4;
                   in bfloat16 they do not agree after 42 residual layers, and
                   neither do the two backends that already shipped;
  * the wiring   — flex is the default, and must NOT reach the language model,
                   where it is measurably slower.

CPU coverage is real coverage: flex runs eagerly there but through the same
mask_mod / BlockMask code, so it catches mask bugs. It cannot catch kernel
scheduling bugs, which is why the cuda tests use the real 1536/12-head geometry.
"""

import pytest
import torch

from dots_mocr.transformers_patch import DotsOCRConfig, DotsOCRForCausalLM
from dots_mocr.transformers_patch.modeling_dots_vision import (
    DOTS_VISION_ATTENTION_CLASSES,
    VisionFlexAttention,
    VisionSdpaAttention,
    _block_diag_bool_mask,
    _flex_block_mask,
    _flex_segment_ids,
    reset_vision_flex_state,
    warmup_vision_flex,
)

from test_transformers_patch import tiny_config

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")

# Real vision geometry: embed_dim 1536 over 12 heads = head_dim 128. The tiny
# config's head_dim of 8 never exercises the inductor kernel.
REAL_HEADS, REAL_HEAD_DIM = 12, 128


def vision_config(attn_implementation):
    config = tiny_config()
    config.vision_config.attn_implementation = attn_implementation
    return config


def paired_towers(dtype=torch.float32, num_hidden_layers=None):
    """Two vision towers, identical weights, differing only in the backend."""
    def build(backend):
        config = vision_config(backend)
        if num_hidden_layers is not None:
            config.vision_config.num_hidden_layers = num_hidden_layers
        return config

    torch.manual_seed(0)
    sdpa = DotsOCRForCausalLM(build("sdpa")).eval().vision_tower.to(dtype)
    flex = DotsOCRForCausalLM(build("flex_attention")).eval().vision_tower.to(dtype)
    flex.load_state_dict(sdpa.state_dict())
    return sdpa, flex


def tower_inputs(grids, dtype=torch.float32):
    """pixel_values / grid_thw for a list of (t, h, w) patch grids."""
    grid_thw = torch.tensor(grids)
    num_patches = int(sum(t * h * w for t, h, w in grids))
    torch.manual_seed(1)
    # patch_size 2, temporal_patch_size 1, 3 channels -> 3*1*2*2 = 12 features
    return torch.randn(num_patches, 12, dtype=dtype), grid_thw


# ---------------------------------------------------------------- mask identity

@pytest.mark.parametrize("bounds", [
    [0, 16],              # one segment
    [0, 4, 16],           # two segments
    [0, 0, 4, 16],        # leading empty segment
    [0, 4, 16, 16],       # trailing empty segment
    [0, 5, 9, 16],        # three uneven segments
])
def test_segment_ids_induce_the_sdpa_bool_mask(bounds):
    """seg[i] == seg[j] must equal the block-diagonal mask, exactly.

    Integer predicate on both sides, so this is torch.equal and not assert_close:
    any difference here is a wrong mask, never rounding.
    """
    seg = _flex_segment_ids(bounds, 16, "cpu")
    from_segments = seg[:, None] == seg[None, :]
    from_loop = _block_diag_bool_mask(torch.tensor(bounds), 16, "cpu")[0]
    assert torch.equal(from_segments, from_loop)


@pytest.mark.parametrize("bounds", [[0, 8], [0, 20], [4, 16], [0, 12, 8, 16], [16]])
def test_block_mask_rejects_cu_seqlens_that_do_not_partition(bounds):
    """Uncovered or overlapping tokens leave rows fully masked, where attention is
    undefined — that is a caller bug and must surface as one."""
    reset_vision_flex_state()
    with pytest.raises(ValueError, match="partition"):
        _flex_block_mask(torch.tensor(bounds), 16, torch.device("cpu"))


# ---------------------------------------------------------------- equivalence

def test_flex_matches_sdpa_single_image():
    sdpa, flex = paired_towers()
    pixel_values, grid_thw = tower_inputs([(1, 4, 4)])
    with torch.inference_mode():
        expected = sdpa(pixel_values, grid_thw)
        actual = flex(pixel_values, grid_thw)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_flex_matches_sdpa_multi_segment():
    """Two images in one batch -> cu_seqlens = [0, 16, 20].

    The only equivalence test that exercises the block-diagonal masking at all:
    with a single image the mask is all-ones and any mask_mod would pass.
    """
    sdpa, flex = paired_towers()
    pixel_values, grid_thw = tower_inputs([(1, 4, 4), (1, 2, 2)])
    with torch.inference_mode():
        expected = sdpa(pixel_values, grid_thw)
        actual = flex(pixel_values, grid_thw)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_flex_matches_sdpa_through_a_deep_tower():
    """Stack several vision layers so a small per-layer mask error accumulates.

    A single layer can hide a subtly wrong mask inside rounding; residual layers
    amplify it. Run in float32, where the backends agree to ~1e-4 relative — in
    bfloat16 they do not agree at all after 42 layers, and neither do the two
    backends that already shipped (measured on the real checkpoint: eager vs sdpa
    0.38 relative, sdpa vs flex 0.46), which is why parity is asserted in fp32.
    """
    sdpa, flex = paired_towers(num_hidden_layers=8)
    pixel_values, grid_thw = tower_inputs([(1, 4, 4), (1, 2, 2)])
    with torch.inference_mode():
        expected = sdpa(pixel_values, grid_thw)
        actual = flex(pixel_values, grid_thw)
    relative = ((actual - expected).abs().max() / expected.abs().max()).item()
    assert relative < 1e-4, f"deep tower drift {relative:.2e}"


def test_block_mask_is_cached_per_page_not_rebuilt_per_layer():
    """The tower hands the same cu_seqlens to all 42 layers; rebuilding the mask
    each time costs tens of ms and a device sync per page."""
    import dots_mocr.transformers_patch.modeling_dots_vision as vision

    reset_vision_flex_state()
    builds = []
    real_fns = vision._flex_fns

    def counting_fns(device):
        flex_fn, create_fn = real_fns(device)

        def counted(*args, **kwargs):
            builds.append(1)
            return create_fn(*args, **kwargs)

        return flex_fn, counted

    monkey = pytest.MonkeyPatch()
    monkey.setattr(vision, "_flex_fns", counting_fns)
    try:
        cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)
        for _ in range(42):
            _flex_block_mask(cu_seqlens, 16, torch.device("cpu"))
    finally:
        monkey.undo()
    assert sum(builds) == 1


def test_block_mask_cache_distinguishes_sequence_length():
    """Same boundary values, different token count: reusing one mask for the other
    would attend over the wrong extent."""
    reset_vision_flex_state()
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)
    first = _flex_block_mask(cu_seqlens, 16, torch.device("cpu"))
    second = _flex_block_mask(torch.tensor([0, 8, 32], dtype=torch.int32), 32,
                              torch.device("cpu"))
    assert first is not second
    assert first.shape[-1] != second.shape[-1]


# ---------------------------------------------------------------- strictness

def test_grad_enabled_call_raises():
    """flex has no CPU backward, and this port is inference-only. An eval() module
    called without inference_mode still builds an autograd graph, so it must say so
    rather than half-work."""
    _, flex = paired_towers()
    pixel_values, grid_thw = tower_inputs([(1, 4, 4)])
    with pytest.raises(RuntimeError, match="inference-only"):
        flex(pixel_values, grid_thw)


@CUDA
def test_float32_on_cuda_raises():
    """dots.mocr ships in bfloat16 (upstream README and the checkpoint's own
    config.json). float32 on CUDA needs non-default triton block sizes at
    head_dim 128; refuse it instead of carrying a tuning ladder for a
    configuration the model is not used in."""
    config = vision_config("flex_attention").vision_config
    dim = REAL_HEADS * REAL_HEAD_DIM
    flex = VisionFlexAttention(config, dim, num_heads=REAL_HEADS, bias=False)
    flex = flex.cuda().to(torch.float32).eval()
    hidden_states = torch.randn(256, dim, device="cuda", dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 256], dtype=torch.int32, device="cuda")
    rotary = torch.randn(256, REAL_HEAD_DIM // 2, device="cuda")
    with pytest.raises(TypeError, match="bfloat16"):
        with torch.inference_mode():
            flex(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary)


def test_indivisible_dim_raises():
    config = vision_config("flex_attention").vision_config
    with pytest.raises(ValueError, match="divisible"):
        VisionFlexAttention(config, 17, num_heads=2)


# ---------------------------------------------------------------- wiring

def test_flex_backend_is_registered():
    assert DOTS_VISION_ATTENTION_CLASSES["flex_attention"] is VisionFlexAttention
    assert set(DOTS_VISION_ATTENTION_CLASSES) == {
        "eager", "sdpa", "flash_attention_2", "flex_attention"}


def test_flex_is_the_default_vision_backend():
    from dots_mocr.transformers_patch.configuration_dots_ocr import DotsVisionConfig

    assert DotsVisionConfig().attn_implementation == "flex_attention"


def test_flex_preserves_checkpoint_parameter_names():
    """qkv/proj names are load-bearing: real checkpoints key on
    vision_tower.blocks.N.attn.{qkv,proj}.*"""
    _, flex = paired_towers()
    keys = set(flex.state_dict())
    assert "blocks.0.attn.qkv.weight" in keys
    assert "blocks.0.attn.proj.weight" in keys


def test_flex_and_sdpa_expose_the_same_module_interface():
    config = vision_config("sdpa").vision_config
    flex = VisionFlexAttention(config, 16, num_heads=2)
    sdpa = VisionSdpaAttention(config, 16, num_heads=2)
    assert flex.qkv.weight.shape == sdpa.qkv.weight.shape
    assert flex.proj.weight.shape == sdpa.proj.weight.shape


def _fake_loader(captured):
    class FakeModel:
        device = torch.device("cpu")

        def eval(self):
            return self

    def fake_from_pretrained(ckpt, **kwargs):
        captured["attn_implementation"] = kwargs.get("attn_implementation")
        captured["vision"] = kwargs["config"].vision_config.attn_implementation
        return FakeModel()

    return fake_from_pretrained


def _patch_loader(monkeypatch, captured):
    import dots_mocr.cli as cli

    monkeypatch.setattr(cli.AutoConfig, "from_pretrained",
                        staticmethod(lambda *a, **k: tiny_config()))
    monkeypatch.setattr(cli.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(_fake_loader(captured)))
    monkeypatch.setattr(cli.AutoProcessor, "from_pretrained",
                        staticmethod(lambda *a, **k: object()))
    return cli


def test_flex_does_not_reach_the_language_model(monkeypatch):
    """The vision tower gets flex; the Qwen2 decoder must keep sdpa.

    flex is measurably slower per decode token (22.21 vs 16.49 ms/tok measured on
    this backbone) because transformers rebuilds a BlockMask on every q_len==1
    step. One string feeds both towers in _load_model, so it is one edit away from
    silently regressing decode.
    """
    captured = {}
    cli = _patch_loader(monkeypatch, captured)
    cli.DotsMOCRParser(ckpt="/nonexistent", device="cuda:0", dtype="bfloat16",
                       attn_implementation="flex_attention")
    assert captured["vision"] == "flex_attention"
    assert captured["attn_implementation"] == "sdpa"


@pytest.mark.parametrize("vision", ["flex_attention", "flash_attention_2", "eager", "sdpa"])
def test_vision_backend_never_moves_the_decoder(monkeypatch, vision):
    """The two knobs are independent.

    They used to be one string, so benchmarking the vision tower silently moved
    the decoder too — which is exactly how a flash_attention_2 vision run got
    credited with a 47 t/s decode against sdpa's 61. Whatever the vision tower
    gets, the decoder keeps its own setting.
    """
    captured = {}
    cli = _patch_loader(monkeypatch, captured)
    cli.DotsMOCRParser(ckpt="/nonexistent", device="cuda:0", dtype="bfloat16",
                       attn_implementation=vision)
    assert captured["vision"] == vision
    assert captured["attn_implementation"] == "sdpa"


def test_decoder_backend_is_settable(monkeypatch):
    captured = {}
    cli = _patch_loader(monkeypatch, captured)
    cli.DotsMOCRParser(ckpt="/nonexistent", device="cuda:0", dtype="bfloat16",
                       attn_implementation="flex_attention",
                       llm_attn_implementation="flash_attention_2")
    assert captured["vision"] == "flex_attention"
    assert captured["attn_implementation"] == "flash_attention_2"


# ---------------------------------------------------------------- cuda kernel

@CUDA
def test_flex_matches_sdpa_on_cuda_bfloat16():
    """Real geometry through the actual inductor kernel.

    bfloat16 tolerance is wide because that is the dtype's noise floor at this
    scale (|output| ~0.07, measured max abs diff ~4.9e-4), not because the mask is
    approximate — the mask is asserted exactly in the CPU tests above.
    """
    config = vision_config("flex_attention").vision_config
    dim = REAL_HEADS * REAL_HEAD_DIM
    dtype = torch.bfloat16
    torch.manual_seed(0)
    flex = VisionFlexAttention(config, dim, num_heads=REAL_HEADS, bias=False).cuda().to(dtype).eval()
    sdpa = VisionSdpaAttention(config, dim, num_heads=REAL_HEADS, bias=False).cuda().to(dtype).eval()
    sdpa.load_state_dict(flex.state_dict())

    seq_length = 2588
    hidden_states = torch.randn(seq_length, dim, device="cuda", dtype=dtype)
    cu_seqlens = torch.tensor([0, 1024, seq_length], dtype=torch.int32, device="cuda")
    rotary = torch.randn(seq_length, REAL_HEAD_DIM // 2, device="cuda")

    reset_vision_flex_state()
    with torch.inference_mode():
        expected = sdpa(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary)
        actual = flex(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)


@CUDA
def test_warmup_compiles_the_production_graph():
    """warmup_vision_flex exists to move the inductor compile out of the first page.

    It only helps if the compiled graph matches production on every dynamo guard
    (two shapes, strides, grad mode, indexed device); if it does, a subsequent call
    at a fresh size is fast rather than paying a full recompile.
    """
    import time

    reset_vision_flex_state()
    assert warmup_vision_flex("cuda", dtype=torch.bfloat16,
                              num_heads=REAL_HEADS, head_dim=REAL_HEAD_DIM) is True

    config = vision_config("flex_attention").vision_config
    dim = REAL_HEADS * REAL_HEAD_DIM
    flex = VisionFlexAttention(config, dim, num_heads=REAL_HEADS,
                               bias=False).cuda().to(torch.bfloat16).eval()
    seq_length = 6270
    hidden_states = torch.randn(seq_length, dim, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, seq_length], dtype=torch.int32, device="cuda")
    rotary = torch.randn(seq_length, REAL_HEAD_DIM // 2, device="cuda")

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        flex(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"first post-warmup page took {elapsed:.2f}s (recompiled?)"

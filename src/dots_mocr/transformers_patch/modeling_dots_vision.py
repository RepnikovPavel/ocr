import functools
import math
import threading
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None
# from flash_attn import flash_attn_varlen_func
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # torch < 2.5
    create_block_mask = flex_attention = None
from torch.nn import LayerNorm
from transformers.modeling_utils import PreTrainedModel
from .configuration_dots_ocr import DotsVisionConfig


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()

    cos = freqs.cos()
    sin = freqs.sin()

    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()

    output = (tensor * cos) + (rotate_half(tensor) * sin)

    output = output.to(orig_dtype)

    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        pre_norm="layernorm",
        init_merger_std=None,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.pre_norm = pre_norm
        if self.pre_norm == "layernorm":
            self.ln_q = LayerNorm(context_dim, eps=1e-6)
        elif self.pre_norm == "rmsnorm":
            self.ln_q = RMSNorm(context_dim, eps=1e-6)
        else:
            print("no norm in patch merger")

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

        if init_merger_std is not None:
            nn.init.normal_(self.mlp[0].weight, mean=0.0, std=init_merger_std)
            nn.init.zeros_(self.mlp[0].bias)
            nn.init.normal_(self.mlp[2].weight, mean=0.0, std=init_merger_std)
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        else:
            x = self.mlp(x.view(-1, self.hidden_size))
        return x


class VisionAttention(nn.Module):
    def __init__(self, config, dim: int, num_heads: int = 16, bias=True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class VisionFlashAttention2(nn.Module):
    def __init__(self, config, dim: int, num_heads: int = 16, bias=True) -> None:
        super().__init__()
        if flash_attn_varlen_func is None:
            raise ImportError("flash-attn is required for flash_attention_2")
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.config = config
        self.is_causal = config.is_causal

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )  # 'shd'
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=self.is_causal
        ).reshape(seq_length, -1)
        attn_output = self.proj(attn_output)

        return attn_output


def _block_diag_bool_mask(cu_seqlens, seq_length: int, device) -> torch.Tensor:
    """The [1, S, S] bool mask the sdpa backend attends under: token i sees token j
    only when both fall in the same [cu_seqlens[k-1], cu_seqlens[k]) segment.

    Named rather than inlined so the regression tests can assert that
    VisionFlexAttention's BlockMask induces exactly this predicate.
    """
    mask = torch.zeros([1, seq_length, seq_length], device=device, dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
    return mask


class VisionSdpaAttention(nn.Module):
    def __init__(self, config, dim: int, num_heads: int = 16, bias=True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = _block_diag_bool_mask(cu_seqlens, seq_length, q.device)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)

        attn_output = self.proj(attn_output)
        return attn_output


_FLEX_AVAILABLE = flex_attention is not None and create_block_mask is not None

# The model ships in bfloat16 (upstream README loads it with torch_dtype=bfloat16,
# and the checkpoint's own config.json says the same). float32 on CUDA would need
# smaller triton block sizes than inductor picks by default at head_dim 128 and is
# not a configuration this model is used in, so it is refused rather than tuned for.
# float32 on CPU is fine: that path runs flex eagerly, with no triton involved.
_FLEX_CUDA_DTYPES = (torch.bfloat16, torch.float16)

_FLEX_CACHE_MAX = 8
_flex_lock = threading.Lock()
_flex_block_masks = {}   # (bounds, seq_length, device, inference_mode) -> BlockMask


def reset_vision_flex_state() -> None:
    """Drop the cached block masks. For tests that assert on cache behaviour."""
    with _flex_lock:
        _flex_block_masks.clear()


@functools.lru_cache(maxsize=1)
def _compiled_flex_fns():
    """flex_attention / create_block_mask compiled once per process.

    Compiling is mandatory, not an optimization: called plainly, flex_attention takes
    an unfused path that materializes the full [H, S, S] score matrix (48x slower,
    200x the memory at S=4096) -- exactly what this backend exists to avoid.

    dynamic=True because page token counts vary continuously (S ~1k-16k). A static
    compile pays ~1s of inductor per distinct S, a recompile storm across a PDF; one
    dynamic graph serves them all (measured: 12 distinct page sizes, 0 recompiles).

    create_block_mask is compiled the same way rather than passed the private
    _compile=True flag, which is deprecated in torch 2.12 and slated for removal.
    Uncompiled it builds the dense S x S mask via vmap: ~1 GB transient at S=10000.

    lru_cache keeps this lazy, so importing this module never triggers inductor.
    """
    return (
        torch.compile(flex_attention, dynamic=True),
        torch.compile(create_block_mask, dynamic=True),
    )


def _flex_fns(device: torch.device):
    # CPU is unit-test territory (a real page needs a GPU either way). Compiling for
    # it needs a C++ toolchain and saves nothing at test sizes, so run flex eagerly
    # there -- still the same mask_mod / BlockMask code, which is what makes the CPU
    # equivalence tests meaningful.
    if device.type == "cpu":
        return flex_attention, create_block_mask
    return _compiled_flex_fns()


def _flex_segment_ids(bounds, seq_length: int, device) -> torch.Tensor:
    """Per-token segment id, so `seg[i] == seg[j]` reproduces the block-diagonal
    predicate _block_diag_bool_mask paints.

    Assignment (not +=) at each boundary means a repeated boundary -- an empty
    segment -- collapses instead of skewing every later id. Empty segments hold no
    tokens, so the induced partition, and hence the mask, is identical. Only equality
    of ids ever matters, never their numbering.
    """
    seg = torch.zeros(seq_length, dtype=torch.int32, device=device)
    interior = [b for b in bounds[1:-1] if 0 < b < seq_length]
    if interior:
        seg[torch.tensor(interior, dtype=torch.long, device=device)] = 1
    return seg.cumsum(0).to(torch.int32)


def _flex_block_mask(cu_seqlens: torch.Tensor, seq_length: int, device):
    """Block-diagonal BlockMask for this page.

    Same predicate as _block_diag_bool_mask, but resolved at 128x128 block
    granularity so fully-masked blocks are skipped by the kernel rather than computed
    and discarded. That skipping is the win.

    Cached because the mask depends only on cu_seqlens, never on weights, while the
    attention forward signature has no slot to hoist it out of the 42-layer loop:
    rebuilding it per layer costs 21-38 ms per page. The key carries seq_length and
    device as well as the boundaries -- two different pages can share boundary values
    without sharing a mask -- plus the grad mode, because a BlockMask built under
    inference_mode holds inference tensors that a grad-enabled forward cannot use.
    """
    bounds = [int(b) for b in cu_seqlens.tolist()]
    if (len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != seq_length
            or any(bounds[i] > bounds[i + 1] for i in range(len(bounds) - 1))):
        # Not a partition of [0, S): some row would be fully masked, which is a
        # caller bug, not something to paper over -- attention there is undefined.
        raise ValueError(
            f"cu_seqlens {bounds} does not partition [0, {seq_length}): every token "
            "must belong to exactly one segment")

    key = (tuple(bounds), seq_length, str(device), torch.is_inference_mode_enabled())
    cached = _flex_block_masks.get(key)
    if cached is not None:
        return cached

    seg_ids = _flex_segment_ids(bounds, seq_length, device)

    def mask_mod(b, h, q_idx, kv_idx):
        # Runs under vmap: tensor ops only. No .item(), print, or python branching.
        return seg_ids[q_idx] == seg_ids[kv_idx]

    _, create_block_mask_fn = _flex_fns(device)
    # S need not be a multiple of BLOCK_SIZE, and mask_mod is never called with an
    # index >= seq_length, so seg_ids needs no padding.
    block_mask = create_block_mask_fn(mask_mod, None, None, seq_length, seq_length,
                                      device=device)

    with _flex_lock:
        # FIFO rather than LRU: pages in a batch are near-uniform, so recency and
        # frequency coincide and the bookkeeping is not worth it.
        if key not in _flex_block_masks and len(_flex_block_masks) >= _FLEX_CACHE_MAX:
            _flex_block_masks.pop(next(iter(_flex_block_masks)))
        _flex_block_masks[key] = block_mask
    return block_mask


def warmup_vision_flex(device, dtype=torch.bfloat16, num_heads: int = 12,
                       head_dim: int = 128, seq_lengths=(1024, 4096)) -> bool:
    """Pay the one-time inductor compile at load time instead of inside the first
    page, where a multi-second stall reads as a hang.

    Every detail below was measured, not reasoned: a warmup that returns True while
    the first real page still recompiles is worse than none, because it looks like it
    worked. Each guard dynamo places on the compiled graph must match production.
      * TWO distinct lengths. Dynamo specializes on the first shape it sees and only
        generalizes on the second, so a one-shape warmup leaves the first real page
        paying a full recompile.
      * Production STRIDES, which differ across q/k/v. q and k are rebuilt by
        apply_rotary_pos_emb_vision and come out contiguous, so transposing gives
        stride (D, H*D, 1); v is never touched by the rotary and stays a view into
        the fused qkv buffer, giving (D, 3*H*D, 1). _warmup_qkv reproduces that.
      * inference_mode, because dynamo guards on GLOBAL_STATE grad_mode and the
        production forward runs under inference_mode (cli.py::_inference).
      * An INDEXED device: production passes q.device ("cuda:0"), and dynamo guards
        on device equality -- torch.device("cuda") != torch.device("cuda", 0).
    With all four aligned the first real page costs ~0.15 s; with any one wrong it
    costs 1.6-3.1 s.

    Returns True if the warmup ran. Never raises: failing to pre-compile must not
    prevent the model from loading -- the first page will just pay for it.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        for s in seq_lengths:
            cu_seqlens = torch.tensor([0, s // 2, s], dtype=torch.int32, device=device)
            with torch.inference_mode():
                qkv = torch.randn(s, 3, num_heads, head_dim, device=device, dtype=dtype)
                q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
                # .contiguous() stands in for apply_rotary_pos_emb_vision, which
                # returns a fresh contiguous tensor for q and k; v passes through.
                q, k = q.contiguous().transpose(0, 1), k.contiguous().transpose(0, 1)
                v = v.transpose(0, 1)
                block_mask = _flex_block_mask(cu_seqlens, s, device)
                flex_fn, _ = _flex_fns(device)
                flex_fn(q[None], k[None], v[None], block_mask=block_mask,
                        scale=head_dim ** -0.5)
            del qkv, q, k, v
        return True
    except Exception as error:  # noqa: BLE001 - load-time probe must not break loading
        warnings.warn(f"dots vision flex_attention warmup failed "
                      f"({type(error).__name__}: {error}); the first page will pay "
                      "the compile", RuntimeWarning, stacklevel=2)
        return False


class VisionFlexAttention(nn.Module):
    """Block-diagonal varlen attention via torch flex_attention.

    Same math as VisionSdpaAttention -- that class stays as the reference the
    regression tests compare against -- but the block-diagonal structure reaches the
    kernel as a BlockMask instead of a dense [1, S, S] bool mask. The off-diagonal
    blocks are then skipped rather than computed and masked away, which is why a page
    that OOMs under sdpa fits here: measured 2.13 Mpx (~10.9k tokens) in 6.2 GiB,
    where sdpa asks for 5.27 GiB of score matrix on top of the weights and dies.

    Inference-only, and deliberately strict: an unsupported configuration raises
    instead of quietly degrading, so a misconfigured run is visible immediately
    rather than showing up later as an unexplained slowdown or an OOM.
    """

    def __init__(self, config, dim: int, num_heads: int = 16, bias=True) -> None:
        super().__init__()
        if not _FLEX_AVAILABLE:
            raise ImportError(
                "flex_attention requires torch >= 2.5 (torch.nn.attention.flex_attention)")
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # Names are load-bearing: checkpoint keys are blocks.N.attn.{qkv,proj}.*
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        if torch.is_grad_enabled():
            # Stricter than `not self.training` on purpose: an eval() module called
            # without inference_mode/no_grad still builds an autograd graph through
            # its parameters, and flex has no CPU backward at all. This port is
            # inference-only, so say so rather than half-supporting training.
            raise RuntimeError(
                "flex_attention vision attention is inference-only; call it under "
                "torch.inference_mode() or torch.no_grad()")

        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        if q.device.type == "cuda" and q.dtype not in _FLEX_CUDA_DTYPES:
            raise TypeError(
                f"flex_attention on CUDA supports {_FLEX_CUDA_DTYPES} here, got {q.dtype}. "
                "dots.mocr ships in bfloat16; load the model with dtype=bfloat16.")

        block_mask = _flex_block_mask(cu_seqlens, seq_length, q.device)
        flex_fn, _ = _flex_fns(q.device)
        # flex_attention wants 4-D [B, H, S, D]; this tower is unbatched, so B=1.
        # Non-contiguous strides out of the transpose are accepted as-is.
        #
        # scale is passed explicitly even though flex and sdpa both default to
        # 1/sqrt(head_dim): pinning it makes parity with VisionSdpaAttention a
        # property of this file rather than of two libraries continuing to agree.
        #
        # config.is_causal is deliberately ignored, exactly as eager and sdpa ignore
        # it. Only flash_attention_2 honours it, and the vision config sets it False.
        attn_output = flex_fn(q[None], k[None], v[None], block_mask=block_mask,
                              scale=self.head_dim ** -0.5)[0]

        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)

        attn_output = self.proj(attn_output)
        return attn_output


DOTS_VISION_ATTENTION_CLASSES = {
    "eager": VisionAttention,
    "flash_attention_2": VisionFlashAttention2,
    "sdpa": VisionSdpaAttention,
    "flex_attention": VisionFlexAttention,
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class DotsSwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.embed_dim
        bias = config.use_bias

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc3(x)
        x = self.fc2(x)
        return x



class DotsPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim
        self.config = config
        self.proj = nn.Conv2d(
            config.num_channels,
            config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
        x = x.view(-1, self.num_channels, self.temporal_patch_size, self.patch_size, self.patch_size)[:, :, 0] 
        x = self.proj(x).view(-1, self.embed_dim)
        x = self.norm(x)
        return x


class DotsViTPreprocessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_h = config.patch_size
        self.patch_w = config.patch_size
        self.embed_dim = config.embed_dim
        self.config = config
        self.patchifier = DotsPatchEmbed(config)

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
        tokens = self.patchifier(x, grid_thw)
        return tokens


class DotsVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "flash_attention_2"):
        super().__init__()
        self.attn = DOTS_VISION_ATTENTION_CLASSES[attn_implementation](
            config, config.embed_dim, num_heads=config.num_attention_heads, bias=config.use_bias
        )
        self.norm1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = DotsSwiGLUFFN(config)
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class DotsVisionTransformer(PreTrainedModel):
    def __init__(self, config: DotsVisionConfig) -> None:
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = DotsViTPreprocessor(config)
        self._init_weights(self.patch_embed.patchifier.proj)

        head_dim = config.embed_dim // config.num_attention_heads

        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        _num_hidden_layers = config.num_hidden_layers
        self.blocks = nn.ModuleList(
            [DotsVisionBlock(config, config.attn_implementation) for _ in range(_num_hidden_layers)]
        )

        if self.config.post_norm:
            self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
            init_merger_std=self.config.init_merger_std,
        )

        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            if not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None and not getattr(module.bias, "_is_hf_initialized", False):
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, VisionRotaryEmbedding):
            inv_freq = 1.0 / (
                module.theta
                ** (
                    torch.arange(0, module.dim, 2, dtype=torch.float, device=module.inv_freq.device)
                    / module.dim
                )
            )
            module.inv_freq.copy_(inv_freq)

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def get_pos_ids_by_grid(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
            )

        return pos_ids

    def rot_pos_emb(self, grid_thw):
        pos_ids = self.get_pos_ids_by_grid(grid_thw)
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states, grid_thw)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__,
                    hidden_states,
                    cu_seqlens,
                    rotary_pos_emb,
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.merger(hidden_states)
        return hidden_states

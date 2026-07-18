# Performance findings — dots.mocr on a single RTX 4070 Ti

Measured on RTX 4070 Ti (12 GiB, sm_89), torch 2.12+cu130, transformers 5.5.4,
bfloat16, page `Searching_for_MobileNet_V3.pdf` at 2.13 Mpx.
Reproduce: `benchmarks/bench_attention.py`, `benchmarks/profile_stages.py`,
`benchmarks/token_budget.py` (raw JSON output is gitignored).

Measured hardware limits (not spec sheet): bf16 GEMM **77.6 TFLOP/s**,
fp8 e4m3 GEMM **148.4 TFLOP/s**, memory bandwidth **391 GB/s**.

## 1. Attention backends — flex gains nothing over the authors' flash

Vision tower only, decoder pinned to sdpa in all three runs:

| vision backend | vision s | page s | peak VRAM |
|---|---|---|---|
| `flash_attention_2` (what the authors use) | **0.408** | 24.20 | **5.891 GiB** |
| `flex_attention` (this repo's default) | 0.446 | 24.31 | 5.892 GiB |
| `sdpa` (previous default) | 1.935 | 26.21 | 8.447 GiB |

- Against `flash_attention_2` flex is **9 % slower** and identical in memory.
  Its only real advantage is needing **no flash-attn dependency** — same numbers
  from stock PyTorch, no nvcc, no long build.
- What actually goes away is **sdpa**, which received a dense `[1, S, S]` mask and
  therefore fell off the fused kernels: 4.7x slower, 2.5 GiB heavier, and unable to
  process a 2.13 Mpx page on a 12 GiB card at all (needs 5.27 GiB of score matrix).
- Decode speed is unchanged by any backend (58.6 / 59.6 / 59.8 tokens/s).
- Outputs are semantically equal but **not bit-identical**: 42 residual vision
  layers in bf16 are chaotically sensitive to summation order. The already-shipped
  `eager` and `sdpa` diverge from each other just as much. Mask equivalence is
  asserted exactly in tests; end-to-end equality is asserted semantically.

## 2. Where a page actually goes

| stage | seconds | share | achieved TFLOP/s | % of measured peak |
|---|---|---|---|---|
| **decoding (1277 tokens)** | **19.3** | **92.7 %** | 0.24 | **0.3 %** |
| vision tower | 1.29 | 6.1 % | 44.1 | 57 % |
| LM prefill | 0.18 | 0.9 % | 49.9 | 64 % |
| preprocessing (CPU) | 0.08 | 0.4 % | — | — |

Inside a decode step: ~80 % is reading weights through matmul/GEMV at batch 1,
4.1 % is `aten::cat` growing the KV cache, and **all attention is 3.6 %**.
Across the whole page attention is ~5 % — which is why no attention backend
moves tokens/s.

## 3. Token budget

Patches are 14x14 px, merged 2x2, so **one LM token = 28x28 px**.

| render | Mpx | patch tokens (vision) | tokens into LM | prompt total |
|---|---|---|---|---|
| 616x784 | 0.48 | 2,464 | 616 | 832 |
| 1288x1652 | 2.13 | 10,856 | 2,714 | 2,930 |

| stage | tokens | seconds | ms/token |
|---|---|---|---|
| vision tower | 10,856 | 1.29 | 0.119 |
| LM prefill | 2,930 | 0.18 | 0.061 |
| **LM decode** | **1,277** | **19.3** | **15.12** |

**A token costs 246x more in decode than in prefill** — same arithmetic, but
prefill does 2,930 tokens in one pass over the weights while decode pays a full
pass per token. The page ingests 13,786 tokens in 1.47 s and emits 1,277 in 19.3 s.

Consequences: lowering `max_pixels` is a quality knob, not a speed knob (input is
7 % of the time). Choosing the prompt mode is a real lever — `layout_only` emits
429 tokens vs `layout_all`'s 1,277, so it is 2.7x faster.

## 4. Decode ceiling — not reached

Each token must read **2944 MiB** of LM weights (the 2315 MiB vision tower is not
read during decoding).

| | ms/token | tokens/s | % of achievable bandwidth |
|---|---|---|---|
| today | 15.12 | **66** | **52 %** |
| at 100 % bandwidth | 7.89 | 127 | 100 % |
| in fp8, at 100 % bandwidth | 3.95 | 253 | 100 % |

Arithmetic intensity at batch 1 is **1 FLOP/byte**; this card needs 198 FLOP/byte
to become compute-bound. Compute sits ~99.4 % idle. Measured on one real layer:
batch 64 runs *faster in absolute terms* than batch 1 while doing 64x the work.

## 5. Ranked next steps

| # | change | expected | why |
|---|---|---|---|
| 1 | **FP8 (e4m3) weights for the LM** | up to **2x** | 2944 → ~1472 MiB/token; sm_89 has fp8 in hardware |
| 2 | **CUDA graphs for the decode step** | 1.3–1.5x | only 52 % of bandwidth used; ~200 kernel launches per step. Decode shapes are static except KV length, which a pre-allocated cache fixes |
| 3 | Static KV cache instead of `cat` | ~4 % | grows with output length |
| 4 | Batching pages | scales pages/min | weights read once per batch |
| 5 | Attention backend | **< 5 %, hard ceiling** | already done |

## Caveats

- One run per configuration, median of 3 timings per stage.
- Achieved TFLOP/s come from an analytic model of the geometry, not hardware counters.
- Profiled on one page of one document; the decode share grows with output length.

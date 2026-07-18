# Performance findings — dots.mocr on a single RTX 4070 Ti

Measured on RTX 4070 Ti (12 GiB, sm_89), torch 2.12+cu130, transformers 5.5.4,
bfloat16, page `Searching_for_MobileNet_V3.pdf` at 2.13 Mpx.
Reproduce: `benchmarks/bench_attention.py`, `benchmarks/profile_stages.py`,
`benchmarks/token_budget.py`, `benchmarks/bench_vllm.py` (raw JSON output is gitignored).

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
- Outputs are semantically equal but **not bit-identical**, and no backend is the
  reference. On page 2 the backends disagree about whether one LaTeX fragment
  carries absolute-value bars — and **`sdpa` lands on both sides**: bars present
  under torch 2.12, absent under torch 2.11, with field-for-field identical configs.
  The tally is 3-3 (present: sdpa/2.12, flash_attention_2, vLLM; absent: sdpa/2.11,
  flex under both builds), so the split cuts through a single backend.

  Greedy decoding is an argmax, so any step with a top-1/top-2 margin thinner than
  the bf16 perturbation is a coin flip. **A single differing token is not evidence
  that a backend is wrong** — at the measured ~0.33 semantic differences per page,
  the 95 % interval on one observed event spans a factor of ~220, and separating a
  doubled error rate from noise would need ~74 pages per arm.

  Worth noting for regression design: **flex is the more reproducible backend.** It
  is byte-identical across both torch builds, while sdpa differs from *itself* in 28
  tokens; flex is 3 diff-chunks from sdpa/2.12 while sdpa/2.12 is 11 from sdpa/2.11.
  So byte-equality against sdpa is not a valid oracle. Mask identity is asserted
  exactly (`torch.equal`), fp32 tower equivalence to ~1e-4, and end-to-end equality
  semantically.

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

## 5. vLLM baseline — we are at half of what the card allows

Same card, same checkpoint, same page, same prompt, greedy, prefix caching
**disabled** on the vLLM side so both engines actually run prefill
(vLLM 0.17.1, which supports `DotsOCRForCausalLM` natively; see `docker/Dockerfile.vllm`).

| page | tokens | vLLM TTFT | vLLM tok/s | vLLM s | ours TTFT | ours tok/s | ours s | gap |
|---|---|---|---|---|---|---|---|---|
| 0 | 1277 | 1.31 | 119.6 | 11.98 | 1.32 | 62.0 | 21.89 | 1.83x |
| 1 | 1549 | 1.28 | 120.5 | 14.13 | 1.31 | 62.9 | 25.92 | 1.83x |
| 2 | 1555 | 1.29 | 119.9 | 14.28 | 1.31 | 64.2 | 25.52 | 1.79x |
| 3 | 1684 | 1.30 | 119.9 | 15.35 | 1.33 | 61.0 | 28.91 | 1.88x |

- **Prefill and the vision tower are at parity**: mean TTFT 1.30 s vs 1.32 s. The
  attention work in this repo holds up — that half of the pipeline is not the problem.
- **Decoding is 1.92x behind**: 120.0 vs 62.5 tokens/s, which drags the whole page
  to ~1.85x.
- Against the bandwidth ceiling computed in §4 (127 tokens/s), **vLLM reaches 94 %
  and we reach 49 %**. That is independent confirmation that the ceiling is real
  and reachable on this hardware — the headroom is not theoretical.
- vLLM's startup log shows `Capturing CUDA graphs`, which is item 2 below. It also
  batches: 4 concurrent pages give **379.8 tokens/s aggregate** (108.3 each), where
  this repo serves strictly one generation at a time.
- **Output agreement**: of 4 pages, 1 byte-identical, 2 equal in text and categories
  with 1-2 px bbox drift, and 1 differing by the single coin-flip token described in
  §1. `benchmarks/bench_vllm.py` now checks this on every run, so a speed comparison
  can never again be reported without the agreement it depends on.

**Answer to "have we extracted everything": no — roughly half.** The remaining ~2x
is entirely in the decode loop, exactly where §2 and §4 said it was.

## 6. Ranked next steps

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
- vLLM was given `--gpu-memory-utilization 0.85` (the desktop session holds ~1.2 GiB)
  and `--max-model-len 12288`; it reported a 25,792-token KV cache.
- The vLLM run needs `auto_map` in `config.json`, which `scripts/prepare_checkpoint.py`
  strips for this repo's own loader. The benchmark serves a symlink directory with
  the original config restored rather than modifying the checkpoint.

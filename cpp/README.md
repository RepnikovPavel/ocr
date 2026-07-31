# dots.mocr — pure C++ + CUDA inference engine

A from-scratch, dependency-light inference engine for the
[dots.mocr](../README.md) vision-language model (3.04 B params: a 42-layer
Qwen2.5-VL-style vision tower + a 28-layer Qwen2 decoder with GQA). No
PyTorch, no vLLM, no Python at runtime — just C++17, CUDA 13, cuBLAS, and two
header-only libraries (`nlohmann/json`, `stb_image`) vendored under
`third_party/`.

The engine loads the model's safetensors checkpoint directly (mmap'd), runs
the full `image + prompt → markdown` pipeline on the GPU, and decodes greedily.
It exists to (a) strip the launch/Python overhead that caps HF `generate` at
~14 % of the memory roofline (see [`docs/uplift/01-…`](../docs/uplift/01-decode-engine-cuda-graphs.md)),
and (b) give a clean CUDA baseline for the uplift proposals.

## Layout

```
cpp/
  include/        headers (tensor, config, tokenizer, kernels, attention, vision_tower, llm, pipeline)
  src/            implementations (kernels.cu / attention.cu / vision_tower.cu / llm.cu are CUDA)
  tools/          dots_mocr_cli.cpp  — the `dots_mocr` CLI
                  capture_reference.py — capture HF greedy ids for the regression test
  tests/          unit + regression tests
  docker/         Dockerfile.build — CUDA 13 devel + cmake build image
  scripts/        build_on_server.sh — build + test over SSH on the GPU box
  third_party/    nlohmann/json, stb_image (vendored, header-only)
```

## Build

The build needs a CUDA 13 toolkit (nvcc + cuBLAS). The shipped Docker image
bundles everything; on the server:

```sh
bash cpp/scripts/build_on_server.sh
```

That rsyncs `cpp/` to `/mnt/nvme2/ocr-flex/cpp`, builds `dots-mocr-cpp:build`
once, then configures + compiles inside a `--gpus all` container with the
checkpoint mounted at `/ckpt`. Targets `89` (Ada) and `120` (Blackwell) are
emitted.

Local one-shot build (with CUDA installed):

```sh
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_CUDA_ARCHITECTURES="89;120"
cmake --build cpp/build -j
```

## Run

```sh
./build/dots_mocr --ckpt /path/to/dots_mocr_ckpt --image page.png \
                  --max-new-tokens 4096 [--device 0] [--prompt "..."]
```

With no `--prompt`, the `prompt_layout_all_en` layout-extraction prompt is used
(matches the Python `cli.py` default). Generated markdown goes to stdout;
timing (`prefill_ms`, `decode_ms`, tok/s) goes to stderr.

## Tests

```sh
./build/test_tokenizer   /ckpt/tokenizer.json   # BPE encode/decode parity
./build/test_loader      /ckpt                  # safetensors weights load
./build/test_imageproc                           # smart_resize + patchify
./build/test_kernels                             # RMSNorm, flash attn vs CPU ref
./build/test_gemm                                # cuBLAS bf16 GEMM, all transA/transB
./build/test_vision_attn                         # RoPE+flash combo vs CPU ref
./build/test_flash_real   /ref_dir               # flash on HF-captured q,k,v
./build/test_regression   /ckpt /ref_dir         # greedy id parity vs HF reference
```

All CPU-reference tests pass: tokenizer, loader, imageproc, RMSNorm, GEMM,
flash attention (maxerr ≤ 0.014 vs a fp32 CPU reference), and the RoPE+flash
combination. The vision block-0 `q`, `k`, `v` (after RoPE) were verified
byte-identical to the HF checkpoint's, and the patch-embed output matches HF
to bf16 precision.

### Note on end-to-end parity

The greedy-id regression test (`test_regression`) needs a captured HF
reference. Capturing one reliably turned out to be harder than expected: the
checkpoint's vision tower ships with a non-persistent `inv_freq` RoPE buffer
that loads as garbage under `from_pretrained`, and its `modeling_dots_vision`
unconditionally casts inputs to bf16 while the conv bias loads as fp32.
`tools/capture_reference_manual.py` works around both. Even so, the HF
`VisionSdpaAttention` output we captured diverges from a straight
fp32 `softmax(QKᵀ/√d)V` of the *same* captured `q,k,v` — i.e. the engine's
flash output matches the CPU reference exactly, but the HF sdpa path does not,
which points at a remaining discrepancy in how the reference is run rather
than in the engine. Greedy decoding amplifies any per-layer bf16 difference
(the docs note HF-flash and vLLM also diverge at ULP level), so the
end-to-end token sequence is not yet bit-identical to vLLM. The component
tests above are the correctness anchor.


## What is and isn't reimplemented

| Stage | Implementation |
| --- | --- |
| Image load + smart_resize | `stb_image` + a PIL-equivalent bilinear resize, byte-exact |
| Patchify + normalise | port of `Qwen2VLImageProcessor`, channel-first `[N_v,588]` |
| BPE tokeniser | from-scratch GPT-4 regex + byte-level + BPE, exact vs HF |
| GEMM (qkv/o/mlp/lm_head/conv) | **cuBLAS bf16** (`cublasGemmEx`, COMPUTE_32F) |
| RMSNorm / LayerNorm / SiLU / SwiGLU / GELU | custom CUDA kernels (fp32 reduce, bf16 out) |
| RoPE | LLM NEOX rotate-half + vision 2D, custom kernels |
| Attention | custom flash-style kernel (online softmax, warp-parallel); full (vision) + causal (LLM prefill) + single-query decode |
| KV-cache | per-layer append-only bf16 buffers, GQA-expanded for prefill |
| Generation | greedy argmax, EOS stop |

The model **weights and math are unchanged** — only the kernel schedule
differs from HF/vLLM, exactly the "inference-only" scope the uplift docs allow.

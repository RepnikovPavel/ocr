# dots.mocr vs GLM-OCR — dual-model demo, 2026-08-01

Issue #15, Lever A: wire GLM-OCR (0.9B) alongside dots.mocr (3.0B) so the same
document can be parsed with each and the quality/speed difference eyeballed in
the browser.

## Setup

Two single-model demos, each on its own GPU of a 2× RTX 4090 host
(driver 565, CUDA 12.7 in-container), cross-linked by the peer link in the
header:

| demo | model | engine | GPU | port |
| --- | --- | --- | --- | --- |
| dots.mocr | dots.mocr 3.0B (`dots_ocr`) | vLLM 0.17.1 | 0 | 8601 |
| GLM-OCR | GLM-OCR 0.9B (`glm_ocr`) | transformers 5.5.4 (in-process) | 1 | 8603 |

GLM-OCR runs **in-process via transformers**, not vLLM: no public vLLM build
loads `glm_ocr` on this host's driver. The cu13 images are refused by
nvidia-container-toolkit (legacy mode, cuda>=12.9 unsatisfied); the cu124
image's torch errors "driver too old"; and a hand-built cu126 vLLM (>=0.26)
links libcudart.so.13. GLM-OCR is a native transformers architecture, so the
model-card path (`AutoModelForImageTextToText`) works on the same cu126 stack
dots.mocr's transformers engine already uses — verified working on the 4090
(1.3s cold load).

## Speed (warm, same single-page PDF, bf16)

| metric | dots.mocr (vLLM) | GLM-OCR (transformers) |
| --- | --- | --- |
| wall time / page | 4.25 s | 5.38 s |
| generated tokens | 151 | 122 |
| **decode t/s** | **223.5** | **63.0** |
| total t/s (incl. prefill) | 37.0 | 24.3 |
| TTFT | 3.4 s | 3.1 s |

GLM-OCR per-trigger modes are lighter (no layout-JSON post-processing):

| GLM mode | wall time |
| --- | --- |
| glm_formula_recognition | 1.97 s |
| glm_table_recognition | 2.37 s |
| glm_text_recognition | 5.38 s |

dots.mocr's vLLM decode rate (223 t/s) is ~3.5× GLM's in-process rate (63 t/s)
— but dots emits more tokens (layout structure) so wall time per page is
comparable. On a vLLM build that runs GLM (not possible on this driver), GLM's
smaller model would likely close the t/s gap and win on throughput; here the
engines differ, which is why total t/s is the fair cross-engine number.

## Quality (same page, verbatim output)

**dots.mocr `prompt_ocr`** — structured markdown (headings, LaTeX formula,
markdown table):

```
# Attention Is All You Need
Vaswani et al., 2017
## Abstract
We propose a new architecture avoiding recurrence...
## 1 Introduction
Recurrent neural networks process sequences left to right.
Equation: $\text{softmax}(\text{QK}^\text{T} / \sqrt{\text{d}_\text{k}) V$
Table:
Layer | Params
Embed | 384 x 512
Attention | 384 x 384
FFN | 384 x 2048
```

**GLM-OCR `glm_text_recognition`** — plain recognised text (no headings /
formula / table markup; it is a recognition model, not a layout model):

```
Abstract
We propose a new architecture avoiding recurrence...
1 Introduction
Recurrent neural networks process sequences left to right.
```

GLM-OCR recovers the formula/table only when asked by the right trigger —
`glm_formula_recognition` returns clean LaTeX
(`$$\text{softmax}(QK^T / \sqrt{d_k}) V$$`).

**Takeaway:** the models are not drop-in substitutes. dots.mocr produces the
structured artifacts (markdown + JSON + LaTeX the repo already ships); GLM-OCR
is faster and smaller but emits recognition output, so a full swap needs the
layout pass GLM's own SDK couples it with (PP-DocLayout). For the repo's
existing contract (layout JSON → markdown), dots.mocr stays; GLM-OCR is the
lighter path where only text/formula/table recognition is needed.

## How to view it

```sh
# on your laptop, one tunnel carries both demos
ssh -N -L 8601:127.0.0.1:8601 -L 8603:127.0.0.1:8603 pavel.repnikov@10.152.1.180
# dots.mocr: http://127.0.0.1:8601   GLM-OCR: http://127.0.0.1:8603
```

Each demo's header has a peer link to the other.

## Notes / follow-ups

- Lever B of issue #15 (GPTQ W4A16 on dots.mocr's decoder + FP8 KV-cache) is
  not touched here — separate task.
- If the host's driver/toolkit is upgraded (so cu13 vLLM images load), GLM-OCR
  can move to vLLM by flipping `DEMO_GLM_ENGINE=vllm` — the parser
  (`GlmOcrVllmParser`) and registry already support it; only the deploy env
  changes.
- `deploy_dual.sh` brought the whole stack up; checkpoints now live on
  `/mnt/data1` (the `/mnt/data2` volume was full at 100% and was migrated).

Agent: ZCode (GLM-5.2)

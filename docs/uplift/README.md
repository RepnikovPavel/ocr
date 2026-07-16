# Uplift proposals — оптимизация инференса dots.mocr

Детальные предложения по каждому пункту §7 отчёта об архитектуре
([../architecture.md](../architecture.md)). Для читателя, который силён в CUDA/PyTorch,
но не в LLM/VLM — начните с primer.

- **[00 — Primer: инференс LLM/VLM для CUDA-инженера](00-primer-llm-inference-for-cuda-engineers.md)**
  — авторегрессия, prefill vs decode, attention в decode, **KV-кэш**, GQA, почему decode
  launch/bandwidth-bound. Мост от «мешанины LLM» к твоему мышлению (shapes/bytes/launches/roofline).

| # | Proposal | Фаза | Тип выигрыша |
|---|---|---|---|
| [01](01-decode-engine-cuda-graphs.md) | Движок decode: CUDA Graphs → vLLM/TRT-LLM | decode | латентность ×2–5, throughput ×5–10 (с батчем) |
| [02](02-vision-flash-attention.md) | Vision: flash-attention вместо sdpa | prefill | TTFT ×1.2–1.5 + снятие потолка разрешения |
| [03](03-kernel-fusion.md) | Фьюзы ядер / `torch.compile` | prefill | ×1.1–1.3 |
| [04](04-tensor-core-gemm-quant.md) | Tensor-core лейауты + fp8/int8 весов | decode | ×1.5–2 (удваивает roofline) |
| [05](05-kv-cache-paged-quant.md) | KV-кэш: paged + квантованный | decode | enabler батча и длины |
| [06](06-batching-parallelism.md) | Батчинг + 2×4090 | decode | throughput ×3–8 на многостраничных |

**Как складываются** (подробно — в конце [06](06-batching-parallelism.md)):
латентность одной страницы лечат §02 + §01 (+§04); throughput пакета страниц — §01-B +
§05 + §06 поверх готового data-parallel 2×4090. Движок из §01-B (vLLM/TRT-LLM) включает
§04/05/06 как встроенные фичи. §02/03 — vision-специфика, которую LLM-движок сам не
ускоряет. Точность не должна пострадать нигде — общий контур регрессии в
[../architecture.md](../architecture.md) §8.

Рекомендованный порядок: **§02 (flash vision) + §01-A (cuda-graphs)** → замер → **§01-B
(движок)** → **§04 (fp8)** → **§06 (батч) поверх §05**.

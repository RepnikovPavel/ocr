# dots.mocr — архитектура end-to-end (карта для оптимизации инференса)

Отчёт описывает полный путь одной страницы: `image + prompt → токены разметки`.
Нотация — как в конспекте *Attention is all you need* и в дипломе: шейпы тензоров
суперскриптом, именованные операции `MATMUL`/`CONCAT`, композиция слоёв через `\circ`,
residual+norm как `A = \mathrm{Norm}\circ(\mathrm{op}+I)`. Все числа сверены по исходникам
(`src/dots_mocr/transformers_patch/*`, `config.json`) адверсариально.

Модель как объект из конспекта: это **vision-encoder → decoder-only LLM**, а не
encoder-decoder seq2seq с teacher forcing. Отличия от оригинального трансформера
проговорены явно в §5: **GQA** вместо MHA, **RoPE** вместо аддитивных позиций,
**RMSNorm** вместо LayerNorm, **SwiGLU** вместо `linear_2 ∘ relu ∘ linear_1`, **pre-norm**
(`x + \mathrm{attn}(\mathrm{norm}(x))`) вместо post-norm, и авторегрессия с **KV-cache**
вместо teacher forcing.

---

## 0. Обозначения

$$
\begin{gather*}
\text{image} \in \mathbb{R}^{3 \times H \times W}\text{: входная страница (после smart\_resize)} \\
P = 14\text{: patch\_size} \qquad m = 2\text{: spatial\_merge\_size} \qquad F = mP = 28\text{: IMAGE\_FACTOR} \\
N_v = t \cdot h \cdot w\text{: число патчей, } h = H/P,\ w = W/P,\ t = 1 \\
N_{img} = N_v / m^2 = N_v/4\text{: число image-токенов, скармливаемых LLM} \\
N_{txt}\text{: число текстовых токенов промпта} \qquad S = N_{txt} + N_{img}\text{: длина префилла} \\
d = 1536\text{: hidden\_size (общий для vision и LLM)} \\
h_q = 12,\ d_{head} = d/h_q = 128\text{: головы и их размерность (обе башни)} \\
h_{kv} = 2,\ g = h_q/h_{kv} = 6\text{: KV-головы и группа GQA (только LLM)} \\
L_v = 42\text{: слоёв vision} \qquad L = 28\text{: слоёв LLM} \\
I_v = 4224,\ I = 8960\text{: SwiGLU hidden (vision / LLM)} \\
V = 151936\text{: vocab\_size} \qquad \theta = 10^6\text{: rope\_theta LLM} \qquad \text{dtype} = \mathrm{bf16}\ (2\text{ B})
\end{gather*}
$$

---

## 1. Конвейер end-to-end

$$
\text{tokens}
= \mathrm{Detok} \circ \underbrace{\mathrm{LLM}_{\text{decode}}}_{\text{авторегрессия, KV-cache}}
\circ\ \mathrm{Fuse}
\circ \big(\ \underbrace{\mathrm{VisionTower}}_{\text{encoder}} \ \big\|\ \mathrm{TextEmbed}\ \big)
\circ\ \mathrm{Preprocess}(\text{image}, \text{prompt})
$$

Прозой: страница режется на патчи и прогоняется через vision-энкодер (§3), давший
$N_{img}$ векторов; они **вставляются** (`masked_scatter`, §4) в текстовую
последовательность на места токенов-заглушек `<imgpad>`; получившаяся смесь
`inputs_embeds` идёт в LLM-декодер (§5), который в режиме **prefill** считает весь
контекст за один проход, а затем **decode** по одному токену с KV-кэшем до `<eos>`.

Reference-конфиг исполнения (`cli.py`): `attn_implementation="sdpa"` (для **обеих**
башен), `dtype=bf16`, `temperature=0` при бенчах (greedy), `max_new_tokens ≤ 16384`,
`num_thread=1`. Это важно для §7: sdpa в vision материализует маску $N_v\times N_v$.

---

## 2. Препроцессинг и токенизация

**smart_resize** (`utils/image_utils.py`): подгоняет $H,W$ так, что оба кратны
$F=28=mP$, площадь зажата в $[\text{MIN},\text{MAX}]=[3136,\ 11.29\cdot10^6]$ пикселей,
соотношение сторон сохраняется, отношение сторон $>200$ запрещено. На 4090 рантайм
дополнительно режет `max_pixels = 2.2\cdot10^6` (иначе OOM, см. §7).

$$
h = H/P,\quad w = W/P,\quad N_v = h\cdot w = \frac{H\cdot W}{P^2},\qquad
N_{img} = \frac{N_v}{m^2} = \frac{H\cdot W}{(mP)^2}
$$

**Патчификация** (`Qwen2VLImageProcessor`) упаковывает изображение в
$$
\text{pixel\_values} \in \mathbb{R}^{\,N_v \times 588},\qquad
588 = C\cdot t\cdot P\cdot P = 3\cdot1\cdot14\cdot14,\qquad
\text{grid\_thw} = [[\,t,h,w\,]] = [[1,h,w]]
$$
Батч-оси нет: патчи всех изображений сложены в один «packed» ряд, границы задаёт
`cu_seqlens` (§3). Промпт оборачивается chat-шаблоном с маркерами
`<|img|><|imgpad|><|endofimg|>`; каждый `<imgpad>` (id `151665`) — заглушка под один
image-токен.

**Конкретика (лимит $2.2\cdot10^6$ px фиксирует $N_v$ независимо от dpi):**
$$
N_v = \frac{\text{max\_pixels}}{P^2} = \frac{2.2\cdot10^6}{196} \approx 11224,\qquad
N_{img} = N_v/4 \approx 2806,\qquad
S \approx 2806 + N_{txt}\ (\approx 250) \approx 3050
$$
CPU-стоимость (resize, нормализация, упаковка) мала относительно GPU; при желании
переносится на GPU, но это не узкое место.

---

## 3. Vision Tower (`DotsVisionTransformer`)

Энкодер: `patch_embed → [42 блока] → post_trunk_norm → PatchMerger`.

**Patch embed** (`DotsPatchEmbed`, срез временной оси до Conv2d):
$$
\text{pixel\_values}^{N_v\times588}
\xrightarrow{\ \text{view}\ } \mathbb{R}^{N_v\times3\times14\times14}
\xrightarrow{\ \mathrm{Conv2d}(3\to1536,\,k{=}14,\,s{=}14)\ } \mathbb{R}^{N_v\times1536}
\xrightarrow{\ \mathrm{RMSNorm}\ } x^{N_v\times d}
$$

**2D-RoPE** из `grid_thw`: позиции $(h,w)$ переставлены в блоки $2\times2$ (чтобы merge
в конце был бесплатным reshape), rotary-размерность $= d_{head}/2 = 64$, $\theta=10^4$:
$$
\text{rope} \in \mathbb{R}^{N_v\times64},\qquad
\text{cu\_seqlens} = [\,0,\ N_v\,]\ \text{(одна страница} \Rightarrow \text{один сегмент)}
$$

**Блок** (pre-norm, $L_v=42$ раз), всё в $\mathbb{R}^{N_v\times d}$:
$$
\begin{gather*}
A_1 = \big(x + \mathrm{Attn}\circ\mathrm{RMSNorm}_1(x)\big) \\
A_2 = \big(A_1 + \mathrm{SwiGLU}\circ\mathrm{RMSNorm}_2(A_1)\big) \\
\text{block} = A_2 \circ A_1,\qquad
\mathrm{VisionTower}_{\text{trunk}} = \text{block}_{42}\circ\dots\circ\text{block}_1
\end{gather*}
$$

**Attention внутри блока — полное (не каузальное), $h_q=12$ голов**, без bias у
`qkv`/`proj` (`use_bias=false`), softmax в float32:
$$
\begin{gather*}
\text{qkv}: x^{N_v\times1536}\ \mathrm{MATMUL}\ W^{1536\times4608} \rightarrow \mathbb{R}^{N_v\times4608}
\xrightarrow{\ \text{reshape}\ } (q,k,v),\ \ q,k,v\in\mathbb{R}^{N_v\times12\times128} \\
q,k \leftarrow \mathrm{RoPE}(q,k;\ \text{rope}) \\
\text{head}_i:\ \mathrm{softmax}\!\Big(\tfrac{1}{\sqrt{128}}\, q_i^{N_v\times128}\ \mathrm{MATMUL}\ (k_i^{N_v\times128})^{T}\Big)^{N_v\times N_v}\ \mathrm{MATMUL}\ v_i^{N_v\times128}
\rightarrow \mathbb{R}^{N_v\times128} \\
\mathrm{Attn} = \mathrm{CONCAT}_i[\text{head}_i]^{N_v\times1536}\ \mathrm{MATMUL}\ W^{O,\ 1536\times1536}
\end{gather*}
$$
Матрица весов внимания $N_v\times N_v$ на голову — **квадратична по числу патчей**; в
режиме flash она не материализуется, в sdpa/eager — материализуется (§7).

**SwiGLU** (три матрицы, `SiLU`), **post_trunk_norm**, **PatchMerger** (сначала
`LayerNorm(1536,\ \varepsilon{=}10^{-6})`, затем слияние $2\times2$):
$$
\begin{gather*}
\mathrm{SwiGLU}(z) = \big(\mathrm{SiLU}(z\,\mathrm{MATMUL}\,W_1^{1536\times4224})\odot(z\,\mathrm{MATMUL}\,W_3^{1536\times4224})\big)\,\mathrm{MATMUL}\,W_2^{4224\times1536} \\
\mathrm{Merger}:\ \mathrm{LayerNorm}(x)^{N_v\times1536}
\xrightarrow{\ \text{view}\ } \mathbb{R}^{\frac{N_v}{4}\times6144}
\xrightarrow{\ 6144\to6144,\ \mathrm{GELU},\ 6144\to1536\ } E_{img}^{\,\frac{N_v}{4}\times1536}
\end{gather*}
$$

Выход энкодера $E_{img}\in\mathbb{R}^{N_{img}\times1536}$ — ровно $N_{img}=N_v/4$ векторов
в размерности LLM.

> Замечание к реализации: `rms_norm_eps` vision $=10^{-5}$, а LLM $=10^{-6}$; `ln_q`
> мерджера — это `LayerNorm` с $\varepsilon=10^{-6}$, а не RMSNorm.

---

## 4. Fusion (вставка vision в текст)

`prepare_inputs_embeds` (`modeling_dots_ocr.py`): текстовые id эмбеддятся, затем
vision-векторы **разбрасываются** по позициям `<imgpad>`:
$$
\begin{gather*}
\text{img\_mask} = [\,\text{input\_ids} = 151665\,] \in \{0,1\}^{B\times S} \\
\text{inputs\_embeds} = \mathrm{Embed}(\text{input\_ids})^{B\times S\times1536}
\ \xleftarrow{\ \text{masked\_scatter}\ }\ E_{img}^{\,N_{img}\times1536} \\
\text{инвариант: } \sum \text{img\_mask} = N_{img} = E_{img}.\text{size}(0)
\end{gather*}
$$
Есть страж усечения: если заглушек больше, чем vision-векторов, маска обрезается до
`size(0)`. Это чистый scatter по строкам (bandwidth-bound, не FLOP); кандидат на
слияние `Embed + scatter` без хостовых синхронизаций.

---

## 5. LLM-декодер (`DotsOCRForCausalLM` = тело Qwen2, $L=28$)

Pre-norm декодер-only, каузальный. Слой в $\mathbb{R}^{B\times S\times d}$:
$$
\begin{gather*}
A_1 = \big(x + \mathrm{GQA}\circ\mathrm{RMSNorm}_1(x)\big) \\
A_2 = \big(A_1 + \mathrm{SwiGLU}\circ\mathrm{RMSNorm}_2(A_1)\big) \\
\mathrm{LLM} = \mathrm{lm\_head}\circ\mathrm{RMSNorm}_f\circ\ \text{layer}_{28}\circ\dots\circ\text{layer}_1
\end{gather*}
$$

**GQA-внимание, $h_q=12$ Q-голов делят $h_{kv}=2$ KV-головы (группа $g=6$)**; bias есть
на `q/k/v`, у `o_proj` bias нет; RoPE применяется ко всей $d_{head}=128$ (NEOX
rotate-half, не interleaved), $\theta=10^6$:
$$
\begin{gather*}
q: x\,\mathrm{MATMUL}\,W^{Q,\,1536\times1536}\rightarrow \mathbb{R}^{B\times S\times12\times128},\quad
k,v: x\,\mathrm{MATMUL}\,W^{K,V,\,1536\times256}\rightarrow \mathbb{R}^{B\times S\times2\times128} \\
k,v \xrightarrow{\ \mathrm{REPEAT}\ g=6\ } \mathbb{R}^{B\times S\times12\times128} \qquad\text{(GQA-расшаривание)} \\
\mathrm{GQA} = \mathrm{CONCAT}_i\Big[\mathrm{softmax}\big(\tfrac{q_iK_i^{T}}{\sqrt{128}} + M\big)V_i\Big]\ \mathrm{MATMUL}\ W^{O,\,1536\times1536}
\end{gather*}
$$
Каузальная маска (как «t-th step» в конспекте, но с KV-кэшем вместо padding):
$$
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

**Prefill vs decode.** `prepare_inputs_for_generation` подаёт `pixel_values`/`grid_thw`
только на первом шаге; дальше vision не запускается:
$$
\begin{gather*}
\textbf{prefill}:\ \text{весь контекст } S \text{ за один проход} \Rightarrow \text{логиты последней позиции} \rightarrow y_1,\ \text{заполнен KV-cache} \\
\textbf{decode } t:\ [\,y_t\,]^{B\times1\times1536}\ \text{+ KV-cache} \rightarrow y_{t+1}\ \text{(один токен на шаг)}
\end{gather*}
$$
$$
\mathrm{lm\_head}: \mathbb{R}^{\dots\times1536}\ \mathrm{MATMUL}\ W^{1536\times V},\quad V=151936\ \text{(untied)}
$$

**KV-cache** (GQA даёт экономию $\times g=6$ против MHA):
$$
\text{bytes/token} = L\cdot h_{kv}\cdot d_{head}\cdot 2_{(K,V)}\cdot 2_{\mathrm{bf16}}
= 28\cdot2\cdot128\cdot2\cdot2 = 28672\ \text{B}
$$
$$
S=3000 \Rightarrow 86\ \text{MB},\qquad S=32768 \Rightarrow 940\ \text{MB}
$$

---

## 6. Где масса вычислений

Параметры (сверено, сумма точно даёт заявленные 3.04B):
$$
\underbrace{P_{vis}\approx1.26\text{B}}_{42\times28.9\text{M}}
+ \underbrace{P_{llm}^{\text{non-emb}}\approx1.31\text{B}}_{28\times46.8\text{M}}
+ \underbrace{\text{Embed}+\text{lm\_head}=2\times233\text{M}}_{\text{vocab }V\text{ большой}}
= 3.04\text{B}\ (6.08\text{ GB bf16})
$$

FLOP-структура (на страницу, $N_v\approx11224$, $S\approx3050$; ×2 FLOP/MAC):

| Этап | Формула | Значение | Доля |
|---|---|---|---|
| Vision attention (квадрат.) | $4 L_v N_v^2 d$ | **32.5 TFLOP** | 54% vision |
| Vision SwiGLU | $6 L_v N_v d I_v$ | 18.4 TFLOP | 31% |
| Vision QKV/proj | $8 L_v N_v d^2$ | 8.9 TFLOP | 15% |
| **Vision итого** | | **≈ 60 TFLOP** | |
| LLM prefill (linear) | $2 L\, S\,(2d^2 + 2\cdot d\cdot d_{kv} + 3 d I)$ | 7.9 TFLOP | |
| LLM prefill (attn) | $\sim 2 L S^2 d$ | 0.8 TFLOP | |
| **Prefill итого (до 1-го токена)** | | **≈ 69 TFLOP** | |
| LLM decode / токен | $2\,(L\cdot46.8\text{M} + d\,V)$ | **≈ 3.1 GFLOP** | |

Связь с измеренным (RTX 4090, dpi 150–200, `reports/benchmark_2x4090_2026-07-15.md`):

- **TTFT ≈ 3.5 с** на ~69 TFLOP $\Rightarrow$ ~20 TFLOP/s $\approx$ 12% от bf16-пика 4090
  (~165 TFLOP/s). Vision-attention — доминирующий и **квадратичный по $N_v$** член:
  удвоение разрешения → ×4 стоимости и памяти (отсюда OOM выше 2.2M px).
- **Decode 40–47 tok/s, util ~38%.** За токен стримится
  $P_{llm}^{\text{non-emb}} + \text{lm\_head} \approx 3.09$ ГБ весов bf16. Bandwidth-roofline
  $= 1008\ \text{ГБ/с} / 3.09\ \text{ГБ} \approx 320$ tok/s. Измеренные 47 = **~14% от
  roofline** $\Rightarrow$ декод упирается **не в память, а в launch-overhead** (per-token
  Python-цикл HF `generate`, мелкие ядра). Это главный резерв.

$$
\mathrm{AI}_{\text{decode}} = \frac{3.1\ \text{GFLOP}}{3.09\ \text{ГБ}} \approx 1\ \frac{\text{FLOP}}{\text{байт}}
\ \Rightarrow\ \text{строго memory/launch-bound (не compute-bound)}
$$

---

## 7. Поверхность оптимизации (в порядке ожидаемой отдачи)

1. **Декод launch-bound (14% от roofline) → устранить per-token overhead.**
   CUDA Graphs + статические шейпы декода; либо подмена движка на **TRT-LLM / vLLM**
   (paged-attention, фьюзы, cuda-graphs из коробки). Цель — приблизиться к ~320 tok/s.
   Наибольший выигрыш end-to-end на длинных выдачах OCR.

2. **Vision использует `sdpa` (дефолт cli) → материализует маску $N_v\times N_v$.**
   При $N_v\approx11224$: bool-маска 126 МБ, но SDPA внутри разворачивает scores
   $h_q\times N_v^2$ (bf16 ~2.4 ГБ) + float32 softmax — это и есть драйвер OOM и потолок
   разрешения. Поставить **flash-attn** и `attn_implementation="flash_attention_2"`
   для vision: варленовое внимание одним сегментом `cu_seqlens=[0,N_v]`, без
   материализации $\Rightarrow$ память $O(N_v)$, выше dpi, быстрее prefill.

3. **Фьюзы ядер (prefill-bound).** (a) `RMSNorm→qkv` и `RMSNorm→SwiGLU` эпилогом, чтобы
   убрать два HBM round-trip $[N_v,1536]$ на блок ×42. (b) SwiGLU: `fc1,fc3` одним GEMM
   $d\to2I$ + слитый SiLU-gate. (c) RoPE: вынести cos/sin из 42-блочного цикла (они
   одинаковы по блокам) и слить во flash-пролог; сейчас `apply_rotary_pos_emb_vision`
   каждый блок аллоцирует `[1,N_v,1,128]` для q и k в float32 — лишний трафик.

4. **Tensor-core GEMM (Ada SM89), bf16.** Проверить лейауты весов под tensor cores
   (как переразметка `Cout×Cin×KV` в `vxcore.cu`); большие GEMM'ы (qkv, SwiGLU,
   lm_head) — на пик TC. `lm_head` $1536\times151936$ — самый дорогой GEMM декода
   (0.47 GFLOP + чтение 467 МБ/токен); кандидат на fp8/int8 или на пропуск через
   top-k если разрешено семантикой.

5. **KV-cache.** GQA уже даёт $\times6$; для длинных выдач — paged-KV и квантование KV
   (int8/fp8), чтобы поднять batch и убрать фрагментацию.

6. **Батчинг и параллелизм.** Несколько страниц — как block-diagonal сегменты
   `cu_seqlens` в одном flash-вызове (внимание между страницами не течёт бесплатно),
   память $O(\sum N_i)$ вместо $O((\sum N_i)^2)$. Data-parallel по 2×4090 (реплика на
   GPU) уже реализован в `benchmarks/bench_throughput.py`.

---

## 8. Опорные точки для регрессии (для «оптимизации с регрессионными тестами»)

Что держать инвариантным при любой оптимизации (match-тесты в духе
`myadvise.md` — «доказать, что ответы совпадают»):

| Уровень | Эталон (референс bf16, greedy) | Критерий |
|---|---|---|
| Vision | $E_{img}\in\mathbb{R}^{N_{img}\times1536}$ | `allclose` (bf16 atol/rtol) |
| Fusion | `inputs_embeds` после scatter | `allclose` |
| Prefill | логиты последней позиции $\in\mathbb{R}^{V}$ | `allclose` + `argmax` совпадает |
| Decode | полная последовательность id при `temperature=0` | **точное совпадение id** |
| End-to-end | синтетика (§edge) + GPU-интеграция | recall не деградирует |

Готовые сторожа уже в репозитории: `tests/test_gpu_integration.py` (валидность
layout/OCR), `benchmarks/synth/` (cell/token/keyword recall с известным ground truth),
`benchmarks/bench_throughput.py` (tok/s, sec/page, util). Любой кернел-порт (Python→
CUDA/TRT) обязан пройти их без регресса метрик и с точным совпадением id при greedy —
это и есть контур «оптимизация → регрессия».

---

*Источники (сверено): `src/dots_mocr/transformers_patch/modeling_dots_vision.py`,
`modeling_dots_ocr.py`, `configuration_dots_ocr.py`, `processing_dots_ocr.py`,
`src/dots_mocr/cli.py`, `utils/image_utils.py`, `utils/consts.py`, checkpoint
`config.json` / `preprocessor_config.json` / `generation_config.json`.*

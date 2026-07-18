# dots.mocr (RepnikovPavel/ocr)

Веб-демка и оффлайн-инференс для **dots.mocr** (мультимодальный OCR / парсинг документов / генерация SVG).

**Архитектура end-to-end (карта для оптимизации инференса):** [docs/architecture.md](docs/architecture.md) —
полный путь `image + prompt → токены`, точные шейпы/FLOP'ы (сверено по исходникам),
где масса вычислений, поверхность CUDA-оптимизации и опорные точки для регрессии.

Пример доведён до уровня демок qwen3_l и qwen3_vl.

Base: `trtllm:1.3.0rc20`, Python 3.12, PyTorch 2.11 nightly, Transformers 5.5.4.

## Какая из двух моделей стоит времени

В репозитории живут две модели. Они очень разного качества, и это стоит знать
до того, как вкладываться:

- **dots.mocr (основная) — рабочая лошадка, на неё имеет смысл тратить время.**
  Парсинг документов (layout + OCR + таблицы + формулы) она делает уверенно и
  предсказуемо: валидный layout JSON, аккуратные bbox, разумный Markdown,
  ~60 t/s на одной 4090/4070 Ti. Оптимизация инференса, бенчмарки и доработки
  окупаются именно здесь. Всё в этом README по умолчанию про неё.
- **dots.mocr-svg (image→SVG) — так себе, вкладываться не стоит.** Она сносно
  берёт графики, диаграммы и простые фигуры, но на плотных текстовых страницах
  разваливается, регулярно отдаёт битый XML, требует temperature 0.9 и при этом
  генерирует на порядок дольше (~140 с/изображение против ~45 с/страницу у
  основной — см. `PAGE_SECONDS_ESTIMATE` в `demo/worker.py`). Держим её в архиве
  как демонстрацию возможности, **не оптимизируем и не развиваем**: цена работы
  несопоставима с отдачей. Все улучшения инференса (включая flex_attention
  ниже) сделаны и измерены только на основной модели.

```sh
./docker/build.sh
```

```sh
docker run --rm --gpus all --ipc=host -v /mnt:/mnt -v "$PWD:/workspace" dots-mocr:trtllm-1.3.0rc20 \
  dots-mocr --device cuda --ckpt /mnt/nvme/huggingface/models--rednote-hilab--dots.mocr/snapshots/main \
  --input_path /mnt/nvme/ocr_data/ko_2025/ko_2025_pages-to-jpg-0003.jpg --output /workspace/output
```

`src/dots_mocr/transformers_patch` is the local port of commit `d2eb02900bcee0cf02b653bbd31c3117b132e060`. Checkpoints and generated results are not committed.

## Локальный запуск (без Docker)

Всё, что раньше жило только в контейнере на сервере, запускается на обычной
рабочей машине с одной картой:

```bash
scripts/setup_local.sh                  # venv + пины (по умолчанию ~/.venvs/dots-mocr)
python3 scripts/check_local_env.py --ckpt "$CKPT"   # диагностика окружения

# веб-демка на http://127.0.0.1:8601
CKPTDIR="$CKPT" scripts/run_local.sh

# одна страница через CLI
PYTHONPATH=src ~/.venvs/dots-mocr/bin/python -m dots_mocr.cli \
    --ckpt "$CKPT" --input_path page.jpg --output ./output
```

`setup_local.sh` переиспользует уже установленный CUDA-torch, если он есть
(`DOTS_MOCR_FRESH_TORCH=1` — поставить свой), и доустанавливает остальное.
Состояние демки складывается в `./local_state` (в `.gitignore`).

> **Целевая платформа — transformers 5.x и bfloat16**, как у авторов
> (`torch_dtype=torch.bfloat16` в их README, то же в `config.json` чекпойнта).
> Под transformers 4.x порт не рассчитан и работать не будет;
> `check_local_env.py` проверяет версию до запуска.

Проверено на этой машине: RTX 4070 Ti (12 ГБ), torch 2.12.0+cu130,
transformers 5.5.4 — демка обрабатывает страницы 2.13 Мп за ~27 с при ~59 t/s.

```sh
python3 scripts/prepare_checkpoint.py /path/to/checkpoint --in-place
MAX_NEW_TOKENS=8 scripts/test_dots_mocr_cpu.sh
MAX_NEW_TOKENS=8 scripts/test_dots_mocr_gpu.sh
MAX_NEW_TOKENS=8 scripts/test_dots_mocr_svg_cpu.sh
MAX_NEW_TOKENS=8 scripts/test_dots_mocr_svg_gpu.sh
```

Measured results: `reports/validation_2026-07-12.json`.

## Веб-демки (dots.mocr и dots.mocr-svg)

Одно FastAPI-приложение (`demo/server.py`) в двух вариантах (`DEMO_VARIANT=mocr|svg`):

- **mocr** — парсинг документов: слева PDF-скроллер (зум, выбор страниц мышкой,
  drag-bbox для grounding OCR), справа очередь задач и результаты по страницам
  (рендер Markdown / raw MD / layout JSON / layout-картинка). Все режимы
  dots.mocr: layout_all, layout_only, ocr, grounding_ocr, web_parsing,
  scene_spotting, general (свой промпт).
- **svg** — image→SVG (dots.mocr-svg): выдача — inline-SVG / raw SVG /
  markdown / сравнение оригинал-рендер. Температура 0.9 (по рекомендации авторов).

Возможности: очередь задач с прогресс-баром и ETA (оценки из бенчмарков,
уточняются по факту), **живая скорость генерации — t/s, число токенов и TTFT
прямо в строке выполняющейся задачи** (и итоговые цифры в карточке каждой
страницы), отмена задач (в том числе на лету — генерация прерывается
за доли секунды, работает и после перезагрузки страницы), sessions/jobs/tasks в
SQLite (история переживает рестарты), мониторинг GPU, ленивое управление моделью:
**по умолчанию GPU свободен** — модель грузится при поступлении задачи и
выгружается после `DEMO_IDLE_UNLOAD_S` (180 c) простоя; чекбокс «не выгружать»
(do_not_unload_model) и кнопки загрузить/выгрузить — в шапке UI.

Запуск двух демок на сервере с 2x4090 (по контейнеру на GPU):

```bash
docker run -d --name demo_mocr --restart unless-stopped \
  --gpus '"device=0"' --ipc=host -p 127.0.0.1:8601:7860 \
  -v /mnt/data2:/mnt/data2 -w $REPO -e PYTHONPATH=$REPO/src \
  -e CKPTDIR=$CKPT_MOCR -e DEMO_STATE_DIR=$STATE/state_mocr \
  -e DEMO_VARIANT=mocr -e DEMO_DEVICE=cuda:0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  dots-mocr:bench-cu126 python3 -m demo.server
# demo_svg аналогично: device=1, порт 8602, CKPT=dots.mocr-svg, DEMO_VARIANT=svg
```

Доступ с локальной машины — только через SSH-туннель (порты слушают 127.0.0.1):

```bash
ssh -N -L 8601:127.0.0.1:8601 -L 8602:127.0.0.1:8602 <user>@<server>
# браузер: http://localhost:8601 (документы) и http://localhost:8602 (SVG)
```

Примечание: инференс идёт при dpi=150 и max_pixels=2.2M
(`DEMO_DPI`/`DEMO_MAX_PIXELS`) — авторские 11.3M px по-прежнему рассчитаны на
vLLM. Сам предел 2.2M раньше упирался в память (плотная маска sdpa), теперь с
flex_attention он упирается только во время декодирования: 2.2M px помещается
даже в 12 ГБ (см. [reports/perf-findings.md](reports/perf-findings.md)).
Бэкенд внимания переключается через `DEMO_ATTN_IMPLEMENTATION`.

## Attention: flex_attention по умолчанию

Vision-башня считает внимание через **`torch.nn.attention.flex_attention`**
(блочно-диагональная varlen-маска из `cu_seqlens` вместо плотной `[1, S, S]`).
Это дефолт: `--attn_implementation flex_attention`; доступны также `sdpa`,
`eager`, `flash_attention_2`.

Замеры на RTX 4070 Ti (12 ГБ), декодер везде на `sdpa` — сводка в
[reports/perf-findings.md](reports/perf-findings.md) (сырой вывод бенчмарков не
коммитится, воспроизводится скриптами из `benchmarks/`):

| vision-башня, 1.0 Мп | vision s | стр. s | peak VRAM |
|---|---|---|---|
| `flash_attention_2` (выбор авторов) | **0.408** | 24.20 | **5.891 ГиБ** |
| `flex_attention` | 0.446 (на 9% медленнее) | 24.31 | 5.892 ГиБ |
| `sdpa` (что стояло раньше) | 1.935 (в 4.7 раза медленнее) | 26.21 | 8.447 ГиБ |

Читать это надо так:

- **Против авторского `flash_attention_2` выигрыша нет** — flex на 9% медленнее,
  по памяти разница 1 МиБ. Ценность flex здесь одна: те же характеристики **на
  голом PyTorch**, без пакета flash-attn, без nvcc и без долгой сборки. В образе
  проекта flash_attn стоит, и там `--attn_implementation flash_attention_2` — не
  хуже, а чуть лучше.
- **Уходит именно `sdpa`**, который получал плотную маску `[1, S, S]` и потому
  сваливался с fused-ядер на math-путь: он в 4.7 раза медленнее и на 2.5 ГиБ
  тяжелее, и на 12 ГБ не тянет страницу 2.13 Мп вообще.
- **Скорость генерации не меняется ни от какого бэкенда** (58.6 / 59.6 / 59.8 t/s).
  Внимание занимает всего 3.6% времени шага декодирования, а сам декодинг — это
  92.7% времени страницы и упирается в чтение весов (2944 МиБ на токен), а не в
  арифметику. Ускорять его надо квантованием весов, не вниманием — разбор в
  [reports/perf-findings.md](reports/perf-findings.md).
- **Башня и декодер настраиваются независимо** (`--attn_implementation` и
  `--llm_attn_implementation`). Раньше это был один флаг, из-за чего A/B
  vision-башни незаметно двигал и декодер. Для декодера лучший замеренный
  вариант — `sdpa` (60.7 t/s против 47.3 у flash и 46.2 у flex).
- **Ответы совпадают семантически, но не побайтово.** Маска доказано та же
  (точное сравнение в тестах; в fp32 башня сходится до ~4e-4), однако в bf16
  42 остаточных слоя хаотически чувствительны к порядку суммирования. Это
  свойство модели, а не flex: уже поставляемые `eager` и `sdpa` расходятся между
  собой того же порядка (0.38 против 0.46 относительных — flex тут чуть дальше
  от sdpa, чем eager). А вот на **сквозном выходе** flex к sdpa заметно ближе:
  у него это ±1 px в bbox и редкие одиночные токены, тогда как eager на тех же
  страницах отдаёт другое число блоков и вдвое больше токенов
  ([reports/perf-findings.md](reports/perf-findings.md) §1). Регрессия проверяется
  тождеством маски, эквивалентностью в fp32 и семантическим сравнением
  сквозного выхода:

```bash
python3 -m pytest tests/test_vision_flex_attention.py -q
python3 -m benchmarks.bench_attention --ckpt "$CKPT" --input doc.pdf \
    --pages 0,1,2 --attn sdpa,flex_attention --output-dir reports/flexattn
```

## Оффлайн CLI (как раньше)

```sh
./docker/build.sh
```

```sh
docker run --rm --gpus all --ipc=host -v /mnt:/mnt -v "$PWD:/workspace" dots-mocr:trtllm-1.3.0rc20 \
  dots-mocr --device cuda --ckpt /mnt/nvme/huggingface/models--rednote-hilab--dots.mocr/snapshots/main \
  --input_path /mnt/nvme/ocr_data/ko_2025/ko_2025_pages-to-jpg-0003.jpg --output /workspace/output
```

## Подготовка чекпойнта / тесты

```sh
python3 scripts/prepare_checkpoint.py /path/to/checkpoint --in-place
MAX_NEW_TOKENS=8 scripts/test_dots_mocr_cpu.sh
...
```

`src/dots_mocr/transformers_patch` — порт transformers кода.

## Структура репозитория (как в qwen3_vl / qwen3_l)

- `demo/` — веб-сервер FastAPI + встроенный UI
- `docker/` — build.sh, Dockerfile, run.sh (оффлайн), **run_demo.sh** (веб-демка)
- `src/dots_mocr/` — основная логика (cli, model, utils, patch)
- `scripts/`, `tests/`, `reports/`

## Полезные команды

```bash
# логи демки
docker logs -f dots_mocr_demo

# остановить
docker rm -f dots_mocr_demo

# пересобрать и перезапустить
./docker/build.sh
docker rm -f dots_mocr_demo; ./docker/run_demo.sh "$CKPT" "$STATE" 8002
```

## Тесты

Юнит-тесты (без GPU и без чекпойнта, ~1 мин):

```bash
python3 -m pytest tests -m "not gpu"
```

Регрессия attention-бэкендов (`tests/test_vision_flex_attention.py`): тождество
маски flex и sdpa (точное, `torch.equal`), эквивалентность башни в fp32, явные
ошибки на неподдерживаемых конфигурациях и проверка, что flex не протекает в
языковую модель. CUDA-часть (реальная геометрия 1536/12 голов, bf16) запускается
сама, если есть карта, — чекпойнт для неё не нужен.

`sdpa` остаётся в дереве **только как эталон для этих тестов**: рантайм-откатов
на него нет, неподдерживаемая конфигурация падает с внятной ошибкой.

GPU-интеграционные тесты против реального чекпойнта — проверяют, что базовые функции
dots.mocr (OCR, layout JSON c легальными bbox и известными категориями, многостраничный
режим) реально отрабатывают на этом порте:

```bash
DOTS_MOCR_CKPT=/path/to/snapshot \
DOTS_MOCR_TEST_PDF=/path/to/mobilenetv3.pdf \
python3 -m pytest tests/test_gpu_integration.py -v -m gpu
```

Окружение для машин с драйвером CUDA 12.x (cu13x-сборки не стартуют на GeForce):

```bash
docker build -f docker/Dockerfile.bench -t dots-mocr:bench-cu126 docker/
docker run --rm --gpus all --ipc=host -v /mnt:/mnt -v "$PWD:/workspace/ocr" \
  -w /workspace/ocr -e PYTHONPATH=/workspace/ocr/src dots-mocr:bench-cu126 \
  python3 -m pytest tests -m "not gpu"
```

## Бенчмарки / сетап 2x RTX 4090

Модель 3B (~6 ГБ bf16) целиком помещается на одну 24 ГБ карту, поэтому максимальная
загрузка двух 4090 — **data parallel**: по одной реплике модели на GPU, страницы PDF
раздаются воркерам round-robin (`benchmarks/bench_throughput.py` поднимает по процессу
на GPU через `CUDA_VISIBLE_DEVICES` и мержит метрики).

```bash
# одна страница / один GPU (латентность, TTFT)
python3 -m benchmarks.bench_throughput --ckpt $CKPT --pdf doc.pdf --gpus 0 --pages 0 \
  --output reports/bench_single.json

# весь PDF на двух GPU (полная загрузка)
python3 -m benchmarks.bench_throughput --ckpt $CKPT --pdf doc.pdf --gpus 0,1 \
  --output reports/bench_2gpu.json

# вся серия: 1 страница, 1 GPU, 2 GPU, 2 GPU с batch=2
CKPT=... PDF=... ./scripts/run_bench_2x4090.sh
```

В отчёте: sec/page (латентность и wall), tokens/s на воркер и суммарно, TTFT,
валидность JSON по страницам, загрузка/память/мощность GPU по сэмплам nvidia-smi.
Результаты измерений: `reports/benchmark_2x4090_2026-07-15.md`.

## Ссылки

Оригинал модели: https://huggingface.co/rednote-hilab/dots.mocr

## Секреты

Секреты (SSH, токены, адреса серверов) в репозиторий не коммитятся — храните их локально.

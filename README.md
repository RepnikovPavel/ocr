# dots.mocr (RepnikovPavel/ocr)

Веб-демка и оффлайн-инференс для **dots.mocr** (мультимодальный OCR / парсинг документов / генерация SVG).

Пример доведён до уровня демок qwen3_l и qwen3_vl.

Base: `trtllm:1.3.0rc20`, Python 3.12, PyTorch 2.11 nightly, Transformers 5.5.4.

```sh
./docker/build.sh
```

```sh
docker run --rm --gpus all --ipc=host -v /mnt:/mnt -v "$PWD:/workspace" dots-mocr:trtllm-1.3.0rc20 \
  dots-mocr --device cuda --ckpt /mnt/nvme/huggingface/models--rednote-hilab--dots.mocr/snapshots/main \
  --input_path /mnt/nvme/ocr_data/ko_2025/ko_2025_pages-to-jpg-0003.jpg --output /workspace/output
```

`src/dots_mocr/transformers_patch` is the local port of commit `d2eb02900bcee0cf02b653bbd31c3117b132e060`. Checkpoints and generated results are not committed.

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
уточняются по факту), отмена задач (в том числе на лету — генерация прерывается
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

Примечание: на 24 GB картах инференс идёт при dpi=150 и max_pixels=2.2M
(`DEMO_DPI`/`DEMO_MAX_PIXELS`) — квадратичный vision attention на авторских
11.3M px не помещается в память (см. отчёт бенчмарков).

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

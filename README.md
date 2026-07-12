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

## Быстрый деплой веб-демки на сервер

Требуется: Linux + Docker + NVIDIA Container Toolkit, GPU, ~ модель  ~10-15 ГБ.

### 1. Клонировать

```bash
git clone https://github.com/RepnikovPavel/ocr.git
cd ocr
```

### 2. Сборка образа

```bash
./docker/build.sh
```

### 3. Запуск демки (пример путей на tuna)

```bash
export CKPT=/mnt/nvme/huggingface/models--rednote-hilab--dots.mocr/snapshots/main
export STATE=/mnt/nvme/ocr_demo_state
mkdir -p "$STATE"

./docker/run_demo.sh "$CKPT" "$STATE" 8002
```

Или вручную:

```bash
docker run -d --name dots_mocr_demo \
  --restart unless-stopped \
  --gpus all --network bridge \
  --publish 127.0.0.1:8002:7860 \
  -e CKPTDIR=/models \
  -e DEMO_STATE_DIR=/state \
  --mount type=bind,src="$CKPT",target=/models,readonly \
  --mount type=bind,src="$STATE",target=/state \
  dots-mocr:trtllm-1.3.0rc20 \
  python3 -m demo.server
```

### 4. Доступ

На сервере: `http://localhost:8002`

С локальной машины:

```bash
ssh -N -L 8002:127.0.0.1:8002 tuna-server
# браузер → http://localhost:8002
```

UI позволяет загружать изображение или PDF, выбирать prompt mode (layout, OCR, SVG, grounding и т.д.), запускать и получать markdown + визуализацию layout + файлы для скачивания.

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

## Ссылки

Оригинал модели: https://huggingface.co/rednote-hilab/dots.mocr

## Секреты / сервер

Смотри `/mnt/nvme/secrets/` (gpt_56 / glm_52) — SSH, токены, пути к workspace и HF на сервере tuna (ru.tuna.am:31932). Не коммитить.

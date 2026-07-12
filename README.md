# dots.mocr offline

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

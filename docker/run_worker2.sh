#!/usr/bin/env bash
# Launch a standalone DemoWorker on GPU 1 (the second card).
#
# Shares /state/demo.db and /state/jobs/ with the demo container (Container A)
# so both workers pull from the same task queue. Each worker claims tasks
# atomically via SQLite; `ocrc --split 2` creates two tasks that the two
# workers pick up independently — true dual-GPU parallelism with zero CUDA
# conflicts because each worker is in its own process.
#
# Usage:
#   bash docker/run_worker2.sh
#
# Requires the same image, same mounts, but --gpus device=1 and no web server.
set -Eeuo pipefail

IMAGE="${DOTS_MOCR_IMAGE:-dots-mocr:trtllm-1.3.0rc20}"
CONTAINER="${DOTS_MOCR_WORKER2_NAME:-dots_mocr_worker2}"

# Paths must match the demo container's mounts exactly.
CKPT_DIR="${CKPT_DIR:-/mnt/nvme2/dots_mocr_ckpt}"
REPO_DIR="${REPO_DIR:-/mnt/nvme2/ocr-flex}"
STATE_DIR="${STATE_DIR:-/mnt/nvme2/ocr_server_state}"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d \
    --name "$CONTAINER" \
    --restart unless-stopped \
    --init \
    --gpus '"device=1"' \
    --ipc host \
    --shm-size 8g \
    --network host \
    --mount type=bind,source="$CKPT_DIR",target=/models,readonly \
    --mount type=bind,source="$REPO_DIR",target=/opt/dots-mocr \
    --mount type=bind,source="$STATE_DIR",target=/state \
    --env CKPTDIR=/models \
    --env DEMO_STATE_DIR=/state \
    --env DEMO_ENGINE=transformers \
    --env DEMO_DEVICE=cuda:1 \
    --env DEMO_AUTOSTART=0 \
    --env DEMO_IDLE_UNLOAD_S=10 \
    --env DEMO_WORKER_NAME=worker-gpu1 \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --workdir /opt/dots-mocr \
    "$IMAGE" \
    python3 -m demo.worker_standalone

echo "started $CONTAINER on cuda:1"
echo "logs: docker logs -f $CONTAINER"

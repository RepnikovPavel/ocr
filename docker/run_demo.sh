#!/usr/bin/env bash
set -Eeuo pipefail

ckpt_dir="${1:-}"
state_dir="${2:-}"
port="${3:-8002}"
image_name="${DOTS_MOCR_IMAGE:-dots-mocr:trtllm-1.3.0rc20}"
container_name="${DOTS_MOCR_CONTAINER_NAME:-dots_mocr_demo}"
gpu_request="${DOTS_MOCR_GPUS:-all}"

[[ -d "${ckpt_dir}" ]] || { echo "checkpoint directory does not exist: ${ckpt_dir}" >&2; exit 2; }
[[ -n "${state_dir}" ]] || { echo "state directory is required (for outputs + sessions)" >&2; exit 2; }
[[ "${port}" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || { echo "invalid port: ${port}" >&2; exit 2; }
mkdir -p "${state_dir}"
ckpt_dir="$(cd "${ckpt_dir}" && pwd -P)"
state_dir="$(cd "${state_dir}" && pwd -P)"

docker rm -f "${container_name}" >/dev/null 2>&1 || true

docker run -d \
    --name "${container_name}" \
    --restart unless-stopped \
    --init \
    --read-only \
    --cap-drop=ALL \
    --security-opt=no-new-privileges \
    --pids-limit=4096 \
    --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    --tmpfs /tmp:rw,nosuid,nodev,exec,size=8g,mode=1777 \
    --gpus "${gpu_request}" \
    --network bridge \
    --publish "127.0.0.1:${port}:7860/tcp" \
    --env HOME=/tmp \
    --env TRITON_CACHE_DIR=/tmp/triton-cache \
    --env HF_HOME=/models \
    --env HF_HUB_CACHE=/models \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env CKPTDIR=/models \
    --env DEMO_STATE_DIR=/state \
    --env PORT=7860 \
    --env DOTS_MOCR_WEB_PORT=7860 \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --mount "type=bind,src=${ckpt_dir},dst=/models,readonly" \
    --mount "type=bind,src=${state_dir},dst=/state" \
    "${image_name}" \
    python3 -m demo.server

echo "demo started: docker logs -f ${container_name}"
echo "open http://127.0.0.1:${port} or ssh -N -L ${port}:127.0.0.1:${port} ..."

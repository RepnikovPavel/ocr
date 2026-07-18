#!/usr/bin/env bash
# Run the web demo backed by vLLM instead of the in-process model.
#
#   CKPTDIR=/path/to/snapshot scripts/run_local_vllm.sh
#
# Starts a vLLM server on :8000 and the same demo UI on :8601 with
# DEMO_ENGINE=vllm, so the queue, artifacts and live tokens/s are identical and
# only the inference path differs — which is what makes the two comparable.
#
# One GPU cannot hold both engines: vLLM reserves most of the card, so the demo
# must not also load the model. That is exactly what DEMO_ENGINE=vllm ensures.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="${DOTS_MOCR_VENV:-$HOME/.venvs/dots-mocr}"
CKPTDIR="${CKPTDIR:-${DOTS_MOCR_CKPT:-}}"
PORT="${PORT:-8601}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_IMAGE="${DOTS_MOCR_VLLM_IMAGE:-dots-mocr:vllm}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
# the demo asks for up to 16384 output tokens by default, and vLLM rejects a
# request whose max_completion_tokens exceeds the context window outright
MAX_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
STATE="${DEMO_STATE_DIR:-$ROOT/local_state/vllm}"
CONTAINER="${VLLM_CONTAINER:-dots_vllm_demo}"

[[ -n "$CKPTDIR" && -d "$CKPTDIR" ]] || { echo "CKPTDIR must point at the checkpoint snapshot" >&2; exit 2; }
[[ -x "$VENV/bin/python" ]] || { echo "no venv at $VENV — run scripts/setup_local.sh" >&2; exit 2; }

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if ! curl -sf --max-time 3 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    # vLLM resolves the architecture through transformers, which needs auto_map in
    # config.json — scripts/prepare_checkpoint.py strips it for this repo's own
    # loader. Serve a symlink shadow with the original config instead of editing
    # the checkpoint, which other tooling here depends on staying stripped.
    SHADOW="$STATE/vllm_ckpt"
    rm -rf "$SHADOW"; mkdir -p "$SHADOW"
    for f in "$CKPTDIR"/*; do
        case "$(basename "$f")" in config.json|config.json.bak) ;; *) ln -s "$f" "$SHADOW/" ;; esac
    done
    if [[ -f "$CKPTDIR/config.json.bak" ]]; then
        cp "$CKPTDIR/config.json.bak" "$SHADOW/config.json"
    else
        cp "$CKPTDIR/config.json" "$SHADOW/config.json"
    fi

    echo "starting vLLM on :${VLLM_PORT} (gpu-memory-utilization ${GPU_UTIL})"
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker run -d --rm --name "$CONTAINER" --gpus all --ipc=host --network host \
        -v /mnt:/mnt:ro -v "$STATE:/state" \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/tmp/hf \
        "$VLLM_IMAGE" /state/vllm_ckpt \
        --served-model-name rednote-hilab/dots.mocr \
        --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_LEN" \
        --chat-template-content-format string --trust-remote-code \
        --host 127.0.0.1 --port "$VLLM_PORT" >/dev/null

    echo -n "waiting for vLLM"
    until curl -sf --max-time 3 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; do
        docker ps --filter "name=$CONTAINER" --format '{{.Names}}' | grep -q . || {
            echo; echo "vLLM exited early:" >&2; docker logs "$CONTAINER" 2>&1 | tail -20 >&2; exit 1; }
        echo -n "."; sleep 5
    done
    echo " up"
else
    echo "reusing the vLLM server already on :${VLLM_PORT}"
    trap - EXIT   # not ours to stop
fi

mkdir -p "$STATE"
export CKPTDIR DEMO_STATE_DIR="$STATE" PORT
export DEMO_ENGINE=vllm
export DEMO_VLLM_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export DEMO_VLLM_MODEL="${DEMO_VLLM_MODEL:-rednote-hilab/dots.mocr}"
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DEMO_AUTOSTART="${DEMO_AUTOSTART:-1}"   # nothing to load, so connect immediately

echo "demo (engine=vllm) on http://127.0.0.1:${PORT}"
exec "$VENV/bin/python" -m demo.server

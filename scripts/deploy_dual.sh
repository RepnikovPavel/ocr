#!/usr/bin/env bash
# Deploy dots.mocr (vLLM, GPU0) AND GLM-OCR (transformers, GPU1) as two demos,
# cross-linked by the peer-link in each header.
#
#   CKPT=/path/to/dots.mocr STATE=/path/to/state scripts/deploy_dual.sh
#   scripts/deploy_dual.sh --down
#
# Why two demos, not one with a selector: GLM-OCR cannot run on this host's vLLM
# (driver/toolkit/CUDA mismatch — see docker/compose.server.yml), so it serves
# in-process via transformers. That needs a GPU in the demo container itself,
# which dots.mocr's vLLM-backed demo does not have. Two single-model demos each
# on their own GPU, linked to each other, keeps both hot and lets you parse the
# same document with each to compare quality and speed.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMPOSE="$ROOT/docker/compose.server.yml"

CKPT="${CKPT:-${CKPTDIR:-${DOTS_MOCR_CKPT:-}}}"
STATE="${STATE:-${DEMO_STATE_DIR:-$ROOT/server_state}}"
DEMO_PORT="${DEMO_PORT:-8601}"
GLM_DEMO_PORT="${GLM_DEMO_PORT:-8603}"
VLLM_PORT="${VLLM_PORT:-8000}"
# Bind on the LAN so the demos are reachable as http://<host>:<port> from other
# machines (bookmarked URLs), not only through an SSH tunnel. This host is on a
# trusted internal network; override with DOTS_MOCR_BIND=127.0.0.1 to lock it
# back down. NOTE: the demo has no auth — anyone who can route to this host can
# use it. 8602 is taken by an existing demo_svg container, so GLM defaults to 8603.
BIND="${DOTS_MOCR_BIND:-0.0.0.0}"
# GLM-OCR HF cache. /mnt/data2/PMRepnikov is the documented "write big here"
# spot; the home volume is small and was wiped once during this work.
GLM_HF_CACHE="${GLM_HF_CACHE:-/mnt/data2/PMRepnikov/glm_ocr/hfhub}"
GLM_STATE_DIR="${GLM_STATE_DIR:-$STATE-glm}"
GLM_MODEL_ID="${GLM_MODEL_ID:-zai-org/GLM-OCR}"

if [[ "${1:-}" == "--down" ]]; then
    REPO_ROOT="$ROOT" CKPT_DIR="${CKPT:-/nonexistent}" STATE_DIR="$STATE" \
    VLLM_CKPT_SHADOW="$STATE/vllm_ckpt" GLM_CKPT_DIR="${GLM_HF_CACHE}" \
        docker compose -f "$COMPOSE" --profile dual down
    echo "stopped"
    exit 0
fi

[[ -n "$CKPT" && -d "$CKPT" ]] || { echo "CKPT must point at the dots.mocr snapshot" >&2; exit 2; }
mkdir -p "$STATE" "$GLM_HF_CACHE" "$GLM_STATE_DIR"
chmod -R a+rwX "$STATE" "$GLM_STATE_DIR" 2>/dev/null || true
CKPT=$(cd "$CKPT" && pwd -P)
STATE=$(cd "$STATE" && pwd -P)

# --- dots.mocr: symlink shadow that restores auto_map (same as deploy_server.sh).
# IMPORTANT: the shadow must live OUTSIDE /mnt. The vllm container bind-mounts
# both $SHADOW:/ckpt AND /mnt:/mnt; when $SHADOW is under /mnt, the nested bind
# leaves /ckpt empty inside the container (a Docker bind-propagation gotcha),
# and vLLM fails with "can't load image processor for /ckpt".
SHADOW="${VLLM_CKPT_SHADOW:-$HOME/ocr_state/vllm_ckpt}"
rm -rf "$SHADOW"; mkdir -p "$SHADOW"
for f in "$CKPT"/*; do
    case "$(basename "$f")" in config.json|config.json.bak) ;; *) ln -s "$f" "$SHADOW/" ;; esac
done
if [[ -f "$CKPT/config.json.bak" ]]; then cp "$CKPT/config.json.bak" "$SHADOW/config.json"
else cp "$CKPT/config.json" "$SHADOW/config.json"; fi
python3 - "$SHADOW/config.json" <<'PY' || { echo "dots.mocr config has no auto_map; vLLM won't resolve dots_ocr" >&2; exit 1; }
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("auto_map") else 1)
PY

# --- GLM-OCR: download into the HF cache if absent, then resolve the snapshot.
GLM_SNAPSHOT="$(find "$GLM_HF_CACHE/models--zai-org--GLM-OCR/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1 || true)"
if [[ -z "$GLM_SNAPSHOT" ]]; then
    echo "GLM-OCR not found in $GLM_HF_CACHE — downloading ${GLM_MODEL_ID} (~2.6 GB)..."
    HF_HUB_CACHE="$GLM_HF_CACHE" python3 - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("$GLM_MODEL_ID", cache_dir="$GLM_HF_CACHE"))
PY
    GLM_SNAPSHOT="$(find "$GLM_HF_CACHE/models--zai-org--GLM-OCR/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)"
fi
# Path to the snapshot FROM INSIDE the container (the cache root is at /hf, so
# the snapshot's symlinks into ../../blobs/ resolve).
GLM_CKPT="/hf/${GLM_SNAPSHOT#"$GLM_HF_CACHE/"}"
echo "  GLM-OCR snapshot (in-container): $GLM_CKPT"

export REPO_ROOT="$ROOT" CKPT_DIR="$CKPT" STATE_DIR="$STATE"
export VLLM_CKPT_SHADOW="$SHADOW" DEMO_PORT VLLM_PORT GLM_DEMO_PORT
export DEMO_HOST="$BIND"
export GLM_CKPT_DIR="$GLM_HF_CACHE" GLM_CKPT GLM_STATE_DIR

echo "deploying dual-model stack from $ROOT"
echo "  dots.mocr : GPU0  vLLM 127.0.0.1:${VLLM_PORT}  demo ${BIND}:${DEMO_PORT}"
echo "  GLM-OCR  : GPU1  transformers (in-process)  demo ${BIND}:${GLM_DEMO_PORT}"
docker compose -f "$COMPOSE" --profile dual up -d

wait_for() {  # <url> <needle> <name> <tries>
    local url="$1" needle="$2" name="$3" tries="${4:-90}"
    echo -n "waiting for $name"
    for _ in $(seq 1 "$tries"); do
        if curl -sf --max-time 3 "$url" 2>/dev/null | grep -q "$needle"; then echo " up"; return 0; fi
        echo -n "."; sleep 5
    done
    echo " FAILED"; return 1
}
wait_for "http://127.0.0.1:${VLLM_PORT}/v1/models" "${VLLM_MODEL_NAME:-rednote-hilab/dots.mocr}" "dots.mocr vLLM" || {
    echo "dots.mocr vLLM did not come up; docker compose logs vllm" >&2; exit 1; }
wait_for "http://127.0.0.1:${DEMO_PORT}/healthz" '"worker_alive":true' "dots.mocr demo" 30 || {
    echo "dots.mocr demo did not come up; docker compose logs demo" >&2; exit 1; }
wait_for "http://127.0.0.1:${GLM_DEMO_PORT}/healthz" '"worker_alive":true' "GLM-OCR demo" 30 || {
    echo "GLM-OCR demo did not come up; docker compose logs demo_glm" >&2; exit 1; }

echo
echo "=== ready ==="
echo "open both demos through a tunnel and compare:"
echo "  ssh -N -L ${DEMO_PORT}:127.0.0.1:${DEMO_PORT} -L ${GLM_DEMO_PORT}:127.0.0.1:${GLM_DEMO_PORT} <server>"
echo "  dots.mocr : http://127.0.0.1:${DEMO_PORT}"
echo "  GLM-OCR  : http://127.0.0.1:${GLM_DEMO_PORT}   (peer link in either header)"

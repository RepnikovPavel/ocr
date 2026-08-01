#!/usr/bin/env bash
# Deploy BOTH models hot on their own GPUs + the model-selector demo.
#
# This is the dual-model counterpart of deploy_server.sh: dots.mocr's vLLM on
# GPU0 and GLM-OCR's vLLM on GPU1, plus the demo with DEMO_MODEL_SELECTOR=1 so
# the UI shows a dropdown to flip between them. Switching is instant — both
# servers stay up, the worker just repoints.
#
#   CKPT=/path/to/dots.mocr STATE=/path/to/state scripts/deploy_dual.sh
#   scripts/deploy_dual.sh --down
#
# Prereqs (auto-handled here):
#   * dots.mocr checkpoint with auto_map stripped (deploy_server.sh's shadow
#     rebuilds it for vLLM) — same as the single-model path.
#   * GLM-OCR downloaded into $GLM_HF_CACHE (downloaded on first run).
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMPOSE="$ROOT/docker/compose.server.yml"

CKPT="${CKPT:-${CKPTDIR:-${DOTS_MOCR_CKPT:-}}}"
STATE="${STATE:-${DEMO_STATE_DIR:-$ROOT/server_state}}"
DEMO_PORT="${DEMO_PORT:-8601}"
VLLM_PORT="${VLLM_PORT:-8000}"
GLM_VLLM_PORT="${GLM_VLLM_PORT:-8001}"
BIND="${DOTS_MOCR_BIND:-127.0.0.1}"
# Where GLM-OCR's HF cache lives. /mnt/data2/PMRepnikov is the documented
# "write big stuff here" spot on this host; the home volume is small and was
# wiped once during this work, so do NOT default to ~/.cache.
GLM_HF_CACHE="${GLM_HF_CACHE:-/mnt/data2/PMRepnikov/glm_ocr/hfhub}"
GLM_MODEL_ID="${GLM_MODEL_ID:-zai-org/GLM-OCR}"

if [[ "${1:-}" == "--down" ]]; then
    REPO_ROOT="$ROOT" CKPT_DIR="${CKPT:-/nonexistent}" STATE_DIR="$STATE" \
    VLLM_CKPT_SHADOW="$STATE/vllm_ckpt" GLM_CKPT_DIR="${GLM_HF_CACHE}" \
        docker compose -f "$COMPOSE" --profile dual down
    echo "stopped"
    exit 0
fi

[[ -n "$CKPT" && -d "$CKPT" ]] || { echo "CKPT must point at the dots.mocr snapshot" >&2; exit 2; }
mkdir -p "$STATE" "$GLM_HF_CACHE"
chmod -R a+rwX "$STATE" 2>/dev/null || true
CKPT=$(cd "$CKPT" && pwd -P)
STATE=$(cd "$STATE" && pwd -P)

# --- dots.mocr: symlink shadow that restores auto_map (same as deploy_server.sh)
SHADOW="$STATE/vllm_ckpt"
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
GLM_SNAPSHOT=$(cd "$GLM_SNAPSHOT" && pwd -P)
echo "  GLM-OCR snapshot: $GLM_SNAPSHOT"
# The compose file mounts the cache root at /hf (so the snapshot's symlinks into
# ../../blobs/ resolve), and vLLM needs the model path FROM INSIDE the container.
GLM_CKPT="/hf/${GLM_SNAPSHOT#"$GLM_HF_CACHE/"}"
export GLM_CKPT

export REPO_ROOT="$ROOT" CKPT_DIR="$CKPT" STATE_DIR="$STATE"
export VLLM_CKPT_SHADOW="$SHADOW" DEMO_PORT VLLM_PORT GLM_VLLM_PORT
export DEMO_HOST="$BIND"
export GLM_CKPT_DIR="$GLM_HF_CACHE"     # mounted read-only; snapshot resolved inside
export DEMO_MODEL_SELECTOR=1
export DEMO_DUAL=1

echo "deploying dual-model stack from $ROOT"
echo "  dots.mocr ckpt : $CKPT  (GPU0 -> 127.0.0.1:${VLLM_PORT})"
echo "  GLM-OCR cache  : $GLM_HF_CACHE  (GPU1 -> 127.0.0.1:${GLM_VLLM_PORT})"
echo "  demo           : ${BIND}:${DEMO_PORT}"
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
wait_for "http://127.0.0.1:${GLM_VLLM_PORT}/v1/models" "${GLM_MODEL_NAME:-glm-ocr}" "GLM-OCR vLLM" 60 || {
    echo "GLM-OCR vLLM did not come up; docker compose logs glm_vllm" >&2; exit 1; }
wait_for "http://127.0.0.1:${DEMO_PORT}/healthz" '"worker_alive":true' "demo" 30 || {
    echo "demo did not come up; docker compose logs demo" >&2; exit 1; }

echo
echo "=== ready ==="
echo "open the demo through a tunnel and flip the model dropdown:"
echo "  ssh -N -L ${DEMO_PORT}:127.0.0.1:${DEMO_PORT} <server>"
echo "  then http://127.0.0.1:${DEMO_PORT}"

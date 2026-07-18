#!/usr/bin/env bash
# Deploy the service on a server: vLLM plus the demo and the agent API.
#
#   CKPT=/path/to/snapshot STATE=/path/to/state scripts/deploy_server.sh
#   CKPT=... STATE=... DOTS_MOCR_BIND=0.0.0.0 scripts/deploy_server.sh   # LAN
#   scripts/deploy_server.sh --down                                      # stop
#
# Brings up two containers with docker compose. They are separate on purpose:
# vLLM owns the GPU and takes ~40 s to be ready, the demo restarts in a second,
# so shipping new code does not reload 6 GB of weights.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMPOSE="$ROOT/docker/compose.server.yml"

CKPT="${CKPT:-${CKPTDIR:-${DOTS_MOCR_CKPT:-}}}"
STATE="${STATE:-${DEMO_STATE_DIR:-$ROOT/server_state}}"
DEMO_PORT="${DEMO_PORT:-8601}"
VLLM_PORT="${VLLM_PORT:-8000}"
BIND="${DOTS_MOCR_BIND:-127.0.0.1}"

if [[ "${1:-}" == "--down" ]]; then
    REPO_ROOT="$ROOT" CKPT_DIR="${CKPT:-/nonexistent}" STATE_DIR="$STATE" \
    VLLM_CKPT_SHADOW="$STATE/vllm_ckpt" \
        docker compose -f "$COMPOSE" down
    echo "stopped"
    exit 0
fi

[[ -n "$CKPT" && -d "$CKPT" ]] || { echo "CKPT must point at the checkpoint snapshot" >&2; exit 2; }
mkdir -p "$STATE"
CKPT=$(cd "$CKPT" && pwd -P)
STATE=$(cd "$STATE" && pwd -P)

# vLLM resolves the architecture through transformers, which needs auto_map in
# config.json. scripts/prepare_checkpoint.py strips it for this repo's own
# loader, so serve a symlink shadow with the original config rather than editing
# a checkpoint other tooling here expects to stay stripped.
SHADOW="$STATE/vllm_ckpt"
rm -rf "$SHADOW"; mkdir -p "$SHADOW"
for f in "$CKPT"/*; do
    case "$(basename "$f")" in config.json|config.json.bak) ;; *) ln -s "$f" "$SHADOW/" ;; esac
done
if [[ -f "$CKPT/config.json.bak" ]]; then
    cp "$CKPT/config.json.bak" "$SHADOW/config.json"
else
    cp "$CKPT/config.json" "$SHADOW/config.json"
fi
python3 - "$SHADOW/config.json" <<'PY' || { echo "checkpoint config has no auto_map; vLLM will not resolve dots_ocr" >&2; exit 1; }
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("auto_map") else 1)
PY

# The demo binds the port itself (network_mode: host), so the bind address is
# handled by uvicorn rather than by a published port. 127.0.0.1 keeps it behind
# an SSH tunnel; 0.0.0.0 exposes it to everyone who can reach this host.
export REPO_ROOT="$ROOT" CKPT_DIR="$CKPT" STATE_DIR="$STATE"
export VLLM_CKPT_SHADOW="$SHADOW" DEMO_PORT VLLM_PORT
export DEMO_HOST="$BIND"

echo "deploying from $ROOT"
echo "  checkpoint : $CKPT"
echo "  state      : $STATE"
echo "  demo       : ${BIND}:${DEMO_PORT}   vLLM: 127.0.0.1:${VLLM_PORT}"
docker compose -f "$COMPOSE" up -d

echo -n "waiting for vLLM"
for _ in $(seq 1 90); do
    if curl -sf --max-time 3 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo " up"; break
    fi
    echo -n "."; sleep 5
done
echo -n "waiting for the demo"
for _ in $(seq 1 30); do
    if curl -sf --max-time 3 "http://127.0.0.1:${DEMO_PORT}/healthz" >/dev/null 2>&1; then
        echo " up"; break
    fi
    echo -n "."; sleep 3
done

echo
curl -sf --max-time 5 "http://127.0.0.1:${DEMO_PORT}/healthz" || {
    echo "the demo did not come up; docker compose -f $COMPOSE logs demo" >&2; exit 1; }
echo
if [[ "$BIND" == "127.0.0.1" ]]; then
    echo "reachable through a tunnel: ssh -N -L ${DEMO_PORT}:127.0.0.1:${DEMO_PORT} <server>"
else
    echo "listening on ${BIND}:${DEMO_PORT} — anyone who can reach this host can use it."
    echo "the service has no authentication; keep it on a trusted network."
fi
echo "agents: OCRC_SERVER=http://<host>:${DEMO_PORT} ocrc parse paper.pdf"

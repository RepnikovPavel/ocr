#!/usr/bin/env bash
# Run the dots.mocr web demo on this machine, without Docker, with the model
# loaded in-process (transformers). For the faster default engine see
# scripts/run_local_vllm.sh, which starts a vLLM server alongside the demo.
#
#   CKPTDIR=/path/to/snapshot scripts/run_local.sh
#   CKPTDIR=... PORT=8601 DEMO_DEVICE=cuda:0 scripts/run_local.sh
#
# State (uploads, artifacts, sqlite) goes to ./local_state by default, which is
# gitignored. Everything else is the same server as in the container; see
# demo/server.py for the full env list.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="${DOTS_MOCR_VENV:-$HOME/.venvs/dots-mocr}"

CKPTDIR="${CKPTDIR:-${DOTS_MOCR_CKPT:-}}"
[[ -n "$CKPTDIR" ]] || { echo "CKPTDIR is required (path to the checkpoint snapshot)" >&2; exit 2; }
[[ -d "$CKPTDIR" ]] || { echo "checkpoint directory does not exist: $CKPTDIR" >&2; exit 2; }

PY_BIN="$VENV/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
    echo "no venv at $VENV — run scripts/setup_local.sh first" >&2
    exit 2
fi

export CKPTDIR
export DEMO_STATE_DIR="${DEMO_STATE_DIR:-$ROOT/local_state}"
export DEMO_VARIANT="${DEMO_VARIANT:-mocr}"
# This launcher is the in-process engine: it loads the model itself and needs no
# server. The demo's own default is vllm, so state the choice rather than inherit
# it — otherwise this script would look for a vLLM server that it never starts.
export DEMO_ENGINE="${DEMO_ENGINE:-transformers}"
export PORT="${PORT:-8601}"
# both $ROOT (for the `demo` package) and $ROOT/src (for `dots_mocr`): relying on
# `python -m` seeding sys.path[0] from the cwd would tie the script to being run
# from the repository root
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# a local box usually has one card and a desktop session on it; keep the model
# off the GPU until there is actual work, exactly like the server deployment
export DEMO_AUTOSTART="${DEMO_AUTOSTART:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$DEMO_STATE_DIR"

"$PY_BIN" "$ROOT/scripts/check_local_env.py" --ckpt "$CKPTDIR" >/dev/null || {
    echo "environment check failed — run: $PY_BIN scripts/check_local_env.py --ckpt $CKPTDIR" >&2
    exit 1
}

echo "dots.mocr demo (variant=$DEMO_VARIANT) on http://127.0.0.1:$PORT"
echo "state: $DEMO_STATE_DIR"
exec "$PY_BIN" -m demo.server

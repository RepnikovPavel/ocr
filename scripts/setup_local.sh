#!/usr/bin/env bash
# Build a local (no-Docker) environment for dots.mocr.
#
#   scripts/setup_local.sh              # venv at ~/.venvs/dots-mocr
#   DOTS_MOCR_VENV=/path scripts/setup_local.sh
#
# By default the venv is created with --system-site-packages when the base
# interpreter already has a working CUDA torch, so a machine that is already set
# up for deep learning does not re-download several GB of wheels. Force a
# self-contained venv with DOTS_MOCR_FRESH_TORCH=1.
#
# transformers is pinned to 5.x: that is what this port targets, and what the
# container image ships, so local and container numerics match.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="${DOTS_MOCR_VENV:-$HOME/.venvs/dots-mocr}"
TRANSFORMERS_PIN="${DOTS_MOCR_TRANSFORMERS:-5.5.4}"
FRESH_TORCH="${DOTS_MOCR_FRESH_TORCH:-0}"

base_python="${DOTS_MOCR_PYTHON:-}"
if [[ -z "$base_python" ]]; then
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then base_python="$candidate"; break; fi
    done
fi
[[ -n "$base_python" ]] || { echo "no python3 found; set DOTS_MOCR_PYTHON" >&2; exit 2; }

"$base_python" - <<'PY' || { echo "python >= 3.10 is required" >&2; exit 2; }
import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY

# Reuse an existing CUDA torch unless told otherwise: installing torch is by far
# the slowest and largest part of this setup.
reuse_torch=0
if [[ "$FRESH_TORCH" != "1" ]] && "$base_python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    reuse_torch=1
    echo "found a working CUDA torch in $base_python — reusing it (DOTS_MOCR_FRESH_TORCH=1 to override)"
fi

if [[ ! -d "$VENV" ]]; then
    if (( reuse_torch )); then
        "$base_python" -m venv --system-site-packages "$VENV"
    else
        "$base_python" -m venv "$VENV"
    fi
    echo "created venv: $VENV"
else
    echo "reusing venv: $VENV"
fi

PY_BIN="$VENV/bin/python"
"$PY_BIN" -m pip install --quiet --upgrade pip

if ! "$PY_BIN" -c "import torch" >/dev/null 2>&1; then
    # Match the wheel to the installed driver: cu13x builds refuse to start on
    # older drivers, and cu12x wheels run fine on a 13.x driver.
    driver_cuda=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    index_url="https://download.pytorch.org/whl/cu126"
    if [[ -n "$driver_cuda" ]]; then
        major=${driver_cuda%%.*}
        (( major >= 580 )) && index_url="https://download.pytorch.org/whl/cu130"
    fi
    echo "installing torch from $index_url (driver ${driver_cuda:-unknown})"
    "$PY_BIN" -m pip install --quiet torch torchvision --index-url "$index_url"
fi

echo "installing dots.mocr dependencies (transformers==$TRANSFORMERS_PIN)"
"$PY_BIN" -m pip install --quiet \
    "transformers==$TRANSFORMERS_PIN" \
    accelerate \
    "CairoSVG==2.8.2" \
    "PyMuPDF==1.28.0" \
    "packaging==25.0" \
    "qwen-vl-utils==0.0.14" \
    fastapi uvicorn python-multipart \
    opencv-python-headless \
    pytest pytest-timeout \
    tqdm requests openai httpx pillow pydantic

echo
"$PY_BIN" "$ROOT/scripts/check_local_env.py" ${CKPTDIR:+--ckpt "$CKPTDIR"}

cat <<EOF

local environment ready: $VENV

  # unit tests
  $PY_BIN -m pytest tests -m "not gpu"

  # one image through the CLI
  PYTHONPATH=$ROOT/src $PY_BIN -m dots_mocr.cli \\
      --ckpt "\${CKPTDIR:-/path/to/snapshot}" --input_path page.jpg --output ./output

  # the web demo on http://127.0.0.1:8601
  CKPTDIR="\${CKPTDIR:-/path/to/snapshot}" scripts/run_local.sh
EOF

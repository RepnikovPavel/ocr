#!/usr/bin/env bash
# Run the same images through every attention backend and report whether they agree.
#
#   CKPT=/path/to/snapshot scripts/run_agreement_matrix.sh img1.jpg img2.png
#
# Each backend needs a different environment, which is the whole reason this is a
# script and not a pytest:
#   sdpa, flex_attention  — the local venv (scripts/setup_local.sh)
#   flash_attention_2     — the project container, which ships flash-attn
#   vLLM                  — the vLLM container (docker/Dockerfile.vllm)
#
# All backends must see identical inputs, so max_pixels is pinned here rather than
# left to each environment's default. 1.0 Mpx is chosen because sdpa materializes a
# dense [1,S,S] mask and OOMs above ~1.5 Mpx on a 12 GiB card — a backend that
# cannot run is not a backend that can be compared.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="${DOTS_MOCR_VENV:-$HOME/.venvs/dots-mocr}"
CKPT="${CKPT:-${DOTS_MOCR_CKPT:-}}"
OUT="${OUT:-$ROOT/reports/agreement}"
MAX_PIXELS="${MAX_PIXELS:-1000000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12288}"
IMAGE="${DOTS_MOCR_IMAGE:-dots-mocr:trtllm-1.3.0rc20}"
VLLM_IMAGE="${DOTS_MOCR_VLLM_IMAGE:-dots-mocr:vllm}"

[[ -n "$CKPT" ]] || { echo "CKPT is required" >&2; exit 2; }
[[ $# -ge 1 ]] || { echo "usage: $0 <image> [image...]" >&2; exit 2; }
IMAGES=("$@")

mkdir -p "$OUT"

# Stage the inputs inside the repo: the container only sees $ROOT and /mnt, so
# an image sitting anywhere else silently breaks the container backends only —
# and a matrix missing one backend looks like agreement rather than a failure.
STAGE="$OUT/images"
mkdir -p "$STAGE"
STAGED=()
for img in "${IMAGES[@]}"; do
    cp -f "$img" "$STAGE/$(basename "$img")"
    STAGED+=("$STAGE/$(basename "$img")")
done
# same files, addressed from inside the container
REL_OUT="${OUT#"$ROOT/"}"
STAGED_IN_CONTAINER=()
for img in "${STAGED[@]}"; do
    STAGED_IN_CONTAINER+=("/workspace/ocr/$REL_OUT/images/$(basename "$img")")
done

common=(--ckpt "$CKPT" --images "${STAGED[@]}" --max-pixels "$MAX_PIXELS" \
        --max-new-tokens "$MAX_NEW_TOKENS" --out "$OUT")

# Run the transformers backends wherever a working interpreter is. A workstation
# set up by scripts/setup_local.sh has the venv; a server usually only has the
# container. Both must produce identical inputs, hence one code path with two hosts.
in_container() {
    docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
        -v /mnt:/mnt:ro -v "$ROOT:/workspace/ocr" -w /workspace/ocr \
        -e PYTHONPATH=/workspace/ocr:/workspace/ocr/src -e HOME=/tmp \
        -e TRITON_CACHE_DIR=/tmp/triton -e HF_HOME=/tmp/hf \
        "$IMAGE" python3 -m benchmarks.agreement_matrix "$@"
}

collect_transformers() {
    local backend="$1"
    if [[ -x "$VENV/bin/python" && "$backend" != "flash_attention_2" ]]; then
        PYTHONPATH="$ROOT:$ROOT/src" "$VENV/bin/python" -m benchmarks.agreement_matrix \
            collect --backend "$backend" "${common[@]}"
    else
        # flash-attn only exists in the image; so does everything else on a server
        in_container collect --backend "$backend" --ckpt "$CKPT" \
            --images "${STAGED_IN_CONTAINER[@]}" --max-pixels "$MAX_PIXELS" \
            --max-new-tokens "$MAX_NEW_TOKENS" --out "/workspace/ocr/$REL_OUT"
    fi
}

echo "== transformers backends =="
for backend in sdpa flex_attention flash_attention_2; do
    echo "-- $backend"
    collect_transformers "$backend"
done

echo "== vLLM =="
if [[ "${SKIP_VLLM:-0}" == "1" ]]; then
    echo "-- skipped (SKIP_VLLM=1)"
elif curl -sf --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "-- using the vLLM server already on :8000"
else
    echo "-- starting vLLM (needs auto_map in config.json, which prepare_checkpoint.py"
    echo "   strips, so serve a symlink shadow of the checkpoint instead)"
    SHADOW="${OUT}/vllm_ckpt"
    rm -rf "$SHADOW"; mkdir -p "$SHADOW"
    for f in "$CKPT"/*; do
        case "$(basename "$f")" in config.json|config.json.bak) ;; *) ln -s "$f" "$SHADOW/" ;; esac
    done
    if [[ -f "$CKPT/config.json.bak" ]]; then cp "$CKPT/config.json.bak" "$SHADOW/config.json"
    else cp "$CKPT/config.json" "$SHADOW/config.json"; fi
    docker rm -f dots_vllm_agreement >/dev/null 2>&1 || true
    docker run -d --rm --name dots_vllm_agreement --gpus all --ipc=host --network host \
        -v /mnt:/mnt:ro -v "$OUT:/ckpt" -e HF_HUB_OFFLINE=1 -e HF_HOME=/tmp/hf \
        "$VLLM_IMAGE" /ckpt/vllm_ckpt --served-model-name rednote-hilab/dots.mocr \
        --gpu-memory-utilization 0.85 --max-model-len 12288 \
        --chat-template-content-format string --trust-remote-code \
        --no-enable-prefix-caching --host 127.0.0.1 --port 8000 >/dev/null
    echo -n "   waiting for vLLM"
    until curl -sf --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1; do
        docker ps --filter name=dots_vllm_agreement --format '{{.Names}}' | grep -q . \
            || { echo; echo "vLLM container died; see docker logs" >&2; exit 1; }
        echo -n "."; sleep 5
    done
    echo " up"
    STARTED_VLLM=1
fi
if [[ "${SKIP_VLLM:-0}" != "1" ]]; then
    if [[ -x "$VENV/bin/python" ]]; then
        PYTHONPATH="$ROOT:$ROOT/src" "$VENV/bin/python" -m benchmarks.agreement_matrix \
            collect --backend vllm --images "${STAGED[@]}" --max-pixels "$MAX_PIXELS" \
            --max-new-tokens "$MAX_NEW_TOKENS" --out "$OUT"
    else
        in_container collect --backend vllm --images "${STAGED_IN_CONTAINER[@]}" \
            --max-pixels "$MAX_PIXELS" --max-new-tokens "$MAX_NEW_TOKENS" \
            --out "/workspace/ocr/$REL_OUT"
    fi
    [[ "${STARTED_VLLM:-0}" == "1" ]] && docker rm -f dots_vllm_agreement >/dev/null 2>&1 || true
fi

echo
echo "== agreement matrix =="
if [[ -x "$VENV/bin/python" ]]; then
    PYTHONPATH="$ROOT:$ROOT/src" "$VENV/bin/python" -m benchmarks.agreement_matrix compare --out "$OUT"
else
    in_container compare --out "/workspace/ocr/$REL_OUT"
fi

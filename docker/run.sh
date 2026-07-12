#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    printf 'usage: %s cpu|gpu command [args...]\n' "$0" >&2
    exit 64
fi

MODE=$1
shift
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${DOTS_MOCR_IMAGE:-dots-mocr:trtllm-1.3.0rc20}
OUTPUT=${DOTS_MOCR_OUTPUT:-$ROOT/output}
mkdir -p "$OUTPUT"

DOCKER_ARGS=(
    run
    --rm
    --ipc=host
    --network=none
    --ulimit memlock=-1
    --ulimit stack=67108864
    --user "$(id -u):$(id -g)"
    --env HOME=/tmp
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env HF_DATASETS_OFFLINE=1
    --env MPLCONFIGDIR=/tmp/matplotlib
    --env PYTHONPATH=/workspace/ocr/src
    --volume /mnt:/mnt:ro
    --volume "$ROOT:/workspace/ocr:ro"
    --volume "$OUTPUT:/workspace/ocr/output:rw"
    --workdir /workspace/ocr
)

case "$MODE" in
    cpu)
        ;;
    gpu)
        DOCKER_ARGS+=(--gpus all)
        ;;
    *)
        printf 'unsupported mode: %s\n' "$MODE" >&2
        exit 64
        ;;
esac

exec docker "${DOCKER_ARGS[@]}" "$IMAGE" "$@"

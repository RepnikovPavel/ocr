#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN="$ROOT/docker/run.sh"
CKPT=${CKPT:-${DOTS_MOCR_CKPT:-/mnt/nvme/huggingface/models--rednote-hilab--dots.mocr/snapshots/main}}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
MAX_PIXELS=${MAX_PIXELS:-1000000}
RESULT_DIR=${RESULT_DIR:-/workspace/ocr/output/benchmarks/dots_mocr_gpu}
METRICS=${METRICS:-$RESULT_DIR/metrics.json}

if [[ -n ${INPUT_PATH:-} ]]; then
    INPUT=$INPUT_PATH
else
    "$RUN" cpu python3 scripts/generate_synthetic.py --output /workspace/ocr/output/fixtures/composite_chart.png
    INPUT=/workspace/ocr/output/fixtures/composite_chart.png
fi

exec "$RUN" gpu python3 -m dots_mocr.benchmark \
    --ckpt "$CKPT" \
    --input "$INPUT" \
    --prompt "${PROMPT:-prompt_layout_all_en}" \
    --device cuda \
    --dtype "${DTYPE:-bfloat16}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-pixels "$MAX_PIXELS" \
    --output-dir "$RESULT_DIR" \
    --metrics "$METRICS"

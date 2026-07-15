#!/usr/bin/env bash
# Full benchmark suite for a 2x RTX 4090 host.
#
# Usage:
#   CKPT=/path/to/snapshot PDF=/path/to/doc.pdf REPORT_DIR=reports ./scripts/run_bench_2x4090.sh
#
# Runs, in order:
#   1. single-page  / single GPU   (latency profile)
#   2. multi-page   / single GPU   (baseline throughput)
#   3. multi-page   / both GPUs    (data-parallel, one model replica per GPU)
#   4. multi-page   / both GPUs, batch=2 per replica (max utilization)
set -euo pipefail

CKPT=${CKPT:?set CKPT to the checkpoint snapshot dir}
PDF=${PDF:?set PDF to the input pdf}
REPORT_DIR=${REPORT_DIR:-reports}
DPI=${DPI:-150}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
PROMPT=${PROMPT:-prompt_layout_all_en}

mkdir -p "$REPORT_DIR"

run() {
    local label=$1; shift
    echo "=== $label ==="
    python3 -m benchmarks.bench_throughput \
        --ckpt "$CKPT" --pdf "$PDF" --prompt "$PROMPT" \
        --dpi "$DPI" --max-new-tokens "$MAX_NEW_TOKENS" \
        --label "$label" \
        --output "$REPORT_DIR/bench_${label}.json" \
        "$@"
}

run single_page_1gpu   --gpus 0   --pages 0 --save-outputs "$REPORT_DIR/outputs_single_page"
run multi_page_1gpu    --gpus 0
run multi_page_2gpu    --gpus 0,1 --save-outputs "$REPORT_DIR/outputs_2gpu"
# batch=2 needs a lower pixel cap: the vision encoder concatenates both pages
# into one sequence and its attention memory is quadratic in the total length —
# at 150 dpi (~2.1M px/page) it asks for ~21 GB and OOMs on a 24 GB card.
# ~1.0M px/page keeps the vision sequence within the batch=1 footprint.
run multi_page_2gpu_b2 --gpus 0,1 --batch-size 2 --max-pixels 1000000

echo "All reports in $REPORT_DIR"

#!/usr/bin/env bash
# Orchestrate the synthetic edge run:
#   1. write .tex documents + manifest      (bench container: pure python)
#   2. compile .tex -> .pdf                  (texlive container)
#   3. render + infer + score                (bench container: torch + fitz)
#
# Env:
#   REPO       repo dir (bind-mounted into both containers)
#   CKPT       dots.mocr checkpoint snapshot dir
#   WORKDIR    scratch dir for tex/pdf/reports
#   KINDS      optional: table,formula,algorithm,code (default all)
#   DEVICE     cuda:0 (default)
set -euo pipefail

REPO=${REPO:?set REPO}
CKPT=${CKPT:?set CKPT}
WORKDIR=${WORKDIR:?set WORKDIR}
KINDS=${KINDS:-}
DEVICE=${DEVICE:-cuda:0}
BENCH_IMAGE=${BENCH_IMAGE:-dots-mocr:bench-cu126}
TEXLIVE_IMAGE=${TEXLIVE_IMAGE:-texlive/texlive:latest}
DATA_ROOT=${DATA_ROOT:-/mnt/data2}

TEXDIR="$WORKDIR/tex"
REPORT="$WORKDIR/synth_edge_report.json"
mkdir -p "$TEXDIR"

echo "=== 1/3 write tex + manifest ==="
docker run --rm -v "$DATA_ROOT:$DATA_ROOT" -w "$REPO" -e PYTHONPATH="$REPO" \
  "$BENCH_IMAGE" python3 -m benchmarks.synth.compile --out "$TEXDIR"

echo "=== 2/3 compile tex -> pdf ==="
for tex in "$TEXDIR"/*.tex; do
    name=$(basename "$tex" .tex)
    echo "  latex: $name"
    # two passes for algorithm/ref numbering; nonstopmode, keep going on warnings
    docker run --rm -v "$DATA_ROOT:$DATA_ROOT" -w "$TEXDIR" \
      "$TEXLIVE_IMAGE" bash -c \
      "pdflatex -interaction=nonstopmode -halt-on-error '$name.tex' >/dev/null 2>&1 || \
       pdflatex -interaction=nonstopmode '$name.tex' >'$name.log' 2>&1; \
       pdflatex -interaction=nonstopmode '$name.tex' >'$name.log' 2>&1 || true"
    if [[ ! -f "$TEXDIR/$name.pdf" ]]; then
        echo "  !! $name failed to compile; tail of log:"
        tail -20 "$TEXDIR/$name.log" || true
        exit 3
    fi
done
echo "  compiled: $(ls "$TEXDIR"/*.pdf | wc -l) pdfs"

echo "=== 3/3 render + infer + score ==="
docker run --rm --gpus all --ipc=host \
  -v "$DATA_ROOT:$DATA_ROOT" -w "$REPO" -e PYTHONPATH="$REPO/src:$REPO" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$BENCH_IMAGE" python3 -m benchmarks.synth.run_edge \
    --pdf-dir "$TEXDIR" --ckpt "$CKPT" --output "$REPORT" \
    --device "$DEVICE" ${KINDS:+--kinds "$KINDS"} --save-outputs

echo "report: $REPORT"

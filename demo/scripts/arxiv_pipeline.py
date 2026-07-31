#!/usr/bin/env python3
"""Host-side runner for the arxiv quant/algo-trading pipeline.

Runs in a host venv (with boto3 + requests), reading the SAME demo.db the OCR
container is bound to, so the container's /arxiv UI reflects this process's
progress live. This is the entrypoint invoked by the operator (or cron):

    python -m demo.scripts.arxiv_pipeline --limit 10

Environment (defaults match the server's single-node deploy):
    DEMO_STATE_DIR        demo.db location (default /state)
    OCR_URL               OCR service base (default http://127.0.0.1:8601)
    OCR_PROMPT_MODE       prompt mode (default prompt_layout_all_en)
    SEAWEED_S3_ENDPOINT   SeaweedFS S3 gateway; unset → local fallback
    SEAWEED_ACCESS_KEY / SECRET_KEY / BUCKET

What it does, in order:
  1. discover — search arxiv for `--query` (default: quant/algo-trading),
     upsert every result into arxiv_papers.
  2. run_pipeline — walk the DAG for each paper (download → submit_ocr →
     wait_ocr → fetch_bundle → store → index), storing PDFs + bundles in
     SeaweedFS and recording every step in pipeline_steps.
  3. print a summary (timing, storage backend, success/failure counts).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow `python demo/scripts/arxiv_pipeline.py` from a checkout as well as
# `python -m demo.scripts.arxiv_pipeline`. Anchor src/ so demo.* resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from demo.arxiv import db as arxiv_db        # noqa: E402
from demo.arxiv import pipeline, source       # noqa: E402
from demo import storage                       # noqa: E402


def _emit(msg: str) -> None:
    print(msg, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the arxiv quant/algo-trading pipeline against the "
                    "OCR service and SeaweedFS.")
    parser.add_argument("--query", default="quant",
                        help="arxiv named query or raw search syntax "
                             "(default: quant = q-fin.* OR algo-trading phrases)")
    parser.add_argument("--limit", type=int, default=10,
                        help="max papers to process this run (default 10)")
    parser.add_argument("--start", type=int, default=0,
                        help="arxiv result offset (default 0)")
    parser.add_argument("--workers", type=int, default=2,
                        help="DAG concurrency (default 2; OCR is single-worker "
                             "so raising this mostly overlaps downloads)")
    parser.add_argument("--ocr-url", default=os.environ.get(
                            "OCR_URL", pipeline.DEFAULT_OCR_URL),
                        help="OCR service base URL")
    parser.add_argument("--prompt-mode", default=os.environ.get(
                            "OCR_PROMPT_MODE", pipeline.DEFAULT_PROMPT_MODE),
                        help="OCR prompt mode (default prompt_layout_all_en)")
    parser.add_argument("--agent", default="arxiv-pipeline",
                        help="agent name shown in the OCR queue")
    parser.add_argument("--state-dir", default=os.environ.get(
                            "DEMO_STATE_DIR", "/state"),
                        help="demo.db location (shared with the OCR container)")
    parser.add_argument("--discover-only", action="store_true",
                        help="search + upsert papers, but do not run the DAG")
    args = parser.parse_args(argv)

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    # The arxiv tables live in their own file (ARXIV_DB_PATH), separate from
    # the container-owned demo.db — the runner is a normal user and demo.db
    # is root-owned in the deploy, so they cannot share a writable file.
    db_path = Path(os.environ.get("ARXIV_DB_PATH", str(state_dir / "arxiv.db")))
    arxiv_db.init(str(db_path))
    # Record the real storage backend so the container's /stats (which has no
    # boto3 and would otherwise report 'local') reflects where blobs go.
    arxiv_db.set_meta("storage_backend", storage.store_kind())
    _emit(f"[runner] db={db_path}  storage={storage.store_kind()}  "
          f"ocr={args.ocr_url}  mode={args.prompt_mode}")

    # 1. discover
    _emit(f"[runner] discovering up to {args.limit} papers for '{args.query}'...")
    t0 = time.time()
    papers = source.search(args.query, max_results=args.limit,
                           start=args.start, polite=True)
    for paper in papers:
        arxiv_db.upsert_paper(paper)
    _emit(f"[runner] discovered {len(papers)} papers in {time.time()-t0:.1f}s")
    if not papers:
        _emit("[runner] no papers found; nothing to do.")
        return 0
    for p in papers:
        _emit(f"    {p['arxiv_id']}\t{p['title'][:70]}")

    if args.discover_only:
        return 0

    # 2. run the DAG
    t1 = time.time()
    run_id = pipeline.run_pipeline(
        papers, query=args.query, ocr_url=args.ocr_url,
        prompt_mode=args.prompt_mode, agent=args.agent, workers=args.workers)
    elapsed = time.time() - t1

    # 3. summary
    run = arxiv_db.get_run(run_id)
    counts = arxiv_db.count_papers()
    _emit("")
    _emit("=" * 60)
    _emit(f"run {run_id}: {run['status']}  ({elapsed:.1f}s)")
    _emit(f"  papers in run : {len(papers)}")
    _emit(f"  steps done    : {run['done']}/{len(papers) * len(arxiv_db.STAGES)}")
    _emit(f"  steps failed  : {run['failed']}")
    _emit(f"  steps skipped : {run['skipped']}")
    _emit(f"  storage       : {storage.store_kind()}")
    _emit(f"  catalogue     : {counts['parsed']} parsed / "
          f"{counts['stored']} stored / {counts['total']} seen "
          f"({counts['bytes']/(1024*1024):.1f} MB)")
    _emit("=" * 60)
    _emit("Open /arxiv on the OCR service for the live grid.")
    return 0 if run["status"] in ("done",) else 1


if __name__ == "__main__":
    # The exit status reflects run health: 0 on a clean run, 1 if any paper
    # failed (the run still completes for the rest).
    sys.exit(main())

"""Standalone worker process: loads model on one GPU, polls the shared queue.

This runs alongside demo.server in a separate Docker container (one GPU each).
Both processes share /state/demo.db (SQLite WAL) and /state/jobs/ via bind mount.
SQLite's atomic claim_next_task ensures each task is picked up by exactly one
worker, so a single document split across two tasks (`ocrc --split 2`) parses
on two GPUs simultaneously without any inter-process coordination beyond the DB.

Usage:
    DEMO_DEVICE=cuda:1 python3 -m demo.worker_standalone

Key difference from demo.server: no FastAPI, no uvicorn, no init_db (the reaper
in init_db would clobber the other worker's in-flight tasks).
"""
from __future__ import annotations

import os
import signal
import sys
import time
import traceback
from pathlib import Path

print("[worker_standalone] starting...", flush=True)

STATE_DIR = Path(os.environ.get("DEMO_STATE_DIR", "/state"))
DB_PATH = STATE_DIR / "demo.db"
JOBS_DIR = STATE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Point db + docstore at the shared SQLite WITHOUT calling init_db,
# which reaps running tasks — that would clobber the other worker.
from demo import db, docstore  # noqa: E402

db._DB_PATH = str(DB_PATH)
docstore.init(str(DB_PATH))  # idempotent: creates tables if missing, no reaper
print(f"[worker_standalone] DB: {DB_PATH}", flush=True)

from demo.worker import DemoWorker  # noqa: E402

worker = DemoWorker(
    ckpt=os.environ.get("CKPTDIR", "/models"),
    jobs_dir=JOBS_DIR,
    device=os.environ.get("DEMO_DEVICE", "cuda:1"),
    engine=os.environ.get("DEMO_ENGINE", "transformers"),
    dpi=int(os.environ.get("DEMO_DPI", "150")),
    max_pixels=int(os.environ.get("DEMO_MAX_PIXELS", "2200000")),
    idle_unload_seconds=int(os.environ.get("DEMO_IDLE_UNLOAD_S", "10")),
    keep_loaded=os.environ.get("DEMO_KEEP_LOADED", "0") == "1",
    autostart=os.environ.get("DEMO_AUTOSTART", "0") == "1",
    attn_implementation=os.environ.get("DEMO_ATTN_IMPLEMENTATION") or None,
    name=os.environ.get("DEMO_WORKER_NAME", "worker-gpu1"),
)
worker.start()
print(f"[worker_standalone] {worker.name} started on {worker.device}, "
      f"engine={worker.engine}, idle={worker.idle_unload_seconds}s", flush=True)


def _shutdown(signum, frame):
    print(f"[worker_standalone] signal {signum} — shutting down", flush=True)
    worker.shutdown(join_timeout=10.0)
    sys.exit(0)


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)

# Main thread blocks here. The daemon worker thread polls db.claim_next_task()
# every ~1 second, loads the model on demand, and unloads it after idle_unload_seconds.
while worker.is_alive():
    time.sleep(5)

print("[worker_standalone] worker thread exited unexpectedly", flush=True)
traceback.print_exc()
sys.exit(1)

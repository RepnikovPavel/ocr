"""Run a single parsing task in a standalone process, then exit.

Launched by demo.server as a subprocess per task. Each invocation:
  1. Claims the task from the shared SQLite DB
  2. Loads the model on the specified GPU
  3. Parses all pages
  4. Exits — process death releases ALL VRAM (zero parasitic load)

Usage:
    python3 -m demo.run_task <task_id>
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

print("[run_task] starting...", flush=True)

STATE_DIR = Path(os.environ.get("DEMO_STATE_DIR", "/state"))
DB_PATH = STATE_DIR / "demo.db"
JOBS_DIR = STATE_DIR / "jobs"

from demo import db, docstore  # noqa: E402

db._DB_PATH = str(DB_PATH)
docstore.init(str(DB_PATH))

task_id = sys.argv[1] if len(sys.argv) > 1 else None
if not task_id:
    print("[run_task] ERROR: task_id required", flush=True)
    sys.exit(2)

task = db.get_task(task_id)
if task is None:
    print(f"[run_task] task {task_id} not found", flush=True)
    sys.exit(1)

if task["status"] not in ("queued", "running"):
    print(f"[run_task] task {task_id} status={task['status']}, skipping", flush=True)
    sys.exit(0)

# Claim if still queued
if task["status"] == "queued":
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    cur = conn.execute(
        "UPDATE tasks SET status='running', started_at=? WHERE id=? AND status='queued'",
        (time.time(), task_id))
    if cur.rowcount != 1:
        print(f"[run_task] could not claim {task_id}", flush=True)
        sys.exit(0)
    conn.commit()
    conn.close()
    task = db.get_task(task_id)

print(f"[run_task] task={task_id} claimed, loading model on "
      f"{os.environ.get('DEMO_DEVICE', 'cuda:0')}...", flush=True)

# Build the parser directly — NO DemoWorker thread, NO run() loop.
# The run() loop would try to claim_next_task() on its own and might grab
# a different task, causing a race with our explicit _execute call below.
from dots_mocr.cli import DotsMOCRParser  # noqa: E402

parser = DotsMOCRParser(
    ckpt=os.environ.get("CKPTDIR", "/models"),
    device=os.environ.get("DEMO_DEVICE", "cuda:0"),
    dtype="bfloat16",
    dpi=int(os.environ.get("DEMO_DPI", "150")),
    max_pixels=int(os.environ.get("DEMO_MAX_PIXELS", "1200000")),
    num_thread=1,
    temperature=0.1,
)
parser._load_model(os.environ.get("CKPTDIR", "/models"))

print("[run_task] model loaded, executing task...", flush=True)

# Reuse DemoWorker's _execute logic without the thread/loop machinery.
# We create a throwaway worker just for _execute + _record_in_docstore,
# but we NEVER start its thread.
from demo.worker import DemoWorker  # noqa: E402

worker = DemoWorker(
    ckpt=os.environ.get("CKPTDIR", "/models"),
    jobs_dir=JOBS_DIR,
    device=os.environ.get("DEMO_DEVICE", "cuda:0"),
    engine="transformers",
    dpi=int(os.environ.get("DEMO_DPI", "150")),
    max_pixels=int(os.environ.get("DEMO_MAX_PIXELS", "1200000")),
    idle_unload_seconds=999999,
    keep_loaded=True,
    name=f"run_task-{task_id}",
)
# Inject our already-loaded parser so _execute uses it.
worker.parser = parser
worker.model_state = "loaded"
worker._last_used = time.time()
# Do NOT call worker.start() — we call _execute directly.

try:
    worker._execute(task)
    print(f"[run_task] task {task_id} done", flush=True)
except Exception as error:  # noqa: BLE001
    traceback.print_exc()
    db.update_task(task_id, status="error",
                   error=f"{type(error).__name__}: {error}",
                   finished_at=time.time())
    print(f"[run_task] task {task_id} failed: {error}", flush=True)
    sys.exit(1)

# Cleanup: release parser references so Python GC frees the model.
del parser
del worker
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except ImportError:
    pass

print("[run_task] exiting — VRAM released", flush=True)
sys.exit(0)

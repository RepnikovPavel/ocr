"""Run a single parsing task in a standalone process, then exit.

Launched by demo.server as a subprocess per task. Each invocation:
  1. Claims the task from the shared SQLite DB
  2. Loads the model on the specified GPU (via DemoWorker)
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

print(f"[run_task] task={task_id} claimed, loading model...", flush=True)

from demo.worker import DemoWorker  # noqa: E402

worker = DemoWorker(
    ckpt=os.environ.get("CKPTDIR", "/models"),
    jobs_dir=JOBS_DIR,
    device=os.environ.get("DEMO_DEVICE", "cuda:0"),
    engine=os.environ.get("DEMO_ENGINE", "transformers"),
    dpi=int(os.environ.get("DEMO_DPI", "150")),
    max_pixels=int(os.environ.get("DEMO_MAX_PIXELS", "2200000")),
    idle_unload_seconds=999999,  # never idle-unload; we exit after the task
    keep_loaded=True,
    autostart=True,  # load model immediately on start
    name=f"run_task-{task_id}",
)

# Start the worker thread — it will load the model (autostart=True)
worker.start()

# Wait for model to load (up to 5 minutes for cold start)
for attempt in range(300):
    if worker.model_state == "loaded":
        break
    if worker.model_state == "error":
        err = worker.model_error or "unknown"
        print(f"[run_task] model load failed: {err}", flush=True)
        db.update_task(task_id, status="error", error=err, finished_at=time.time())
        worker.shutdown()
        sys.exit(1)
    time.sleep(1)
    if attempt % 30 == 0 and attempt > 0:
        print(f"[run_task] still loading model... ({attempt}s)", flush=True)

if worker.model_state != "loaded":
    print("[run_task] model load timeout (300s)", flush=True)
    db.update_task(task_id, status="error", error="model load timeout", finished_at=time.time())
    worker.shutdown()
    sys.exit(1)

print("[run_task] model loaded, executing task...", flush=True)

try:
    worker._run_task(task)
    print(f"[run_task] task {task_id} done", flush=True)
except Exception as error:  # noqa: BLE001
    traceback.print_exc()
    db.update_task(task_id, status="error",
                   error=f"{type(error).__name__}: {error}",
                   finished_at=time.time())

worker.shutdown(join_timeout=5)
print("[run_task] exiting — VRAM released", flush=True)
sys.exit(0)

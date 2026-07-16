"""SQLite persistence for the demo: sessions, uploaded jobs, inference tasks.

Single-writer usage: the FastAPI handlers and the worker thread share one
database file; every function opens a short-lived connection (WAL mode), so
cross-thread use is safe without a shared connection object.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path

_DB_PATH = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    last_seen_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    kind TEXT NOT NULL,               -- 'pdf' | 'image'
    num_pages INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    prompt_mode TEXT NOT NULL,
    pages TEXT NOT NULL,              -- json list of 0-based page ids
    params TEXT NOT NULL,             -- json: temperature, custom_prompt, bbox, ...
    status TEXT NOT NULL,             -- queued | running | done | error | cancelled
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    progress TEXT NOT NULL DEFAULT '{}',
    result TEXT NOT NULL DEFAULT '[]',-- json list of per-page results
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id, created_at);
"""


def init_db(path):
    global _DB_PATH
    _DB_PATH = str(path)
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(SCHEMA)
        # tasks stuck in 'running' from a previous process are orphans
        conn.execute(
            "UPDATE tasks SET status='error', error='interrupted by restart', "
            "finished_at=? WHERE status='running'",
            (time.time(),),
        )


def _connect():
    conn = sqlite3.connect(_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _row_to_task(row):
    if row is None:
        return None
    task = dict(row)
    task["pages"] = json.loads(task["pages"])
    task["params"] = json.loads(task["params"])
    task["progress"] = json.loads(task["progress"])
    task["result"] = json.loads(task["result"])
    task["cancel_requested"] = bool(task["cancel_requested"])
    return task


# ---------------------------------------------------------------- sessions

def get_or_create_session(session_id=None):
    now = time.time()
    with _connect() as conn:
        if session_id:
            row = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row:
                conn.execute("UPDATE sessions SET last_seen_at=? WHERE id=?", (now, session_id))
                return session_id
        new_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO sessions (id, created_at, last_seen_at) VALUES (?, ?, ?)",
            (new_id, now, now),
        )
        return new_id


# ---------------------------------------------------------------- jobs

def create_job(session_id, filename, kind, num_pages):
    job_id = uuid.uuid4().hex[:12]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, session_id, filename, kind, num_pages, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, session_id, filename, kind, num_pages, time.time()),
        )
    return job_id


def get_job(job_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None


def list_jobs(session_id, limit=20):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]


# ---------------------------------------------------------------- tasks

def create_task(session_id, job_id, prompt_mode, pages, params):
    task_id = uuid.uuid4().hex[:12]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO tasks (id, session_id, job_id, prompt_mode, pages, params, "
            "status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)",
            (task_id, session_id, job_id, prompt_mode,
             json.dumps(pages), json.dumps(params), time.time()),
        )
    return task_id


def get_task(task_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return _row_to_task(row)


def list_tasks(limit=50, session_id=None, statuses=None):
    where, params = [], []
    if session_id:
        where.append("session_id=?")
        params.append(session_id)
    if statuses:
        where.append("status IN (%s)" % ",".join("?" for _ in statuses))
        params.extend(statuses)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM tasks {clause} ORDER BY created_at DESC LIMIT ?", params,
        ).fetchall()
        return [_row_to_task(row) for row in rows]


def list_active_tasks(limit=50):
    """Only queued/running tasks — the live queue (finished ones drop off)."""
    return list_tasks(limit=limit, statuses=("queued", "running"))


def has_queued_tasks():
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM tasks WHERE status='queued' LIMIT 1").fetchone()
        return row is not None


def claim_next_task():
    """Atomically move the oldest queued task to running; None if empty."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM tasks WHERE status='queued' ORDER BY created_at LIMIT 1",
        ).fetchone()
        if row is None:
            return None
        updated = conn.execute(
            "UPDATE tasks SET status='running', started_at=? "
            "WHERE id=? AND status='queued'",
            (time.time(), row["id"]),
        )
        if updated.rowcount != 1:
            return None
    return get_task(row["id"])


def update_task(task_id, **fields):
    columns = []
    values = []
    for key, value in fields.items():
        if key in ("pages", "params", "progress", "result"):
            value = json.dumps(value)
        columns.append(f"{key}=?")
        values.append(value)
    values.append(task_id)
    with _connect() as conn:
        conn.execute(f"UPDATE tasks SET {', '.join(columns)} WHERE id=?", values)


def request_cancel(task_id):
    """Cancel a queued task immediately; flag a running task for the worker.

    Returns the resulting status or None if the task does not exist.
    """
    with _connect() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            return None
        if row["status"] == "queued":
            updated = conn.execute(
                "UPDATE tasks SET status='cancelled', cancel_requested=1, finished_at=? "
                "WHERE id=? AND status='queued'",
                (time.time(), task_id),
            )
            if updated.rowcount == 1:
                return "cancelled"
            # the worker claimed the task between our SELECT and UPDATE:
            # fall through to the running branch
        if row["status"] in ("queued", "running"):
            conn.execute("UPDATE tasks SET cancel_requested=1 WHERE id=?", (task_id,))
            return "cancelling"
        return row["status"]


def is_cancel_requested(task_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT cancel_requested FROM tasks WHERE id=?", (task_id,),
        ).fetchone()
        return bool(row and row["cancel_requested"])

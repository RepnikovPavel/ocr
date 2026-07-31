"""Arxiv pipeline persistence: papers, pipeline runs, and per-step progress.

Shares the same SQLite file as the OCR queue (demo.db), so the container's
API can read run/step status that the host-side pipeline runner writes,
without either process needing a network call to the other. WAL mode + one
short-lived connection per call keeps cross-process writes safe.

Schema
------
    arxiv_papers     one row per arxiv id seen (deduped on arxiv_id). Carries
                     metadata + sha256 of the downloaded PDF + storage status.
    pipeline_runs    one row per `arxiv_pipeline` invocation. A run fans out
                     over N papers; status aggregates from its steps.
    pipeline_steps   one row per (run, paper, stage). The stage sequence is the
                     DAG: download -> submit_ocr -> wait_ocr -> fetch_bundle ->
                     store -> index. Each step's status/error/duration is here,
                     which is exactly what the progress UI renders.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Optional

_DB_PATH: Optional[str] = None

# The DAG, in execution order. Every paper walks these stages; a run is done
# when all its steps have a terminal status. Exposed as a constant so the API
# and UI agree on the canonical order and names.
STAGES = ("download", "submit_ocr", "wait_ocr", "fetch_bundle", "store", "index")

# Step statuses. `running` and `queued` are live; the rest are terminal.
STEP_QUEUED = "queued"
STEP_RUNNING = "running"
STEP_DONE = "done"
STEP_ERROR = "error"
STEP_SKIPPED = "skipped"  # e.g. download skipped because PDF already in storage

# Paper storage/parse states.
PAPER_NEW = "new"
PAPER_DOWNLOADED = "downloaded"
PAPER_STORED = "stored"
PAPER_PARSED = "parsed"
PAPER_FAILED = "failed"

SCHEMA = """
CREATE TABLE IF NOT EXISTS arxiv_papers (
    arxiv_id        TEXT PRIMARY KEY,        -- e.g. 2306.12345 (no version)
    title           TEXT NOT NULL,
    authors         TEXT NOT NULL DEFAULT '',-- JSON list
    summary         TEXT NOT NULL DEFAULT '',
    categories      TEXT NOT NULL DEFAULT '',-- e.g. "q-fin.TR q-fin.CP"
    published_at    TEXT,                    -- ISO from arxiv
    pdf_url         TEXT NOT NULL DEFAULT '',
    abs_url         TEXT NOT NULL DEFAULT '',
    sha256          TEXT,                    -- set once the PDF is downloaded
    num_pages       INTEGER,
    size_bytes      INTEGER,
    storage_status  TEXT NOT NULL DEFAULT 'new',  -- new|downloaded|stored|parsed|failed
    ocr_task_id     TEXT,                    -- the OCR service's task id, once submitted
    parsed          INTEGER NOT NULL DEFAULT 0,
    first_seen_at   REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_arxiv_papers_sha ON arxiv_papers(sha256);
CREATE INDEX IF NOT EXISTS idx_arxiv_papers_storage ON arxiv_papers(storage_status);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id          TEXT PRIMARY KEY,
    query       TEXT NOT NULL DEFAULT '',    -- the search that seeded this run
    max_papers  INTEGER NOT NULL DEFAULT 0,  -- intended size
    status      TEXT NOT NULL DEFAULT 'running',  -- running|done|error|cancelled
    error       TEXT,
    created_at  REAL NOT NULL,
    finished_at REAL,
    -- aggregate counters (denormalised for cheap UI reads)
    done        INTEGER NOT NULL DEFAULT 0,
    failed      INTEGER NOT NULL DEFAULT 0,
    skipped     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON pipeline_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS pipeline_steps (
    run_id      TEXT NOT NULL,
    arxiv_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,               -- one of STAGES
    status      TEXT NOT NULL DEFAULT 'queued',  -- queued|running|done|error|skipped
    detail      TEXT NOT NULL DEFAULT '',    -- free-form progress/error text
    error       TEXT,
    started_at  REAL,
    finished_at REAL,
    PRIMARY KEY (run_id, arxiv_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON pipeline_steps(run_id, arxiv_id);
"""


def init(path) -> None:
    """Bind to the shared db file and ensure the arxiv tables exist.

    Idempotent: safe to call from both the container and the runner; the
    first to start creates the schema, the second is a no-op.
    """
    global _DB_PATH
    _DB_PATH = str(path)
    with _connect() as conn:
        conn.executescript(SCHEMA)


def _connect():
    if _DB_PATH is None:
        raise RuntimeError("arxiv.db.init(path) was not called")
    conn = sqlite3.connect(_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _now():
    return time.time()


# ----------------------------------------------------------------- papers

def upsert_paper(paper: dict) -> bool:
    """Insert or refresh a paper row from arxiv metadata.

    Returns True when the paper is new to the db. On refresh we overwrite the
    volatile metadata (title/summary/categories can be revised on arxiv) but
    keep sha256/storage_status/ocr_task_id — once we have the bytes and a
    parse, a metadata change must not throw that work away.
    """
    now = _now()
    with _connect() as conn:
        row = conn.execute("SELECT arxiv_id FROM arxiv_papers WHERE arxiv_id=?",
                           (paper["arxiv_id"],)).fetchone()
        if row:
            conn.execute(
                "UPDATE arxiv_papers SET title=?, authors=?, summary=?, "
                "categories=?, published_at=?, pdf_url=?, abs_url=?, updated_at=? "
                "WHERE arxiv_id=?",
                (paper.get("title", ""), json.dumps(paper.get("authors", [])),
                 paper.get("summary", ""), paper.get("categories", ""),
                 paper.get("published_at"), paper.get("pdf_url", ""),
                 paper.get("abs_url", ""), now, paper["arxiv_id"]))
            return False
        conn.execute(
            "INSERT INTO arxiv_papers (arxiv_id, title, authors, summary, categories, "
            "published_at, pdf_url, abs_url, storage_status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, ?)",
            (paper["arxiv_id"], paper.get("title", ""),
             json.dumps(paper.get("authors", [])), paper.get("summary", ""),
             paper.get("categories", ""), paper.get("published_at"),
             paper.get("pdf_url", ""), paper.get("abs_url", ""), now, now))
        return True


def get_paper(arxiv_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM arxiv_papers WHERE arxiv_id=?",
                           (arxiv_id,)).fetchone()
        return _paper_row(row)


def get_paper_by_sha(sha256: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM arxiv_papers WHERE sha256=?",
                           (sha256,)).fetchone()
        return _paper_row(row)


def _paper_row(row) -> Optional[dict]:
    if row is None:
        return None
    paper = dict(row)
    try:
        paper["authors"] = json.loads(paper.get("authors") or "[]")
    except (ValueError, TypeError):
        paper["authors"] = []
    paper["parsed"] = bool(paper.get("parsed"))
    return paper


def set_paper_downloaded(arxiv_id: str, sha256: str, size_bytes: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE arxiv_papers SET sha256=?, size_bytes=?, "
            "storage_status='downloaded', updated_at=? WHERE arxiv_id=?",
            (sha256, size_bytes, _now(), arxiv_id))


def set_paper_pages(arxiv_id: str, num_pages: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE arxiv_papers SET num_pages=?, updated_at=? WHERE arxiv_id=?",
            (num_pages, _now(), arxiv_id))


def set_paper_stored(arxiv_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE arxiv_papers SET storage_status='stored', updated_at=? "
            "WHERE arxiv_id=?", (_now(), arxiv_id))


def set_paper_submitted(arxiv_id: str, ocr_task_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE arxiv_papers SET ocr_task_id=?, updated_at=? WHERE arxiv_id=?",
            (ocr_task_id, _now(), arxiv_id))


def set_paper_parsed(arxiv_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE arxiv_papers SET parsed=1, storage_status='parsed', "
            "updated_at=? WHERE arxiv_id=?", (_now(), arxiv_id))


def set_paper_failed(arxiv_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE arxiv_papers SET storage_status='failed', updated_at=? "
            "WHERE arxiv_id=?", (_now(), arxiv_id))


def list_papers(limit: int = 100, parsed_only: bool = False) -> list[dict]:
    where = "WHERE parsed=1" if parsed_only else ""
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM arxiv_papers {where} ORDER BY updated_at DESC LIMIT ?",
            (limit,)).fetchall()
        return [_paper_row(r) for r in rows]


def count_papers() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM arxiv_papers").fetchone()["n"]
        parsed = conn.execute(
            "SELECT COUNT(*) AS n FROM arxiv_papers WHERE parsed=1").fetchone()["n"]
        stored = conn.execute(
            "SELECT COUNT(*) AS n FROM arxiv_papers WHERE storage_status "
            "IN ('stored','parsed')").fetchone()["n"]
        bytes_stored = conn.execute(
            "SELECT COALESCE(SUM(size_bytes),0) AS b FROM arxiv_papers "
            "WHERE sha256 IS NOT NULL").fetchone()["b"]
    return {"total": total, "parsed": parsed, "stored": stored,
            "bytes": bytes_stored}


# ----------------------------------------------------------------- runs

def create_run(query: str, max_papers: int, paper_ids: list[str]) -> str:
    """Open a run and pre-seed a `queued` step row for every paper's every stage.

    Pre-seeding means the UI can render the full grid (paper × stage) before
    anything starts, and a step that has never run is distinguishable from one
    that errored.
    """
    run_id = uuid.uuid4().hex[:12]
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO pipeline_runs (id, query, max_papers, status, created_at) "
            "VALUES (?, ?, ?, 'running', ?)",
            (run_id, query, max_papers, now))
        rows = [(run_id, pid, stage) for pid in paper_ids for stage in STAGES]
        conn.executemany(
            "INSERT OR IGNORE INTO pipeline_steps (run_id, arxiv_id, stage, status) "
            "VALUES (?, ?, ?, 'queued')", rows)
    return run_id


def get_run(run_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM pipeline_runs WHERE id=?",
                           (run_id,)).fetchone()
        return dict(row) if row else None


def list_runs(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in rows]


def steps_for_run(run_id: str) -> list[dict]:
    """All step rows for a run, ordered for a stable paper×stage grid."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT s.*, p.title, p.sha256, p.storage_status FROM pipeline_steps s "
            "JOIN arxiv_papers p ON p.arxiv_id = s.arxiv_id "
            "WHERE s.run_id=? ORDER BY s.arxiv_id, s.stage", (run_id,)).fetchall()
        return [dict(r) for r in rows]


def steps_by_paper(run_id: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for step in steps_for_run(run_id):
        out.setdefault(step["arxiv_id"], []).append(step)
    return out


# ----------------------------------------------------------------- steps

def step_start(run_id: str, arxiv_id: str, stage: str, detail: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE pipeline_steps SET status='running', detail=?, error=NULL, "
            "started_at=?, finished_at=NULL WHERE run_id=? AND arxiv_id=? AND stage=?",
            (detail, _now(), run_id, arxiv_id, stage))


def step_detail(run_id: str, arxiv_id: str, stage: str, detail: str) -> None:
    """Update only the free-form progress text (no status change).

    Used for live progress like 'page 3/12' during the wait_ocr stage.
    """
    with _connect() as conn:
        conn.execute(
            "UPDATE pipeline_steps SET detail=? WHERE run_id=? AND arxiv_id=? AND stage=?",
            (detail, run_id, arxiv_id, stage))


def step_done(run_id: str, arxiv_id: str, stage: str, detail: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE pipeline_steps SET status='done', detail=?, finished_at=? "
            "WHERE run_id=? AND arxiv_id=? AND stage=?",
            (detail, _now(), run_id, arxiv_id, stage))
    _recompute_run(run_id)


def step_skip(run_id: str, arxiv_id: str, stage: str, detail: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE pipeline_steps SET status='skipped', detail=?, finished_at=? "
            "WHERE run_id=? AND arxiv_id=? AND stage=?",
            (detail, _now(), run_id, arxiv_id, stage))
    _recompute_run(run_id)


def step_error(run_id: str, arxiv_id: str, stage: str, error: str,
               detail: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE pipeline_steps SET status='error', detail=?, error=?, finished_at=? "
            "WHERE run_id=? AND arxiv_id=? AND stage=?",
            (detail, error, _now(), run_id, arxiv_id, stage))
    _recompute_run(run_id)


def _recompute_run(run_id: str) -> None:
    """Re-derive a run's aggregate counters and terminal status from its steps.

    A run is 'done' when every step reached a terminal state (done/error/skipped);
    'error' only if at least one step errored AND nothing is still running.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT status FROM pipeline_steps WHERE run_id=?", (run_id,)).fetchall()
        if not rows:
            return
        done = sum(1 for r in rows if r["status"] == "done")
        failed = sum(1 for r in rows if r["status"] == "error")
        skipped = sum(1 for r in rows if r["status"] == "skipped")
        pending = sum(1 for r in rows if r["status"] in ("queued", "running"))
        if pending:
            return  # still in flight — leave the run running
        status = "error" if failed else "done"
        conn.execute(
            "UPDATE pipeline_runs SET done=?, failed=?, skipped=?, status=?, "
            "finished_at=? WHERE id=?",
            (done, failed, skipped, status, _now(), run_id))


def finish_run(run_id: str, status: str = "done", error: str = "") -> None:
    """Force-close a run (e.g. the runner process caught a fatal error)."""
    _recompute_run(run_id)
    with _connect() as conn:
        conn.execute(
            "UPDATE pipeline_runs SET status=?, error=?, finished_at=? WHERE id=?",
            (status, error, _now(), run_id))

"""Content-addressed store of parsed documents.

Agents resubmit the same paper constantly — the same arxiv PDF asked for by three
agents is three identical 20-second parses of the same bytes. Keying results on
the SHA-256 of the uploaded file (plus the prompt mode and the page selection,
because those change the answer) turns every repeat into a lookup.

The store lives in the same SQLite file as the queue, so deployment gains nothing
to run and a backup is still one file. Search uses FTS5, which ships with the
stdlib sqlite3 and handles Cyrillic as well as Latin.
"""

import hashlib
import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    sha256 TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    kind TEXT NOT NULL,               -- 'pdf' | 'image'
    num_pages INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    first_seen_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    times_submitted INTEGER NOT NULL DEFAULT 1
);
-- One document parsed with different prompts, or different page subsets, gives
-- different answers, so the cache key is all three.
CREATE TABLE IF NOT EXISTS document_results (
    sha256 TEXT NOT NULL,
    prompt_mode TEXT NOT NULL,
    pages_key TEXT NOT NULL,          -- canonical page list, '' means every page
    task_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    markdown TEXT NOT NULL,
    pages_done INTEGER NOT NULL,
    generated_tokens INTEGER NOT NULL DEFAULT 0,
    seconds REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    PRIMARY KEY (sha256, prompt_mode, pages_key)
);
CREATE INDEX IF NOT EXISTS idx_results_created ON document_results(created_at DESC);
CREATE VIRTUAL TABLE IF NOT EXISTS document_search USING fts5(
    sha256 UNINDEXED, prompt_mode UNINDEXED, filename, body
);
"""

_DB_PATH = None


def init(path):
    global _DB_PATH
    _DB_PATH = str(path)
    with _connect() as conn:
        conn.executescript(SCHEMA)


def _connect():
    conn = sqlite3.connect(_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def sha256_of(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def pages_key(pages):
    """Canonical form of a page selection, so [2,1] and [1,2] share a cache entry."""
    if not pages:
        return ""
    return ",".join(str(p) for p in sorted(set(int(p) for p in pages)))


# ---------------------------------------------------------------- documents

def remember_document(sha256, filename, kind, num_pages, size_bytes):
    """Record the upload; returns True when this content is new to the store.

    On resubmission we also refresh `filename`, `kind`, `num_pages`, and
    `size_bytes` — a corrected page count (or a renamed file) from a later
    upload should propagate, otherwise stale metadata is served forever from
    the original row. Content-addressing by sha256 still guarantees
    idempotency: same bytes → same row.
    """
    now = time.time()
    with _connect() as conn:
        row = conn.execute("SELECT sha256 FROM documents WHERE sha256=?", (sha256,)).fetchone()
        if row:
            conn.execute(
                "UPDATE documents SET last_seen_at=?, times_submitted=times_submitted+1, "
                "filename=?, kind=?, num_pages=?, size_bytes=? "
                "WHERE sha256=?",
                (now, filename, kind, num_pages, size_bytes, sha256))
            return False
        conn.execute(
            "INSERT INTO documents (sha256, filename, kind, num_pages, size_bytes, "
            "first_seen_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sha256, filename, kind, num_pages, size_bytes, now, now))
        return True


def get_document(sha256):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------- results

def find_result(sha256, prompt_mode, pages):
    """The cached parse for exactly this document + prompt + page selection."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM document_results WHERE sha256=? AND prompt_mode=? AND pages_key=?",
            (sha256, prompt_mode, pages_key(pages))).fetchone()
        return dict(row) if row else None


def find_latest_result(sha256, prompt_mode):
    """The newest parse of this document with this prompt, whatever pages it covered.

    Used when the caller did not say which page selection it means. It returns a
    full row — the bundle needs job_id to find the artifacts on disk, and a
    partial row here is how that endpoint came to raise KeyError.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM document_results WHERE sha256=? AND prompt_mode=? "
            "ORDER BY created_at DESC LIMIT 1", (sha256, prompt_mode)).fetchone()
        return dict(row) if row else None


def store_result(sha256, prompt_mode, pages, task_id, job_id, markdown,
                 pages_done, generated_tokens=0, seconds=0.0, filename=""):
    key = pages_key(pages)
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO document_results (sha256, prompt_mode, pages_key, "
            "task_id, job_id, markdown, pages_done, generated_tokens, seconds, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sha256, prompt_mode, key, task_id, job_id, markdown, pages_done,
             generated_tokens, seconds, time.time()))
        # keep the index in step: FTS5 has no upsert, so replace the row outright
        conn.execute("DELETE FROM document_search WHERE sha256=? AND prompt_mode=?",
                     (sha256, prompt_mode))
        conn.execute(
            "INSERT INTO document_search (sha256, prompt_mode, filename, body) "
            "VALUES (?, ?, ?, ?)", (sha256, prompt_mode, filename, markdown))


def search(query, limit=20):
    """Full-text search over parsed documents.

    The query goes to FTS5 verbatim first, so callers keep its syntax (phrases,
    NEAR, prefixes, OR). If that is a syntax error the query is retried as a
    quoted phrase: FTS5 reads `-` as NOT, which makes ordinary terms like
    "open-source" or "GPT-4V" fail, and an agent searching for a hyphenated word
    should get results rather than a 400.

    As a last resort, if FTS returns zero matches (e.g. the unicode61 tokenizer
    split a term in a way the query didn't anticipate), we fall back to a
    case-insensitive LIKE over both body and filename. This guarantees a
    substring match a user would reasonably expect.
    """
    try:
        rows = _run_search(query, limit)
    except ValueError:
        escaped = '"' + query.replace('"', '""') + '"'
        rows = _run_search(escaped, limit)
    if rows:
        return rows
    return _run_search_like(query, limit)


def _run_search_like(query, limit):
    """Substring fallback used when FTS5 returns 0 rows. Slower but exact."""
    pattern = f"%{query}%"
    with _connect() as conn:
        rows = conn.execute(
            "SELECT s.sha256, s.prompt_mode, s.filename, "
            "       substr(s.body, 1, 240) AS snippet, "
            "       d.num_pages, d.times_submitted, r.created_at "
            "FROM document_search s "
            "JOIN documents d ON d.sha256 = s.sha256 "
            "LEFT JOIN document_results r ON r.sha256 = s.sha256 "
            "     AND r.prompt_mode = s.prompt_mode "
            "WHERE s.body LIKE ? COLLATE NOCASE OR s.filename LIKE ? COLLATE NOCASE "
            "ORDER BY r.created_at DESC LIMIT ?",
            (pattern, pattern, limit)).fetchall()
        return [dict(row) for row in rows]


def _run_search(query, limit):
    with _connect() as conn:
        try:
            rows = conn.execute(
                "SELECT s.sha256, s.prompt_mode, s.filename, "
                "       snippet(document_search, 3, '[', ']', '…', 12) AS snippet, "
                "       d.num_pages, d.times_submitted, r.created_at "
                "FROM document_search s "
                "JOIN documents d ON d.sha256 = s.sha256 "
                "LEFT JOIN document_results r ON r.sha256 = s.sha256 "
                "     AND r.prompt_mode = s.prompt_mode "
                "WHERE document_search MATCH ? ORDER BY rank LIMIT ?",
                (query, limit)).fetchall()
        except sqlite3.OperationalError as error:
            raise ValueError(f"bad search query: {error}") from error
        return [dict(row) for row in rows]


def stats():
    with _connect() as conn:
        documents = conn.execute("SELECT COUNT(*) AS n, SUM(size_bytes) AS b, "
                                 "SUM(times_submitted) AS s FROM documents").fetchone()
        results = conn.execute("SELECT COUNT(*) AS n FROM document_results").fetchone()
        return {
            "documents": documents["n"] or 0,
            "bytes": documents["b"] or 0,
            "submissions": documents["s"] or 0,
            "cached_results": results["n"] or 0,
            # every submission beyond the first that hit a cached result is a
            # parse that did not have to run
            "reuse_ratio": round(
                1 - (results["n"] or 0) / max(documents["s"] or 1, 1), 3),
        }


def recent(limit=20):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT r.sha256, r.prompt_mode, r.pages_done, r.generated_tokens, "
            "       r.seconds, r.created_at, d.filename, d.num_pages "
            "FROM document_results r JOIN documents d ON d.sha256 = r.sha256 "
            "ORDER BY r.created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

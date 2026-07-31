# Arxiv quant / algorithmic-trading pipeline

An end-to-end pipeline that discovers quant-finance / algorithmic-trading
papers on arxiv, parses them to Markdown with the dots.mocr OCR service, and
caches both the source PDFs and the parsed result bundles in **SeaweedFS** so a
repeat request is a lookup, not a re-parse.

It adds a status UI, a content-addressed blob store, and a multi-stage DAG on
top of the existing OCR service — without changing the service's own cache
semantics.

```
                    ┌─────────── arxiv API ───────────┐
                    │  q-fin.*  OR  algo-trading       │
                    └──────────────┬───────────────────┘
                                   ▼
   download ──▶ submit_ocr ──▶ wait_ocr ──▶ fetch_bundle ──▶ store ──▶ index
       │             │             │              │               │
       ▼             ▼             ▼              ▼               ▼
   SeaweedFS     OCR service    (poll)        OCR service     SeaweedFS
   (source PDF)  /api/v1/...                  /bundle zip     (result bundle)
```

## Architecture

Two processes cooperate over a shared filesystem (SQLite + SeaweedFS), with no
network RPC between them:

| layer | runs where | has boto3? | role |
|-------|-----------|-----------|------|
| **OCR service** (`demo.server`) | the `dots_mocr_demo` container | no | parses docs, serves the UI + `/api/v1/arxiv/*` status, owns the markdown cache in `demo.db` |
| **pipeline runner** (`demo.scripts.arxiv_pipeline`) | a host venv | yes | discovers papers, downloads PDFs, drives the OCR service, stores blobs in SeaweedFS, writes run/step progress |

The seam between them is the **`arxiv.db` SQLite file** (created by whichever
process starts first; both read/write it) and the **SeaweedFS bucket** for
binaries. The container never needs boto3: it reads run/step rows the runner
writes, and the parsed markdown already lives in its own `demo.db` cache from
the OCR parse itself.

## What gets cached, and where

| artefact | location | key | lifetime |
|----------|----------|-----|----------|
| parsed **markdown** (text + FTS index) | OCR service's `demo.db` (existing) | `(sha256, prompt_mode, pages)` | permanent |
| source **PDF** (binary) | SeaweedFS bucket `arxiv-papers` | `sha256` | permanent |
| result **bundle** zip (md + images + layout) | SeaweedFS bucket `arxiv-papers` | `sha256` | permanent |

A re-run that hits a paper already parsed is a cache lookup at every stage:
the PDF comes from SeaweedFS (`download` → skipped), the OCR is already done
(`wait_ocr` → skipped, `status: cached`), and the bundle is already stored.

## DAG stages

`download → submit_ocr → wait_ocr → fetch_bundle → store → index`

Each stage of each paper is one row in `pipeline_steps` (`queued` → `running`
→ `done`/`error`/`skipped`). A run's aggregate status and counters are derived
from its steps; a failure in one paper isolates (the rest still complete).

## Quick start (on the server)

The OCR service + SeaweedFS are already running on the dev server
(`192.168.0.1` / `192.168.1.68`). From there:

```sh
# 1. one-time: host venv with boto3 (the container doesn't have it)
cd /mnt/nvme2/ocr-arxiv            # a clone of this branch
python3 -m venv .venv
.venv/bin/pip install boto3 requests PyMuPDF Pillow

# 2. point the runner at SeaweedFS + the shared state dir + the OCR service
export SEAWEED_S3_ENDPOINT=http://127.0.0.1:8333
export SEAWEED_ACCESS_KEY=agent_key
export SEAWEED_SECRET_KEY=agent_secret_dev_change_me
export SEAWEED_BUCKET=arxiv-papers
export DEMO_STATE_DIR=/mnt/nvme2/ocr_server_state     # shared with the container
export ARXIV_DB_PATH=$DEMO_STATE_DIR/arxiv.db         # default; spelled out
export OCR_URL=http://127.0.0.1:8601

# 3. run the pipeline on 10 papers
.venv/bin/python demo/scripts/arxiv_pipeline.py --limit 10

# 4. watch it live
open http://192.168.0.1:8601/arxiv        # or via the SSH tunnel
```

### Important: arxiv.db permissions

`arxiv.db` is written by the runner (a normal user) **and** read by the
container (root). Create it once with world-write so both can use WAL:

```sh
docker stop dots_mocr_demo
rm -f $DEMO_STATE_DIR/arxiv.db*
( umask 000; .venv/bin/python -c "from demo.arxiv import db; db.init('$DEMO_STATE_DIR/arxiv.db')" )
chmod 666 $DEMO_STATE_DIR/arxiv.db
docker start dots_mocr_demo
```

(`demo.db` itself stays root-owned and is only written by the container — the
runner never touches it.)

## Environment variables

| var | default | meaning |
|-----|---------|---------|
| `SEAWEED_S3_ENDPOINT` | unset → local disk | SeaweedFS S3 gateway URL |
| `SEAWEED_ACCESS_KEY` / `SECRET_KEY` | — | S3 credentials (agent identity) |
| `SEAWEED_BUCKET` | `arxiv-papers` | bucket name (auto-created) |
| `ARXIV_DB_PATH` | `<state>/arxiv.db` | pipeline SQLite file |
| `DEMO_STATE_DIR` | `/state` | state dir (shared with container) |
| `OCR_URL` | `http://127.0.0.1:8601` | OCR service base |
| `OCR_PROMPT_MODE` | `prompt_layout_all_en` | OCR prompt mode |

## CLI flags

```
--query quant        arxiv named query, or raw syntax (abs:"...", cat:q-fin.TR, ...)
--limit 10           max papers per run
--workers 2          DAG concurrency (OCR is single-worker regardless)
--discover-only      populate arxiv_papers without running the DAG
--start N            arxiv result offset
```

## HTTP API (`/api/v1/arxiv/*`)

All read-only, served by the container from `arxiv.db`:

| method | path | returns |
|--------|------|---------|
| GET | `/runs` | recent runs |
| GET | `/runs/{id}` | one run + its paper×stage grid |
| GET | `/papers` | parsed-paper catalogue |
| GET | `/papers/{arxiv_id}` | one paper |
| GET | `/papers/{arxiv_id}/bundle` | the stored bundle zip (from SeaweedFS) |
| GET | `/stats` | totals + storage backend name |

## UI

`/arxiv` — a single page that polls `/api/v1/arxiv/*` every 2s and renders a
paper×stage status grid, an aggregate progress bar, the recent-runs list, and a
filterable parsed-papers catalogue (click a row to download its bundle).

## Files

```
demo/storage.py                 BlobStore: SeaweedFS (boto3) + local fallback
demo/arxiv/db.py                papers / runs / steps tables (shared arxiv.db)
demo/arxiv/source.py            arxiv API client + Atom feed parser
demo/arxiv/pipeline.py          DAG executor + OCR client
demo/arxiv_api.py               /api/v1/arxiv/* router (container-side, read-only)
demo/scripts/arxiv_pipeline.py  host-side CLI runner
demo/static/arxiv.{html,css,js} status UI
tests/test_storage.py           blob store contract (network-free)
tests/test_arxiv_db.py          step state machine + monotonic status
tests/test_arxiv_source.py      feed parser (network-free) + live marker
tests/test_arxiv_pipeline.py    full DAG against a fake OCR service
```

## Design notes

- **No celery / redis / airflow.** The executor is a `ThreadPoolExecutor`
  with bounded workers, matching the service's existing in-process design.
  Download/store/index parallelise; submit/wait are gated by the OCR service's
  own single-worker queue anyway.
- **Content-addressed everything.** PDFs and bundles are keyed on SHA-256,
  mirroring the OCR service's own markdown cache. The cache is *additive* — it
  sits behind the existing dedup, never replaces its semantics.
- **Graceful degradation.** If boto3 or SeaweedFS is unavailable, the store
  falls back to a local sharded directory; the UI surfaces which backend is
  active. The pipeline runs anywhere there is a writable directory.
- **Rate-limit-aware.** arxiv throttles harder than its documented 1 req/3s;
  `search()` and `fetch_pdf()` retry on 429/timeouts with exponential backoff,
  honouring `Retry-After`.

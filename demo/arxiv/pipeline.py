"""DAG executor for the arxiv -> OCR -> SeaweedFS pipeline.

A run = one search, N papers. Each paper walks the DAG:

    download -> submit_ocr -> wait_ocr -> fetch_bundle -> store -> index

Every stage is recorded in pipeline_steps; failures are isolated per paper
(one bad download does not abort the run). The executor is a thin
ThreadPoolExecutor wrapper — there is no celery/redis/airflow on purpose,
matching the service's in-process design.

Cross-process contract: this runs in the host venv (has boto3 + requests),
the OCR container reads the same demo.db over the SQLite file. The container
never needs a network client for storage because the *binary* blobs go to
SeaweedFS via this process, while the parsed *markdown* already lives in the
container's SQLite cache from the OCR parse.
"""

from __future__ import annotations

import io
import json
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

from demo.arxiv import db, source
from demo.arxiv.db import STAGES
from demo import storage

# How the pipeline talks to the OCR service. Same protocol ocrc uses:
# multipart POST /api/v1/documents, then poll /documents/{sha}, then GET the
# bundle zip. Defaults point at the local container; override for tunnels.
DEFAULT_OCR_URL = "http://127.0.0.1:8601"
DEFAULT_PROMPT_MODE = "prompt_layout_all_en"

# Polling cadence and resilience for the OCR wait. Long PDFs (50+ pages) take
# many minutes; a transient 5xx must not abort a 10-minute parse.
POLL_INTERVAL_S = 5.0
MAX_TRANSIENT_ERRORS = 12


class PipelineError(Exception):
    """A stage failed terminally (4xx, bad PDF, OCR error)."""


class _Transient(Exception):
    """Ride-through error: retry the next poll instead of giving up."""


# ----------------------------------------------------------------- OCR client

def _ocr_request(ocr_url: str, method: str, path: str, *,
                 timeout: float = 60.0, **kwargs):
    url = ocr_url.rstrip("/") + path
    response = requests.request(method, url, timeout=timeout, **kwargs)
    if response.status_code >= 500:
        raise _Transient(f"{method} {path} -> {response.status_code}")
    response.raise_for_status()
    return response


def submit_to_ocr(ocr_url: str, pdf_bytes: bytes, sha256: str,
                  prompt_mode: str, agent: str) -> dict:
    """POST the PDF to the OCR service; returns the submit response.

    The service dedups on (sha256, prompt_mode, pages), so a paper already
    parsed returns `status: cached` immediately — that is the cache hit that
    makes re-runs cheap.
    """
    files = {"file": (f"{sha256}.pdf", pdf_bytes, "application/pdf")}
    data = {"prompt_mode": prompt_mode, "pages": "all", "agent": agent}
    response = _ocr_request(ocr_url, "POST", "/api/v1/documents",
                            files=files, data=data, timeout=300)
    return response.json()


def ocr_status(ocr_url: str, sha256: str, prompt_mode: str) -> dict:
    q = {"prompt_mode": prompt_mode}
    response = _ocr_request(ocr_url, "GET", f"/api/v1/documents/{sha256}",
                            params=q, timeout=30)
    return response.json()


def ocr_bundle(ocr_url: str, sha256: str, prompt_mode: str) -> bytes:
    q = {"prompt_mode": prompt_mode}
    response = _ocr_request(ocr_url, "GET", f"/api/v1/documents/{sha256}/bundle",
                            params=q, timeout=300)
    return response.content


def wait_for_ocr(ocr_url: str, sha256: str, prompt_mode: str,
                 on_progress=None, poll: float = POLL_INTERVAL_S,
                 max_transient: int = MAX_TRANSIENT_ERRORS) -> dict:
    """Block until the OCR parse finishes, reporting progress along the way.

    Rides through transient 5xx/network errors (a long parse must survive a
    brief blip) but exits immediately on a terminal status (done/error/
    cancelled) or a hard 4xx.
    """
    last_detail = None
    consecutive = 0
    while True:
        try:
            state = ocr_status(ocr_url, sha256, prompt_mode)
            consecutive = 0
        except _Transient as error:
            consecutive += 1
            if consecutive >= max_transient:
                raise PipelineError(
                    f"gave up after {consecutive} transient errors: {error}")
            time.sleep(min(poll * consecutive, 30))
            continue
        status = state.get("status")
        if state.get("cached") or status == "done":
            return state
        if status in ("error", "cancelled"):
            raise PipelineError(f"OCR {status} for {sha256[:12]}")
        progress = state.get("progress") or {}
        done = progress.get("done", 0)
        total = progress.get("total", "?")
        detail = f"page {done}/{total}"
        if on_progress and detail != last_detail:
            on_progress(detail)
            last_detail = detail
        time.sleep(poll)


# ----------------------------------------------------------------- DAG stages

def stage_download(paper: dict, ocr_url: str, prompt_mode: str,
                   run_id: str, *, on_detail=None) -> tuple[bytes, str]:
    """Ensure the PDF is in blob storage; return (bytes, sha256).

    If the bytes are already cached (SeaweedFS/local has this sha256), the
    download stage is `skipped` and the bytes come from storage — a re-run of
    the pipeline never re-downloads from arxiv.
    """
    arxiv_id = paper["arxiv_id"]
    if on_detail:
        on_detail("fetching from arxiv")
    pdf_bytes = source.fetch_pdf(paper["pdf_url"])
    sha = storage.sha256_bytes(pdf_bytes)
    db.set_paper_downloaded(arxiv_id, sha, len(pdf_bytes))
    if storage.has_pdf(sha):
        storage.put_pdf(sha, pdf_bytes)  # idempotent overwrite
    else:
        storage.put_pdf(sha, pdf_bytes)
    if on_detail:
        on_detail(f"{len(pdf_bytes)//1024} KB, sha {sha[:8]}")
    return pdf_bytes, sha


def stage_ensure_pdf(paper: dict, run_id: str, *, on_detail=None) -> bytes:
    """Return the PDF bytes for a paper, downloading only if not cached.

    Split from stage_download so a re-run that finds the PDF already in
    storage skips the network entirely (the `download` step is marked
    `skipped`) but still yields the bytes the OCR submit needs.
    """
    arxiv_id = paper["arxiv_id"]
    existing = db.get_paper(arxiv_id)
    sha = existing.get("sha256") if existing else None
    if sha and storage.has_pdf(sha):
        if on_detail:
            on_detail("cached in storage")
        return storage.get_pdf(sha)
    if on_detail:
        on_detail("fetching from arxiv")
    pdf_bytes = source.fetch_pdf(paper["pdf_url"])
    sha = storage.sha256_bytes(pdf_bytes)
    db.set_paper_downloaded(arxiv_id, sha, len(pdf_bytes))
    storage.put_pdf(sha, pdf_bytes)
    if on_detail:
        on_detail(f"{len(pdf_bytes)//1024} KB")
    return pdf_bytes


def stage_submit_ocr(ocr_url: str, prompt_mode: str, agent: str,
                     pdf_bytes: bytes, sha256: str, arxiv_id: str,
                     *, on_detail=None) -> dict:
    if on_detail:
        on_detail("submitting to OCR")
    result = submit_to_ocr(ocr_url, pdf_bytes, sha256, prompt_mode, agent)
    status = result.get("status")
    if status == "cached":
        db.set_paper_parsed(arxiv_id)
    task_id = result.get("task_id") or ""
    db.set_paper_submitted(arxiv_id, task_id)
    if on_detail:
        on_detail(f"status={status}")
    return result


def stage_wait_ocr(ocr_url: str, prompt_mode: str, sha256: str,
                   arxiv_id: str, run_id: str, *, on_detail=None) -> dict:
    state = wait_for_ocr(ocr_url, sha256, prompt_mode, on_progress=on_detail)
    db.set_paper_parsed(arxiv_id)
    return state


def stage_fetch_bundle(ocr_url: str, prompt_mode: str, sha256: str,
                       *, on_detail=None) -> bytes:
    if on_detail:
        on_detail("downloading bundle")
    bundle = ocr_bundle(ocr_url, sha256, prompt_mode)
    if on_detail:
        on_detail(f"{len(bundle)//1024} KB")
    return bundle


def stage_store(bundle_bytes: bytes, sha256: str, arxiv_id: str, parser: str,
                *, on_detail=None) -> str:
    if on_detail:
        on_detail("storing bundle")
    # keyed by (sha, parser) so dots_mocr and classic bundles coexist
    key = storage.put_bundle(sha256, bundle_bytes, parser=parser)
    db.set_paper_stored(arxiv_id)
    if on_detail:
        on_detail("stored")
    return key


def stage_index(bundle_bytes: bytes, sha256: str, arxiv_id: str, parser: str,
                *, on_detail=None) -> int:
    """Extract page count + record the parse, tagged by parser.

    For dots.mocr the markdown body is already in the OCR service's own SQLite
    FTS index from the parse; here we lift the page count and write a
    paper_parses row so the catalogue knows WHICH parser produced this bundle.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
            with archive.open("meta.json") as meta_file:
                meta = json.loads(meta_file.read())
        pages = int(meta.get("pages_done") or 0)
        if pages:
            db.set_paper_pages(arxiv_id, pages)
        # Presign so the boto3-less container can redirect /bundle to SeaweedFS;
        # empty for the local backend (the container reads local files directly).
        bundle_url = storage.get_store().presign(sha256, storage.KIND_BUNDLE, parser) or ""
        db.record_parse(arxiv_id, parser, sha256, status="parsed",
                        pages_done=pages, bundle_key=f"<sha>.{parser}.bundle",
                        bundle_url=bundle_url)
        if on_detail:
            on_detail(f"{pages} pages indexed by {parser}")
        return pages
    except Exception as error:  # noqa: BLE001 — indexing is best-effort
        db.record_parse(arxiv_id, parser, sha256, status="failed", error=str(error))
        if on_detail:
            on_detail(f"index skipped: {error}")
        return 0


# ----------------------------------------------------------------- per-paper DAG

def stage_local_parse(parser: str, pdf_bytes: bytes, sha256: str, arxiv_id: str,
                      *, on_detail=None) -> bytes:
    """Run a GPU-less classic parser on the PDF bytes and build a bundle.

    Mirrors the OCR path's output shape (document.md + meta.json) so the rest
    of the pipeline is parser-agnostic. Used for classic_fitz / classic_pdfplumber.
    """
    from demo.arxiv import parsers
    if on_detail:
        on_detail(f"parsing with {parser}")
    result = parsers.parse_locally(parser, pdf_bytes)
    bundle = parsers.build_bundle(result["markdown"], result["meta"], result["images"])
    if on_detail:
        on_detail(f"{result['meta'].get('pages_done', 0)} pages")
    return bundle


# ----------------------------------------------------------------- per-paper DAG

def run_paper_dag(paper: dict, ocr_url: str, prompt_mode: str, agent: str,
                  run_id: str, parser: str = "dots_mocr",
                  download_only: bool = False) -> dict:
    """Walk all stages for one paper, recording each in pipeline_steps.

    `parser` selects the algorithm: dots_mocr goes through the OCR service
    over HTTP (submit -> wait -> fetch bundle); a local parser
    (classic_fitz / classic_pdfplumber) runs in-process and replaces the
    submit/wait/fetch trio with a single parse stage.

    `download_only` stops after the PDF is in storage — used to pre-fetch the
    whole corpus before any parsing, so later parse runs are I/O-free.

    A failure short-circuits the remaining stages (marked `skipped` with the
    upstream reason) so the run's grid shows where it stopped instead of
    hiding the tail. Returns a summary dict for the executor to aggregate.
    """
    arxiv_id = paper["arxiv_id"]

    def detail(stage, text):
        db.step_detail(run_id, arxiv_id, stage, text)

    try:
        # 1. download (skipped if PDF already in storage)
        cached_pdf = (db.get_paper(arxiv_id) or {}).get("sha256")
        if cached_pdf and storage.has_pdf(cached_pdf):
            db.step_skip(run_id, arxiv_id, "download", "cached in storage")
            pdf_bytes = storage.get_pdf(cached_pdf)
            sha = cached_pdf
        else:
            db.step_start(run_id, arxiv_id, "download")
            pdf_bytes, sha = stage_download(paper, ocr_url, prompt_mode, run_id,
                                            on_detail=lambda t: detail("download", t))
            db.step_done(run_id, arxiv_id, "download")

        # download-only runs stop here: PDF is in storage, parse skipped.
        if download_only:
            for stage in ("submit_ocr", "wait_ocr", "fetch_bundle", "store", "index"):
                db.step_skip(run_id, arxiv_id, stage, "download-only run")
            return {"arxiv_id": arxiv_id, "ok": True, "sha256": sha,
                    "download_only": True}

        # 2-4. parse — three shapes depending on parser:
        #   - dots_mocr : submit to OCR service, wait, fetch the bundle it built
        #   - local parser (classic_fitz, ...): parse in-process into a bundle
        #   - already-parsed by THIS parser: skip everything (cache hit)
        if storage.has_bundle(sha, parser):
            for stage in ("submit_ocr", "wait_ocr", "fetch_bundle"):
                db.step_skip(run_id, arxiv_id, stage, f"{parser} bundle cached")
            bundle = storage.get_bundle(sha, parser)
        elif parser == "dots_mocr":
            db.step_start(run_id, arxiv_id, "submit_ocr")
            submit = stage_submit_ocr(ocr_url, prompt_mode, agent, pdf_bytes, sha,
                                      arxiv_id, on_detail=lambda t: detail("submit_ocr", t))
            db.step_done(run_id, arxiv_id, "submit_ocr", f"status={submit.get('status')}")
            if submit.get("status") == "cached":
                db.step_skip(run_id, arxiv_id, "wait_ocr", "OCR cache hit")
            else:
                db.step_start(run_id, arxiv_id, "wait_ocr")
                stage_wait_ocr(ocr_url, prompt_mode, sha, arxiv_id, run_id,
                               on_detail=lambda t: detail("wait_ocr", t))
                db.step_done(run_id, arxiv_id, "wait_ocr")
            db.step_start(run_id, arxiv_id, "fetch_bundle")
            bundle = stage_fetch_bundle(ocr_url, prompt_mode, sha,
                                        on_detail=lambda t: detail("fetch_bundle", t))
            db.step_done(run_id, arxiv_id, "fetch_bundle")
        else:
            # local parser collapses submit+wait+fetch into one in-process parse
            db.step_start(run_id, arxiv_id, "submit_ocr")
            db.step_detail(run_id, arxiv_id, "submit_ocr", f"local {parser}")
            db.step_done(run_id, arxiv_id, "submit_ocr", f"local:{parser}")
            db.step_skip(run_id, arxiv_id, "wait_ocr", "local parser (no OCR)")
            db.step_start(run_id, arxiv_id, "fetch_bundle")
            bundle = stage_local_parse(parser, pdf_bytes, sha, arxiv_id,
                                       on_detail=lambda t: detail("fetch_bundle", t))
            db.step_done(run_id, arxiv_id, "fetch_bundle")

        # 5. store bundle in SeaweedFS (keyed by sha+parser)
        db.step_start(run_id, arxiv_id, "store")
        stage_store(bundle, sha, arxiv_id, parser, on_detail=lambda t: detail("store", t))
        db.step_done(run_id, arxiv_id, "store")

        # 6. index (records paper_parses row tagged with parser)
        db.step_start(run_id, arxiv_id, "index")
        stage_index(bundle, sha, arxiv_id, parser, on_detail=lambda t: detail("index", t))
        db.step_done(run_id, arxiv_id, "index")

        return {"arxiv_id": arxiv_id, "ok": True, "sha256": sha, "parser": parser}
    except Exception as error:  # noqa: BLE001 — isolate per paper
        # mark the failed stage error, skip the rest
        _fail_remaining(run_id, arxiv_id, error)
        db.set_paper_failed(arxiv_id)
        traceback.print_exc()
        return {"arxiv_id": arxiv_id, "ok": False, "error": str(error),
                "sha256": (db.get_paper(arxiv_id) or {}).get("sha256")}


def _fail_remaining(run_id: str, arxiv_id: str, error: Exception) -> None:
    """Mark the first non-terminal stage `error` and the rest `skipped`."""
    msg = f"{type(error).__name__}: {error}"
    found_running = False
    for stage in STAGES:
        steps = db.steps_for_run(run_id)
        match = next((s for s in steps if s["arxiv_id"] == arxiv_id
                      and s["stage"] == stage), None)
        if match is None:
            continue
        if match["status"] in ("queued", "running"):
            if not found_running:
                db.step_error(run_id, arxiv_id, stage, msg)
                found_running = True
            else:
                db.step_skip(run_id, arxiv_id, stage, "upstream failed")
    if not found_running:
        # everything was already terminal but the paper raised anyway
        db.step_error(run_id, arxiv_id, STAGES[-1], msg)


# ----------------------------------------------------------------- run driver

def discover(query: str, max_papers: int, polite: bool = True) -> list[dict]:
    """Search arxiv and upsert every result into arxiv_papers.

    Returns the list of arxiv_ids to process. Upserting here (before the run
    is even created) means the paper table is the source of truth for what
    has been seen, independent of any run.
    """
    papers = source.search(query, max_results=max_papers, polite=polite)
    ids = []
    for paper in papers:
        db.upsert_paper(paper)
        ids.append(paper["arxiv_id"])
    return papers


def run_pipeline(papers: list[dict], *, query: str, ocr_url: str,
                 prompt_mode: str = DEFAULT_PROMPT_MODE,
                 agent: str = "arxiv-pipeline", workers: int = 2,
                 parser: str = "dots_mocr", download_only: bool = False) -> str:
    """Execute the DAG over N papers with bounded concurrency.

    `workers` caps concurrency. For dots_mocr, download/store are cheap and
    parallel-friendly while submit/wait are gated by the OCR service's own
    single-worker queue, so workers>~2 mostly overlaps downloads with the OCR
    of the previous paper. For local parsers (classic_*), each paper's parse
    is CPU-bound and independent — callers can safely raise workers to use the
    whole box. `parser` selects the algorithm; `download_only` pre-fetches
    PDFs into storage without parsing. Returns the run_id.
    """
    paper_ids = [p["arxiv_id"] for p in papers]
    run_id = db.create_run(query, len(paper_ids), paper_ids,
                           parser=parser, download_only=download_only)
    kind = "download-only" if download_only else parser
    print(f"[pipeline] run {run_id}: {len(paper_ids)} papers, "
          f"workers={workers}, parser={kind}, ocr={ocr_url}, mode={prompt_mode}",
          flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(run_paper_dag, paper, ocr_url, prompt_mode, agent, run_id,
                        parser, download_only): paper
            for paper in papers
        }
        for future in as_completed(futures):
            paper = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 — never crash the driver
                traceback.print_exc()
                db.set_paper_failed(paper["arxiv_id"])
                _fail_remaining(run_id, paper["arxiv_id"], error)
                results.append({"arxiv_id": paper["arxiv_id"], "ok": False,
                                "error": str(error)})

    ok = sum(1 for r in results if r.get("ok"))
    db.finish_run(run_id)
    print(f"[pipeline] run {run_id} finished: {ok}/{len(results)} ok "
          f"(storage={storage.store_kind()}, parser={kind})", flush=True)
    return run_id

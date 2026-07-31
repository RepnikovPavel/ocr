"""HTTP surface for the arxiv pipeline — read-only views on the shared db.

These routes live in the OCR container (which has no boto3 and never talks to
SeaweedFS directly). They read the same demo.db that the host-side pipeline
runner writes, so the UI reflects runner progress in near-real time without
either process needing a network call to the other.

The container never starts a run — runs are started by the host runner
(demo/scripts/arxiv_pipeline.py) because it owns boto3 + arxiv network access.
A POST /trigger endpoint exists but is optional/off by default (set
ARXIV_ALLOW_TRIGGER=1 to enable an in-process trigger that calls the runner
functions directly; usually you want the external runner for isolation).

Routes (all under /api/v1/arxiv):
    GET  /runs                  list recent runs
    GET  /runs/{id}             one run + its paper×stage grid
    GET  /papers                parsed-paper catalogue (with sha + storage)
    GET  /papers/{arxiv_id}     one paper
    GET  /papers/{arxiv_id}/bundle   proxy the stored bundle from SeaweedFS
    GET  /stats                 pipeline totals + storage backend
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from demo import storage
from demo.arxiv import db as arxiv_db
from demo.arxiv.db import STAGES

router = APIRouter(prefix="/api/v1/arxiv", tags=["arxiv-pipeline"])

# Whether the container may start runs itself. Off by default: the host runner
# is the canonical launcher (it has boto3 + the arxiv client); the container
# only reads status. Flip on for single-process dev/test.
ALLOW_TRIGGER = False


def configure(*, allow_trigger: bool = False) -> None:
    global ALLOW_TRIGGER
    ALLOW_TRIGGER = allow_trigger


# ----------------------------------------------------------------- runs

@router.get("/runs")
def list_runs(limit: int = 20):
    runs = arxiv_db.list_runs(limit=limit)
    return {"runs": runs, "count": len(runs)}


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    run = arxiv_db.get_run(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    steps = arxiv_db.steps_for_run(run_id)
    # group into per-paper rows: one paper -> its stage statuses (in STAGES order)
    by_paper: dict[str, dict] = {}
    order: list[str] = []
    for step in steps:
        aid = step["arxiv_id"]
        if aid not in by_paper:
            by_paper[aid] = {
                "arxiv_id": aid, "title": step.get("title"),
                "sha256": step.get("sha256"),
                "storage_status": step.get("storage_status"),
                "stages": {},
            }
            order.append(aid)
        by_paper[aid]["stages"][step["stage"]] = {
            "status": step["status"], "detail": step.get("detail"),
            "error": step.get("error"),
            "started_at": step.get("started_at"),
            "finished_at": step.get("finished_at"),
        }
    papers = [by_paper[aid] for aid in order]
    return {**run, "stages": list(STAGES), "papers": papers}


# ----------------------------------------------------------------- papers

@router.get("/papers")
def list_papers(limit: int = 100, parsed_only: bool = False):
    papers = arxiv_db.list_papers(limit=limit, parsed_only=parsed_only)
    return {"papers": papers, "count": len(papers)}


@router.get("/papers/{arxiv_id}")
def get_paper(arxiv_id: str):
    paper = arxiv_db.get_paper(arxiv_id)
    if paper is None:
        raise HTTPException(404, "unknown paper")
    return paper


@router.get("/papers/{arxiv_id}/bundle")
def paper_bundle(arxiv_id: str):
    """Serve the stored result bundle (zip) for a paper.

    Reads bytes from the BlobStore (SeaweedFS in prod, local in dev). The
    container needs boto3 only if the backend is SeaweedFS — when the runner
    has been storing to SeaweedFS and boto3 is absent here, this returns 502
    with a clear message rather than crashing.
    """
    paper = arxiv_db.get_paper(arxiv_id)
    if paper is None:
        raise HTTPException(404, "unknown paper")
    sha = paper.get("sha256")
    if not sha:
        raise HTTPException(404, "paper has no stored bundle (sha256 unknown)")
    try:
        bundle = storage.get_bundle(sha)
    except Exception as error:  # noqa: BLE001 — surface backend errors cleanly
        raise HTTPException(502, f"storage backend error: {error}")
    if bundle is None:
        raise HTTPException(404, "bundle not in storage")
    filename = f"{arxiv_id}.zip"
    return Response(
        content=bundle, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# ----------------------------------------------------------------- stats

@router.get("/stats")
def stats():
    counts = arxiv_db.count_papers()
    runs = arxiv_db.list_runs(limit=5)
    try:
        backend = storage.store_kind()
    except Exception:  # noqa: BLE001 — stats must always answer
        backend = "unavailable"
    return {
        "papers": counts,
        "recent_runs": runs,
        "storage_backend": backend,
        "server_time": time.time(),
    }

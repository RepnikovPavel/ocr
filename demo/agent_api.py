"""HTTP surface for agents, as opposed to the browser UI.

An agent wants three things the UI does not: one call that takes a PDF and
returns something addressable, one artifact instead of a directory of URLs, and a
way to see what else is queued before deciding to wait. Everything here is built
on the same queue, worker and artifacts the UI uses — this is a different door
into one service, not a second implementation.

Routes (all under /api/v1):
    POST /documents              submit a file; returns cached instantly on a repeat
    GET  /documents/{sha256}     status and metrics for the newest parse
    GET  /documents/{sha256}/bundle   the result as one ZIP: markdown + images/
    GET  /documents/search?q=    full-text search over everything parsed
    GET  /queue                  who is waiting, in order
    GET  /events                 SSE: queued / started / progress / done
    GET  /stats                  store size and cache reuse
"""

import asyncio
import io
import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from demo import db, docstore

router = APIRouter(prefix="/api/v1", tags=["agents"])

# Filled in by demo.server at import time: it owns the paths and the worker.
CONTEXT = {"jobs_dir": None, "worker": None, "prompt_modes": (), "default_mode": None}


def configure(jobs_dir, worker, prompt_modes, default_mode):
    CONTEXT.update(jobs_dir=Path(jobs_dir), worker=worker,
                   prompt_modes=tuple(prompt_modes), default_mode=default_mode)


def _require_mode(prompt_mode):
    mode = prompt_mode or CONTEXT["default_mode"]
    if mode not in CONTEXT["prompt_modes"]:
        raise HTTPException(400, f"prompt_mode must be one of {list(CONTEXT['prompt_modes'])}")
    return mode


# ---------------------------------------------------------------- submit

@router.post("/documents")
def submit_document(
    request: Request,
    file: UploadFile = File(...),
    prompt_mode: str = Form(None),
    pages: str = Form(None),
    agent: str = Form(None),
):
    """Queue a document, or return the cached parse of identical bytes.

    Deduplication is on the SHA-256 of the uploaded file together with the prompt
    mode and page selection, because those change the answer. Agents resubmit the
    same paper constantly; without this each repeat is another full parse.
    """
    import fitz
    from PIL import Image

    mode = _require_mode(prompt_mode)
    suffix = Path(file.filename or "upload.bin").suffix.lower()
    if suffix not in {".pdf", ".jpg", ".jpeg", ".png"}:
        raise HTTPException(400, f"unsupported file type: {suffix}")

    jobs_dir = CONTEXT["jobs_dir"]
    staging = jobs_dir / f"staging-{uuid.uuid4().hex[:12]}"
    staging.mkdir(parents=True)
    stored = staging / f"input{suffix}"
    try:
        with stored.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        sha256 = docstore.sha256_of(stored)
        kind = "pdf" if suffix == ".pdf" else "image"
        if kind == "pdf":
            try:
                with fitz.open(str(stored)) as doc:
                    num_pages = doc.page_count
            except Exception as error:
                raise HTTPException(400, f"broken pdf: {error}")
        else:
            try:
                with Image.open(stored) as image:
                    image.verify()
            except Exception as error:
                raise HTTPException(400, f"broken image: {error}")
            num_pages = 1

        page_list = _parse_pages(pages, num_pages)

        cached = docstore.find_result(sha256, mode, page_list)
        docstore.remember_document(sha256, file.filename or stored.name, kind,
                                   num_pages, stored.stat().st_size)
        if cached:
            # the bundle is rebuilt from the stored job, so nothing is re-run
            return {
                "sha256": sha256, "status": "cached", "prompt_mode": mode,
                "pages": page_list, "num_pages": num_pages,
                "task_id": cached["task_id"],
                "bundle_url": f"/api/v1/documents/{sha256}/bundle?prompt_mode={mode}",
            }

        job_id = db.create_job(request.state.sid, file.filename or stored.name,
                               kind, num_pages)
        job_dir = jobs_dir / job_id
        staging.rename(job_dir)
        staging = None

        params = {"agent": agent or "", "sha256": sha256, "temperature": None,
                  "custom_prompt": None, "bbox": None, "bbox_view_size": None,
                  "max_new_tokens": None}
        task_id = db.create_task(request.state.sid, job_id, mode, page_list, params)
        CONTEXT["worker"].notify_new_task()
        queue = db.list_active_tasks(limit=100)
        position = next((i for i, t in enumerate(queue) if t["id"] == task_id), 0)
        return {
            "sha256": sha256, "status": "queued", "prompt_mode": mode,
            "pages": page_list, "num_pages": num_pages,
            "task_id": task_id, "job_id": job_id,
            "queue_position": position, "queue_length": len(queue),
            "bundle_url": f"/api/v1/documents/{sha256}/bundle?prompt_mode={mode}",
        }
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _parse_pages(pages, num_pages):
    if not pages or pages.strip() in {"", "all"}:
        return list(range(num_pages))
    try:
        wanted = sorted({int(x) for x in pages.split(",") if x.strip() != ""})
    except ValueError:
        raise HTTPException(400, f"bad pages: {pages}")
    bad = [p for p in wanted if p < 0 or p >= num_pages]
    if bad:
        raise HTTPException(400, f"pages out of range (document has {num_pages}): {bad}")
    return wanted


# ---------------------------------------------------------------- status

@router.get("/documents/{sha256}")
def document_status(sha256: str, prompt_mode: str = None):
    document = docstore.get_document(sha256)
    if document is None:
        raise HTTPException(404, "unknown document")
    mode = prompt_mode or CONTEXT["default_mode"]
    task = _task_for(sha256, mode)
    # A result for a different page selection is not this request's answer:
    # reporting it as cached made an agent that asked for page 5 fetch the parse
    # of pages 0-2 and then fail on a bundle that had never been produced.
    cached = (docstore.find_result(sha256, mode, task["pages"]) if task
              else docstore.find_latest_result(sha256, mode))
    return {
        "sha256": sha256, "filename": document["filename"],
        "num_pages": document["num_pages"], "times_submitted": document["times_submitted"],
        "prompt_mode": mode,
        "status": task["status"] if task else ("done" if cached else "unknown"),
        "progress": task["progress"] if task else None,
        "cached": bool(cached),
        "generated_tokens": cached["generated_tokens"] if cached else None,
        "seconds": cached["seconds"] if cached else None,
    }


def _task_for(sha256, mode):
    for task in db.list_tasks(limit=200):
        if task["params"].get("sha256") == sha256 and task["prompt_mode"] == mode:
            return task
    return None


# ---------------------------------------------------------------- bundle

@router.get("/documents/{sha256}/bundle")
def document_bundle(sha256: str, prompt_mode: str = None):
    """The whole result as one ZIP: markdown plus the images it references.

    Agents get a single artifact they can unpack and read, with the relative
    image links in the markdown already resolving inside the archive — the same
    layout the UI renders, without walking a directory of URLs.
    """
    mode = prompt_mode or CONTEXT["default_mode"]
    task = _task_for(sha256, mode)
    result = (docstore.find_result(sha256, mode, task["pages"]) if task
              else docstore.find_latest_result(sha256, mode))
    if result is None:
        # the caller may mean an older page selection; serve the newest parse
        result = docstore.find_latest_result(sha256, mode)
    if result is None:
        raise HTTPException(404, "no parsed result for this document and prompt mode")

    job_dir = CONTEXT["jobs_dir"] / result["job_id"]
    out_dir = job_dir / "out"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("document.md", result["markdown"])
        archive.writestr("meta.json", json.dumps({
            "sha256": sha256,
            "prompt_mode": mode,
            "pages_done": result["pages_done"],
            "generated_tokens": result.get("generated_tokens"),
            "seconds": result.get("seconds"),
            "task_id": result["task_id"],
        }, ensure_ascii=False, indent=2))
        if out_dir.is_dir():
            for path in sorted(out_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(out_dir)
                # the markdown links pictures as images/<name>; keep that path so
                # the archive is self-contained when unpacked
                if relative.parts and relative.parts[0] == "images":
                    archive.write(path, str(relative))
                elif path.suffix == ".json":
                    archive.write(path, f"layout/{relative.name}")
    buffer.seek(0)
    filename = f"{sha256[:12]}-{mode}.zip"
    return StreamingResponse(
        buffer, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# ---------------------------------------------------------------- search

@router.get("/documents")
def search_documents(q: str = None, limit: int = 20):
    if not q:
        return {"results": docstore.recent(limit=limit), "query": None}
    try:
        return {"results": docstore.search(q, limit=limit), "query": q}
    except ValueError as error:
        raise HTTPException(400, str(error))


@router.get("/stats")
def store_stats():
    return {"store": docstore.stats(), "worker": CONTEXT["worker"].status()}


# ---------------------------------------------------------------- queue

def _queue_snapshot():
    queue = []
    for position, task in enumerate(db.list_active_tasks(limit=100)):
        queue.append({
            "task_id": task["id"],
            "agent": (task["params"] or {}).get("agent") or None,
            "sha256": (task["params"] or {}).get("sha256"),
            "prompt_mode": task["prompt_mode"],
            "pages": len(task["pages"]),
            "status": task["status"],
            "position": position,
            "progress": task["progress"],
        })
    return queue


@router.get("/queue")
def queue():
    """Who is waiting, in the order they will run.

    An agent reads this to decide whether to wait or come back later, and to see
    that another agent already asked for the same document.
    """
    return {"queue": _queue_snapshot(), "length": len(_queue_snapshot())}


@router.get("/events")
async def events(request: Request):
    """Server-Sent Events for the queue.

    Emits only on change, plus a keep-alive comment, so an idle agent holding the
    stream costs nothing and a busy one sees transitions without polling.
    """
    async def stream():
        previous = None
        last_beat = 0.0
        while True:
            if await request.is_disconnected():
                break
            snapshot = await asyncio.to_thread(_queue_snapshot)
            payload = json.dumps({"queue": snapshot, "at": time.time()}, ensure_ascii=False)
            fingerprint = json.dumps(
                [(t["task_id"], t["status"], (t["progress"] or {}).get("done")) for t in snapshot])
            now = time.time()
            if fingerprint != previous:
                previous = fingerprint
                last_beat = now
                yield f"event: queue\ndata: {payload}\n\n"
            elif now - last_beat > 15:
                last_beat = now
                yield ": keep-alive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # nginx would otherwise buffer the stream away
    })

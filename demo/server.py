#!/usr/bin/env python3
"""dots.mocr demo server: task queue + sessions (SQLite) + model control.

Two variants of the same app (env DEMO_VARIANT):
  mocr — document parsing demo for the dots.mocr checkpoint
  svg  — image->SVG demo for the dots.mocr-svg checkpoint

Environment:
  CKPTDIR         checkpoint snapshot dir (required for real inference)
  DEMO_STATE_DIR  writable state dir (uploads, artifacts, sqlite)
  DEMO_VARIANT    mocr | svg (default mocr)
  PORT            listen port (default 7860)
  DEMO_DEVICE     cuda:0 / cuda:1 / auto (default: GPU with most free memory)
  DEMO_DPI        PDF render dpi for inference (default 200, authors' choice)
  DEMO_AUTOSTART  load model at startup (default 1)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from demo import db
from demo.worker import PAGE_SECONDS_ESTIMATE, DemoWorker, default_temperature

VARIANT = os.environ.get("DEMO_VARIANT", "mocr")
CKPTDIR = os.environ.get("CKPTDIR", "/models")
STATE_DIR = Path(os.environ.get("DEMO_STATE_DIR", "/state"))
PORT = int(os.environ.get("PORT", "7860"))
INFER_DPI = int(os.environ.get("DEMO_DPI", "150"))
MAX_PIXELS = int(os.environ.get("DEMO_MAX_PIXELS", "2200000"))
# the sibling demo (the other model) — rendered as a link in the header
PEER_PORT = os.environ.get("DEMO_PEER_PORT")
PEER_TITLE = os.environ.get("DEMO_PEER_TITLE", "вторая модель")
VIEW_DPI = 144  # page images shown in the viewer
# per-variant cookie: both demos run on one host (different ports), and
# cookies are host-scoped — a shared name would rotate sessions on each switch
SESSION_COOKIE = f"demo_sid_{os.environ.get('DEMO_VARIANT', 'mocr')}"
MAX_UPLOAD_MB = 512

VARIANTS = {
    "mocr": {
        "title": "dots.mocr — document parsing",
        "prompt_modes": [
            "prompt_layout_all_en",
            "prompt_layout_only_en",
            "prompt_ocr",
            "prompt_grounding_ocr",
            "prompt_web_parsing",
            "prompt_scene_spotting",
            "prompt_general",
        ],
        "default_mode": "prompt_layout_all_en",
    },
    "svg": {
        "title": "dots.mocr-svg — image to SVG",
        "prompt_modes": ["prompt_image_to_svg"],
        "default_mode": "prompt_image_to_svg",
    },
}
if VARIANT not in VARIANTS:
    raise SystemExit(f"unknown DEMO_VARIANT: {VARIANT}")

JOBS_DIR = STATE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
db.init_db(STATE_DIR / "demo.db")

WORKER = DemoWorker(
    ckpt=CKPTDIR,
    jobs_dir=JOBS_DIR,
    device=os.environ.get("DEMO_DEVICE", "auto"),
    dpi=INFER_DPI,
    max_pixels=MAX_PIXELS,
    # lazy by default: the GPU stays free until a task arrives
    autostart=os.environ.get("DEMO_AUTOSTART", "0") == "1",
    keep_loaded=os.environ.get("DEMO_KEEP_LOADED", "0") == "1",
    idle_unload_seconds=int(os.environ.get("DEMO_IDLE_UNLOAD_S", "180")),
)

app = FastAPI(title=VARIANTS[VARIANT]["title"])


@app.on_event("startup")
def _start_worker():
    if not WORKER.is_alive():
        WORKER.start()


@app.middleware("http")
async def session_middleware(request: Request, call_next):
    # sqlite write must not run on the event loop
    sid = await run_in_threadpool(db.get_or_create_session, request.cookies.get(SESSION_COOKIE))
    request.state.sid = sid
    response = await call_next(request)
    if request.cookies.get(SESSION_COOKIE) != sid:
        response.set_cookie(SESSION_COOKIE, sid, max_age=365 * 24 * 3600, samesite="lax")
    return response


# ---------------------------------------------------------------- helpers

def _file_url(path):
    """Absolute artifact path -> /files URL (only inside JOBS_DIR)."""
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(JOBS_DIR.resolve())
    except ValueError:
        return None
    return f"/files/{rel.as_posix()}"


def _task_public(task):
    if task is None:
        return None
    out = dict(task)
    # other visitors' identifiers and prompts are not part of the public shape
    out.pop("session_id", None)
    out.pop("params", None)
    out["result"] = [
        {
            **{k: v for k, v in page.items() if not k.endswith("_path")},
            "urls": {
                k.replace("_path", ""): _file_url(v)
                for k, v in page.items() if k.endswith("_path") and _file_url(v)
            },
        }
        for page in task["result"]
    ]
    return out


def gpu_snapshot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, timeout=3,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue

        def num(value):
            try:
                return float(value)
            except ValueError:
                return None

        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "util_pct": num(parts[2]),
            "memory_used_mb": num(parts[3]),
            "memory_total_mb": num(parts[4]),
            "power_w": num(parts[5]),
        })
    return gpus


# ---------------------------------------------------------------- routes

@app.get("/healthz")
def healthz():
    return {"status": "ok", "variant": VARIANT, **WORKER.status()}


@app.get("/api/state")
def api_state(request: Request):
    import time as _time

    sid = request.state.sid
    config = VARIANTS[VARIANT]
    gpus = gpu_snapshot()
    # selectable inference devices: every visible GPU + auto (+ cpu for fallback)
    devices = ["auto"] + [f"cuda:{g['index']}" for g in gpus] + ["cpu"]
    return {
        "variant": VARIANT,
        "title": config["title"],
        "peer": {"port": PEER_PORT, "title": PEER_TITLE} if PEER_PORT else None,
        "server_time": _time.time(),
        "ckpt": CKPTDIR,
        "devices": devices,
        "prompt_modes": [
            {
                "mode": mode,
                "default_temperature": default_temperature(mode),
                "page_seconds_estimate": PAGE_SECONDS_ESTIMATE.get(mode, 45),
            }
            for mode in config["prompt_modes"]
        ],
        "default_mode": config["default_mode"],
        "worker": WORKER.status(),
        "gpus": gpus,
        "session": {"id": sid, "jobs": db.list_jobs(sid)},
        "tasks": [
            {**_task_public(task), "own": task["session_id"] == sid}
            for task in db.list_tasks(limit=30)
        ],
    }


@app.post("/api/upload")
def api_upload(request: Request, file: UploadFile = File(...)):
    # sync def: FastAPI runs it in the threadpool — copying a 512MB upload and
    # rendering every PDF page must not block the event loop for other clients
    suffix = Path(file.filename or "upload.bin").suffix.lower()
    if suffix not in {".pdf", ".jpg", ".jpeg", ".png"}:
        raise HTTPException(400, f"unsupported file type: {suffix}")
    if file.size and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file too large (> {MAX_UPLOAD_MB}MB)")

    import fitz

    kind = "pdf" if suffix == ".pdf" else "image"
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    dest = job_dir / f"input{suffix}"
    with dest.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    views = []
    if kind == "pdf":
        try:
            doc = fitz.open(str(dest))
        except Exception as error:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(400, f"broken pdf: {error}")
        num_pages = doc.page_count
        for index in range(num_pages):
            pix = doc[index].get_pixmap(dpi=VIEW_DPI)
            view_path = job_dir / f"view_{index:03d}.jpg"
            pix.save(str(view_path), jpg_quality=85)
            views.append({
                "page": index,
                "url": f"/files/{job_id}/view_{index:03d}.jpg",
                "width": pix.width,
                "height": pix.height,
            })
        doc.close()
    else:
        from PIL import Image

        try:
            with Image.open(dest) as img:
                img = img.convert("RGB")
                width, height = img.size
                view_path = job_dir / "view_000.jpg"
                img.save(view_path, "JPEG", quality=90)
        except Exception as error:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(400, f"broken image: {error}")
        num_pages = 1
        views.append({
            "page": 0,
            "url": f"/files/{job_id}/view_000.jpg",
            "width": width,
            "height": height,
        })

    # persist metadata; the directory is renamed to the db job id
    real_job_id = db.create_job(request.state.sid, file.filename or dest.name, kind, num_pages)
    new_dir = JOBS_DIR / real_job_id
    job_dir.rename(new_dir)
    for view in views:
        view["url"] = view["url"].replace(f"/files/{job_id}/", f"/files/{real_job_id}/")

    (new_dir / "views.json").write_text(json.dumps(views), encoding="utf-8")
    return {"job_id": real_job_id, "kind": kind, "num_pages": num_pages,
            "filename": file.filename, "views": views}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    views_path = JOBS_DIR / job_id / "views.json"
    views = json.loads(views_path.read_text(encoding="utf-8")) if views_path.exists() else []
    return {**job, "views": views}


@app.post("/api/tasks")
def api_create_task(
    request: Request,
    job_id: str = Form(...),
    prompt_mode: str = Form(None),
    pages: str = Form("0"),
    custom_prompt: str = Form(None),
    temperature: float = Form(None),
    max_new_tokens: int = Form(None),
    bbox: str = Form(None),
):
    config = VARIANTS[VARIANT]
    prompt_mode = prompt_mode or config["default_mode"]
    if prompt_mode not in config["prompt_modes"]:
        raise HTTPException(400, f"prompt mode {prompt_mode} is not available in this demo")
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")

    try:
        page_list = sorted({int(x) for x in pages.split(",") if x.strip() != ""})
    except ValueError:
        raise HTTPException(400, f"bad pages: {pages}")
    if not page_list:
        raise HTTPException(400, "no pages selected")
    bad = [p for p in page_list if p < 0 or p >= job["num_pages"]]
    if bad:
        raise HTTPException(400, f"pages out of range: {bad}")

    bbox_list = None
    bbox_view_size = None
    if bbox:
        try:
            bbox_list = [int(float(x)) for x in bbox.split(",")]
            assert len(bbox_list) == 4
        except (ValueError, AssertionError):
            raise HTTPException(400, f"bad bbox: {bbox}")
        # the UI drags on the viewer render; the worker rescales the bbox by
        # the ACTUAL inference render size of that page (fitz may silently
        # fall back to a lower dpi for oversized pages)
        views_path = JOBS_DIR / job_id / "views.json"
        if views_path.exists():
            views = json.loads(views_path.read_text(encoding="utf-8"))
            match = [v for v in views if v["page"] == page_list[0]]
            if match:
                bbox_view_size = [match[0]["width"], match[0]["height"]]
    if prompt_mode == "prompt_grounding_ocr":
        if bbox_list is None:
            raise HTTPException(400, "prompt_grounding_ocr requires a bbox (drag on the page)")
        if len(page_list) != 1:
            raise HTTPException(400, "grounding OCR works on exactly one page")

    params = {
        "temperature": temperature,
        "custom_prompt": custom_prompt,
        "bbox": bbox_list,
        "bbox_view_size": bbox_view_size,
        "max_new_tokens": max_new_tokens,
    }
    task_id = db.create_task(request.state.sid, job_id, prompt_mode, page_list, params)
    WORKER.notify_new_task()
    return {"task_id": task_id, "status": "queued"}


@app.get("/api/tasks/{task_id}")
def api_get_task(task_id: str):
    task = db.get_task(task_id)
    if task is None:
        raise HTTPException(404, "task not found")
    return _task_public(task)


@app.post("/api/tasks/{task_id}/cancel")
def api_cancel_task(task_id: str):
    status = WORKER.cancel_task(task_id)
    if status is None:
        raise HTTPException(404, "task not found")
    return {"task_id": task_id, "status": status}


@app.post("/api/model/start")
def api_model_start():
    WORKER.request_start()
    return WORKER.status()


@app.post("/api/model/stop")
def api_model_stop():
    WORKER.request_stop()
    return WORKER.status()


@app.post("/api/model/keep_loaded")
def api_model_keep_loaded(value: bool = Form(...)):
    """do_not_unload_model: keep the model on the GPU between tasks."""
    WORKER.set_keep_loaded(value)
    return WORKER.status()


@app.post("/api/model/device")
def api_model_device(device: str = Form(...)):
    """Choose which GPU (or auto/cpu) runs inference; reloads the model there."""
    allowed = {"auto", "cpu"} | {f"cuda:{g['index']}" for g in gpu_snapshot()}
    if device not in allowed:
        raise HTTPException(400, f"device must be one of {sorted(allowed)}")
    WORKER.set_device(device)
    return WORKER.status()


@app.get("/api/raw")
def api_raw(path: str):
    """Return a text artifact (md / json / svg) by its /files URL."""
    if not path.startswith("/files/"):
        raise HTTPException(400, "path must be a /files URL")
    target = (JOBS_DIR / path[len("/files/"):]).resolve()
    if not str(target).startswith(str(JOBS_DIR.resolve())):
        raise HTTPException(400, "path escapes the state dir")
    if not target.is_file():
        raise HTTPException(404, "not found")
    if target.stat().st_size > 20 * 1024 * 1024:
        raise HTTPException(413, "artifact too large")
    return JSONResponse({"path": path, "content": target.read_text(encoding="utf-8", errors="replace")})


app.mount("/files", StaticFiles(directory=str(JOBS_DIR)), name="files")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/")
def index():
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    html = html.replace("__VARIANT__", VARIANT)
    html = html.replace("__TITLE__", VARIANTS[VARIANT]["title"])
    return HTMLResponse(html)


if __name__ == "__main__":
    print(f"Starting {VARIANTS[VARIANT]['title']} on 0.0.0.0:{PORT} (ckpt={CKPTDIR})")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

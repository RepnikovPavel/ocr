#!/usr/bin/env python3
"""
FastAPI web demo for dots.mocr (OCR / document parsing / SVG).

Matches the style and deploy pattern of qwen3_l / qwen3_vl demos.

- Lazy model load on first request (DotsMOCRParser)
- Upload image or PDF
- Choose prompt mode + basic params
- Results saved under DEMO_STATE_DIR (or /state)
- Returns structured results + served files (layout.jpg, .md, .json, .svg, ...)
- Simple self-contained HTML/JS UI
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

try:
    import fitz  # PyMuPDF for PDF page thumbs
except ImportError:
    fitz = None

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Ensure package is importable inside container
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lazy imports to avoid pulling torch/transformers/fitz on every health check.
# Actual heavy modules are imported only on first real inference request.
def _get_parser_cls():
    from dots_mocr.cli import DotsMOCRParser  # type: ignore
    return DotsMOCRParser

def _get_prompts():
    from dots_mocr.utils.prompts import dict_promptmode_to_prompt  # type: ignore
    return dict_promptmode_to_prompt

app = FastAPI(title="dots.mocr Demo")

STATE_DIR = Path(os.environ.get("DEMO_STATE_DIR", "/state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = STATE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

CKPTDIR = os.environ.get("CKPTDIR", "/models")
PORT = int(os.environ.get("PORT", os.environ.get("DOTS_MOCR_WEB_PORT", "7860")))

# Global lazy parser
PARSER: Any | None = None
PARSER_LOCK = __import__("threading").Lock()

PROMPT_CHOICES: list[str] = []
DEFAULT_PROMPT = "prompt_layout_all_en"

MAX_UPLOAD_MB = 512  # support large multi-page PDFs (100+ pages at reasonable dpi)


def get_parser():
    global PARSER
    if PARSER is not None:
        return PARSER
    DotsMOCRParser = _get_parser_cls()
    with PARSER_LOCK:
        if PARSER is None:
            print(f"[demo] Lazy loading dots.mocr from {CKPTDIR} (multi-GPU auto)...")
            t0 = time.time()
            num_threads = int(os.environ.get("DEMO_NUM_THREADS", "4"))
            print(f"[demo] using num_thread={num_threads} for PDF parallelism (good for 100+ page docs)")

            # Load the model on the GPU with the most free memory.
            # This gives maximum headroom for the page inference and avoids OOM when other demos are running.
            # (Full layer split "balanced"/"auto" for this model tends to put the bulk on one card anyway.)
            try:
                import subprocess
                out = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                    text=True, timeout=2
                )
                frees = [int(x.strip()) for x in out.strip().split('\n')]
                best_idx = frees.index(max(frees))
                dev = f"cuda:{best_idx}"
                print(f"[demo] loading model on {dev} (most free: {max(frees)} MiB) for headroom")
            except Exception:
                dev = "cuda"

            PARSER = DotsMOCRParser(
                ckpt=CKPTDIR,
                temperature=0.1,
                top_p=1.0,
                max_completion_tokens=16384,
                num_thread=num_threads,
                dpi=200,
                output_dir=str(JOBS_DIR),
                attn_implementation="sdpa",
                device=dev,
                dtype="bfloat16",
                max_pixels=2_500_000,  # lower than default to reduce VRAM for vision features on 16GB GPUs
            )
            print(f"[demo] Model loaded in {time.time() - t0:.1f}s (device_map=auto for 32GB parallel)")
    return PARSER


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:80]


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload.bin").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".pdf"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / f"input{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest, job_dir, job_id


def parse_pages_selection(sel: str) -> list[int] | None:
    if not sel or sel.lower() in ("all", ""):
        return None
    res = set()
    for token in sel.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            try:
                a, b = map(int, token.split("-"))
                res.update(range(a-1, b))
            except:
                pass
        else:
            try:
                res.add(int(token) - 1)
            except:
                pass
    return sorted(res) if res else None


@app.get("/api/gpu")
def api_gpu():
    """GPU resource monitor (used by UI)"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=3
        )
        gpus = []
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_mb": int(parts[2]),
                    "used_mb": int(parts[3]),
                    "free_mb": int(parts[4]),
                })
        return {"gpus": gpus, "note": "GPU0 often busy with other demos; GPU1 usually free for parallel"}
    except Exception as e:
        return {"gpus": [], "error": str(e)}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": PARSER is not None}


@app.get("/api/status")
def api_status():
    parser = None
    loaded = False
    try:
        parser = get_parser()
        loaded = True
    except Exception as e:
        return {"loaded": False, "error": str(e), "ckpt": CKPTDIR}
    # populate prompts on first status if not yet
    global PROMPT_CHOICES
    if not PROMPT_CHOICES:
        try:
            PROMPT_CHOICES = list(_get_prompts().keys())
        except Exception:
            PROMPT_CHOICES = ["prompt_layout_all_en", "prompt_layout_only_en", "prompt_image_to_svg", "prompt_ocr", "prompt_general"]
    return {
        "loaded": loaded,
        "ckpt": CKPTDIR,
        "device": getattr(parser, "device", "unknown"),
        "state_dir": str(STATE_DIR),
        "prompts": PROMPT_CHOICES,
    }


@app.post("/api/prepare")
async def api_prepare(file: UploadFile = File(...)):
    """Prepare file: save, for PDF generate page thumbs + return page info for UI selector."""
    if file.size and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (> {MAX_UPLOAD_MB}MB)")

    input_path, job_dir, job_id = _save_upload(file)
    suffix = input_path.suffix.lower()
    is_pdf = suffix == ".pdf"
    num_pages = 1
    thumb_urls = []

    view_urls = []
    if is_pdf and fitz is not None:
        try:
            doc = fitz.open(str(input_path))
            num_pages = len(doc)
            for i in range(num_pages):
                page = doc[i]
                # small thumbs for list
                pix = page.get_pixmap(dpi=60)
                tp = job_dir / f"thumb_{i:03d}.jpg"
                pix.save(str(tp))
                thumb_urls.append(f"/files/{job_id}/thumb_{i:03d}.jpg")

                # larger view images for proper PDF viewer on left
                pix = page.get_pixmap(dpi=120)
                vp = job_dir / f"view_{i:03d}.jpg"
                pix.save(str(vp))
                view_urls.append(f"/files/{job_id}/view_{i:03d}.jpg")
            doc.close()
        except Exception as e:
            print(f"thumb/view error: {e}")
    else:
        # image as single "page"
        try:
            from PIL import Image as PILImage
            im = PILImage.open(input_path)
            # thumb
            im_thumb = im.copy()
            im_thumb.thumbnail((300, 400))
            tp = job_dir / "thumb_000.jpg"
            im_thumb.save(tp, "JPEG")
            thumb_urls = [f"/files/{job_id}/thumb_000.jpg"]

            # view
            im_view = im.copy()
            im_view.thumbnail((900, 1200))
            vp = job_dir / "view_000.jpg"
            im_view.save(vp, "JPEG")
            view_urls = [f"/files/{job_id}/view_000.jpg"]
        except Exception:
            pass

    return {
        "job_id": job_id,
        "filename": file.filename,
        "is_pdf": is_pdf,
        "num_pages": num_pages,
        "thumb_urls": thumb_urls,
        "view_urls": view_urls,
    }


@app.post("/api/parse")
async def api_parse(
    file: UploadFile | None = File(None),
    job_id: str = Form(None),
    pages: str = Form("all"),
    prompt: str = Form(DEFAULT_PROMPT),
    temperature: float = Form(0.1),
    max_tokens: int = Form(16384),
    dpi: int = Form(200),
    custom_prompt: str | None = Form(None),
):
    global PROMPT_CHOICES
    if not PROMPT_CHOICES:
        try:
            PROMPT_CHOICES = list(_get_prompts().keys())
        except Exception:
            PROMPT_CHOICES = ["prompt_layout_all_en", "prompt_layout_only_en", "prompt_image_to_svg", "prompt_ocr", "prompt_general"]
    if prompt not in PROMPT_CHOICES and prompt != "prompt_general":
        raise HTTPException(400, f"Unknown prompt: {prompt}")

    # Support either direct file or pre-prepared job_id (for page selection + no reupload)
    if job_id:
        job_dir = JOBS_DIR / job_id
        candidates = list(job_dir.glob("input.*"))
        if not candidates:
            raise HTTPException(404, "job not found or no input")
        input_path = candidates[0]
        out_dir = job_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if file is None or not file.filename:
            raise HTTPException(400, "file or job_id required")
        if file.size and file.size > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File too large (> {MAX_UPLOAD_MB}MB)")
        input_path, job_dir, job_id = _save_upload(file)
        out_dir = job_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

    pages_list = parse_pages_selection(pages)

    parser = get_parser()

    # Override some runtime params (note: parser stores them)
    parser.temperature = float(temperature)
    parser.max_completion_tokens = int(max_tokens)
    parser.dpi = int(dpi)

    t0 = time.time()
    try:
        if pages_list and len(pages_list) > 1:
            # Process pages one-by-one (or very small batches) to stay within 16GB GPU limits
            # This allows 100-page PDFs to complete without OOM, at the cost of longer wall time.
            import torch
            batch_size = 1
            all_results = []
            for i in range(0, len(pages_list), batch_size):
                batch = pages_list[i : i + batch_size]
                batch_res = parser.parse_file(
                    str(input_path),
                    output_dir=str(out_dir),
                    prompt_mode=prompt,
                    custom_prompt=custom_prompt if prompt == "prompt_general" else None,
                    pages=batch,
                )
                all_results.extend(batch_res)
                torch.cuda.empty_cache()
            results = all_results
        else:
            results = parser.parse_file(
                str(input_path),
                output_dir=str(out_dir),
                prompt_mode=prompt,
                custom_prompt=custom_prompt if prompt == "prompt_general" else None,
                pages=pages_list,
            )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}") from e

    elapsed = time.time() - t0
    num_pages = max(1, len(results))
    sec_per_page = round(elapsed / num_pages, 2)

    # Build response with file URLs relative to /files/<job_id>/...
    # We will mount /files -> JOBS_DIR
    file_base = f"/files/{job_id}/out"

    enriched = []
    for r in results:
        item = dict(r)
        item["elapsed_sec"] = sec_per_page
        # Add downloadable URLs
        for key in ("layout_image_path", "md_content_path", "layout_info_path", "svg_content_path"):
            p = item.get(key)
            if p:
                rel = Path(p).relative_to(out_dir) if Path(p).is_relative_to(out_dir) else Path(p).name
                item[key.replace("_path", "_url")] = f"{file_base}/{rel}"
        enriched.append(item)

    # Write a summary jsonl already done by parser, also write a job meta
    meta = {
        "job_id": job_id,
        "filename": file.filename,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "dpi": dpi,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_page": sec_per_page,
        "num_pages": len(enriched),
        "results": enriched,
    }
    (job_dir / "job.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return JSONResponse({
        "job_id": job_id,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_page": sec_per_page,
        "num_pages": len(enriched),
        "results": enriched,
        "download_base": file_base,
    })


# Serve generated artifacts
app.mount("/files", StaticFiles(directory=str(JOBS_DIR)), name="files")


INDEX_HTML = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>dots.mocr — веб-демо</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, Inter, sans-serif; }
    body { margin: 0; padding: 12px; background: #0f1117; color: #e6e8f0; }
    .container { display: flex; gap: 12px; max-width: 1400px; margin: 0 auto; }
    .left { flex: 0 0 38%; min-width: 320px; }
    .right { flex: 1; min-width: 400px; }
    .panel { background: #161a24; border: 1px solid #2a3040; border-radius: 12px; padding: 14px; margin-bottom: 12px; }
    h1 { margin: 0 0 4px; font-size: 1.4em; }
    .muted { color: #9aa3b8; font-size: 0.85em; }
    .gpu-bar { height: 8px; background: #333; border-radius: 4px; margin: 4px 0; position: relative; }
    .gpu-fill { height: 100%; background: #5c78ff; border-radius: 4px; }
    .dropzone { border: 2px dashed #5c78ff; border-radius: 12px; padding: 30px; text-align: center; cursor: pointer; background: #10141d; margin-bottom: 8px; }
    .dropzone.drag { background: #1a2030; }
    .pages-list { max-height: 380px; overflow: auto; border: 1px solid #2a3040; border-radius: 8px; padding: 6px; background: #10141d; }
    .page-item { display: flex; align-items: center; gap: 8px; padding: 4px; margin: 2px 0; }
    .page-item img { width: 120px; height: auto; border: 1px solid #2a3040; border-radius: 4px; } /* larger for better document viewing on left */
    .result-card { border: 1px solid #2a3040; border-radius: 10px; padding: 10px; margin: 8px 0; background: #11151f; }
    .result-card h4 { margin: 0 0 6px; }
    img.thumb { max-width: 100%; border-radius: 4px; }
    pre, .md { background: #0a0d14; padding: 8px; border-radius: 4px; white-space: pre-wrap; font-size: 0.9em; max-height: 320px; overflow: auto; }
    button { background: #5c78ff; color: white; border: 0; padding: 8px 14px; border-radius: 8px; cursor: pointer; font-weight: 600; }
    button.secondary { background: #343b4d; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; }
    .status-bar { font-size: 0.8em; padding: 4px 8px; background: #10141d; border-radius: 6px; }
    .loading { opacity: 0.6; pointer-events: none; }
    .gpu { font-size: 0.8em; }
  </style>
</head>
<body>
  <div style="max-width:1400px;margin:0 auto 8px;">
    <h1>dots.mocr Demo <span class="muted">— multi-GPU (32GB) • page-by-page • drag & drop</span></h1>
    <div id="gpu-panel" class="panel" style="margin:0 0 8px;padding:8px 12px;">
      <div class="muted">GPU Resources (обновляется автоматически)</div>
      <div id="gpu-content">Загрузка GPU...</div>
    </div>
    <div id="status" class="status-bar">Загрузка статуса…</div>
  </div>

  <div class="container">
    <!-- LEFT: workspace / drop / pages -->
    <div class="left">
      <div class="panel">
        <div id="dropzone" class="dropzone">
          <strong>Перетащите PDF или изображение сюда</strong><br>
          <small>или кликните для выбора файла</small>
          <input id="file-input" type="file" accept="image/*,.pdf" style="display:none">
        </div>

        <div id="file-info" class="muted" style="margin:6px 0;"></div>

        <div style="margin:8px 0;">
          <label>Выберите страницы для распознавания (1-based, пример: 1-5,10,15-20)</label>
          <input id="pages-input" type="text" value="all" style="width:100%;padding:6px;background:#10141d;border:1px solid #3a4255;color:#e6e8f0;border-radius:6px;">
        </div>

        <!-- Proper PDF viewer on left -->
        <div id="pdf-viewer" style="border:1px solid #2a3040; border-radius:8px; padding:8px; background:#10141d; margin:8px 0; min-height:280px; display:flex; align-items:center; justify-content:center;">
          <img id="current-page-img" style="max-width:100%; max-height:260px; display:none; border:1px solid #2a3040;" />
          <span id="viewer-placeholder" class="muted">После загрузки файла здесь будет крупный просмотр выбранной страницы</span>
        </div>

        <div id="pages-list" class="pages-list" style="display:none;"></div>

        <div class="row" style="margin-top:8px;">
          <label>Prompt
            <select id="prompt" style="width:100%;"></select>
          </label>
        </div>

        <div class="row" style="margin-top:6px;">
          <label>Temp <input id="temperature" type="number" value="0.1" step="0.05" min="0" max="2" style="width:70px;"></label>
          <label>Max tok <input id="max_tokens" type="number" value="16384" style="width:90px;"></label>
          <label>DPI <input id="dpi" type="number" value="200" style="width:70px;"></label>
        </div>

        <div style="margin-top:10px;">
          <button id="parse-btn" disabled>Распознать выбранные страницы</button>
          <button id="clear-btn" class="secondary">Очистить</button>
        </div>
        <div id="error" style="color:#ff8e98;margin-top:6px;font-size:0.85em;"></div>
      </div>
    </div>

    <!-- RIGHT: results -->
    <div class="right">
      <div class="panel">
        <div id="results" style="min-height:200px;">
          <span class="muted">Здесь появятся результаты парсинга (слева выбирайте страницы → жмите «Распознать»).</span>
        </div>
      </div>
    </div>
  </div>

<script>
let currentFile = null;
let currentJobId = null;
let currentThumbs = [];
let currentNumPages = 1;
let prompts = [];

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const pagesList = document.getElementById('pages-list');
const pagesInput = document.getElementById('pages-input');
const promptSel = document.getElementById('prompt');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const errorEl = document.getElementById('error');
const parseBtn = document.getElementById('parse-btn');
const clearBtn = document.getElementById('clear-btn');
const gpuContent = document.getElementById('gpu-content');

function updateGPU() {
  fetch('/api/gpu').then(r => r.json()).then(d => {
    if (!d.gpus || !d.gpus.length) {
      gpuContent.textContent = 'GPU info unavailable';
      return;
    }
    let html = '';
    d.gpus.forEach(g => {
      const pct = Math.round((g.used_mb / g.total_mb) * 100);
      html += `<div class="gpu">GPU${g.index}: ${g.used_mb} / ${g.total_mb} MB used (${pct}%) — free ${g.free_mb}MB</div>`;
      html += `<div class="gpu-bar"><div class="gpu-fill" style="width:${pct}%"></div></div>`;
    });
    gpuContent.innerHTML = html + (d.note ? `<div class="muted" style="font-size:0.75em;margin-top:4px;">${d.note}</div>` : '');
  }).catch(() => { gpuContent.textContent = 'GPU monitor error'; });
}
setInterval(updateGPU, 5000);
updateGPU();

fetch('/api/status').then(r => r.json()).then(s => {
  statusEl.textContent = s.loaded ? `Модель загружена • ${s.device || 'auto (parallel)'} • ckpt: ${s.ckpt}` : `Модель не загружена: ${s.error || ''}`;
  prompts = s.prompts || [];
  promptSel.innerHTML = '';
  prompts.forEach(p => {
    const o = document.createElement('option');
    o.value = p; o.textContent = p;
    if (p === 'prompt_layout_all_en') o.selected = true;
    promptSel.appendChild(o);
  });
}).catch(e => statusEl.textContent = 'Статус: ' + e);

let currentViewUrls = [];
let currentPageIndex = 0; // 0-based

function showPageInViewer(idx) {
  currentPageIndex = idx;
  const img = document.getElementById('current-page-img');
  const placeholder = document.getElementById('viewer-placeholder');
  const viewUrl = currentViewUrls[idx];
  if (viewUrl) {
    img.src = viewUrl;
    img.style.display = 'block';
    placeholder.style.display = 'none';
  }
}

function renderPageList(thumbs, views, num) {
  currentViewUrls = views || thumbs || [];
  pagesList.innerHTML = '';
  pagesList.style.display = 'block';

  for (let i = 0; i < num; i++) {
    const li = document.createElement('div');
    li.className = 'page-item';
    const checked = 'checked';
    const displayImg = (views && views[i]) || thumbs[i] || '';  // use larger view image for better visibility of page content
    li.innerHTML = `
      <input type="checkbox" value="${i+1}" ${checked}>
      <img src="${displayImg}" alt="p${i+1}" style="cursor:pointer;">
      <span>Page ${i+1}</span>
    `;
    // click on thumb or span to view
    const imgEl = li.querySelector('img');
    const spanEl = li.querySelector('span');
    const clickHandler = () => showPageInViewer(i);
    if (imgEl) imgEl.onclick = clickHandler;
    if (spanEl) spanEl.onclick = clickHandler;

    pagesList.appendChild(li);
  }

  // show first page by default in viewer
  if (num > 0) {
    setTimeout(() => showPageInViewer(0), 50);
  }
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
  currentFile = file;
  fileInfo.textContent = `${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`;
  errorEl.textContent = '';
  resultsEl.innerHTML = '<span class="muted">Подготовка (генерация превью страниц)...</span>';

  const fd = new FormData();
  fd.append('file', file);
  try {
    const resp = await fetch('/api/prepare', {method:'POST', body: fd});
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    currentJobId = data.job_id;
    currentNumPages = data.num_pages || 1;
    currentThumbs = data.thumb_urls || [];
    const views = data.view_urls || data.thumb_urls || [];
    renderPageList(currentThumbs, views, currentNumPages);
    parseBtn.disabled = false;
    resultsEl.innerHTML = `<span class="muted">Готово. Выберите страницы слева и нажмите «Распознать».</span>`;
  } catch (e) {
    errorEl.textContent = 'Prepare error: ' + e;
    parseBtn.disabled = true;
  }
}

parseBtn.addEventListener('click', async () => {
  if (!currentJobId) return;
  errorEl.textContent = '';
  resultsEl.innerHTML = '<div class="muted">Выполняется распознавание выбранных страниц…</div>';
  parseBtn.disabled = true;

  // collect selected
  const checks = pagesList.querySelectorAll('input[type=checkbox]');
  const selected = [];
  checks.forEach(c => { if (c.checked) selected.push(c.value); });
  const pagesStr = selected.length ? selected.join(',') : 'all';

  const fd = new FormData();
  fd.append('job_id', currentJobId);
  fd.append('pages', pagesStr);
  fd.append('prompt', promptSel.value);
  fd.append('temperature', document.getElementById('temperature').value);
  fd.append('max_tokens', document.getElementById('max_tokens').value);
  fd.append('dpi', document.getElementById('dpi').value);

  try {
    const res = await fetch('/api/parse', {method: 'POST', body: fd});
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t);
    }
    const data = await res.json();
    renderResults(data);
  } catch (e) {
    errorEl.textContent = 'Ошибка: ' + e;
    resultsEl.innerHTML = '';
  } finally {
    parseBtn.disabled = false;
  }
});

clearBtn.onclick = () => {
  resultsEl.innerHTML = '<span class="muted">Результаты очищены.</span>';
  pagesList.style.display = 'none';
  fileInfo.textContent = '';
  currentJobId = null;
  currentFile = null;
  errorEl.textContent = '';
};

function renderResults(data) {
  const spp = data.sec_per_page || (data.elapsed_sec / Math.max(1, data.num_pages || 1));
  let html = `<div class="muted">job ${data.job_id} • ${data.num_pages} стр. • ${data.elapsed_sec}s • <b>~${spp}s/стр.</b></div>`;
  (data.results || []).forEach((r, i) => {
    const pageNo = r.page_no ?? (i+1);
    html += `<div class="result-card">`;
    html += `<h4>Страница ${pageNo} <button class="secondary" style="font-size:0.8em;padding:2px 6px;margin-left:8px;" onclick="showPageInViewer(${pageNo-1})">показать в просмотрщике</button></h4>`;
    if (r.layout_image_url) html += `<img class="thumb" src="${r.layout_image_url}" alt="layout">`;
    if (r.md_content_url) {
      html += `<div><strong>Результат</strong> <a href="${r.md_content_url}" target="_blank">скачать .md</a></div>`;
      html += `<pre class="md" id="md-${i}">Загрузка...</pre>`;
    }
    if (r.layout_info_url) html += `<div><a href="${r.layout_info_url}" target="_blank">layout.json</a></div>`;
    html += `</div>`;
  });
  resultsEl.innerHTML = html;

  // load previews
  (data.results || []).forEach((r, i) => {
    if (r.md_content_url) {
      fetch(r.md_content_url).then(t => t.text()).then(txt => {
        const p = document.getElementById(`md-${i}`);
        if (p) p.textContent = txt.substring(0, 2500) + (txt.length > 2500 ? '\n…' : '');
      });
    }
  });
}

// initial status already fetched above
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTMLResponse(INDEX_HTML)


if __name__ == "__main__":
    print(f"Starting dots.mocr demo on 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

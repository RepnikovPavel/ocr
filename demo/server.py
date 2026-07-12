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
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

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
            print(f"[demo] Lazy loading dots.mocr from {CKPTDIR} ...")
            t0 = time.time()
            num_threads = int(os.environ.get("DEMO_NUM_THREADS", "4"))
            print(f"[demo] using num_thread={num_threads} for PDF parallelism (good for 100+ page docs)")
            PARSER = DotsMOCRParser(
                ckpt=CKPTDIR,
                temperature=0.1,
                top_p=1.0,
                max_completion_tokens=16384,
                num_thread=num_threads,
                dpi=200,
                output_dir=str(JOBS_DIR),  # will override per call
                attn_implementation="sdpa",
                device="auto",
                dtype="auto",
            )
            print(f"[demo] Model loaded in {time.time() - t0:.1f}s")
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


@app.post("/api/parse")
async def api_parse(
    file: UploadFile = File(...),
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

    if file.size and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (> {MAX_UPLOAD_MB}MB)")

    input_path, job_dir, job_id = _save_upload(file)
    out_dir = job_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = get_parser()

    # Override some runtime params (note: parser stores them)
    parser.temperature = float(temperature)
    parser.max_completion_tokens = int(max_tokens)
    parser.dpi = int(dpi)

    t0 = time.time()
    try:
        results = parser.parse_file(
            str(input_path),
            output_dir=str(out_dir),
            prompt_mode=prompt,
            custom_prompt=custom_prompt if prompt == "prompt_general" else None,
        )
    except Exception as e:
        # Cleanup on error? keep input for debug
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
    body { max-width: 1100px; margin: 0 auto; padding: 20px; background: #0f1117; color: #e6e8f0; }
    h1 { margin: 0 0 4px; } .muted { color: #9aa3b8; font-size: 0.9em; }
    .panel { background: #161a24; border: 1px solid #2a3040; border-radius: 12px; padding: 16px; margin: 14px 0; }
    textarea, input, select, button { font: inherit; }
    input[type=file], select, input[type=number] { width: 100%; box-sizing: border-box; background: #10141d; color: inherit; border: 1px solid #3a4255; border-radius: 8px; padding: 8px; }
    textarea { width: 100%; min-height: 60px; background: #10141d; border: 1px solid #3a4255; border-radius: 8px; padding: 8px; color: inherit; }
    button { background: #5c78ff; color: white; border: 0; padding: 10px 18px; border-radius: 8px; cursor: pointer; font-weight: 600; }
    button.secondary { background: #343b4d; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .row > * { flex: 1 1 180px; min-width: 160px; }
    .result { margin-top: 18px; }
    .page { border: 1px solid #2a3040; border-radius: 10px; padding: 12px; margin: 10px 0; background: #11151f; }
    .page h3 { margin: 0 0 8px; font-size: 1em; }
    img { max-width: 100%; border-radius: 6px; border: 1px solid #2a3040; }
    pre, .md { background: #0a0d14; padding: 10px; border-radius: 6px; overflow: auto; white-space: pre-wrap; }
    .links a { margin-right: 12px; color: #8ab4ff; }
    #status { font-size: 0.85em; }
    .loading { opacity: .6; pointer-events: none; }
  </style>
</head>
<body>
  <h1>dots.mocr Demo</h1>
  <div class="muted">Document / layout / OCR / SVG parsing • FP8 on server • same pattern as qwen3_vl</div>

  <div id="status" class="muted">Загрузка статуса…</div>

  <div class="panel">
    <form id="form">
      <div class="row">
        <label>Файл (изображение или PDF)
          <input id="file" name="file" type="file" accept="image/*,.pdf" required>
        </label>
        <label>Prompt mode
          <select id="prompt" name="prompt"></select>
        </label>
      </div>

      <div class="row" style="margin-top:10px">
        <label>Temperature <input id="temperature" name="temperature" type="number" value="0.1" step="0.05" min="0" max="2"></label>
        <label>Max tokens <input id="max_tokens" name="max_tokens" type="number" value="16384" min="256" max="32768"></label>
        <label>DPI (PDF) <input id="dpi" name="dpi" type="number" value="200" min="72" max="400"></label>
      </div>

      <div style="margin-top:10px">
        <label>Custom prompt (только для prompt_general)
          <textarea id="custom_prompt" name="custom_prompt" placeholder="Опиши содержимое документа подробно..."></textarea>
        </label>
      </div>

      <div style="margin-top:14px">
        <button id="submit" type="submit">Распознать</button>
        <button id="clear" type="button" class="secondary">Очистить</button>
      </div>
    </form>
    <div id="error" style="color:#ff8e98;margin-top:8px"></div>
  </div>

  <div id="results" class="result"></div>

  <div class="panel muted">
    Результаты сохраняются в state dir. Поддержка PDF до 100+ страниц (параллельная обработка страниц).<br>
    Для деплоя используйте <code>docker/run_demo.sh</code>.<br>
    Пример: <code>./docker/run_demo.sh /path/to/dots.mocr /path/to/ocr_demo_state 8002</code><br>
    Бенчмарк: после обработки показывается <b>sec_per_page</b>.
  </div>

<script>
const promptSelect = document.getElementById('prompt');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const errorEl = document.getElementById('error');
const form = document.getElementById('form');

let prompts = [];

fetch('/api/status').then(r => r.json()).then(s => {
  statusEl.textContent = s.loaded
    ? `Модель загружена • ${s.device} • ckpt: ${s.ckpt}`
    : `Модель не загружена (${s.error || ''})`;
  prompts = s.prompts || [];
  promptSelect.innerHTML = '';
  for (const p of prompts) {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p;
    if (p === 'prompt_layout_all_en') opt.selected = true;
    promptSelect.appendChild(opt);
  }
}).catch(e => {
  statusEl.textContent = 'Статус недоступен: ' + e;
});

form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  errorEl.textContent = '';
  resultsEl.innerHTML = '<div class="muted">Выполняется распознавание… (может занять 10-60с)</div>';
  const submitBtn = document.getElementById('submit');
  submitBtn.disabled = true;
  form.classList.add('loading');

  const fd = new FormData(form);
  // ensure file from the input
  const fileInput = document.getElementById('file');
  fd.set('file', fileInput.files[0]);

  try {
    const res = await fetch('/api/parse', { method: 'POST', body: fd });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || res.status);
    }
    const data = await res.json();
    renderResults(data);
  } catch (e) {
    errorEl.textContent = 'Ошибка: ' + e;
    resultsEl.innerHTML = '';
  } finally {
    submitBtn.disabled = false;
    form.classList.remove('loading');
  }
});

document.getElementById('clear').onclick = () => {
  resultsEl.innerHTML = '';
  errorEl.textContent = '';
};

function renderResults(data) {
  const spp = data.sec_per_page || (data.elapsed_sec / Math.max(1, data.num_pages || 1));
  resultsEl.innerHTML = `<div class="muted">job ${data.job_id} • ${data.num_pages} стр. • всего ${data.elapsed_sec}s • <b>~${spp}s / стр.</b></div>`;
  for (const [i, r] of data.results.entries()) {
    const div = document.createElement('div');
    div.className = 'page';
    let html = `<h3>Страница ${r.page_no ?? (i+1)}</h3>`;
    if (r.layout_image_url) {
      html += `<img src="${r.layout_image_url}" alt="layout">`;
    }
    if (r.md_content_url) {
      html += `<div style="margin-top:8px"><strong>Markdown / текст</strong><br><a href="${r.md_content_url}" target="_blank">скачать .md</a></div>`;
      // Try to fetch and show preview
      html += `<pre class="md" id="mdprev-${i}">Загрузка превью…</pre>`;
    }
    if (r.svg_content_url) {
      html += `<div><a href="${r.svg_content_url}" target="_blank">скачать .svg</a></div>`;
    }
    if (r.layout_info_url) {
      html += `<div class="links"><a href="${r.layout_info_url}" target="_blank">layout.json</a></div>`;
    }
    div.innerHTML = html;
    resultsEl.appendChild(div);

    // fetch md preview
    if (r.md_content_url) {
      fetch(r.md_content_url).then(resp => resp.text()).then(txt => {
        const pre = document.getElementById(`mdprev-${i}`);
        if (pre) pre.textContent = txt.slice(0, 2000) + (txt.length > 2000 ? '\n…' : '');
      }).catch(()=>{});
    }
  }
}
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

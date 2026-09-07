/* Minimal demo UI: PDF scroller + page selection, task queue, MD/SVG results. */
"use strict";

const VARIANT = document.body.dataset.variant;
const $ = (id) => document.getElementById(id);

const state = {
  job: null,            // {job_id, kind, num_pages, filename, views: [...]}
  selected: new Set(),  // 0-based page ids
  bbox: null,           // {page, x1, y1, x2, y2} in view-image pixels
  watching: null,       // task id whose results are rendered
  watchTimer: null,
  resultTaskId: null,   // task whose cards are currently in #results
  renderedPages: new Set(), // page_no already appended (incremental render)
  promptModes: [],
  estimates: {},        // prompt_mode -> seconds per page (from benchmarks)
  timeOffset: 0,        // server_time - client_time, seconds
};

/* ------------------------------------------------ state polling */

async function pollState() {
  try {
    const res = await fetch("/api/state");
    const data = await res.json();
    if (data.server_time) state.timeOffset = data.server_time - Date.now() / 1000;
    renderModel(data.worker);
    renderDevices(data.devices, data.worker);
    renderGpus(data.gpus);
    renderTasks(data.tasks);
    // painted after renderTasks, and separately from it: the tokens/s readout
    // changes on every poll, and folding it into the task signature would
    // rebuild the queue DOM twice a second just to update one number
    renderLiveTps(data.worker && data.worker.live);
    initPeer(data);
    if (!state.promptModes.length) initModes(data);
  } catch (err) { /* server restarting; keep polling */ }
}
setInterval(pollState, 2000);

function renderModel(worker) {
  const el = $("model-state");
  let text = worker.model_state + (worker.device ? ` @ ${worker.device}` : "");
  // the device already reads "vllm" for that engine; do not say it twice
  if (worker.engine && worker.engine !== "transformers" && worker.engine !== worker.device) {
    text += ` · ${worker.engine}`;
  }
  if (worker.attn_implementation) text += ` · ${worker.attn_implementation}`;
  if (worker.model_state === "stopped") {
    text += worker.paused ? " (пауза)" : " (загрузится по запросу)";
  }
  if (worker.model_state === "loaded" && worker.unload_in_seconds != null) {
    text += ` · выгрузка через ${worker.unload_in_seconds}s`;
  }
  el.textContent = text;
  el.className = "badge " + ({loaded: "ok", loading: "warn", error: "err"}[worker.model_state] || "");
  if (worker.model_state === "error") el.title = worker.model_error || "";
  $("model-start").disabled = ["loaded", "loading"].includes(worker.model_state);
  $("model-stop").disabled = worker.model_state === "stopped";
  const keep = $("keep-loaded");
  if (document.activeElement !== keep) keep.checked = !!worker.keep_loaded;
}

function renderDevices(devices, worker) {
  const sel = $("device-select");
  const target = worker.configured_device || "auto";
  const opts = (devices || ["auto"]).join(",");
  // rebuild options only when the set changes, and never while the user is picking
  if (sel.dataset.opts !== opts && document.activeElement !== sel) {
    sel.innerHTML = "";
    (devices || ["auto"]).forEach((d) => {
      const o = document.createElement("option");
      o.value = d; o.textContent = d;
      sel.appendChild(o);
    });
    sel.dataset.opts = opts;
  }
  if (document.activeElement !== sel) sel.value = target;
}

function renderGpus(gpus) {
  $("gpu-panel").innerHTML = (gpus || []).map((g) => {
    const memPct = g.memory_total_mb ? Math.round(100 * g.memory_used_mb / g.memory_total_mb) : 0;
    return `<div class="gpu">GPU${g.index} · ${g.util_pct ?? "?"}% · ` +
      `${Math.round(g.memory_used_mb ?? 0)}/${Math.round(g.memory_total_mb ?? 0)}MB · ${Math.round(g.power_w ?? 0)}W` +
      `<div class="bar"><i style="width:${memPct}%"></i></div></div>`;
  }).join("");
}

$("model-start").onclick = () => fetch("/api/model/start", {method: "POST"}).then(pollState);
$("model-stop").onclick = () => fetch("/api/model/stop", {method: "POST"}).then(pollState);
$("keep-loaded").onchange = () => {
  const body = new FormData();
  body.append("value", $("keep-loaded").checked);
  fetch("/api/model/keep_loaded", {method: "POST", body}).then(pollState);
};
$("device-select").onchange = () => {
  const body = new FormData();
  body.append("device", $("device-select").value);
  fetch("/api/model/device", {method: "POST", body})
    .then((r) => { if (!r.ok) return r.text().then((t) => { throw new Error(t); }); })
    .then(pollState)
    .catch((e) => { $("run-error").textContent = "смена GPU: " + e; });
};

/* ------------------------------------------------ help & peer link */

const HELP = {
  mocr: `
    <p>Это демка модели <b>dots.mocr</b> — парсинг документов. Генерация SVG —
    у отдельной модели <b>dots.mocr-svg</b>, это другая демка (ссылка в шапке;
    для неё нужен проброшенный второй порт).</p>
    <ol>
      <li>Перетащите <b>PDF или картинки</b> (jpg/png) в зону слева, нажмите
      <b>«прикрепить файлы»</b> или <b>кликните по зоне и вставьте скриншот (Ctrl+V)</b>.
      Изображения накапливаются пачкой: каждое следующее становится новой страницей —
      потом обрабатываются по очереди.</li>
      <li>Отметьте галочками страницы для инференса (можно кликать по подписи страницы).</li>
      <li>Выберите режим (скилл модели) и нажмите «Запустить».</li>
    </ol>
    <p>Результат можно выгрузить кнопками <b>⬇ zip</b> (markdown + картинки + layout JSON)
    и <b>⬇ pdf</b> (готовый PDF-файл с отрендеренными формулами — скачивается сразу,
    без диалога печати) в секции «Результаты».</p>
    <p>Скиллы:</p>
    <ul>
      <li><b>layout_all</b> — блоки страницы: bbox + категория + текст → Markdown (основной режим);</li>
      <li><b>layout_only</b> — только детекция блоков (JSON);</li>
      <li><b>ocr</b> — весь текст страницы;</li>
      <li><b>grounding_ocr</b> — текст из области: выберите режим, затем <b>потяните мышкой прямоугольник прямо по странице</b>;</li>
      <li><b>web_parsing</b> — разметка скриншота веб-страницы;</li>
      <li><b>scene_spotting</b> — текст на фото/вывесках (координаты + текст);</li>
      <li><b>general</b> — свободный вопрос по странице (своё поле промпта).</li>
    </ul>
    <p>Модель сама загрузится на GPU при первой задаче и выгрузится после простоя
    (галочка «не выгружать» отключает выгрузку). Задачи можно останавливать в очереди —
    даже после перезагрузки страницы.</p>`,
  svg: `
    <p>Это демка модели <b>dots.mocr-svg</b> — генерация SVG-кода по изображению.
    Парсинг документов (OCR, layout) — у базовой модели dots.mocr, это другая демка
    (ссылка в шапке).</p>
    <ol>
      <li>Загрузите <b>картинку (png/jpg)</b> — основной сценарий. PDF тоже можно:
      каждая выбранная страница рендерится в картинку и превращается в SVG.</li>
      <li>Нажмите «Запустить». Генерация SVG небыстрая (~1-3 мин на изображение).</li>
    </ol>
    <p>Результат: вкладка <b>SVG</b> — отрисованный вектор, <b>raw svg</b> — код,
    <b>сравнение</b> — оригинал против рендера. Модель сильна на графиках, диаграммах
    и простых фигурах; плотные текстовые страницы даются ей хуже (возможен битый XML —
    смотрите raw svg).</p>
    <p>Модель грузится на GPU при первой задаче и выгружается после простоя.</p>`,
};

function initHelp() {
  $("help-body").innerHTML = HELP[VARIANT] || "";
}
initHelp();

function initPeer(data) {
  const link = $("peer-link");
  if (!link.hidden || !data.peer || !data.peer.port) return;
  link.textContent = `→ ${data.peer.title}`;
  link.href = `${location.protocol}//${location.hostname}:${data.peer.port}/`;
  link.title = "вторая демка на соседнем порту (туннель должен пробрасывать оба порта)";
  link.hidden = false;
}

/* ------------------------------------------------ prompt modes */

function initModes(data) {
  state.promptModes = data.prompt_modes;
  for (const item of data.prompt_modes) state.estimates[item.mode] = item.page_seconds_estimate || 45;
  const select = $("prompt-mode");
  select.innerHTML = "";
  for (const item of data.prompt_modes) {
    const option = document.createElement("option");
    option.value = item.mode;
    option.textContent = `${item.mode} (t=${item.default_temperature})`;
    if (item.mode === data.default_mode) option.selected = true;
    select.appendChild(option);
  }
  select.onchange = onModeChange;
  onModeChange();
}

function currentMode() { return $("prompt-mode").value; }

function onModeChange() {
  const mode = currentMode();
  $("custom-prompt-row").hidden = mode !== "prompt_general";
  $("bbox-row").hidden = mode !== "prompt_grounding_ocr";
  const hints = {
    prompt_layout_all_en: "полный layout: bbox + категория + текст (JSON → Markdown)",
    prompt_layout_only_en: "только детекция layout (JSON)",
    prompt_ocr: "извлечение текста страницы",
    prompt_grounding_ocr: "текст внутри bbox — выделите область мышкой на странице",
    prompt_web_parsing: "парсинг скриншота веб-страницы (JSON)",
    prompt_scene_spotting: "детекция текста в сцене (координаты + текст)",
    prompt_general: "свободный вопрос по странице",
    prompt_image_to_svg: "генерация SVG-кода по изображению (t=0.9, как у авторов)",
  };
  $("mode-hint").textContent = hints[mode] || "";
}

/* ------------------------------------------------ upload & viewer */

const dropzone = $("dropzone");
// Click focuses the zone so the next Ctrl+V pastes a screenshot straight in;
// the file picker lives on the separate "прикрепить" button.
dropzone.onclick = () => dropzone.focus();
$("attach-btn").onclick = (e) => { e.stopPropagation(); $("file-input").click(); };
$("file-input").onchange = () => {
  handleFiles([...$("file-input").files]);
  $("file-input").value = ""; // same file again must still fire onchange
};
dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add("drag"); };
dropzone.ondragleave = () => dropzone.classList.remove("drag");
dropzone.ondrop = (e) => {
  e.preventDefault(); dropzone.classList.remove("drag");
  handleFiles([...e.dataTransfer.files]);
};
// Paste is listened for at document level: non-editable focus targets (our
// dropzone div) do not receive paste events in every browser, and a missed
// paste reads as "the feature is broken". Text fields keep their normal paste
// (custom prompt, etc.) — everywhere else an image in the clipboard goes
// straight into the upload flow.
document.addEventListener("paste", (e) => {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
  const files = [...(e.clipboardData ? e.clipboardData.files : [])];
  if (files.length) { e.preventDefault(); handleFiles(files); }
});

const isImageFile = (f) => /^image\/(png|jpe?g)$/i.test(f.type) || /\.(png|jpe?g)$/i.test(f.name);
const isPdfFile = (f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name);

async function handleFiles(files) {
  files = files.filter((f) => isImageFile(f) || isPdfFile(f));
  if (!files.length) return;
  const images = files.filter(isImageFile);
  if (files.some(isPdfFile) || !images.length) {
    // a PDF always starts a fresh job (the server rejects pdf+image mixes)
    await upload(files);
  } else if (state.job && state.job.kind === "image") {
    // screenshots keep accumulating: each one becomes the next page
    await appendImages(images);
  } else {
    await upload(images);
  }
}

async function upload(files) {
  $("run-error").textContent = "";
  const body = new FormData();
  for (const f of files) body.append("file", f);
  const res = await fetch("/api/upload", {method: "POST", body});
  if (!res.ok) { $("run-error").textContent = await res.text(); return; }
  const job = await res.json();
  state.job = job;
  state.selected = new Set(job.views.map((v) => v.page));
  state.bbox = null;
  $("doc-name").textContent = `${job.filename} · ${job.num_pages} стр.`;
  $("doc-toolbar").hidden = false;
  renderViewer();
}

async function appendImages(files) {
  $("run-error").textContent = "";
  const body = new FormData();
  for (const f of files) body.append("file", f);
  const res = await fetch(`/api/jobs/${state.job.job_id}/images`, {method: "POST", body});
  if (!res.ok) { $("run-error").textContent = await res.text(); return; }
  const data = await res.json();
  const before = new Set(state.job.views.map((v) => v.page));
  state.job.num_pages = data.num_pages;
  state.job.views = data.views;
  for (const v of data.views) if (!before.has(v.page)) state.selected.add(v.page);
  state.bbox = null;
  $("doc-name").textContent = `${state.job.filename} · ${data.num_pages} стр.`;
  renderViewer();
}

function renderViewer() {
  const viewer = $("viewer");
  viewer.innerHTML = "";
  for (const view of state.job.views) {
    const wrap = document.createElement("div");
    wrap.className = "page-wrap" + (state.selected.has(view.page) ? " selected" : "");
    wrap.dataset.page = view.page;
    wrap.innerHTML =
      `<div class="page-label"><input type="checkbox" ${state.selected.has(view.page) ? "checked" : ""}> стр. ${view.page + 1}</div>` +
      `<img src="${view.url}" loading="lazy" draggable="false" data-w="${view.width}" data-h="${view.height}">`;
    wrap.querySelector(".page-label").onclick = (e) => { e.preventDefault(); togglePage(view.page); };
    attachBboxDrag(wrap.querySelector("img"), view.page);
    viewer.appendChild(wrap);
  }
  applyZoom();
  updateSelectionInfo();
}

function togglePage(page) {
  if (state.selected.has(page)) state.selected.delete(page); else state.selected.add(page);
  const wrap = document.querySelector(`.page-wrap[data-page="${page}"]`);
  wrap.classList.toggle("selected", state.selected.has(page));
  wrap.querySelector("input[type=checkbox]").checked = state.selected.has(page);
  updateSelectionInfo();
}

function updateSelectionInfo() {
  $("selection-info").textContent = `выбрано: ${state.selected.size}/${state.job ? state.job.num_pages : 0}`;
}

$("select-all").onclick = () => { state.job.views.forEach((v) => state.selected.add(v.page)); renderViewer(); };
$("select-none").onclick = () => { state.selected.clear(); renderViewer(); };
$("zoom").oninput = applyZoom;

function applyZoom() {
  const pct = Number($("zoom").value) / 100;
  document.querySelectorAll(".page-wrap").forEach((wrap) => {
    wrap.style.setProperty("--page-w", `${Math.round(640 * pct)}px`);
  });
}

/* bbox drag for grounding OCR: coordinates in the view image pixel space */
function attachBboxDrag(img, page) {
  let start = null;
  img.addEventListener("mousedown", (e) => {
    if (currentMode() !== "prompt_grounding_ocr") return;
    e.preventDefault();
    const rect = img.getBoundingClientRect();
    const scaleX = Number(img.dataset.w) / rect.width;
    const scaleY = Number(img.dataset.h) / rect.height;
    start = {x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY, rect, scaleX, scaleY};

    const overlay = document.createElement("div");
    overlay.className = "bbox-overlay";
    img.parentElement.querySelectorAll(".bbox-overlay").forEach((el) => el.remove());
    img.parentElement.appendChild(overlay);

    const move = (ev) => {
      const cx = Math.min(Math.max(ev.clientX, rect.left), rect.right);
      const cy = Math.min(Math.max(ev.clientY, rect.top), rect.bottom);
      const x2 = (cx - rect.left) * scaleX, y2 = (cy - rect.top) * scaleY;
      const bbox = normBbox(start.x, start.y, x2, y2);
      state.bbox = {page, ...bbox};
      drawOverlay(overlay, img, bbox);
      $("bbox-value").textContent = `стр.${page + 1} [${bbox.x1},${bbox.y1},${bbox.x2},${bbox.y2}]`;
    };
    const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  });
}

function normBbox(x1, y1, x2, y2) {
  return {x1: Math.round(Math.min(x1, x2)), y1: Math.round(Math.min(y1, y2)),
          x2: Math.round(Math.max(x1, x2)), y2: Math.round(Math.max(y1, y2))};
}

function drawOverlay(overlay, img, bbox) {
  const rect = img.getBoundingClientRect();
  const kx = rect.width / Number(img.dataset.w), ky = rect.height / Number(img.dataset.h);
  overlay.style.left = `${bbox.x1 * kx}px`;
  overlay.style.top = `${bbox.y1 * ky}px`;
  overlay.style.width = `${(bbox.x2 - bbox.x1) * kx}px`;
  overlay.style.height = `${(bbox.y2 - bbox.y1) * ky}px`;
}

/* ------------------------------------------------ run task */

$("run").onclick = async () => {
  $("run-error").textContent = "";
  if (!state.job) { $("run-error").textContent = "сначала загрузите файл"; return; }
  const mode = currentMode();
  let pages = [...state.selected].sort((a, b) => a - b);
  if (mode === "prompt_grounding_ocr") {
    if (!state.bbox) { $("run-error").textContent = "выделите bbox мышкой на странице"; return; }
    pages = [state.bbox.page];
  }
  if (!pages.length) { $("run-error").textContent = "не выбраны страницы"; return; }

  const body = new FormData();
  body.append("job_id", state.job.job_id);
  body.append("prompt_mode", mode);
  body.append("pages", pages.join(","));
  if ($("custom-prompt").value) body.append("custom_prompt", $("custom-prompt").value);
  if ($("temperature").value) body.append("temperature", $("temperature").value);
  if ($("max-tokens").value) body.append("max_new_tokens", $("max-tokens").value);
  if (mode === "prompt_grounding_ocr") {
    body.append("bbox", [state.bbox.x1, state.bbox.y1, state.bbox.x2, state.bbox.y2].join(","));
  }
  const res = await fetch("/api/tasks", {method: "POST", body});
  if (!res.ok) { $("run-error").textContent = await res.text(); return; }
  const {task_id} = await res.json();
  watchTask(task_id);
  pollState();
};

/* ------------------------------------------------ task queue */

function pageSeconds(task) {
  const measured = (task.result || []).map((r) => r.seconds).filter((s) => s > 0);
  if (measured.length) return measured.reduce((a, b) => a + b, 0) / measured.length;
  return state.estimates[task.prompt_mode] || 45;
}

function taskProgress(task) {
  const p = task.progress || {};
  const total = p.total || task.pages.length || 1;
  let done = p.done || 0;
  const perPage = pageSeconds(task);
  let etaSeconds = perPage * (total - done);
  if (task.status === "running" && p.page_started_at) {
    const now = Date.now() / 1000 + state.timeOffset;
    const elapsed = Math.max(0, now - p.page_started_at);
    const pageFrac = Math.min(elapsed / perPage, 0.95);
    done += pageFrac;
    etaSeconds -= pageFrac * perPage;
  }
  return {pct: Math.min(100, Math.round(100 * done / total)), eta: Math.max(0, Math.round(etaSeconds))};
}

function formatEta(seconds) {
  if (seconds >= 90) return `~${Math.round(seconds / 60)} мин`;
  return `~${seconds}s`;
}

function renderTasks(tasks) {
  // only the live queue (server already filters, guard client-side too)
  const active = (tasks || []).filter((t) => ["queued", "running"].includes(t.status));
  // skip the DOM rebuild when nothing changed -> no flicker on idle polls
  const sig = active.map((t) => `${t.id}:${t.status}:${(t.progress || {}).done}:${(t.progress || {}).page_started_at}`).join("|");
  if ($("tasks").dataset.sig === sig) return;
  $("tasks").dataset.sig = sig;
  $("tasks").innerHTML = active.map((t) => {
    const counts = t.progress && t.progress.total ? ` ${t.progress.done}/${t.progress.total}` : "";
    const cancellable = ["queued", "running"].includes(t.status);
    let bar = "";
    if (t.status === "running") {
      const {pct, eta} = taskProgress(t);
      bar = `<div class="progress"><i style="width:${pct}%"></i></div>
             <span class="muted">${pct}% · осталось ${formatEta(eta)}</span>
             <span class="tps" data-task="${t.id}"></span>`;
    } else if (t.status === "queued") {
      const eta = pageSeconds(t) * t.pages.length;
      bar = `<span class="muted">оценка: ${formatEta(Math.round(eta))}</span>`;
    }
    return `<div class="task">
      <span class="status-${t.status}">●</span>
      <span class="grow">${t.own ? "" : "<span class=muted>(чужая)</span> "}${t.prompt_mode}
        <span class="muted">стр. ${t.pages.map((p) => p + 1).join(",")}${counts}</span>
        ${bar}</span>
      <span class="status-${t.status}">${t.status}</span>
      ${cancellable ? `<button class="tiny secondary" onclick="cancelTask('${t.id}')">стоп</button>` : ""}
      ${t.result.length ? `<button class="tiny" onclick="watchTask('${t.id}')">показать</button>` : ""}
    </div>`;
  }).join("") || '<span class="muted">нет активных задач</span>';
}

/* Live generation speed for the page being decoded right now. The worker
   publishes the counters straight off the running generate() call, so this is
   the real decode rate, not an average over the task. */
function formatTps(live) {
  if (!live || live.done) return "";
  if (!live.generated_tokens) return "префилл (кодируем страницу)…";
  const rate = live.decode_tps || live.total_tps;
  const parts = [];
  if (rate) parts.push(`<b>${rate.toFixed(1)} t/s</b>`);
  parts.push(`${live.generated_tokens} ток.`);
  if (live.ttft_seconds) parts.push(`TTFT ${live.ttft_seconds.toFixed(1)}s`);
  return parts.join(" · ");
}

function renderLiveTps(live) {
  document.querySelectorAll(".tps").forEach((el) => {
    const mine = live && live.task_id === el.dataset.task;
    el.innerHTML = mine ? formatTps(live) : "";
  });
}

window.cancelTask = (id) => fetch(`/api/tasks/${id}/cancel`, {method: "POST"}).then(pollState);

/* ------------------------------------------------ export */

function exportTask(fmt) {
  if (!state.resultTaskId) return;
  window.open(`/api/tasks/${state.resultTaskId}/export.${fmt}`, "_blank");
}
$("export-zip").onclick = () => exportTask("zip");
$("export-pdf").onclick = () => exportPdf();

/* PDF export with typeset formulas and NO browser chrome. The markdown is
   rendered by the same marked+MathJax pipeline as the preview, but MathJax
   runs in a hidden iframe with fontCache "none" so every formula SVG is
   self-contained paths; the preview HTML plus those SVGs go to the server,
   which embeds them into a fitz-built PDF that simply downloads as a file.
   No new window and no print dialog — hence no URL/date/page-number header
   lines either. */
async function exportPdf() {
  const taskId = state.resultTaskId;
  if (!taskId) return;
  $("run-error").textContent = "";
  try {
    const task = await (await fetch(`/api/tasks/${taskId}`)).json();
    const pages = [];
    for (const page of (task.result || [])) {
      const urls = page.urls || {};
      if (!urls.md_content) continue;
      const md = (await (await fetch(`/api/raw?path=${encodeURIComponent(urls.md_content)}`)).json()).content;
      // keep image links RELATIVE (images/foo.png) — the server resolves them
      // against the task's out dir inside the PDF archive
      pages.push(renderMarkdownWithMath(md));
    }
    if (!pages.length) { $("run-error").textContent = "нет markdown для экспорта"; return; }
    const html = pages.map((h, i) =>
      `<div${i < pages.length - 1 ? ' style="page-break-after: always"' : ""}>${h}</div>`).join("");

    const iframe = await typesetInHiddenFrame(html);
    const svgs = await rasterizeMath(iframe);
    const body = iframe.contentDocument.body.innerHTML;
    iframe.remove();

    const res = await fetch(`/api/tasks/${taskId}/export.pdf`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({html: body, math: svgs}),
    });
    if (!res.ok) { $("run-error").textContent = await res.text(); return; }
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${taskId}-${task.prompt_mode}.pdf`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (err) {
    $("run-error").textContent = "экспорт pdf: " + err;
  }
}

/* Hidden iframe running its own MathJax instance. visibility:hidden (not
   display:none) so layout still happens and formula sizes are measurable.

   CRITICAL INVARIANT: the iframe's base font size must equal the PDF's base
   font size (demo/mdexport.py _CSS: body 10pt). MathJax scales formulas to
   the surrounding font, so typesetting in an unstyled iframe (16px default)
   produced formulas ~1.6x bigger than the PDF's 10pt text. 10pt here makes
   the measured px height convert to PDF points exactly (1pt = 96/72 px). */
function typesetInHiddenFrame(html) {
  return new Promise((resolve, reject) => {
    const iframe = document.createElement("iframe");
    iframe.setAttribute("style",
      "position:absolute; left:-10000px; top:0; width:697px; height:600px; visibility:hidden;");
    // width 697px = the PDF text column (A4 595pt − 2×36pt margins, 1pt =
    // 96/72 px): MathJax display equations are width:100% svgs, so they must
    // be typeset at the width they will occupy in the PDF
    iframe.setAttribute("aria-hidden", "true");
    document.body.appendChild(iframe);
    const doc = iframe.contentDocument;
    doc.open();
    doc.write(`<!doctype html><html><head><meta charset="utf-8">
<style>
  /* mirror demo/mdexport.py _CSS font sizes — the invariant above */
  body { font-size: 10pt; font-family: sans-serif; margin: 0; }
  h1 { font-size: 16pt; } h2 { font-size: 14pt; } h3 { font-size: 12pt; }
  code, pre { font-size: 9pt; }
</style>
<script>
  window.MathJax = {
    tex: { inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] },
    svg: { fontCache: 'none' },
    options: { enableMenu: false, skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] },
    startup: { typeset: false },
  };
<\/script>
<script src="/static/tex-svg.js"><\/script>
</head><body>${html}</body></html>`);
    doc.close();
    const deadline = Date.now() + 30000;
    const poll = () => {
      const w = iframe.contentWindow;
      if (w && w.MathJax && w.MathJax.startup) {
        w.MathJax.startup.promise
          .then(() => w.MathJax.typesetPromise())
          .then(() => resolve(iframe), reject);
      } else if (Date.now() > deadline) {
        iframe.remove();
        reject(new Error("MathJax не загрузился (30s)"));
      } else {
        setTimeout(poll, 100);
      }
    };
    poll();
  });
}

/* Replace every typeset mjx-container with an <img data-math="K"> placeholder
   carrying the measured size; rasterize each formula to PNG in the browser and
   return per-formula {svg, png, w, h} records in placeholder order.
   Rasterizing here (canvas) rather than on the server sidesteps the svgs that
   defeat cairosvg: MathJax display equations are width="100%" with nested
   <svg> elements and no viewBox — only a real browser engine lays them out
   correctly. The svg is still sent as a server-side fallback. */
async function rasterizeMath(iframe) {
  const doc = iframe.contentDocument;
  const SCALE = 3;
  const items = [];
  for (const container of doc.querySelectorAll("mjx-container")) {
    const svg = container.querySelector("svg");
    if (!svg) { container.remove(); continue; }
    const rect = container.getBoundingClientRect();
    const display = container.hasAttribute("display");
    const heightMatch = /([\d.]+)ex/.exec(svg.getAttribute("height") || "");
    const pxPerEx = (heightMatch && rect.height) ? rect.height / parseFloat(heightMatch[1]) : 8;
    const widthPt = (rect.width * 0.75).toFixed(1);
    const heightPt = (rect.height * 0.75).toFixed(1);
    const valignMatch = /vertical-align:\s*([-\d.]+)ex/.exec(svg.getAttribute("style") || "");
    // vertical-align is the INLINE baseline shift; on a display equation (an
    // img alone in its own block) it shifts the image below the box the PDF
    // layout reserves, so the next paragraph overlaps the image
    const valignPt = (!display && valignMatch)
      ? (parseFloat(valignMatch[1]) * pxPerEx * 0.75).toFixed(1) : "0";

    // the display-equation svg has width="100%" — pin the measured px size so
    // the canvas rasterization matches what was on screen
    const clone = svg.cloneNode(true);
    clone.setAttribute("width", `${Math.max(1, rect.width)}px`);
    clone.setAttribute("height", `${Math.max(1, rect.height)}px`);
    clone.removeAttribute("style");
    const url = URL.createObjectURL(new Blob([clone.outerHTML], {type: "image/svg+xml"}));
    const png = await new Promise((resolve) => {
      const image = new Image();
      image.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(rect.width * SCALE));
        canvas.height = Math.max(1, Math.round(rect.height * SCALE));
        const ctx = canvas.getContext("2d");
        ctx.scale(SCALE, SCALE);
        ctx.drawImage(image, 0, 0, rect.width, rect.height);
        URL.revokeObjectURL(url);
        resolve(canvas.toDataURL("image/png"));
      };
      image.onerror = () => { URL.revokeObjectURL(url); resolve(null); };
      image.src = url;
    });

    items.push({svg: svg.outerHTML, png, w: rect.width, h: rect.height});
    const img = doc.createElement("img");
    img.setAttribute("data-math", String(items.length - 1));
    img.setAttribute("style", `width:${widthPt}pt; height:${heightPt}pt; vertical-align:${valignPt}pt;`);
    img.setAttribute("alt", "(формула)");
    if (display) {
      const wrap = doc.createElement("div");
      wrap.setAttribute("style", "text-align:center; margin:6pt 0;");
      container.replaceWith(wrap);
      wrap.appendChild(img);
    } else {
      container.replaceWith(img);
    }
  }
  return items;
}

/* ------------------------------------------------ results */

/* Render model markdown with LaTeX math.
   marked.js mangles TeX ($x_i^l$ -> emphasis), so we lift math spans out
   BEFORE markdown, run marked + sanitize on the rest, then splice the math
   back as MathJax \(...\)/\[...\] delimiters and typeset. */
function renderMarkdownWithMath(src) {
  const math = [];
  const stash = (display, tex) => `MJXMATH${math.push({ display, tex }) - 1}END`;
  // display $$...$$ first, then inline $...$ (allow escaped \$ inside)
  let protectedSrc = (src || "")
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => stash(true, tex))
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, tex) => stash(true, tex))
    .replace(/(?<![\\$])\$((?:[^$\\\n]|\\.)+?)\$(?!\$)/g, (_, tex) => stash(false, tex))
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => stash(false, tex));

  let html = sanitizeHtml(marked.parse(protectedSrc));

  // restore placeholders (which survived markdown/sanitize) as escaped TeX
  html = html.replace(/MJXMATH(\d+)END/g, (_, i) => {
    const { display, tex } = math[Number(i)];
    const esc = tex.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return display
      ? `<span class="math-display">\\[${esc}\\]</span>`
      : `<span class="math-inline">\\(${esc}\\)</span>`;
  });
  return html;
}

/* Picture links in the markdown are relative (images/foo.png). Resolve them
   against the md file's served directory so the preview loads the crop from
   its folder — the markdown itself keeps only the relative link. */
function rewriteRelativeImages(el, baseUrl) {
  el.querySelectorAll("img[src]").forEach((img) => {
    const src = img.getAttribute("src") || "";
    if (/^(https?:|data:|\/)/i.test(src)) return; // absolute / data: untouched
    img.setAttribute("src", baseUrl + src.replace(/^\.\//, ""));
    img.setAttribute("loading", "lazy");
  });
}

function typesetMath(el) {
  if (!(window.MathJax && MathJax.typesetPromise)) return;
  MathJax.typesetPromise([el]).then(() => {
    // MathJax \href can emit clickable links from untrusted TeX — strip
    // javascript:/vbscript:/data: schemes MathJax generated during typeset.
    el.querySelectorAll("a").forEach((a) => {
      for (const name of ["href", "xlink:href"]) {
        const v = a.getAttribute(name);
        if (v && /^(javascript|vbscript|data):/i.test(v.replace(/[\u0000-\u0020]/g, "").toLowerCase())) {
          a.removeAttribute(name);
        }
      }
    });
  }).catch((e) => console.warn("MathJax:", e));
}

/* model output (markdown/svg) is injected as HTML: strip active content */
function sanitizeHtml(html) {
  const tpl = document.createElement("template");
  tpl.innerHTML = html;
  tpl.content.querySelectorAll("script, iframe, object, embed").forEach((n) => n.remove());
  tpl.content.querySelectorAll("*").forEach((node) => {
    for (const attr of [...node.attributes]) {
      const isHandler = /^on/i.test(attr.name);
      // browsers strip whitespace/control chars inside the scheme,
      // so "jav\tascript:" still executes — normalize before matching
      const normalized = attr.value.replace(/[\u0000-\u0020]/g, "").toLowerCase();
      const isUrlAttr = ["href", "src", "xlink:href", "action", "formaction"].includes(attr.name);
      const isBadUrl = isUrlAttr && (normalized.startsWith("javascript:")
        || normalized.startsWith("vbscript:") || normalized.startsWith("data:text/html"));
      if (isHandler || isBadUrl) node.removeAttribute(attr.name);
    }
  });
  return tpl.innerHTML;
}

window.watchTask = function watchTask(taskId) {
  state.watching = taskId;
  if (state.watchTimer) clearInterval(state.watchTimer);
  let timer = null;
  const tick = async () => {
    const res = await fetch(`/api/tasks/${taskId}`);
    // a late response for a task the user switched away from must not
    // repaint the panel or kill the new task's timer
    if (state.watching !== taskId) { if (timer) clearInterval(timer); return; }
    if (!res.ok) return;
    const task = await res.json();
    renderResults(task);
    if (!["queued", "running"].includes(task.status) && timer) clearInterval(timer);
  };
  timer = setInterval(tick, 1500);
  state.watchTimer = timer;
  tick();
};

/* Incremental render: append only newly-arrived page cards and update the
   header in place. Never rebuild existing cards, so the results scroller does
   not jump and already-typeset math / open tabs stay put during generation. */
function renderResults(task) {
  $("result-task").textContent = `· ${task.prompt_mode} · ${task.status}`;
  $("export-row").hidden = !task.result.length;
  const box = $("results");

  if (state.resultTaskId !== task.id) {
    // switched to a different task -> start fresh
    state.resultTaskId = task.id;
    state.renderedPages = new Set();
    box.innerHTML = "";
  }

  if (!task.result.length) {
    if (state.renderedPages.size === 0) {
      box.innerHTML = `<span class="muted">${task.status === "error" ? (task.error || "ошибка") : "задача выполняется…"}</span>`;
    }
    return;
  }

  for (const page of task.result) {
    if (state.renderedPages.has(page.page_no)) continue;
    if (state.renderedPages.size === 0) box.innerHTML = ""; // drop the placeholder
    box.appendChild(resultCard(page));
    state.renderedPages.add(page.page_no);
  }
}

function resultCard(page) {
  const card = document.createElement("div");
  card.className = "result-card";
  const urls = page.urls || {};
  const gen = page.generation || {};
  const genInfo = gen.generated_tokens
    ? ` · ${gen.generated_tokens} ток.` +
      (gen.decode_tps ? ` · ${gen.decode_tps.toFixed(1)} t/s` : "") +
      (gen.ttft_seconds ? ` · TTFT ${gen.ttft_seconds.toFixed(1)}s` : "")
    : "";
  card.innerHTML = `<h4>Страница ${page.page_no + 1} <span class="muted">${page.seconds ?? "?"}s${genInfo}</span></h4>
    <div class="result-tabs"></div><div class="result-body"><span class="muted">…</span></div>`;
  const tabs = card.querySelector(".result-tabs");
  const body = card.querySelector(".result-body");

  const addTab = (label, render, primary) => {
    const btn = document.createElement("button");
    btn.className = "tiny" + (primary ? "" : " secondary");
    btn.textContent = label;
    btn.onclick = () => render(body);
    tabs.appendChild(btn);
    return btn;
  };

  const rawFetch = async (url) => (await (await fetch(`/api/raw?path=${encodeURIComponent(url)}`)).json()).content;

  let first = null;
  if (urls.svg_content) {
    first = addTab("SVG", async (el) => {
      el.innerHTML = sanitizeHtml(await rawFetch(urls.svg_content));
    }, true);
    addTab("raw svg", async (el) => {
      el.innerHTML = `<pre></pre>`; el.querySelector("pre").textContent = await rawFetch(urls.svg_content);
    });
  }
  if (urls.md_content) {
    // the markdown stores picture links relative to its own folder (images/…);
    // resolve them against the md file's served directory at preview time
    const mdBase = urls.md_content.replace(/[^/]*$/, "");
    const tab = addTab(urls.svg_content ? "md" : "MD", async (el) => {
      el.innerHTML = `<div class="md-render"></div>`;
      const target = el.querySelector(".md-render");
      target.innerHTML = renderMarkdownWithMath(await rawFetch(urls.md_content));
      rewriteRelativeImages(target, mdBase);
      typesetMath(target);
    }, !first);
    first = first || tab;
    addTab("raw md", async (el) => {
      el.innerHTML = `<pre></pre>`; el.querySelector("pre").textContent = await rawFetch(urls.md_content);
    });
  }
  if (urls.layout_info) {
    addTab("json", async (el) => {
      let text = await rawFetch(urls.layout_info);
      try { text = JSON.stringify(JSON.parse(text), null, 2); } catch (err) { /* raw */ }
      el.innerHTML = `<pre></pre>`; el.querySelector("pre").textContent = text;
    });
  }
  if (urls.layout_image) {
    addTab(VARIANT === "svg" ? "сравнение" : "layout", (el) => {
      el.innerHTML = `<img src="${urls.layout_image}">`;
    });
  }
  if (first) first.click(); else body.innerHTML = '<span class="muted">нет артефактов</span>';
  return card;
}

pollState();

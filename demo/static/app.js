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
    initPeer(data);
    if (!state.promptModes.length) initModes(data);
  } catch (err) { /* server restarting; keep polling */ }
}
setInterval(pollState, 2000);

function renderModel(worker) {
  const el = $("model-state");
  let text = worker.model_state + (worker.device ? ` @ ${worker.device}` : "");
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
      <li>Перетащите <b>PDF или картинку</b> (jpg/png) в зону слева.</li>
      <li>Отметьте галочками страницы для инференса (можно кликать по подписи страницы).</li>
      <li>Выберите режим (скилл модели) и нажмите «Запустить».</li>
    </ol>
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
dropzone.onclick = () => $("file-input").click();
$("file-input").onchange = () => { if ($("file-input").files[0]) upload($("file-input").files[0]); };
dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add("drag"); };
dropzone.ondragleave = () => dropzone.classList.remove("drag");
dropzone.ondrop = (e) => {
  e.preventDefault(); dropzone.classList.remove("drag");
  if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
};

async function upload(file) {
  $("run-error").textContent = "";
  const body = new FormData();
  body.append("file", file);
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
  $("tasks").innerHTML = (tasks || []).map((t) => {
    const counts = t.progress && t.progress.total ? ` ${t.progress.done}/${t.progress.total}` : "";
    const cancellable = ["queued", "running"].includes(t.status);
    let bar = "";
    if (t.status === "running") {
      const {pct, eta} = taskProgress(t);
      bar = `<div class="progress"><i style="width:${pct}%"></i></div>
             <span class="muted">${pct}% · осталось ${formatEta(eta)}</span>`;
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
      ${t.result.length || t.status === "done" ? `<button class="tiny" onclick="watchTask('${t.id}')">показать</button>` : ""}
    </div>`;
  }).join("") || '<span class="muted">пока пусто</span>';
}

window.cancelTask = (id) => fetch(`/api/tasks/${id}/cancel`, {method: "POST"}).then(pollState);

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

function renderResults(task) {
  $("result-task").textContent = `· ${task.prompt_mode} · ${task.status}`;
  const box = $("results");
  if (!task.result.length) {
    box.innerHTML = `<span class="muted">${task.status === "error" ? (task.error || "ошибка") : "задача выполняется…"}</span>`;
    return;
  }
  box.innerHTML = "";
  for (const page of task.result) {
    box.appendChild(resultCard(page));
  }
}

function resultCard(page) {
  const card = document.createElement("div");
  card.className = "result-card";
  const urls = page.urls || {};
  card.innerHTML = `<h4>Страница ${page.page_no + 1} <span class="muted">${page.seconds ?? "?"}s</span></h4>
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

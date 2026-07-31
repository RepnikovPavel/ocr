// Arxiv pipeline status UI. Polls the container's read-only /api/v1/arxiv/*
// endpoints, which reflect what the host-side runner is writing to the shared
// demo.db. No framework; plain fetch + innerHTML, like app.js.
"use strict";

const API = "/api/v1/arxiv";
const POLL_MS = 2000;

const state = {
  activeRunId: null,   // which run the grid shows (last running, or clicked)
  runs: [],
  papers: [],
  paperFilter: "",
  stats: null,
};

// ----------------------------------------------------------------- helpers

function fmtBytes(n) {
  if (!n && n !== 0) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let v = n, i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
}

function fmtWhen(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  const ago = (Date.now() - d.getTime()) / 1000;
  if (ago < 60) return `${Math.floor(ago)}s ago`;
  if (ago < 3600) return `${Math.floor(ago / 60)}m ago`;
  if (ago < 86400) return `${Math.floor(ago / 3600)}h ago`;
  return d.toLocaleDateString();
}

function el(id) { return document.getElementById(id); }

// ----------------------------------------------------------------- poll loop

async function poll() {
  try {
    await refreshStats();
    await refreshRuns();
    await refreshPapers();
    // follow the most recent still-running run automatically
    const running = state.runs.find(r => r.status === "running");
    if (running && state.activeRunId !== running.id) {
      state.activeRunId = running.id;
    }
    if (state.activeRunId) await refreshRun(state.activeRunId);
  } catch (e) {
    console.warn("poll failed", e);
  }
}

async function getJSON(url) {
  const r = await fetch(url, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

// ----------------------------------------------------------------- stats

async function refreshStats() {
  const s = await getJSON(`${API}/stats`);
  state.stats = s;
  const p = s.papers || {};
  const backend = s.storage_backend || "?";
  const badge = el("storage-badge");
  badge.textContent = `storage: ${backend}`;
  badge.className = "badge " + (backend === "seaweedfs" ? "ok" : "warn");
  const cells = [
    ["papers seen", p.total || 0],
    ["parsed", p.parsed || 0],
    ["stored", p.stored || 0],
    ["bytes", fmtBytes(p.bytes || 0)],
  ];
  el("stats").innerHTML = cells.map(([k, v]) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
}

// ----------------------------------------------------------------- runs list

async function refreshRuns() {
  const data = await getJSON(`${API}/runs?limit=20`);
  state.runs = data.runs || [];
  if (!state.activeRunId && state.runs.length) {
    state.activeRunId = state.runs[0].id;
  }
  const html = state.runs.map(r => {
    const cls = "runs-item" + (r.id === state.activeRunId ? " active" : "");
    const tag = runStatusBadge(r);
    return `<div class="${cls}" data-id="${r.id}">
        <span class="rid">${r.id}</span>
        ${tag}
        <span class="muted">${r.done}/${r.max_papers * 6 || "—"} steps</span>
        <span class="when">${fmtWhen(r.created_at)}</span>
      </div>`;
  }).join("");
  el("runs-list").innerHTML = html || `<p class="empty">No runs yet.</p>`;
  el("runs-list").querySelectorAll(".runs-item").forEach(node => {
    node.onclick = () => { state.activeRunId = node.dataset.id; refreshRun(state.activeRunId); refreshRuns(); };
  });
  // header badge for the freshest run
  const freshest = state.runs[0];
  el("last-run-badge").textContent = freshest
    ? `${freshest.id} · ${freshest.status}`
    : "no run yet";
  el("last-run-badge").className = "badge " + runBadgeClass(freshest && freshest.status);
}

function runStatusBadge(r) {
  if (!r) return "";
  return `<span class="badge ${runBadgeClass(r.status)}">${r.status}</span>`;
}
function runBadgeClass(status) {
  if (status === "done") return "ok";
  if (status === "error") return "err";
  if (status === "running") return "warn";
  return "secondary";
}

// ----------------------------------------------------------------- active run grid

async function refreshRun(runId) {
  let run;
  try {
    run = await getJSON(`${API}/runs/${runId}`);
  } catch (e) {
    el("grid-empty").textContent = `Failed to load run ${runId}: ${e.message}`;
    return;
  }
  el("grid-empty").style.display = (run.papers && run.papers.length) ? "none" : "block";

  // progress bar = done steps / total steps. Prefer the server-computed live
  // counters (steps_done/total include in-flight progress); fall back to a
  // grid-derived count for older responses that don't carry them.
  const stages = run.stages || [];
  const total = (typeof run.steps_total === "number")
    ? run.steps_total
    : (run.papers || []).length * (stages.length || 6);
  const done = (typeof run.steps_done === "number")
    ? run.steps_done
    : (run.papers || []).reduce((acc, p) =>
        acc + stages.filter(s => (p.stages[s] || {}).status === "done").length, 0);
  const failed = (typeof run.steps_failed === "number")
    ? run.steps_failed
    : (run.papers || []).reduce((acc, p) =>
        acc + stages.filter(s => (p.stages[s] || {}).status === "error").length, 0);
  const pct = total ? Math.round((done / total) * 100) : 0;
  const bar = el("run-bar");
  bar.style.width = `${pct}%`;
  bar.className = failed > 0 ? "err" : "";
  const parsed = (typeof run.papers_parsed === "number")
    ? run.papers_parsed
    : (run.papers || []).filter(p =>
        stages.every(s => (p.stages[s] || {}).status === "done")).length;
  el("run-counts").textContent = `${done}/${total} steps · ${failed} failed · ${parsed} papers done`;
  el("run-summary").textContent = `· ${run.id} · ${run.query || ""} · ${run.status}`;

  // header: paper | title | one column per stage
  const stages = run.stages || [];
  const head = `<th>paper</th><th>title</th>` +
    stages.map(s => `<th>${s}</th>`).join("");
  el("grid-head").innerHTML = head;

  const body = (run.papers || []).map(p => {
    const cells = stages.map(s => {
      const st = (p.stages || {})[s] || {};
      const cls = `pip s-${st.status || "queued"}` + (st.detail ? " has-detail" : "");
      const title = st.detail ? `${st.status}: ${st.detail}` : st.status || "queued";
      return `<td class="cell"><span class="${cls}" title="${title}"></span></td>`;
    }).join("");
    return `<tr>
        <td class="paper-id"><a href="https://arxiv.org/abs/${p.arxiv_id}" target="_blank">${p.arxiv_id}</a></td>
        <td class="paper-title" title="${escapeHtml(p.title || "")}">${escapeHtml(p.title || "")}</td>
        ${cells}
      </tr>`;
  }).join("");
  el("grid-body").innerHTML = body;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

// ----------------------------------------------------------------- papers catalogue

async function refreshPapers() {
  const data = await getJSON(`${API}/papers?limit=100`);
  state.papers = data.papers || [];
  renderPapers();
}

function renderPapers() {
  const q = state.paperFilter.trim().toLowerCase();
  const list = state.papers.filter(p => {
    if (!q) return true;
    return (p.title || "").toLowerCase().includes(q)
      || (p.arxiv_id || "").toLowerCase().includes(q)
      || (p.categories || "").toLowerCase().includes(q);
  });
  const html = list.map(p => {
    const pages = p.num_pages ? `<span class="pages">${p.num_pages}p</span>` : "";
    return `<div class="paper-row" data-id="${p.arxiv_id}">
        <div class="ptitle">${escapeHtml(p.title || p.arxiv_id)} ${pages}</div>
        <div class="pmeta">
          <code>${p.arxiv_id}</code>
          <span>${(p.categories || "").split(" ").slice(0, 3).join(" ")}</span>
          ${p.sha256 ? `<span class="muted">sha ${p.sha256.slice(0, 8)}</span>` : ""}
          ${p.storage_status ? `<span class="muted">${p.storage_status}</span>` : ""}
        </div>
      </div>`;
  }).join("");
  el("papers-list").innerHTML = html || `<p class="empty">No parsed papers yet.</p>`;
  el("papers-list").querySelectorAll(".paper-row").forEach(node => {
    node.onclick = () => {
      // download the stored bundle zip
      window.location.href = `${API}/papers/${node.dataset.id}/bundle`;
    };
  });
}

// ----------------------------------------------------------------- boot

el("paper-filter").addEventListener("input", e => {
  state.paperFilter = e.target.value;
  renderPapers();
});

poll();
setInterval(poll, POLL_MS);

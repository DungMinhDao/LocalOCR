"use strict";
/* LocalOCR front-end: vanilla JS, talks to the FastAPI backend in localocr/server.py */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : String(kid));
  return el;
};
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtS = (s) => s == null ? "–" : s < 1 ? `${(s * 1000).toFixed(0)} ms` : s < 60 ? `${s.toFixed(s < 10 ? 2 : 1)} s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
const fmtN = (n) => n >= 10000 ? `${(n / 1000).toFixed(0)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
const fmtB = (b) => b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`;
const estTokens = (s) => Math.round(new TextEncoder().encode(s).length / 4);   // same heuristic as the server
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem("localocr." + k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("localocr." + k, JSON.stringify(v)); } catch {} },
};

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

let toastTimer;
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "show" + (err ? " err" : "");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.className = ""), err ? 6000 : 2200);
}
function download(blob, name) {
  const a = h("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

/* ───────────────────────── state ───────────────────────── */
const S = {
  appMode: new URLSearchParams(location.search).get("mode") || store.get("appMode", "extract"),
  engines: [], workers: {}, files: [], jobs: [],
  file: store.get("file", null),
  jobByFile: store.get("jobByFile", {}),
  selected: new Set(store.get("engines", ["apple-vision", "mlx:paddleocr-vl"])),
  opts: store.get("opts", {}),
  openEngine: null,
  mode: store.get("mode", "preview"),
  zoom: 1, activePage: null,
  detail: {}, rendered: {}, exports: {},
  diffAgainst: null,
  // publish
  formats: {}, templates: [],
  pubFormat: store.get("pubFormat", "docx"),
  pubOpts: store.get("pubOpts", {}),
  suggestions: [], suggest: null,
};
const saveSel = () => { store.set("engines", [...S.selected]); store.set("opts", S.opts); };
const curFile = () => S.files.find((f) => f.id === S.file);
const fileJobs = () => S.jobs.filter((j) => j.file_id === S.file);
const curJob = () => { const id = S.jobByFile[S.file]; return fileJobs().find((j) => j.id === id) || fileJobs()[0]; };
const active = (j) => j && (j.status === "running" || j.status === "queued");

/* ───────────────────────── panes: resize & collapse ───────────────────────── */
// Fixed-width panes keep a width; the "flex" pane of each mode takes the remaining space.
const PANE_DEFAULTS = { sidebar: 300, viewer: 520, editor: 560 };
const FLEX = { extract: "output", publish: "preview" };
const MIN_W = 200, RAIL_W = 34;
// Narrower than this (e.g. a window snapped to half a screen), the two content panes of a mode take turns:
// one is open, the other is a rail. The saved desktop layout is left untouched.
const NARROW_W = 1000;
let paneState = store.get("panes", {});
const narrowOpen = {};   // mode → the content pane open in narrow mode
const pstate = (id) => (paneState[id] ||= { w: PANE_DEFAULTS[id] || 480, collapsed: false });
const visiblePanes = () => $$("#app > .pane").filter((p) => !p.dataset.mode || p.dataset.mode === S.appMode);
const isNarrow = () => window.innerWidth < NARROW_W;
const contentPanes = () => visiblePanes().map((p) => p.dataset.pane).filter((id) => id !== "sidebar");
function isCollapsed(id) {
  if (!isNarrow() || id === "sidebar") return pstate(id).collapsed;
  const ids = contentPanes(), pick = narrowOpen[S.appMode];
  return id !== (ids.includes(pick) ? pick : FLEX[S.appMode]);
}

function flexPaneId() {
  const open = visiblePanes().filter((p) => !isCollapsed(p.dataset.pane));
  const pref = FLEX[S.appMode];
  if (open.some((p) => p.dataset.pane === pref)) return pref;
  return open.length ? open[open.length - 1].dataset.pane : null;
}

function layout() {
  const app = $("#app");
  app.dataset.mode = S.appMode;
  $$(".gutter", app).forEach((g) => g.remove());
  const panes = visiblePanes();
  panes.forEach((p, i) => {
    if (i === 0) return;
    const g = h("div", { class: "gutter", title: "Drag to resize · double-click to collapse/expand" });
    g.addEventListener("pointerdown", (e) => startDrag(e, panes[i - 1], p, g));
    g.addEventListener("dblclick", () => {
      const left = panes[i - 1].dataset.pane, flex = flexPaneId();
      toggleCollapse(left !== flex ? left : p.dataset.pane);
    });
    p.before(g);
  });
  applyColumns();
}

// Only updates column sizes, so it is safe to call on every pointermove (gutters stay in place).
function applyColumns() {
  const flex = flexPaneId();
  const panes = visiblePanes();
  // Shrink the fixed-width panes (not their saved widths) so the flex pane always keeps MIN_W on screen.
  const fixed = panes.filter((p) => !isCollapsed(p.dataset.pane) && p.dataset.pane !== flex);
  const rails = panes.filter((p) => isCollapsed(p.dataset.pane)).length;
  const room = window.innerWidth - (panes.length - 1) * 5 - rails * RAIL_W - MIN_W;
  const want = fixed.reduce((a, p) => a + pstate(p.dataset.pane).w, 0);
  const k = want > room ? Math.max(0, room) / want : 1;
  const cols = [];
  panes.forEach((p, i) => {
    const id = p.dataset.pane, st = pstate(id), shut = isCollapsed(id);
    if (i > 0) cols.push("5px");
    p.classList.toggle("collapsed", shut);
    cols.push(shut ? `${RAIL_W}px` : id === flex ? "minmax(0, 1fr)" : `${Math.max(160, Math.floor(st.w * k))}px`);
  });
  $("#app").style.gridTemplateColumns = cols.join(" ");
  store.set("panes", paneState);
}

function toggleCollapse(id, force) {
  const want = force ?? !isCollapsed(id);
  if (isNarrow() && id !== "sidebar") {   // the content panes take turns
    narrowOpen[S.appMode] = want ? contentPanes().find((x) => x !== id) : id;
    return applyColumns();
  }
  const st = pstate(id);
  if (want) {
    const stillOpen = visiblePanes().filter((p) => p.dataset.pane !== id && p.dataset.pane !== "sidebar" && !pstate(p.dataset.pane).collapsed);
    if (id !== "sidebar" && !stillOpen.length) return toast("Keep at least one pane open");
  }
  st.collapsed = want;
  applyColumns();
}

function startDrag(e, left, right, gutter) {
  const flex = flexPaneId();
  const lid = left.dataset.pane, rid = right.dataset.pane;
  if (isNarrow() && lid !== "sidebar" && rid !== "sidebar") return;   // one content pane at a time
  // Resize whichever neighbour has a fixed width; expanding a collapsed one by dragging opens it.
  let target, sign;
  if (lid !== flex) { target = lid; sign = 1; } else if (rid !== flex) { target = rid; sign = -1; } else return;
  const st = pstate(target);
  let moved = false;
  const startX = e.clientX, startW = (target === lid ? left : right).getBoundingClientRect().width;
  const total = $("#app").getBoundingClientRect().width;
  gutter.setPointerCapture(e.pointerId);
  gutter.classList.add("dragging");
  document.body.classList.add("resizing");
  const move = (ev) => {
    const others = visiblePanes().filter((p) => p.dataset.pane !== target && p.dataset.pane !== flex)
      .reduce((a, p) => a + p.getBoundingClientRect().width, 0);
    const max = total - others - MIN_W - 20;
    if (!moved && Math.abs(ev.clientX - startX) < 3) return;   // let clicks / double-clicks through
    moved = true;
    st.collapsed = false;   // dragging a collapsed rail opens it
    st.w = Math.round(Math.min(max, Math.max(MIN_W - 40, startW + sign * (ev.clientX - startX))));
    applyColumns();
  };
  const up = () => {
    gutter.removeEventListener("pointermove", move);
    gutter.classList.remove("dragging");
    document.body.classList.remove("resizing");
    if (moved && st.w < MIN_W - 20) { st.w = PANE_DEFAULTS[target] || 400; st.collapsed = true; }   // dragged shut
    applyColumns();
  };
  gutter.addEventListener("pointermove", move);
  gutter.addEventListener("pointerup", up, { once: true });
}

function setupPanes() {
  for (const p of $$("#app > .pane")) {
    p.prepend(h("div", { class: "rail", title: `Expand ${p.dataset.title}`, onclick: () => toggleCollapse(p.dataset.pane, false) },
      h("span", {}, "›"), h("span", { class: "rail-label" }, p.dataset.title)));
  }
  $$(".collapse-btn").forEach((b) => b.addEventListener("click", () => toggleCollapse(b.dataset.pane, true)));
  // foldable sidebar sections
  const folds = store.get("folds", {});
  for (const panel of $$(".panel[data-fold]")) {
    panel.classList.toggle("folded", !!folds[panel.dataset.fold]);
    $(".panel-head", panel).addEventListener("click", () => {
      folds[panel.dataset.fold] = panel.classList.toggle("folded");
      store.set("folds", folds);
    });
  }
}

function setAppMode(m) {
  S.appMode = m; store.set("appMode", m);
  $$("#mode-switch button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  layout();
  if (m === "publish") { renderFormats(); refreshPreview(); }
}

/* ───────────────────────── engines ───────────────────────── */
async function loadEngines() {
  const d = await api("/api/engines");
  S.engines = d.engines; S.workers = d.workers; S.suggestions = d.suggestions || [];
  const known = new Set(S.engines.map((e) => e.id));   // forget engines that were removed from the app
  for (const id of S.selected) if (!known.has(id)) S.selected.delete(id);
  renderEngines();
}

function engineDot(e) {
  if (!e.worker || e.id === "markitdown") return ["loaded", "Always ready"];
  const w = S.workers[e.worker];
  if (e.worker === "mlx") {
    if (w && w.loaded.includes(e.model)) return ["loaded", "Loaded in memory (warm)"];
    return e.cached ? ["cached", "Downloaded, not loaded"] : ["", e.note || "Not downloaded yet"];
  }
  if (w && w.loaded.length) return ["loaded", "Loaded (warm)"];
  if (w && (w.state === "starting" || w.state === "busy")) return ["busy", "Starting…"];
  return ["", "Not loaded"];
}

// "What's in the file?": tick the engine the benchmark found best for that kind of document.
function renderSuggest() {
  const root = $("#suggest");
  root.innerHTML = "";
  const byId = Object.fromEntries(S.engines.map((e) => [e.id, e]));
  const cur = S.suggestions.find((x) => x.id === S.suggest);
  root.append(h("div", { class: "chips" }, ...S.suggestions.map((x) => h("button", {
    class: "chip" + (cur === x ? " on" : ""), title: `Tick ${byId[x.pick]?.name}: ${x.why}`,
    onclick: () => {
      S.suggest = cur === x ? null : x.id;
      if (S.suggest) {
        S.selected = new Set([x.pick]);
        if (!byId[x.pick].featured) store.set("moreModels", true);
        saveSel();
      }
      renderEngines();
    } }, x.label))));
  if (!cur) return;
  const add = (id, label) => h("button", { class: "link" + (S.selected.has(id) ? " on" : ""),
    title: S.selected.has(id) ? "Untick" : "Tick it too, to compare",
    onclick: () => { S.selected.has(id) ? S.selected.delete(id) : S.selected.add(id); saveSel(); renderEngines(); } },
    (S.selected.has(id) ? "✓ " : "+ ") + byId[id].name);
  root.append(h("div", { class: "suggest-why" },
    h("div", {}, h("b", {}, byId[cur.pick].name), ": ", cur.why, "."),
    ...cur.alts.map(([id, why]) => h("div", {}, add(id), " ", h("span", { class: "muted" }, why)))));
}

function renderEngines() {
  renderSuggest();
  const root = $("#engines");
  root.innerHTML = "";
  const kind = curFile()?.kind;
  let group = null, more = null;
  const extra = S.engines.filter((e) => !e.featured);
  for (const e of S.engines) {
    if (e.group !== group) { group = e.group; root.append(h("div", { class: "group-title" }, group)); }
    if (!e.featured && !more) {   // the less useful models: folded away, still one click from use
      const picked = extra.filter((x) => S.selected.has(x.id)).length;
      more = h("details", { class: "more-models", open: store.get("moreModels", false) || undefined,
          ontoggle: (ev) => store.set("moreModels", ev.target.open) },
        h("summary", { title: "Older, niche or slower models. See Benchmarks for how they compare." },
          `More models (${extra.length})`, picked ? h("span", { class: "badge" }, `${picked} ticked`) : null));
      root.append(more);
    }
    const on = S.selected.has(e.id);
    const na = kind && !e.accepts.includes(kind);
    const [dotCls, dotTitle] = engineDot(e);
    const box = h("input", { type: "checkbox", checked: on, disabled: !e.available,
      onchange: (ev) => { ev.target.checked ? S.selected.add(e.id) : S.selected.delete(e.id); saveSel(); renderEngines(); } });
    const row = h("div", { class: "engine-row" },
      h("label", { title: na ? `Can't read ${kind} files (skipped)` : e.description }, box,
        h("span", { class: "dot " + dotCls, title: dotTitle }), h("span", { class: "nmw" }, h("span", { class: "nm" }, e.name),
          e.tag ? h("span", { class: "tag" }, e.tag) : null)),
      e.speed ? h("span", { class: "badge" + (e.speed.startsWith("⚡") ? " ok" : "") }, e.speed.replace("⚡ ", "")) : null,
      h("button", { class: "gear", title: "Options", onclick: () => { S.openEngine = S.openEngine === e.id ? null : e.id; renderEngines(); } }, "⚙"));
    const opts = h("div", { class: "engine-opts" }, h("div", { class: "desc" }, e.description),
      e.note ? h("div", { class: "desc", style: "color:var(--warn)" }, e.note) : null,
      e.model ? h("div", { class: "desc" }, h("code", {}, e.model)) : null,
      ...e.options.map((o) => optInput(o, optVal(S.opts[e.id], o), (v) => { (S.opts[e.id] ||= {})[o.key] = v; saveSel(); })));
    const acts = h("div", { class: "opt-actions" },
      h("button", { class: "small", onclick: () => { delete S.opts[e.id]; saveSel(); renderEngines(); toast("Options reset"); } }, "Reset options"));
    if (e.worker && e.id !== "markitdown") {
      acts.append(h("button", { class: "small", onclick: async () => { await post(`/api/engines/${encodeURIComponent(e.id)}/warmup`); toast("Loading model in background…"); setTimeout(loadEngines, 800); } }, "Pre-load"));
      acts.append(h("button", { class: "small", title: "Stops the worker process and frees its memory", onclick: async () => { await post(`/api/workers/${e.worker}/stop`); loadEngines(); toast("Unloaded"); } }, "Unload"));
    }
    opts.append(acts);
    (e.featured ? root : more).append(h("div", { class: `engine${on ? " on" : ""}${S.openEngine === e.id ? " open" : ""}${na ? " na" : ""}` }, row, opts));
  }
  for (const w of Object.values(S.workers)) if (w.state === "error") root.append(h("div", { class: "error-box" }, w.error.slice(-800)));
}

const optVal = (bag, o) => { const v = (bag || {})[o.key]; return v === undefined ? o.default : v; };

function optInput(o, val, set) {
  let input;
  if (o.type === "select") {
    input = h("select", { onchange: (ev) => set(isNaN(+ev.target.value) || ev.target.value === "" ? ev.target.value : +ev.target.value) },
      ...o.options.map((op) => { const [v, l] = Array.isArray(op) ? op : [op, op]; return h("option", { value: v, selected: String(v) === String(val ?? "") }, l); }));
  } else if (o.type === "bool") {
    return h("label", { class: "check opt", title: o.help || "" }, h("input", { type: "checkbox", checked: !!val, onchange: (ev) => set(ev.target.checked) }), h("span", {}, o.label));
  } else if (o.type === "textarea") {
    input = h("textarea", { spellcheck: "false", placeholder: o.placeholder || "", oninput: (ev) => set(ev.target.value) }, val ?? "");
  } else {
    input = h("input", { type: o.type === "number" ? "number" : "text", value: val ?? "", placeholder: o.placeholder || "", spellcheck: "false",
      oninput: (ev) => set(o.type === "number" ? +ev.target.value : ev.target.value) });
  }
  return h("label", { class: "opt" }, h("span", {}, o.label), input, o.help ? h("span", { class: "help" }, o.help) : null);
}

/* ───────────────────────── files ───────────────────────── */
async function loadFiles() {
  S.files = await api("/api/files");
  if (!curFile()) S.file = S.files[0]?.id || null;
  renderFiles();
}

function renderFiles() {
  const ul = $("#files");
  ul.innerHTML = "";
  $("#file-count").textContent = S.files.length ? `${S.files.length}` : "";
  for (const f of S.files) {
    const running = S.jobs.some((j) => j.file_id === f.id && active(j));
    const n = S.jobs.filter((j) => j.file_id === f.id && j.status === "done").length;
    ul.append(h("li", { class: f.id === S.file ? "sel" : "", onclick: () => selectFile(f.id) },
      h("img", { src: `/api/files/${f.id}/page/0?w=80`, loading: "lazy", alt: "" }),
      h("div", { class: "meta" },
        h("div", { class: "name", title: f.name }, f.name),
        h("div", { class: "small muted" },
          f.kind === "office" ? `${f.ext.slice(1).toUpperCase()} · ${fmtB(f.size)} ` : `${f.pages} page${f.pages > 1 ? "s" : ""} `,
          f.kind === "pdf" && f.has_text_layer ? h("span", { class: "badge accent", title: "Has an embedded text layer — the converters read it instantly" }, "text layer") : null, " ",
          running ? h("span", { class: "badge warn" }, "running") : n ? h("span", { class: "badge" }, `${n} result${n > 1 ? "s" : ""}`) : null)),
      h("button", { class: "x", title: "Remove file and its results", onclick: async (ev) => {
        ev.stopPropagation();
        if (!confirm(`Remove ${f.name} and its results?`)) return;
        await api(`/api/files/${f.id}`, { method: "DELETE" });
        S.jobs = S.jobs.filter((j) => j.file_id !== f.id);
        if (S.file === f.id) S.file = null;
        await loadFiles(); selectFile(S.file);
      } }, "✕")));
  }
}

async function upload(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const fd = new FormData();
  files.forEach((f, i) => fd.append("files", f, f.name || `pasted-${Date.now()}-${i}.png`));
  toast(`Uploading ${files.length} file${files.length > 1 ? "s" : ""}…`);
  try {
    const metas = await api("/api/files", { method: "POST", body: fd });
    await loadFiles();
    selectFile(metas[metas.length - 1].id);
    toast(`Added ${metas.map((m) => m.name).join(", ")}`);
  } catch (e) { toast(e.message, true); }
}

function selectFile(id) {
  S.file = id; store.set("file", id);
  S.activePage = null;
  renderFiles(); renderEngines(); renderViewer(); renderOutput();
}

/* ───────────────────────── viewer ───────────────────────── */
function renderViewer() {
  const f = curFile();
  const pv = $("#pages-view");
  $("#viewer-title").textContent = f ? f.name : "No file selected";
  $("#open-original").href = f ? `/api/files/${f.id}/original` : "#";
  if (!f) { pv.innerHTML = `<div class="empty"><div class="empty-big">Drop a document to start</div><p class="muted">Pages show here, next to the extracted Markdown.<br>Click a page to jump to its text.</p></div>`; return; }
  pv.innerHTML = "";
  pv.classList.toggle("doc-view", f.kind === "office");
  $$("#zoom-out, #zoom-fit, #zoom-in").forEach((b) => (b.hidden = f.kind === "office"));
  if (f.kind === "office") return renderDocView(f, pv);
  for (let i = 0; i < f.pages; i++) {
    const layer = h("div", { class: "textlayer" });
    const wrap = h("div", { class: "page-wrap", id: `src-p${i}`, "data-page": i,
        onclick: () => { if (window.getSelection().isCollapsed) focusPage(i, "viewer"); },
        ondblclick: (e) => { if (!e.target.closest(".textlayer span")) window.open(`/api/files/${f.id}/page/${i}?dpi=200`); } },
      h("img", { src: `/api/files/${f.id}/page/${i}?dpi=${S.zoom > 1.5 ? 200 : 120}`, loading: "lazy", alt: `page ${i + 1}`,
        onload: (e) => loadTextLayer(f.id, i, e.target, layer) }),
      layer,
      h("span", { class: "pno" }, `${i + 1} / ${f.pages}`));
    pv.append(wrap);
  }
  applyZoom();
}

/* Office / CSV / EPUB…: the server's HTML rendering (Quick Look or pandoc), sandboxed. */
async function renderDocView(f, pv) {
  pv.append(h("p", { class: "muted small center" }, "Rendering…"));
  try {
    const v = await api(`/api/files/${f.id}/view`);
    if (S.file !== f.id) return;
    pv.innerHTML = "";
    pv.append(h("iframe", { class: "doc-frame", title: f.name, src: `/api/files/${f.id}/view/${v.entry}`,
      sandbox: v.scripts ? "allow-scripts" : "", onload: applyZoom }));
    $$("#zoom-out, #zoom-fit, #zoom-in").forEach((b) => (b.hidden = !v.scripts));   // zoom needs the frame's script
  } catch (e) {
    pv.innerHTML = "";
    pv.append(h("div", { class: "error-box" }, `No preview: ${e.message}`));
  }
}

/* Invisible text over a page image, for selecting and copying: the PDF's own text, or Apple Vision
   line boxes for scans. Sizes are in container units, so the layer follows zoom without re-layout. */
const measureCtx = document.createElement("canvas").getContext("2d");
async function loadTextLayer(fid, i, img, layer) {
  if (layer.dataset.loaded) return;
  layer.dataset.loaded = "1";
  let d;
  try { d = await api(`/api/files/${fid}/text/${i}`); } catch { return; }
  const aspect = img.naturalHeight / img.naturalWidth;
  const font = getComputedStyle(layer).fontFamily;
  measureCtx.font = `100px ${font}`;
  const frag = document.createDocumentFragment();
  d.lines.forEach((l, k) => {
    if (k && l.nl) frag.append(h("br"));                   // copied text keeps the line breaks
    const size = l.h * aspect * 100;                       // cqw
    const natural = measureCtx.measureText(l.t).width / 100 * size;
    const sep = d.lines[k + 1] && !d.lines[k + 1].nl ? " " : "";
    frag.append(h("span", { style: `left:${l.x * 100}%;top:${l.y * 100}%;font-size:${size}cqw;` +
      `transform:scaleX(${natural ? (l.w * 100) / natural : 1})` }, l.t + sep));
  });
  layer.title = d.source === "ocr" ? "Text recognised by Apple Vision; select to copy" : "";
  layer.append(frag);
}

function applyZoom() {
  $(".doc-frame")?.contentWindow.postMessage({ zoom: S.zoom }, "*");
  $$(".page-wrap").forEach((w) => (w.style.width = `${Math.round(S.zoom * 100)}%`));
  $("#zoom-label").textContent = S.zoom === 1 ? "fit" : `${Math.round(S.zoom * 100)}%`;
}

function focusPage(i, from) {
  S.activePage = i;
  $$(".page-wrap").forEach((w) => w.classList.toggle("active", +w.dataset.page === i));
  $$(".page-out").forEach((w) => w.classList.toggle("active", +w.dataset.page === i));
  const target = from === "viewer" ? $(`#out-p${i}`) : $(`#src-p${i}`);
  target?.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ───────────────────────── output ───────────────────────── */
function renderTabs() {
  const tabs = $("#job-tabs");
  const js = fileJobs().slice().reverse();
  const cur = curJob();
  const done = js.filter((j) => j.status === "done" && j.total_seconds != null);
  const best = done.length > 1 ? Math.min(...done.map((j) => j.total_seconds)) : null;
  tabs.innerHTML = "";
  for (const j of js) {
    let right;
    if (active(j)) right = h("span", { class: "spin" });
    else if (j.status === "error") right = h("span", { class: "status-err", title: j.error }, "failed");
    else if (j.status === "cancelled") right = h("span", { class: "status-cancel" }, "stopped");
    else right = h("span", { class: "t" + (j.total_seconds === best ? " best" : ""), title: "Total wall time" }, fmtS(j.total_seconds));
    tabs.append(h("div", { class: "tab" + (cur && j.id === cur.id ? " on" : ""), title: `${j.engine_name}${j.model ? " · " + j.model : ""}\n${new Date(j.created * 1000).toLocaleString()}`,
      onclick: () => { S.jobByFile[S.file] = j.id; store.set("jobByFile", S.jobByFile); renderOutput(); } },
      h("span", {}, j.engine_name), right));
  }
}

function renderStats(j) {
  const st = $("#stats");
  if (!j) { st.innerHTML = ""; return; }
  const items = [];
  const add = (v, l, title) => items.push(`<span class="s" title="${esc(title || "")}"><b>${v}</b><span>${l}</span></span>`);
  add(fmtS(j.total_seconds), "total", "Wall-clock time from start to finish, including model load");
  if (j.load_seconds > 0.05) add(fmtS(j.load_seconds), "model load", "Cold start: loading weights into memory. Warm runs skip this.");
  if (!j.whole_doc) add(fmtS(j.seconds_per_page), "/ page", "Average time per page (excludes model load)");
  add(`${j.pages_done}/${j.pages.length}`, "pages");
  if (j.tokens_per_second) add(j.tokens_per_second.toFixed(0), "tok/s", "Generation speed");
  if (j.tokens_est != null) add("≈" + fmtN(j.tokens_est), "LLM tokens", `Rough size of this Markdown in an LLM prompt (~4 UTF-8 bytes/token). The original file is ${fmtB(j.file_size || 0)}.`);
  const peers = fileJobs().filter((x) => x.status === "done" && x.id !== j.id && x.total_seconds);
  if (j.status === "done" && peers.length) {
    const fastest = Math.min(...peers.map((x) => x.total_seconds));
    const r = j.total_seconds / fastest;
    add(r <= 1 ? "fastest" : `${r < 100 ? r.toFixed(1) : Math.round(r)}×`, r <= 1 ? "" : "vs fastest");
  }
  if (active(j)) add(esc(j.stage || j.status), j.current_page != null ? `page ${j.current_page + 1}` : "");
  st.innerHTML = items.join("");
}

let lastSig = "";
async function renderOutput(force = true) {
  renderTabs();
  const j = curJob();
  renderStats(j);
  $$("#view-mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === S.mode));
  $("#diff-against").classList.toggle("hidden", S.mode !== "diff");
  $("#btn-cancel").classList.toggle("hidden", !active(j));
  $$("#download-list .tess-only").forEach((a) => a.classList.toggle("disabled", !j || j.engine_id !== "tesseract"));
  const res = $("#result");
  if (!j) { res.innerHTML = `<div class="empty"><p class="muted">${curFile() ? "Tick engines on the left and press <b>Run</b>." : "Results appear here."}</p></div>`; lastSig = ""; return; }

  let d = S.detail[j.id];
  if (!d || active(j) || d.status !== j.status) d = S.detail[j.id] = await api(`/api/jobs/${j.id}`);
  const sig = `${j.id}:${S.mode}:${S.diffAgainst}:${d.status}:${Object.keys(d.results).length}:${(d.partial || "").length}`;
  if (!force && sig === lastSig) return;
  lastSig = sig;
  const scroll = res.scrollTop;

  if (d.status === "error") {
    res.innerHTML = "";
    res.append(h("div", { class: "error-box" }, d.error || "Failed"));
    if (!Object.keys(d.results).length) return;
  }
  if (S.mode === "preview" || S.mode === "markdown") await renderPages(d, res);
  else if (S.mode === "text" || S.mode === "html") await renderExport(d, res, S.mode === "text" ? "txt" : "html");
  else if (S.mode === "diff") await renderDiff(d, res);
  if (!force) res.scrollTop = scroll;
}

async function renderPages(d, res) {
  const frag = document.createDocumentFragment();
  const pages = Object.keys(d.results).map(Number).sort((a, b) => a - b);
  for (const p of pages) {
    const r = d.results[p];
    const stats = r.stats || {};
    const head = h("div", { class: "page-out-head", onclick: () => focusPage(p, "output") },
      h("b", {}, d.whole_doc ? (d.pages.length > 1 ? `Pages ${d.pages[0] + 1}–${d.pages[d.pages.length - 1] + 1}` : "Document") : `Page ${p + 1}`),
      h("span", {}, fmtS(r.seconds)),
      stats.ocr_calls ? h("span", { title: "Images sent to the OCR model" }, `${stats.ocr_calls} OCR call${stats.ocr_calls > 1 ? "s" : ""}`) : null,
      stats.skipped_images ? h("span", { title: "Images too small to contain text" }, `${stats.skipped_images} tiny images skipped`) : null,
      stats.generation_tokens ? h("span", {}, `${stats.generation_tokens} tok · ${(stats.generation_tps || 0).toFixed(0)} tok/s`) : null,
      stats.prompt_tokens && stats.prompt_tps ? h("span", { title: "Prompt (mostly image) tokens and prefill speed" }, `${stats.prompt_tokens} in · ${stats.prompt_tps.toFixed(0)} tok/s prefill`) : null,
      stats.finish_reason === "length" ? h("span", { class: "warn" }, "hit max tokens") : null,
      stats.finish_reason === "repetition" ? h("span", { class: "warn" }, "stopped: repetition loop") : null);
    const body = h("div", { class: "page-out-body" + (S.mode === "preview" ? " md" : "") });
    if (S.mode === "preview") body.innerHTML = await renderMd(d.id, p, r.markdown);
    else body.append(h("pre", { class: "raw" }, r.markdown));
    frag.append(h("div", { class: "page-out" + (S.activePage === p ? " active" : ""), id: `out-p${p}`, "data-page": p }, head, body));
  }
  if (active(d)) {
    const p = d.current_page;
    const label = d.status === "queued" ? "Queued — waiting for the GPU lane…" : p != null ? `Page ${p + 1} · ${d.stage || "working"}` : d.stage || "Working…";
    frag.append(h("div", { class: "page-out" }, h("div", { class: "page-out-head" }, h("span", { class: "spin" }), h("b", {}, label)),
      h("div", { class: "page-out-body" }, h("pre", { class: "raw streaming" }, d.partial || ""))));
  } else if (!pages.length && d.status !== "error") {
    frag.append(h("div", { class: "empty muted" }, d.status === "cancelled" ? "Stopped." : "No output."));
  }
  res.innerHTML = "";
  res.append(frag);
}

const sanitize = (html) => DOMPurify.sanitize(html, { ADD_TAGS: ["math", "semantics", "annotation", "section"] });

async function renderMd(jid, p, md) {
  const key = `${jid}:${p}`;
  if (!(key in S.rendered)) {
    try { S.rendered[key] = sanitize(await post("/api/render", { markdown: md })); }
    catch { S.rendered[key] = `<pre class="raw">${esc(md)}</pre>`; }
  }
  return S.rendered[key];
}

async function getExport(d, fmt) {
  const key = `${d.id}:${fmt}`;
  if (active(d)) return api(`/api/jobs/${d.id}/export?fmt=${fmt}`);
  if (!(key in S.exports)) S.exports[key] = await api(`/api/jobs/${d.id}/export?fmt=${fmt}`);
  return S.exports[key];
}

async function renderExport(d, res, fmt) {
  const txt = await getExport(d, fmt);
  res.innerHTML = "";
  res.append(h("div", { class: "page-out" }, h("div", { class: "page-out-body" }, h("pre", { class: "raw" }, txt))));
}

function wordDiff(a, b) {
  const dmp = new diff_match_patch();
  const map = new Map(), arr = [""];
  const enc = (s) => s.split(/(\s+)/).filter(Boolean).map((w) => {
    if (!map.has(w)) { map.set(w, arr.length); arr.push(w); }
    return String.fromCharCode(map.get(w));
  }).join("");
  const diffs = dmp.diff_main(enc(a), enc(b), false);
  dmp.diff_cleanupSemantic(diffs);
  let same = 0, total = 0;
  const html = diffs.map(([op, s]) => {
    const words = [...s].map((c) => arr[c.charCodeAt(0)]).join("");
    const n = [...s].filter((c) => arr[c.charCodeAt(0)].trim()).length;
    total += n; if (op === 0) same += n;
    return op === 0 ? esc(words) : op < 0 ? `<del>${esc(words)}</del>` : `<ins>${esc(words)}</ins>`;
  }).join("");
  return { html, similarity: total ? same / total : 1 };
}

async function renderDiff(d, res) {
  const others = fileJobs().filter((x) => x.id !== d.id && x.status === "done");
  const sel = $("#diff-against");
  if (!others.some((o) => o.id === S.diffAgainst)) S.diffAgainst = others[0]?.id || null;
  sel.innerHTML = others.map((o) => `<option value="${o.id}" ${o.id === S.diffAgainst ? "selected" : ""}>vs ${esc(o.engine_name)} (${fmtS(o.total_seconds)})</option>`).join("");
  res.innerHTML = "";
  if (!others.length) { res.append(h("div", { class: "empty muted" }, "Run another engine on this file to diff results.")); return; }
  if (active(d)) { res.append(h("div", { class: "empty muted" }, "Wait for this run to finish.")); return; }
  const other = others.find((o) => o.id === S.diffAgainst);
  const [a, b] = await Promise.all([getExport(other, "txt"), getExport(d, "txt")]);
  const { html, similarity } = wordDiff(a, b);
  res.append(h("div", { class: "page-out" },
    h("div", { class: "page-out-head" }, h("b", {}, `${(similarity * 100).toFixed(1)}% words shared`),
      h("span", {}, h("del", {}, other.engine_name), " → ", h("ins", {}, d.engine_name))),
    h("div", { class: "page-out-body diff" }, h("pre", { class: "raw" }))));
  $("pre.raw", res).innerHTML = html;
}

/* ───────────────────────── run & poll ───────────────────────── */
async function run() {
  const ids = $("#run-all").checked ? S.files.map((f) => f.id) : S.file ? [S.file] : [];
  if (!ids.length) return toast("Add a file first", true);
  const engines = S.engines.filter((e) => S.selected.has(e.id) && e.available)
    .map((e) => ({ id: e.id, opts: Object.fromEntries(e.options.map((o) => [o.key, optVal(S.opts[e.id], o)])) }));
  if (!engines.length) return toast("Tick at least one engine", true);
  try {
    const r = await post("/api/jobs", { file_ids: ids, engines, pages: $("#pages").value });
    if (r.errors.length) toast(`Skipped ${r.errors.length} incompatible combination${r.errors.length > 1 ? "s" : ""}:\n` + r.errors.slice(0, 6).join("\n"), !r.jobs.length);
    const mine = r.jobs.filter((j) => j.file_id === S.file);
    if (mine.length) { S.jobByFile[S.file] = mine[0].id; store.set("jobByFile", S.jobByFile); }
    if (isNarrow() && mine.length) toggleCollapse("output", false);   // show the result that is on its way
    await refreshJobs(true);
  } catch (e) { toast(e.message, true); }
}

async function refreshJobs(force = false) {
  S.jobs = await api("/api/jobs");
  renderFiles();
  await renderOutput(force);
}

let tick = 0;
async function poll() {
  tick++;
  try {
    const anyActive = S.jobs.some(active);
    if (anyActive) await refreshJobs(false);
    if (tick % (anyActive ? 4 : 10) === 0) {
      const d = await api("/api/engines");
      const before = JSON.stringify(S.workers);
      S.workers = d.workers; S.engines = d.engines;
      if (JSON.stringify(S.workers) !== before && !document.activeElement?.closest?.("#engines")) renderEngines();
    }
  } catch (e) { /* server restarting */ }
  setTimeout(poll, S.jobs.some(active) ? 400 : 1000);
}

/* ───────────────────────── publish (Markdown → documents) ───────────────────────── */
const PUB_OPTIONS = [
  { key: "title", label: "Title (if the Markdown has no YAML title)", type: "text", default: "", placeholder: "optional" },
  { key: "toc", label: "Table of contents", type: "bool", default: false },
  { key: "number_sections", label: "Number sections", type: "bool", default: false },
  { key: "hard_breaks", label: "Single newline = line break", type: "bool", default: false, help: "Useful for OCR output" },
  { key: "slide_level", label: "Slide per heading level", type: "select", default: "", only: ["pptx"],
    options: [["", "auto"], [1, "# (H1)"], [2, "## (H2)"], [3, "### (H3)"]] },
];

async function loadFormats() {
  const d = await api("/api/publish/formats");
  S.formats = d.formats; S.templates = d.templates;
  if (!S.formats[S.pubFormat]) S.pubFormat = "docx";
}

function renderFormats() {
  const root = $("#formats");
  root.innerHTML = "";
  for (const [k, f] of Object.entries(S.formats)) {
    root.append(h("div", { class: `fmt${k === S.pubFormat ? " on" : ""}${f.available ? "" : " off"}`,
      onclick: () => { S.pubFormat = k; store.set("pubFormat", k); renderFormats(); refreshPreview(); } },
      h("input", { type: "radio", name: "fmt", checked: k === S.pubFormat }), h("span", {}, f.label), h("span", { class: "ext" }, "." + f.ext)));
    if (f.note && (k === S.pubFormat || !f.available)) root.append(h("div", { class: "fmt-note" }, f.note));
  }
  const o = $("#pub-options");
  o.innerHTML = "";
  const set = (key) => (v) => { S.pubOpts[key] = v; store.set("pubOpts", S.pubOpts); refreshPreview(); };
  for (const opt of PUB_OPTIONS) if (!opt.only || opt.only.includes(S.pubFormat)) o.append(optInput(opt, optVal(S.pubOpts, opt), set(opt.key)));
  const fmt = S.formats[S.pubFormat];
  if (fmt?.template) {
    const mine = S.templates.filter((t) => t.format === S.pubFormat);
    const key = `template_${S.pubFormat}`;
    if (!mine.some((t) => t.id === S.pubOpts[key])) S.pubOpts[key] = "";
    o.append(optInput({ key, label: `Style template (reference .${fmt.ext})`, type: "select",
      options: [["", "pandoc default"], ...mine.map((t) => [t.id, t.name])] }, S.pubOpts[key], set(key)));
    const up = h("input", { type: "file", accept: "." + fmt.ext, hidden: true, onchange: async (ev) => {
      const file = ev.target.files[0]; if (!file) return;
      const fd = new FormData(); fd.append("file", file);
      try {
        const t = await api("/api/templates", { method: "POST", body: fd });
        S.templates.push(t); S.pubOpts[key] = t.id; store.set("pubOpts", S.pubOpts); renderFormats(); toast(`Template ${t.name} added`);
      } catch (e) { toast(e.message, true); }
    } });
    o.append(h("div", { class: "opt-actions" },
      h("label", { class: "btn small", title: "Fonts, colours, layouts and master slides come from this file" }, "Upload template…", up),
      h("a", { class: "btn small", href: `/api/templates/default/${S.pubFormat}`, title: "pandoc's default reference file: restyle it in Word/PowerPoint, then upload" }, "Get blank"),
      S.pubOpts[key] ? h("button", { class: "small", onclick: async () => {
        await api(`/api/templates/${S.pubOpts[key]}`, { method: "DELETE" });
        S.templates = S.templates.filter((t) => t.id !== S.pubOpts[key]); S.pubOpts[key] = ""; renderFormats();
      } }, "Delete") : null));
  }
  $("#btn-export").firstChild.textContent = `Export ${fmt ? fmt.label.replace(/ \(.*/, "") : ""} `;
}

function pubOptions() {
  const o = Object.fromEntries(PUB_OPTIONS.map((x) => [x.key, optVal(S.pubOpts, x)]));
  o.template = S.pubOpts[`template_${S.pubFormat}`] || "";
  return o;
}

function updateMdStatus() {
  const md = $("#md").value;
  const words = (md.match(/\S+/g) || []).length;
  const tables = (md.match(/^\s*\|?\s*:?-{3,}/gm) || []).length;
  $("#md-status").innerHTML = `<span>${words.toLocaleString()} words</span><span>≈${fmtN(estTokens(md))} tokens</span><span>${tables} table${tables === 1 ? "" : "s"}</span><span>autosaved</span>`;
}

let previewTimer, previewSeq = 0;
function refreshPreview() {
  if (S.appMode !== "publish") return;
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    const seq = ++previewSeq;
    const box = $("#pub-preview");
    box.classList.toggle("slides", S.pubFormat === "pptx");
    box.classList.toggle("tables", S.pubFormat === "xlsx");
    $("#preview-hint").textContent = { pptx: "· one card per slide", xlsx: "· highlighted tables become sheets", pdf: "· approximate layout" }[S.pubFormat] || "";
    const md = $("#md").value;
    if (!md.trim()) { $(".doc", box).innerHTML = `<p class="muted">Preview appears here.</p>`; return; }
    try {
      const html = await post("/api/publish/preview", { markdown: md, options: pubOptions() });
      if (seq === previewSeq) $(".doc", box).innerHTML = sanitize(html);
    } catch (e) { if (seq === previewSeq) $(".doc", box).innerHTML = `<div class="error-box">${esc(e.message)}</div>`; }
  }, 250);
}

function docName() {
  const typed = $("#pub-name").value.trim();
  if (typed) return typed;
  const md = $("#md").value;
  const t = md.match(/^title:\s*["']?(.+?)["']?\s*$/m) || md.match(/^#\s+(.+)$/m);
  return t ? t[1].slice(0, 60) : "document";
}

async function exportDoc() {
  const md = $("#md").value;
  if (!md.trim()) return toast("The editor is empty", true);
  const status = $("#pub-status");
  status.textContent = "Converting…";
  try {
    const r = await fetch("/api/publish", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ markdown: md, format: S.pubFormat, options: pubOptions(), filename: docName() }) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const name = r.headers.get("X-Filename"), secs = +r.headers.get("X-Seconds");
    download(await r.blob(), name);
    status.textContent = `${name} · ${fmtS(secs)}`;
  } catch (e) { status.textContent = ""; toast(e.message, true); }
}

function setEditor(md, name) {
  $("#md").value = md;
  if (name !== undefined) $("#pub-name").value = name;
  store.set("md", md);
  updateMdStatus(); refreshPreview();
}

async function sendToPublish() {
  const j = curJob(); if (!j || j.status !== "done") return toast("Pick a finished result first", true);
  const cur = $("#md").value.trim();
  if (cur && !confirm("Replace the Markdown currently in the Publish editor?")) return;
  const md = await getExport(S.detail[j.id] || await api(`/api/jobs/${j.id}`), "md");
  setEditor(md.replace(/<!-- page \d+ -->\n*/g, ""), j.file_name.replace(/\.[^.]+$/, ""));
  setAppMode("publish");
}

const SAMPLE = `---
title: Quarterly report
author: Your Name
---

# Summary

Revenue grew **12%** quarter on quarter. Formulas work too: $E = mc^2$.

| Region | Q1  | Q2  |
|--------|-----|-----|
| EU     | 120 | 135 |
| US     | 340 | 310 |

# Next steps

- Hire two engineers
- Open the APAC office

::: notes
Speaker notes like this go into PowerPoint's notes pane.
:::
`;

/* ───────────────────────── benchmarks ───────────────────────── */
let benchSort = { key: "created", dir: -1 };
function renderBench() {
  const rows = S.jobs.filter((j) => j.status === "done");
  const cols = [
    ["file_name", "File"], ["engine_name", "Engine"], ["pages_done", "Pages"], ["total_seconds", "Total"],
    ["load_seconds", "Load"], ["seconds_per_page", "s / page"], ["tokens_per_second", "tok/s"], ["tokens_est", "≈ LLM tokens"], ["created", "When"],
  ];
  rows.sort((a, b) => ((a[benchSort.key] ?? -1) > (b[benchSort.key] ?? -1) ? 1 : -1) * benchSort.dir);
  const maxSpp = Math.max(...rows.map((r) => r.seconds_per_page || 0), 0.001);
  const t = $("#bench-table");
  t.innerHTML = "";
  t.append(h("tr", {}, ...cols.map(([k, l]) => h("th", { onclick: () => { benchSort = { key: k, dir: benchSort.key === k ? -benchSort.dir : 1 }; renderBench(); } },
    l + (benchSort.key === k ? (benchSort.dir > 0 ? " ▲" : " ▼") : ""))), h("th", {}, "")));
  for (const r of rows) {
    t.append(h("tr", { onclick: () => { $("#bench").close(); setAppMode("extract"); S.jobByFile[r.file_id] = r.id; selectFile(r.file_id); } },
      h("td", {}, r.file_name), h("td", { title: r.model || "" }, r.engine_name), h("td", { class: "num" }, r.pages_done),
      h("td", { class: "num" }, fmtS(r.total_seconds)), h("td", { class: "num" }, r.load_seconds > 0.05 ? fmtS(r.load_seconds) : "warm"),
      h("td", { class: "num" }, fmtS(r.seconds_per_page)), h("td", { class: "num" }, r.tokens_per_second ? r.tokens_per_second.toFixed(0) : "–"),
      h("td", { class: "num" }, r.tokens_est != null ? r.tokens_est.toLocaleString() : "–"), h("td", { class: "small muted" }, new Date(r.created * 1000).toLocaleString()),
      h("td", { style: "width:120px" }, h("div", { class: "bar", style: `width:${Math.max(2, ((r.seconds_per_page || 0) / maxSpp) * 100)}%` }))));
  }
  if (!rows.length) t.append(h("tr", {}, h("td", { colspan: 10, class: "muted" }, "No finished runs yet.")));
}

/* ───────────────────────── wiring ───────────────────────── */
const isMdFile = (f) => /\.(md|markdown|txt|text)$/i.test(f.name);

async function readMdFile(f) { setEditor(await f.text(), f.name.replace(/\.[^.]+$/, "")); toast(`Opened ${f.name}`); }

function wire() {
  $("#mode-switch").addEventListener("click", (e) => { const m = e.target.closest("button")?.dataset.mode; if (m) setAppMode(m); });
  $("#file-input").addEventListener("change", (e) => { upload(e.target.files); e.target.value = ""; });
  let depth = 0;
  window.addEventListener("dragenter", (e) => { e.preventDefault(); depth++; document.body.classList.add("dragging"); });
  window.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; document.body.classList.remove("dragging"); } });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault(); depth = 0; document.body.classList.remove("dragging");
    const files = [...e.dataTransfer.files];
    if (S.appMode === "publish" && files.length === 1 && isMdFile(files[0])) return readMdFile(files[0]);
    if (S.appMode === "publish") setAppMode("extract");
    upload(files);
  });
  window.addEventListener("paste", (e) => {
    if (e.target.closest("input, textarea")) return;
    const files = [...(e.clipboardData?.files || [])];
    if (files.length) { e.preventDefault(); setAppMode("extract"); upload(files); }
  });
  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); S.appMode === "publish" ? exportDoc() : run(); }
    if ((e.metaKey || e.ctrlKey) && e.key === "\\") { e.preventDefault(); toggleCollapse("sidebar"); }
  });
  $("#btn-run").addEventListener("click", run);
  $("#view-mode").addEventListener("click", (e) => {
    const m = e.target.dataset.mode; if (!m) return;
    S.mode = m; store.set("mode", m); renderOutput();
  });
  $("#diff-against").addEventListener("change", (e) => { S.diffAgainst = e.target.value; renderOutput(); });
  $("#zoom-in").addEventListener("click", () => { S.zoom = Math.min(3, S.zoom + 0.25); applyZoom(); });
  $("#zoom-out").addEventListener("click", () => { S.zoom = Math.max(0.25, S.zoom - 0.25); applyZoom(); });
  $("#zoom-fit").addEventListener("click", () => { S.zoom = 1; applyZoom(); });
  $("#btn-cancel").addEventListener("click", async () => { const j = curJob(); if (j) { await post(`/api/jobs/${j.id}/cancel`); refreshJobs(true); } });
  $("#btn-delete-job").addEventListener("click", async () => {
    const j = curJob(); if (!j) return;
    await api(`/api/jobs/${j.id}`, { method: "DELETE" });
    delete S.jobByFile[S.file]; refreshJobs(true);
  });
  $("#btn-copy").addEventListener("click", async () => {
    const j = curJob(); if (!j) return;
    const d = S.detail[j.id] || await api(`/api/jobs/${j.id}`);
    const fmt = S.mode === "text" ? "txt" : S.mode === "html" ? "html" : "md";
    const txt = await getExport(d, fmt);
    await navigator.clipboard.writeText(txt);
    toast(`Copied ${fmt.toUpperCase()} (≈${fmtN(estTokens(txt))} tokens)`);
  });
  $("#btn-to-publish").addEventListener("click", sendToPublish);
  const menu = $(".menu");
  $("#btn-download").addEventListener("click", (e) => { e.stopPropagation(); menu.classList.toggle("open"); });
  document.addEventListener("click", () => menu.classList.remove("open"));
  $("#download-list").addEventListener("click", (e) => {
    const a = e.target.closest("a"); const j = curJob(); if (!a || !j) return;
    if (a.dataset.zip) {
      const ids = fileJobs().filter((x) => x.status === "done").map((x) => x.id).join(",");
      location.href = `/api/export.zip?ids=${ids}&fmt=${a.dataset.zip}`;
    } else location.href = `/api/jobs/${j.id}/export?fmt=${a.dataset.fmt}&download=1`;
  });
  $("#btn-bench").addEventListener("click", () => { renderBench(); $("#bench").showModal(); });
  $("#bench-close").addEventListener("click", () => $("#bench").close());

  // publish
  const ed = $("#md");
  ed.addEventListener("input", () => { store.set("md", ed.value); updateMdStatus(); refreshPreview(); });
  ed.addEventListener("keydown", (e) => {
    if (e.key === "Tab") { e.preventDefault(); document.execCommand("insertText", false, "  "); }
  });
  ed.addEventListener("scroll", () => {   // proportional scroll sync editor → preview
    const p = $("#pub-preview");
    const r = ed.scrollTop / Math.max(1, ed.scrollHeight - ed.clientHeight);
    p.scrollTop = r * (p.scrollHeight - p.clientHeight);
  });
  $("#md-input").addEventListener("change", (e) => { if (e.target.files[0]) readMdFile(e.target.files[0]); e.target.value = ""; });
  $("#btn-md-save").addEventListener("click", () => download(new Blob([ed.value], { type: "text/markdown" }), docName() + ".md"));
  $("#btn-md-sample").addEventListener("click", () => { if (!ed.value.trim() || confirm("Replace the editor content with the example?")) setEditor(SAMPLE, ""); });
  $("#btn-md-clear").addEventListener("click", () => { if (confirm("Clear the editor?")) setEditor("", ""); });
  $("#btn-export").addEventListener("click", exportDoc);
  $("#pub-name").addEventListener("input", () => store.set("pubName", $("#pub-name").value));
}

(async function init() {
  setupPanes();
  let resizeFrame;
  window.addEventListener("resize", () => { cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(applyColumns); });
  wire();
  $("#md").value = store.get("md", "");
  $("#pub-name").value = store.get("pubName", "");
  updateMdStatus();
  await Promise.all([loadEngines(), loadFiles(), loadFormats()]);
  S.jobs = await api("/api/jobs");
  setAppMode(S.appMode);
  selectFile(S.file);
  poll();
})();

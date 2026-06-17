// Vector Graph — embeds the unique words in the text box with a small in-browser
// model (Transformers.js / all-MiniLM-L6-v2), reduces to 3D with PCA, clusters
// with k-means, and draws a rotatable X/Y/Z scatter. Runs live as you type:
// each word's embedding is cached so typing only embeds new words. The model
// loads automatically on first input (one-time ~25 MB download).

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3";
const MODEL = "Xenova/all-MiniLM-L6-v2";
const MAX_WORDS = 200;
const DEBOUNCE_MS = 450;

const CLUSTER_COLORS = [
  "#3fd3a6", "#6ea8ff", "#f0883e", "#d2a8ff",
  "#f7768e", "#e3b341", "#56d364", "#ff7eb6",
];

const cEls = {
  k: document.getElementById("kClusters"),
  status: document.getElementById("clusterStatus"),
  plot: document.getElementById("clusterPlot"),
  input: document.getElementById("input"),
};

let _extractor = null;             // cached model pipeline
let _modelLoading = null;          // in-flight model load promise
const _vecCache = new Map();       // word -> Float64Array(384)

// Latest computed scene, kept so rotation re-projects without recomputing.
let scene = null; // { words, coords3, labels, k }
const view = { yaw: 0.6, pitch: 0.5 };

let debounceTimer = null;
let busy = false;
let pending = false;

function cStatus(msg, kind) {
  cEls.status.textContent = msg;
  cEls.status.className = "cluster-status" + (kind ? " " + kind : "");
}

// --- Text -> unique words -------------------------------------------------
function extractWords(text) {
  const matches = text.toLowerCase().match(/[\p{L}\p{N}]+(?:'[\p{L}]+)?/gu) || [];
  const seen = new Set();
  const out = [];
  for (const w of matches) {
    if (!seen.has(w)) { seen.add(w); out.push(w); }
  }
  return out.slice(0, MAX_WORDS);
}

// --- Model + embeddings (per-word cache) ----------------------------------
async function getExtractor() {
  if (_extractor) return _extractor;
  if (_modelLoading) return _modelLoading;
  _modelLoading = (async () => {
    const { pipeline, env } = await import(TRANSFORMERS_URL);
    env.allowLocalModels = false; // fetch the model from the Hugging Face hub
    _extractor = await pipeline("feature-extraction", MODEL, {
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          const pct = Math.round((p.loaded / p.total) * 100);
          cStatus(`Downloading model… ${pct}% (one-time, then cached)`, "busy");
        }
      },
    });
    return _extractor;
  })();
  return _modelLoading;
}

async function embedWords(words) {
  const missing = words.filter((w) => !_vecCache.has(w));
  if (missing.length) {
    const extractor = await getExtractor();
    const output = await extractor(missing, { pooling: "mean", normalize: true });
    const vecs = output.tolist(); // missing.length x 384
    missing.forEach((w, i) => _vecCache.set(w, Float64Array.from(vecs[i])));
  }
  return words.map((w) => _vecCache.get(w));
}

// --- PCA to K dims (power iteration on the covariance) ---------------------
function pcaK(data, K) {
  const n = data.length, d = data[0].length;
  const mean = new Float64Array(d);
  for (const row of data) for (let j = 0; j < d; j++) mean[j] += row[j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  const X = data.map((row) => Float64Array.from(row, (v, j) => v - mean[j]));

  const normalize = (v) => {
    let s = 0; for (const x of v) s += x * x;
    s = Math.sqrt(s) || 1;
    return Float64Array.from(v, (x) => x / s);
  };
  const Cv = (v) => { // C v = X^T (X v), no explicit covariance matrix
    const Xv = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0; const row = X[i];
      for (let j = 0; j < d; j++) s += row[j] * v[j];
      Xv[i] = s;
    }
    const out = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      const row = X[i], xv = Xv[i];
      for (let j = 0; j < d; j++) out[j] += row[j] * xv;
    }
    return out;
  };
  const powerIter = (deflate) => {
    let v = normalize(Float64Array.from({ length: d }, () => Math.random() - 0.5));
    for (let it = 0; it < 80; it++) {
      let cv = Cv(v);
      for (const u of deflate) {
        let dot = 0; for (let j = 0; j < d; j++) dot += cv[j] * u[j];
        for (let j = 0; j < d; j++) cv[j] -= dot * u[j];
      }
      v = normalize(cv);
    }
    return v;
  };
  const comps = [];
  for (let c = 0; c < K; c++) comps.push(powerIter(comps.slice()));
  return X.map((row) => comps.map((pc) => {
    let s = 0; for (let j = 0; j < d; j++) s += row[j] * pc[j];
    return s;
  }));
}

// --- k-means (k-means++ init) on the full embeddings ----------------------
function dist2(a, b) {
  let s = 0; for (let i = 0; i < a.length; i++) { const x = a[i] - b[i]; s += x * x; }
  return s;
}
function kmeans(data, k, iters = 30) {
  const n = data.length, d = data[0].length;
  k = Math.min(k, n);
  const centers = [data[Math.floor(Math.random() * n)]];
  while (centers.length < k) {
    const dists = data.map((p) => Math.min(...centers.map((c) => dist2(p, c))));
    const sum = dists.reduce((a, b) => a + b, 0) || 1;
    let r = Math.random() * sum, idx = 0;
    for (; idx < n; idx++) { r -= dists[idx]; if (r <= 0) break; }
    centers.push(data[Math.min(idx, n - 1)]);
  }
  const labels = new Array(n).fill(0);
  for (let it = 0; it < iters; it++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < k; c++) {
        const dd = dist2(data[i], centers[c]);
        if (dd < bd) { bd = dd; best = c; }
      }
      if (labels[i] !== best) { labels[i] = best; moved = true; }
    }
    const sums = Array.from({ length: k }, () => new Float64Array(d));
    const cnt = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      const l = labels[i]; cnt[l]++; const row = data[i];
      for (let j = 0; j < d; j++) sums[l][j] += row[j];
    }
    for (let c = 0; c < k; c++) {
      if (cnt[c] > 0) { for (let j = 0; j < d; j++) sums[c][j] /= cnt[c]; centers[c] = Array.from(sums[c]); }
    }
    if (!moved) break;
  }
  return labels;
}

// --- 3D rotation + SVG projection -----------------------------------------
function rotate([x, y, z], yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const x1 = x * cy + z * sy;
  const z1 = -x * sy + z * cy;
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const y2 = y * cp - z1 * sp;
  const z2 = y * sp + z1 * cp;
  return [x1, y2, z2];
}

function svgEl(name, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, val] of Object.entries(attrs)) el.setAttribute(key, val);
  return el;
}

function renderScene() {
  if (!scene) return;
  const { words, coords3, labels, k } = scene;
  const W = 820, H = 580, pad = 56;
  const cx = W / 2, cy = H / 2;

  // Scale so the most distant point fits regardless of rotation.
  let R = 0;
  for (const p of coords3) R = Math.max(R, Math.hypot(p[0], p[1], p[2]));
  R = R || 1;
  const half = Math.min(W, H) / 2 - pad;
  const scale = half / R;
  const proj = (p) => {
    const [rx, ry, rz] = rotate(p, view.yaw, view.pitch);
    return { x: cx + rx * scale, y: cy - ry * scale, z: rz };
  };

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "cluster-svg" });

  // Axes (X, Y, Z) from origin.
  const axes = [
    { v: [R, 0, 0], name: "X", color: "#f7768e" },
    { v: [0, R, 0], name: "Y", color: "#56d364" },
    { v: [0, 0, R], name: "Z", color: "#6ea8ff" },
  ];
  const origin = proj([0, 0, 0]);
  for (const ax of axes) {
    const end = proj(ax.v);
    svg.appendChild(svgEl("line", {
      x1: origin.x, y1: origin.y, x2: end.x, y2: end.y,
      stroke: ax.color, "stroke-width": "1.5", "stroke-opacity": "0.55",
    }));
    const lab = svgEl("text", {
      x: end.x, y: end.y, fill: ax.color, "font-size": "13", "font-weight": "700",
    });
    lab.textContent = ax.name;
    svg.appendChild(lab);
  }

  // Points: draw far-to-near so nearer dots sit on top.
  const pts = coords3.map((p, i) => ({ ...proj(p), i }));
  pts.sort((a, b) => a.z - b.z);
  const minZ = pts[0].z, maxZ = pts[pts.length - 1].z;
  for (const pt of pts) {
    const i = pt.i;
    const color = CLUSTER_COLORS[labels[i] % CLUSTER_COLORS.length];
    const depth = (pt.z - minZ) / ((maxZ - minZ) || 1); // 0 far, 1 near
    const r = 3.5 + depth * 3;
    const g = svgEl("g", { class: "pt" });
    const dot = svgEl("circle", { cx: pt.x, cy: pt.y, r, fill: color, "fill-opacity": (0.5 + depth * 0.5).toFixed(2) });
    const title = svgEl("title", {});
    title.textContent = `${words[i]} — cluster ${labels[i] + 1}`;
    dot.appendChild(title);
    const label = svgEl("text", { x: pt.x + r + 2, y: pt.y + 4, fill: "#c9d1d9", "font-size": "11", "fill-opacity": (0.45 + depth * 0.55).toFixed(2) });
    label.textContent = words[i];
    g.append(dot, label);
    svg.appendChild(g);
  }

  // Legend.
  for (let c = 0; c < k; c++) {
    const ly = pad + c * 18;
    svg.appendChild(svgEl("circle", { cx: W - pad - 80, cy: ly, r: 5, fill: CLUSTER_COLORS[c % CLUSTER_COLORS.length] }));
    const t = svgEl("text", { x: W - pad - 68, y: ly + 4, fill: "#8b949e", "font-size": "11" });
    t.textContent = `Cluster ${c + 1}`;
    svg.appendChild(t);
  }

  enableRotate(svg);
  cEls.plot.replaceChildren(svg);
}

// --- Drag to rotate (re-projects cached 3D coords, no recompute) -----------
function enableRotate(svg) {
  let dragging = false, lastX = 0, lastY = 0;
  svg.style.cursor = "grab";
  svg.addEventListener("pointerdown", (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    svg.setPointerCapture(e.pointerId); svg.style.cursor = "grabbing";
  });
  svg.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    view.yaw += (e.clientX - lastX) * 0.01;
    view.pitch += (e.clientY - lastY) * 0.01;
    view.pitch = Math.max(-1.4, Math.min(1.4, view.pitch));
    lastX = e.clientX; lastY = e.clientY;
    renderScene();
  });
  const stop = () => { dragging = false; svg.style.cursor = "grab"; };
  svg.addEventListener("pointerup", stop);
  svg.addEventListener("pointercancel", stop);
}

// --- Live orchestration ---------------------------------------------------
async function recompute() {
  if (busy) { pending = true; return; }
  busy = true;
  try {
    const words = extractWords(cEls.input.value);
    if (words.length < 4) {
      scene = null;
      cEls.plot.replaceChildren();
      cStatus("Type at least 4 different words to build the vector graph.", null);
      return;
    }
    const vecs = await embedWords(words);
    cStatus("Projecting & clustering…", "busy");
    await new Promise((r) => setTimeout(r, 0)); // let status paint
    const coords3 = pcaK(vecs, 3);
    const k = Math.min(parseInt(cEls.k.value, 10), words.length);
    const labels = kmeans(vecs, k);
    scene = { words, coords3, labels, k };
    renderScene();
    cStatus(`${words.length} words · ${k} clusters · drag to rotate`, "ready");
  } catch (err) {
    console.error(err);
    cStatus("Vector graph error: " + err.message, "error");
  } finally {
    busy = false;
    if (pending) { pending = false; recompute(); }
  }
}

function schedule() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(recompute, DEBOUNCE_MS);
}

cEls.input.addEventListener("input", schedule);
cEls.k.addEventListener("change", recompute);
cStatus("Start typing — the vector graph builds live (first use downloads a ~25 MB model once).", null);

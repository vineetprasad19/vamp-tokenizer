// Word cluster map — embeds unique words with a small in-browser model
// (Transformers.js / all-MiniLM-L6-v2), reduces to 2D with PCA, clusters with
// k-means, and draws an interactive scatter plot. Loaded lazily on click so the
// ~25 MB model only downloads if the user actually wants the graph.

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3";
const MODEL = "Xenova/all-MiniLM-L6-v2";
const MAX_WORDS = 200;

const CLUSTER_COLORS = [
  "#3fd3a6", "#6ea8ff", "#f0883e", "#d2a8ff",
  "#f7768e", "#e3b341", "#56d364", "#ff7eb6",
];

const cEls = {
  btn: document.getElementById("genCluster"),
  k: document.getElementById("kClusters"),
  status: document.getElementById("clusterStatus"),
  plot: document.getElementById("clusterPlot"),
  input: document.getElementById("input"),
};

let _extractor = null;          // cached model pipeline
let _cache = { words: null, vecs: null }; // cache embeddings for the current word set

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

// --- Model ----------------------------------------------------------------
async function getExtractor() {
  if (_extractor) return _extractor;
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false; // fetch the model from the Hugging Face hub
  _extractor = await pipeline("feature-extraction", MODEL, {
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        const pct = Math.round((p.loaded / p.total) * 100);
        cStatus(`Downloading model… ${pct}% (one-time, then cached)`, "busy");
      } else if (p.status === "ready" || p.status === "done") {
        cStatus("Model ready — embedding your words…", "busy");
      }
    },
  });
  return _extractor;
}

async function embed(words) {
  if (_cache.words && sameWords(_cache.words, words)) return _cache.vecs;
  const extractor = await getExtractor();
  cStatus("Embedding your words…", "busy");
  const output = await extractor(words, { pooling: "mean", normalize: true });
  const vecs = output.tolist(); // N x 384
  _cache = { words: words.slice(), vecs };
  return vecs;
}

function sameWords(a, b) {
  return a.length === b.length && a.every((w, i) => w === b[i]);
}

// --- PCA to 2D (power iteration on the covariance, no full matrix) ---------
function pca2(data) {
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
  // C v = X^T (X v), computed without materializing the 384x384 covariance.
  const Cv = (v) => {
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
      for (const u of deflate) { // keep orthogonal to earlier components
        let dot = 0; for (let j = 0; j < d; j++) dot += cv[j] * u[j];
        for (let j = 0; j < d; j++) cv[j] -= dot * u[j];
      }
      v = normalize(cv);
    }
    return v;
  };
  const pc1 = powerIter([]);
  const pc2 = powerIter([pc1]);
  return X.map((row) => {
    let a = 0, b = 0;
    for (let j = 0; j < d; j++) { a += row[j] * pc1[j]; b += row[j] * pc2[j]; }
    return [a, b];
  });
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

// --- SVG scatter ----------------------------------------------------------
function svgEl(name, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, val] of Object.entries(attrs)) el.setAttribute(key, val);
  return el;
}

function renderPlot(words, coords, labels, k) {
  const W = 800, H = 520, pad = 48;
  const xs = coords.map((c) => c[0]), ys = coords.map((c) => c[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (x) => pad + ((x - minX) / ((maxX - minX) || 1)) * (W - 2 * pad);
  const sy = (y) => H - pad - ((y - minY) / ((maxY - minY) || 1)) * (H - 2 * pad);

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "cluster-svg", role: "img" });

  for (let i = 0; i < words.length; i++) {
    const color = CLUSTER_COLORS[labels[i] % CLUSTER_COLORS.length];
    const cx = sx(coords[i][0]), cy = sy(coords[i][1]);
    const g = svgEl("g", { class: "pt" });
    const dot = svgEl("circle", { cx, cy, r: 5, fill: color, "fill-opacity": "0.85" });
    const title = svgEl("title", {});
    title.textContent = `${words[i]} — cluster ${labels[i] + 1}`;
    dot.appendChild(title);
    const label = svgEl("text", { x: cx + 7, y: cy + 4, fill: "#c9d1d9", "font-size": "11" });
    label.textContent = words[i];
    g.append(dot, label);
    svg.appendChild(g);
  }

  // simple legend
  const legend = svgEl("g", {});
  for (let c = 0; c < k; c++) {
    const ly = pad + c * 18;
    legend.appendChild(svgEl("circle", { cx: W - pad - 90, cy: ly, r: 5, fill: CLUSTER_COLORS[c % CLUSTER_COLORS.length] }));
    const t = svgEl("text", { x: W - pad - 78, y: ly + 4, fill: "#8b949e", "font-size": "11" });
    t.textContent = `Cluster ${c + 1}`;
    legend.appendChild(t);
  }
  svg.appendChild(legend);

  cEls.plot.replaceChildren(svg);
}

// --- Orchestration --------------------------------------------------------
async function generate() {
  const words = extractWords(cEls.input.value);
  if (words.length < 3) {
    cStatus("Add at least 3 different words in the text box to map them.", "error");
    cEls.plot.replaceChildren();
    return;
  }
  cEls.btn.disabled = true;
  cEls.k.disabled = true;
  try {
    const vecs = await embed(words);
    cStatus("Projecting & clustering…", "busy");
    // Yield so the status paints before the synchronous math.
    await new Promise((r) => setTimeout(r, 0));
    const coords = pca2(vecs);
    const k = parseInt(cEls.k.value, 10);
    const labels = kmeans(vecs, k);
    renderPlot(words, coords, labels, Math.min(k, words.length));
    cStatus(`Mapped ${words.length} unique words into ${Math.min(k, words.length)} clusters.`, "ready");
  } catch (err) {
    console.error(err);
    cStatus("Could not build the cluster map: " + err.message, "error");
  } finally {
    cEls.btn.disabled = false;
    cEls.k.disabled = false;
  }
}

cEls.btn.addEventListener("click", generate);

// Attention arcs — visualizes self-attention as curved links between words.
//
// We reuse the shared in-browser MiniLM model (window.Embeddings) to get a
// contextual vector per token, then compute a *real* single-head self-attention:
// for each word i, score against every word j with the dot product of their
// (L2-normalized) vectors, scale by a temperature, and softmax. Clicking a word
// draws arcs to the words it attends to (thicker / brighter = stronger focus).
//
// This is deliberately simplified: real transformers learn separate Query / Key /
// Value projections for each of many heads. Here the word vectors play the role of
// queries and keys directly, so the core idea — every word looks at every word and
// decides how much to focus — is visible without any extra download.

const SVG_NS = "http://www.w3.org/2000/svg";
const ATTN_TOPN = 6;    // max arcs drawn per word
const ATTN_MIN = 0.04;  // ignore very weak links

const aEls = {
  input: document.getElementById("attnInput"),
  btn: document.getElementById("attnBtn"),
  scale: document.getElementById("attnScale"),
  scaleVal: document.getElementById("attnScaleVal"),
  causal: document.getElementById("attnCausal"),
  status: document.getElementById("attnStatus"),
  canvas: document.getElementById("attnCanvas"),
  svg: document.getElementById("attnArcs"),
  words: document.getElementById("attnWords"),
  legend: document.getElementById("attnLegend"),
};

let aBusy = false;
let _aTokens = null;    // array of display strings
let _aVecs = null;      // normalized contextual vector per token (cached)
let _aWeights = null;   // N×N attention rows (Float64Array per row)
let _aSelected = null;  // index of the currently selected word

// Slider value: higher = sharper softmax (attention concentrates on fewer words).
function attnScale() { return Math.max(1, parseFloat(aEls.scale.value) || 8); }

function aStatus(msg, kind) {
  aEls.status.textContent = msg;
  aEls.status.className = "cluster-status" + (kind ? " " + kind : "");
}

// Softmax over a row; -Infinity entries (masked) map to 0.
function softmaxRow(scores) {
  let max = -Infinity;
  for (const v of scores) if (v > max) max = v;
  const out = new Float64Array(scores.length);
  let sum = 0;
  for (let i = 0; i < scores.length; i++) {
    const e = scores[i] === -Infinity ? 0 : Math.exp(scores[i] - max);
    out[i] = e; sum += e;
  }
  if (sum > 0) for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

function normalize(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

// Run the model on the current text and cache normalized per-token vectors.
async function embedTokens() {
  const text = aEls.input.value.trim();
  if (!text) { aStatus("Type a sentence first.", "error"); return false; }

  const ex = await Embeddings.getExtractor((p) =>
    aStatus(`Downloading model… ${p}% (one-time, then cached)`, "busy"));
  aStatus("Computing attention…", "busy");

  // Token IDs (includes special tokens like [CLS]/[SEP]) aligned with the vectors.
  const enc = await ex.tokenizer(text);
  const ids = Array.from(enc.input_ids.data).map(Number);
  const special = new Set((ex.tokenizer.all_special_ids || []).map(Number));

  const out = await ex(text, { pooling: "none" }); // [1, seq, dim]
  const rows = out.tolist()[0];

  // Keep only real (non-special) tokens; normalize each vector for cosine scores.
  const tokens = [], vecs = [];
  for (let i = 0; i < ids.length; i++) {
    if (special.has(ids[i])) continue;
    let t = ex.tokenizer.decode([ids[i]]).trim().replace(/^##/, "");
    tokens.push(t || "·");
    vecs.push(normalize(rows[i]));
  }

  _aTokens = tokens;
  _aVecs = vecs;
  aStatus("", "hidden");
  return true;
}

// Rebuild the attention matrix from cached vectors using the current Focus scale
// and causal toggle. Instant — no model re-run, so the slider feels live.
function recomputeWeights() {
  if (!_aVecs) return;
  const N = _aVecs.length;
  const scale = attnScale();
  const W = [];
  for (let i = 0; i < N; i++) {
    const scores = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      if (aEls.causal.checked && j > i) { scores[j] = -Infinity; continue; }
      let s = 0; const vi = _aVecs[i], vj = _aVecs[j];
      for (let k = 0; k < vi.length; k++) s += vi[k] * vj[k];
      scores[j] = s * scale;
    }
    W.push(softmaxRow(scores));
  }
  _aWeights = W;
}

function renderWords() {
  aEls.words.replaceChildren();
  _aTokens.forEach((t, i) => {
    const w = document.createElement("span");
    w.className = "attn-word";
    w.textContent = t;
    w.dataset.idx = String(i);
    w.addEventListener("click", () => selectWord(i));
    aEls.words.appendChild(w);
  });
}

function selectWord(i) {
  _aSelected = i;
  drawArcs();
}

function drawArcs() {
  const svg = aEls.svg;
  svg.replaceChildren();

  const w = aEls.canvas.offsetWidth;
  const h = aEls.canvas.offsetHeight;
  svg.setAttribute("width", w);
  svg.setAttribute("height", h);
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);

  const chips = Array.from(aEls.words.children);
  chips.forEach((c) => {
    c.classList.remove("attn-source", "attn-target");
    c.style.removeProperty("--w");
  });
  if (_aSelected == null || !_aWeights) return;

  const base = aEls.canvas.getBoundingClientRect();
  const centerX = (el) => {
    const r = el.getBoundingClientRect();
    return r.left - base.left + r.width / 2;
  };
  const topY = (el) => el.getBoundingClientRect().top - base.top;

  const i = _aSelected;
  const row = _aWeights[i];
  const targets = Array.from(row.keys())
    .filter((j) => j !== i && isFinite(row[j]) && row[j] >= ATTN_MIN)
    .sort((a, b) => row[b] - row[a])
    .slice(0, ATTN_TOPN);

  chips[i].classList.add("attn-source");
  if (!targets.length) return;

  const max = Math.max(...targets.map((j) => row[j]));
  const x0 = centerX(chips[i]);
  const baseY = topY(chips[i]); // arcs spring from the top edge of the words

  for (const j of targets) {
    const norm = row[j] / max;            // 0..1 relative strength
    const x1 = centerX(chips[j]);
    // Bow height grows with distance but stays inside the reserved top band.
    const peak = Math.max(6, baseY - 18 - Math.min(baseY - 10, Math.abs(x1 - x0) * 0.42));

    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", `M ${x0} ${baseY} Q ${(x0 + x1) / 2} ${peak} ${x1} ${baseY}`);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "#3fd3a6");
    path.setAttribute("stroke-width", (1.2 + norm * 6).toFixed(2));
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("opacity", (0.25 + 0.65 * norm).toFixed(2));
    svg.appendChild(path);

    chips[j].classList.add("attn-target");
    chips[j].style.setProperty("--w", norm.toFixed(3));
  }
}

async function runAttention() {
  if (aBusy) return;
  aBusy = true; aEls.btn.disabled = true;
  try {
    aStatus("Loading model…", "busy");
    const ok = await embedTokens();
    if (!ok) return;
    recomputeWeights();
    renderWords();
    aEls.legend.classList.remove("hidden");
    // Auto-select a fun default: "it" if present, else the last word.
    const guess = _aTokens.findIndex((t) => t.toLowerCase() === "it");
    selectWord(guess >= 0 ? guess : _aTokens.length - 1);
  } catch (err) {
    console.error(err);
    aStatus("Error: " + err.message, "error");
  } finally {
    aBusy = false; aEls.btn.disabled = false;
  }
}

aEls.btn.addEventListener("click", runAttention);
// The Focus slider and causal toggle re-filter cached vectors instantly.
aEls.scale.addEventListener("input", () => {
  aEls.scaleVal.textContent = attnScale().toFixed(1) + "×";
  if (_aVecs) { recomputeWeights(); drawArcs(); }
});
aEls.causal.addEventListener("change", () => {
  if (_aVecs) { recomputeWeights(); drawArcs(); }
});
// Editing the text invalidates the current arcs.
aEls.input.addEventListener("input", () => {
  _aVecs = null; _aWeights = null; _aSelected = null;
  aEls.words.replaceChildren();
  aEls.svg.replaceChildren();
  aEls.legend.classList.add("hidden");
});
// Keep arcs aligned with the words when the layout changes.
window.addEventListener("resize", () => { if (_aSelected != null) drawArcs(); });

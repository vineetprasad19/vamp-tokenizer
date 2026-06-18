// Next-token predictor — runs distilGPT-2 in the browser and shows the top
// predicted next tokens with probability bars. A temperature slider reshapes the
// distribution (low = sharp/safe, high = flat/creative); "Sample" draws a token
// from that distribution and appends it. Clicking a bar appends that token.

const PREDICT_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3";
const PREDICT_MODEL = "Xenova/distilgpt2";
const TOP_K = 10;

const pEls = {
  input: document.getElementById("predictInput"),
  btn: document.getElementById("predictBtn"),
  sampleBtn: document.getElementById("sampleBtn"),
  temp: document.getElementById("temp"),
  tempVal: document.getElementById("tempVal"),
  topk: document.getElementById("topk"),
  topkVal: document.getElementById("topkVal"),
  topp: document.getElementById("topp"),
  toppVal: document.getElementById("toppVal"),
  pool: document.getElementById("predictPool"),
  status: document.getElementById("predictStatus"),
  result: document.getElementById("predictResult"),
};

let _tok = null;
let _model = null;
let _loading = null;
let pBusy = false;
let _lastLogits = null; // raw logits for the last position, cached for the slider

function pStatus(msg, kind) {
  pEls.status.textContent = msg;
  pEls.status.className = "cluster-status" + (kind ? " " + kind : "");
}

function temperature() {
  return Math.max(0.1, parseFloat(pEls.temp.value) || 1);
}
function getTopK() { return parseInt(pEls.topk.value, 10) || 0; }   // 0 = off
function getTopP() { return parseFloat(pEls.topp.value) || 1; }     // 1 = off

async function loadModel() {
  if (_model) return;
  const { tok, model } = await GPT2.load((pct) =>
    pStatus(`Downloading model… ${pct}% (one-time, then cached)`, "busy"));
  _tok = tok;
  _model = model;
}

// Temperature-scaled softmax probabilities (Float64Array) over all logits.
function softmaxT(logits, temp) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) { const v = logits[i] / temp; if (v > max) max = v; }
  const probs = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { const e = Math.exp(logits[i] / temp - max); probs[i] = e; sum += e; }
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  return probs;
}

function topK(probs, k) {
  const idx = Array.from({ length: probs.length }, (_, i) => i);
  idx.sort((a, b) => probs[b] - probs[a]);
  return idx.slice(0, Math.min(k, idx.length)).map((j) => ({ index: j, prob: probs[j] }));
}

// Apply temperature, then top-k and top-p (nucleus) filtering. Returns the
// renormalized probability distribution and how many tokens survived.
function filteredProbs(logits, temp, topk, topp) {
  const probs = softmaxT(logits, temp);
  const idx = Array.from({ length: probs.length }, (_, i) => i)
    .sort((a, b) => probs[b] - probs[a]);
  const kLimit = topk > 0 ? topk : idx.length;
  const keep = [];
  let cum = 0;
  for (let i = 0; i < idx.length && keep.length < kLimit; i++) {
    keep.push(idx[i]);
    cum += probs[idx[i]];
    if (topp < 1 && cum >= topp) break; // nucleus: stop once we've covered p
  }
  const out = new Float64Array(probs.length);
  let sum = 0;
  for (const j of keep) sum += probs[j];
  for (const j of keep) out[j] = probs[j] / (sum || 1);
  return { probs: out, poolSize: keep.length };
}

function sampleIndex(probs) {
  let r = Math.random(), acc = 0;
  for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (acc >= r) return i; }
  return probs.length - 1;
}

// Run the model on the current text and cache the last-position logits.
async function computeLogits() {
  await loadModel();
  pStatus("Predicting…", "busy");
  const inputs = await _tok(pEls.input.value);
  const output = await _model(inputs);
  const logitsT = output.logits;            // dims [1, seq, vocab]
  const dims = logitsT.dims;
  const vocab = dims[dims.length - 1];
  const seq = dims[dims.length - 2];
  const offset = (seq - 1) * vocab;
  _lastLogits = Float64Array.from(logitsT.data.subarray(offset, offset + vocab));
  pStatus("", "hidden");
}

function renderFromCache() {
  if (!_lastLogits) return;
  const { probs, poolSize } = filteredProbs(_lastLogits, temperature(), getTopK(), getTopP());
  pEls.pool.textContent =
    `Candidate pool: ${poolSize.toLocaleString()} of 50,257 tokens can be chosen` +
    (getTopK() || getTopP() < 1 ? " (trimmed by top-k / top-p)" : "");
  renderPredictions(topK(probs, TOP_K).filter((t) => t.prob > 0));
}

async function predict() {
  if (!pEls.input.value) { pStatus("Type some text first.", "error"); return; }
  if (pBusy) return;
  pBusy = true; pEls.btn.disabled = true; pEls.sampleBtn.disabled = true;
  try {
    pStatus("Loading model…", "busy");
    await computeLogits();
    renderFromCache();
  } catch (err) {
    console.error(err);
    pStatus("Error: " + err.message, "error");
  } finally {
    pBusy = false; pEls.btn.disabled = false; pEls.sampleBtn.disabled = false;
  }
}

async function sampleNext() {
  if (!pEls.input.value) { pStatus("Type some text first.", "error"); return; }
  if (pBusy) return;
  pBusy = true; pEls.btn.disabled = true; pEls.sampleBtn.disabled = true;
  try {
    pStatus("Loading model…", "busy");
    await computeLogits();
    const { probs } = filteredProbs(_lastLogits, temperature(), getTopK(), getTopP());
    const idx = sampleIndex(probs);
    pEls.input.value += _tok.decode([idx]);
    _lastLogits = null; // text changed; recompute on next action
    await computeLogits();
    renderFromCache();
  } catch (err) {
    console.error(err);
    pStatus("Error: " + err.message, "error");
  } finally {
    pBusy = false; pEls.btn.disabled = false; pEls.sampleBtn.disabled = false;
  }
}

function renderPredictions(top) {
  pEls.result.replaceChildren();
  for (const t of top) {
    const tokenText = _tok.decode([t.index]);
    const row = document.createElement("button");
    row.className = "pred-row";
    row.type = "button";
    row.title = "Click to append this token and predict again";

    const label = document.createElement("span");
    label.className = "pred-tok";
    label.textContent = JSON.stringify(tokenText).slice(1, -1); // reveal whitespace

    const bar = document.createElement("span");
    bar.className = "pred-bar";
    const fill = document.createElement("span");
    fill.className = "pred-fill";
    fill.style.width = (t.prob * 100).toFixed(1) + "%";
    bar.appendChild(fill);

    const pct = document.createElement("span");
    pct.className = "pred-pct";
    pct.textContent = (t.prob * 100).toFixed(1) + "%";

    row.append(label, bar, pct);
    row.addEventListener("click", async () => {
      pEls.input.value += tokenText;
      _lastLogits = null;
      await predict();
    });
    pEls.result.appendChild(row);
  }
}

pEls.btn.addEventListener("click", predict);
pEls.sampleBtn.addEventListener("click", sampleNext);

// All three sliders re-filter the cached logits instantly (no model re-run).
pEls.temp.addEventListener("input", () => {
  pEls.tempVal.textContent = parseFloat(pEls.temp.value).toFixed(1);
  renderFromCache();
});
pEls.topk.addEventListener("input", () => {
  pEls.topkVal.textContent = getTopK() === 0 ? "off" : String(getTopK());
  renderFromCache();
});
pEls.topp.addEventListener("input", () => {
  pEls.toppVal.textContent = getTopP() >= 1 ? "1.00" : getTopP().toFixed(2);
  renderFromCache();
});
// Re-running prediction is needed if the text is edited by hand.
pEls.input.addEventListener("input", () => { _lastLogits = null; });

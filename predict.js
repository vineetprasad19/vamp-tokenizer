// Next-token predictor — runs distilGPT-2 in the browser and shows the top
// predicted next tokens with probability bars. Click a prediction to append it
// and predict again. This is the core idea of how LLMs generate text.

const PREDICT_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3";
const PREDICT_MODEL = "Xenova/distilgpt2";
const TOP_K = 10;

const pEls = {
  input: document.getElementById("predictInput"),
  btn: document.getElementById("predictBtn"),
  status: document.getElementById("predictStatus"),
  result: document.getElementById("predictResult"),
};

let _tok = null;
let _model = null;
let _loading = null;
let pBusy = false;

function pStatus(msg, kind) {
  pEls.status.textContent = msg;
  pEls.status.className = "cluster-status" + (kind ? " " + kind : "");
}

async function loadModel() {
  if (_model) return;
  if (_loading) return _loading;
  _loading = (async () => {
    const { AutoTokenizer, AutoModelForCausalLM, env } = await import(PREDICT_URL);
    env.allowLocalModels = false;
    const opts = {
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          pStatus(`Downloading model… ${Math.round((p.loaded / p.total) * 100)}% (one-time, then cached)`, "busy");
        }
      },
    };
    _tok = await AutoTokenizer.from_pretrained(PREDICT_MODEL, opts);
    _model = await AutoModelForCausalLM.from_pretrained(PREDICT_MODEL, opts);
  })();
  return _loading;
}

// Softmax over all logits, return the top-k {index, prob}.
function softmaxTopK(logits, k) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const exps = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { const e = Math.exp(logits[i] - max); exps[i] = e; sum += e; }
  const idx = Array.from({ length: logits.length }, (_, i) => i);
  idx.sort((a, b) => exps[b] - exps[a]);
  const out = [];
  for (let i = 0; i < Math.min(k, idx.length); i++) {
    const j = idx[i];
    out.push({ index: j, prob: exps[j] / sum });
  }
  return out;
}

async function predict() {
  const text = pEls.input.value;
  if (!text) { pStatus("Type some text first.", "error"); return; }
  if (pBusy) return;
  pBusy = true;
  pEls.btn.disabled = true;
  try {
    pStatus("Loading model…", "busy");
    await loadModel();
    pStatus("Predicting…", "busy");
    const inputs = await _tok(text);
    const output = await _model(inputs);
    const logitsT = output.logits;          // dims [1, seq, vocab]
    const dims = logitsT.dims;
    const vocab = dims[dims.length - 1];
    const seq = dims[dims.length - 2];
    const offset = (seq - 1) * vocab;        // logits for the LAST position
    const last = logitsT.data.subarray(offset, offset + vocab);
    renderPredictions(softmaxTopK(last, TOP_K));
    pStatus("", "hidden");
  } catch (err) {
    console.error(err);
    pStatus("Error: " + err.message, "error");
  } finally {
    pBusy = false;
    pEls.btn.disabled = false;
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
    // Reveal leading spaces/newlines by quoting the raw token text.
    label.textContent = JSON.stringify(tokenText).slice(1, -1);

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
    row.addEventListener("click", () => { pEls.input.value += tokenText; predict(); });
    pEls.result.appendChild(row);
  }
}

pEls.btn.addEventListener("click", predict);

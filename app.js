// Token Counter — runs OpenAI's tiktoken in the browser via Pyodide.

const MODELS = [
  { id: "o200k_base", label: "GPT-4o · GPT-4.1 · o1 / o3 / o4-mini" },
  { id: "cl100k_base", label: "GPT-4 · GPT-3.5 Turbo · text-embedding-3" },
  { id: "p50k_base", label: "Codex · text-davinci-002 / 003" },
  { id: "r50k_base", label: "GPT-3 (davinci, curie, babbage, ada)" },
];

const els = {
  model: document.getElementById("model"),
  input: document.getElementById("input"),
  tokenCount: document.getElementById("tokenCount"),
  charCount: document.getElementById("charCount"),
  wordCount: document.getElementById("wordCount"),
  tokens: document.getElementById("tokens"),
  ids: document.getElementById("ids"),
  status: document.getElementById("status"),
};

let pyodide = null;
let pyInstallVocab = null;
let pyTokenize = null;
const loadedEncodings = new Set();
let debounceTimer = null;

function setStatus(msg, kind) {
  els.status.textContent = msg;
  els.status.className = "status" + (kind ? " " + kind : "");
}

// Populate the model dropdown.
for (const m of MODELS) {
  const opt = document.createElement("option");
  opt.value = m.id;
  opt.textContent = m.label;
  els.model.appendChild(opt);
}

async function boot() {
  try {
    setStatus("Loading Python runtime… (first load downloads ~10 MB, then it's cached)");
    pyodide = await loadPyodide();

    setStatus("Loading the tiktoken tokenizer…");
    await pyodide.loadPackage(["tiktoken", "regex"]);

    const pySource = await (await fetch("tokenizer.py")).text();
    await pyodide.runPythonAsync(pySource);
    pyInstallVocab = pyodide.globals.get("install_vocab");
    pyTokenize = pyodide.globals.get("tokenize");

    await ensureEncoding(MODELS[0].id);

    // Enable the UI.
    els.model.disabled = false;
    els.input.disabled = false;
    setStatus("Ready — running entirely in your browser.", "ready");
    setTimeout(() => els.status.classList.add("hidden"), 2500);

    render();
    els.input.focus();
  } catch (err) {
    console.error(err);
    setStatus("Failed to load the tokenizer: " + err.message, "error");
  }
}

// Fetch a bundled .tiktoken vocab file and hand it to Python (once per encoding).
async function ensureEncoding(name) {
  if (loadedEncodings.has(name)) return;
  const buf = await (await fetch(`encodings/${name}.tiktoken`)).arrayBuffer();
  pyInstallVocab(name, new Uint8Array(buf));
  loadedEncodings.add(name);
}

function countWords(text) {
  const m = text.trim().match(/\S+/g);
  return m ? m.length : 0;
}

function renderTokens(pieces) {
  els.tokens.replaceChildren();
  const frag = document.createDocumentFragment();
  for (const piece of pieces) {
    const span = document.createElement("span");
    span.className = "tok" + (/^\s+$/.test(piece) ? " ws" : "");
    // Show newlines/tabs as themselves inside pre-wrap; keep raw text otherwise.
    span.textContent = piece;
    frag.appendChild(span);
  }
  els.tokens.appendChild(frag);
}

async function render() {
  const text = els.input.value;

  // Character / word counts are pure JS — instant.
  els.charCount.textContent = text.length.toLocaleString();
  els.wordCount.textContent = countWords(text).toLocaleString();

  if (!pyTokenize) return;

  const name = els.model.value;
  await ensureEncoding(name);

  let result;
  try {
    result = JSON.parse(pyTokenize(name, text));
  } catch (err) {
    console.error(err);
    setStatus("Tokenization error: " + err.message, "error");
    return;
  }

  els.tokenCount.textContent = result.count.toLocaleString();
  renderTokens(result.pieces);
  els.ids.textContent = "[" + result.ids.join(", ") + "]";
}

function scheduleRender() {
  // Counts update instantly; the (heavier) token view is debounced.
  els.charCount.textContent = els.input.value.length.toLocaleString();
  els.wordCount.textContent = countWords(els.input.value).toLocaleString();
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(render, 120);
}

// Events
els.input.addEventListener("input", scheduleRender);
els.model.addEventListener("change", render);

boot();

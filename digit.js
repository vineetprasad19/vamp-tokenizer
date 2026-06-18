// Draw-a-digit recognizer — doodle a number 0–9, a tiny convolutional model
// guesses it. The model (MNIST, ~26 KB) and the onnxruntime-web engine load on
// first use. Same idea as the vision demo: pixels in → numbers → prediction.
//
// MNIST expects a white digit on a black background, so the pad below is black
// with a white pen — no inversion needed.

(function () {
  const pad = document.getElementById("digitPad");
  const clearBtn = document.getElementById("digitClear");
  const guessBtn = document.getElementById("digitGuess");
  const status = document.getElementById("digitStatus");
  const result = document.getElementById("digitResult");
  if (!pad) return;

  const ctx = pad.getContext("2d");
  const SIZE = pad.width; // square
  const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";
  const MODEL_URL = "https://raw.githubusercontent.com/onnx/models/main/validated/vision/classification/mnist/model/mnist-12.onnx";

  let drawing = false, dirty = false, session = null, ortLib = null;

  function clearPad() {
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, SIZE, SIZE);
    dirty = false; result.replaceChildren();
  }
  clearPad();

  function pos(e) {
    const r = pad.getBoundingClientRect();
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
    return { x: (cx / r.width) * SIZE, y: (cy / r.height) * SIZE };
  }
  function start(e) { drawing = true; dirty = true; const p = pos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); e.preventDefault(); }
  function move(e) {
    if (!drawing) return;
    const p = pos(e);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = SIZE * 0.09;
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.lineTo(p.x, p.y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(p.x, p.y);
    e.preventDefault();
  }
  function end() { drawing = false; }
  pad.addEventListener("pointerdown", start);
  pad.addEventListener("pointermove", move);
  window.addEventListener("pointerup", end);

  clearBtn.addEventListener("click", clearPad);

  function dStatus(msg, kind) {
    status.textContent = msg;
    status.className = "cluster-status" + (kind ? " " + kind : "");
  }

  function loadOrt() {
    if (window.ort) return Promise.resolve(window.ort);
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = ORT_URL;
      s.onload = () => resolve(window.ort);
      s.onerror = () => reject(new Error("failed to load onnxruntime-web"));
      document.head.appendChild(s);
    });
  }

  // Downscale the 280px pad to 28×28 grayscale floats (0–255), as MNIST expects.
  function preprocess() {
    const off = document.createElement("canvas");
    off.width = 28; off.height = 28;
    const octx = off.getContext("2d");
    octx.drawImage(pad, 0, 0, 28, 28);
    const px = octx.getImageData(0, 0, 28, 28).data;
    const data = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) data[i] = px[i * 4]; // R channel (gray)
    return data;
  }

  function softmax(arr) {
    let m = -Infinity; for (const v of arr) if (v > m) m = v;
    const ex = arr.map((v) => Math.exp(v - m));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map((v) => v / s);
  }

  async function guess() {
    if (!dirty) { dStatus("Draw a digit first.", "error"); return; }
    guessBtn.disabled = true;
    try {
      if (!session) {
        dStatus("Loading model… (one-time)", "busy");
        ortLib = await loadOrt();
        ortLib.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        session = await ortLib.InferenceSession.create(MODEL_URL);
      }
      dStatus("Recognizing…", "busy");
      const input = new ortLib.Tensor("float32", preprocess(), [1, 1, 28, 28]);
      const feeds = {}; feeds[session.inputNames[0]] = input;
      const out = await session.run(feeds);
      const scores = Array.from(out[session.outputNames[0]].data);
      const probs = softmax(scores);
      render(probs);
      dStatus("", "hidden");
    } catch (err) {
      console.error(err);
      dStatus("Error: " + err.message, "error");
    } finally {
      guessBtn.disabled = false;
    }
  }

  function render(probs) {
    const ranked = probs.map((p, d) => ({ d, p })).sort((a, b) => b.p - a.p);
    result.replaceChildren();
    const big = document.createElement("div");
    big.className = "digit-big";
    big.textContent = ranked[0].d;
    result.appendChild(big);
    const list = document.createElement("div");
    list.className = "metric-list";
    for (const { d, p } of ranked.slice(0, 3)) {
      const row = document.createElement("div");
      row.className = "metric-row" + (d === ranked[0].d ? " top" : "");
      row.innerHTML =
        `<span class="metric-name">${d}</span>` +
        `<span class="metric-bar"><span class="metric-fill" style="width:${(p * 100).toFixed(1)}%"></span></span>` +
        `<span class="metric-val">${(p * 100).toFixed(1)}%</span>`;
      list.appendChild(row);
    }
    result.appendChild(list);
  }

  guessBtn.addEventListener("click", guess);
})();

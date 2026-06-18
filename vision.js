// Image recognition — runs a Vision Transformer (ViT, ImageNet) in the browser
// to label an uploaded image with its top guesses. Demonstrates computer vision.

const VIS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3";
const VIS_MODEL = "Xenova/vit-base-patch16-224";

const visEls = {
  drop: document.getElementById("visDrop"),
  file: document.getElementById("visFile"),
  img: document.getElementById("visImg"),
  numbers: document.getElementById("visNumbers"),
  status: document.getElementById("visStatus"),
  result: document.getElementById("visResult"),
};

const PIXEL_GRID = 12;

// Show that "an image is just a grid of numbers": downsample to a small grid and
// print each pixel's grayscale value (0-255).
function renderPixelNumbers() {
  const G = PIXEL_GRID;
  const canvas = document.createElement("canvas");
  canvas.width = G; canvas.height = G;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(visEls.img, 0, 0, G, G);
  const data = ctx.getImageData(0, 0, G, G).data;

  const cap = document.createElement("div");
  cap.className = "num-cap";
  cap.textContent = `The model resizes this to 224×224×3 = ${(224 * 224 * 3).toLocaleString()} numbers. ` +
    `Here's a ${G}×${G} grayscale preview of those pixel values (0–255):`;

  const grid = document.createElement("div");
  grid.className = "pixel-grid";
  grid.style.gridTemplateColumns = `repeat(${G}, 1fr)`;
  for (let i = 0; i < G * G; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
    const v = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    const cell = document.createElement("div");
    cell.className = "pixel-cell";
    cell.style.background = `rgb(${v},${v},${v})`;
    cell.style.color = v > 140 ? "#000" : "#fff";
    cell.textContent = v;
    grid.appendChild(cell);
  }
  visEls.numbers.replaceChildren(cap, grid);
}

let _vis = null, _visLoading = null, visBusy = false;

function visStatus(msg, kind) {
  visEls.status.textContent = msg;
  visEls.status.className = "cluster-status" + (kind ? " " + kind : "");
}

async function getVis() {
  if (_vis) return _vis;
  if (_visLoading) return _visLoading;
  _visLoading = (async () => {
    const { pipeline, env } = await import(VIS_URL);
    env.allowLocalModels = false;
    _vis = await pipeline("image-classification", VIS_MODEL, {
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          visStatus(`Downloading model… ${Math.round((p.loaded / p.total) * 100)}% (one-time, then cached)`, "busy");
        }
      },
    });
    return _vis;
  })();
  return _visLoading;
}

async function classifyImage(url) {
  if (visBusy) return;
  visBusy = true;
  try {
    visStatus("Loading model…", "busy");
    const model = await getVis();
    visStatus("Looking at the image…", "busy");
    const out = await model(url, { topk: 5 });
    renderVis(out);
    visStatus("", "hidden");
  } catch (err) {
    console.error(err);
    visStatus("Error: " + err.message, "error");
  } finally {
    visBusy = false;
  }
}

function renderVis(preds) {
  visEls.result.replaceChildren();
  for (const p of preds) {
    const pct = Math.round(p.score * 100);
    const row = document.createElement("div");
    row.className = "metric-row";
    const name = document.createElement("span");
    name.className = "metric-name metric-name-wide";
    name.textContent = p.label;
    const bar = document.createElement("span");
    bar.className = "metric-bar";
    const fill = document.createElement("span");
    fill.className = "metric-fill";
    fill.style.width = pct + "%";
    bar.appendChild(fill);
    const num = document.createElement("span");
    num.className = "metric-num";
    num.textContent = pct + "%";
    row.append(name, bar, num);
    visEls.result.appendChild(row);
  }
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const url = URL.createObjectURL(file);
  visEls.img.hidden = false;
  visEls.img.onload = renderPixelNumbers;
  visEls.img.src = url;
  classifyImage(url);
}

visEls.file.addEventListener("change", (e) => handleFile(e.target.files[0]));
visEls.drop.addEventListener("dragover", (e) => { e.preventDefault(); visEls.drop.classList.add("drag"); });
visEls.drop.addEventListener("dragleave", () => visEls.drop.classList.remove("drag"));
visEls.drop.addEventListener("drop", (e) => {
  e.preventDefault();
  visEls.drop.classList.remove("drag");
  handleFile(e.dataTransfer.files[0]);
});

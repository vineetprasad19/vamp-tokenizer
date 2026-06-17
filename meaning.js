// Meaning explorer — compare the semantic similarity of two phrases using the
// shared embedding model. Similarity is the cosine of the two normalized vectors
// (1 = same meaning, 0 = unrelated).

const mEls = {
  a: document.getElementById("phraseA"),
  b: document.getElementById("phraseB"),
  btn: document.getElementById("simBtn"),
  status: document.getElementById("simStatus"),
  result: document.getElementById("simResult"),
};

let mBusy = false;

function mStatus(msg, kind) {
  mEls.status.textContent = msg;
  mEls.status.className = "cluster-status" + (kind ? " " + kind : "");
}

async function compareMeaning() {
  const a = mEls.a.value.trim();
  const b = mEls.b.value.trim();
  if (!a || !b) {
    mStatus("Enter two words or phrases to compare.", "error");
    mEls.result.replaceChildren();
    return;
  }
  if (mBusy) return;
  mBusy = true;
  mEls.btn.disabled = true;
  try {
    mStatus("Embedding…", "busy");
    const [va, vb] = await window.Embeddings.embed([a, b], (pct) =>
      mStatus(`Downloading model… ${pct}% (one-time, then cached)`, "busy"));
    const sim = Math.max(0, window.Embeddings.cosine(va, vb)); // clamp tiny negatives
    const pct = Math.round(sim * 100);
    const verdict =
      pct >= 80 ? "Very similar meaning" :
      pct >= 55 ? "Related" :
      pct >= 35 ? "Loosely related" : "Unrelated";

    const text = document.createElement("div");
    text.className = "sim-text";
    text.textContent = `${pct}% — ${verdict}`;
    const bar = document.createElement("div");
    bar.className = "sim-bar";
    const fill = document.createElement("div");
    fill.className = "sim-fill";
    fill.style.width = pct + "%";
    bar.appendChild(fill);
    mEls.result.replaceChildren(text, bar);
    mStatus("", "hidden");
  } catch (err) {
    console.error(err);
    mStatus("Error: " + err.message, "error");
  } finally {
    mBusy = false;
    mEls.btn.disabled = false;
  }
}

mEls.btn.addEventListener("click", compareMeaning);

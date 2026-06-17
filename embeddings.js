// Shared in-browser embedding model (Transformers.js / all-MiniLM-L6-v2).
// Used by the Vector Graph and the Meaning Explorer so the ~25 MB model is
// loaded only once. Vectors are mean-pooled and L2-normalized, so cosine
// similarity between two vectors is just their dot product.

window.Embeddings = (function () {
  const URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3";
  const MODEL = "Xenova/all-MiniLM-L6-v2";

  let _extractor = null;
  let _loading = null;
  const cache = new Map(); // text -> Float64Array(384)

  async function getExtractor(onProgress) {
    if (_extractor) return _extractor;
    if (_loading) return _loading;
    _loading = (async () => {
      const { pipeline, env } = await import(URL);
      env.allowLocalModels = false; // fetch from the Hugging Face hub
      _extractor = await pipeline("feature-extraction", MODEL, {
        progress_callback: (p) => {
          if (onProgress && p.status === "progress" && p.total) {
            onProgress(Math.round((p.loaded / p.total) * 100));
          }
        },
      });
      return _extractor;
    })();
    return _loading;
  }

  // Embed an array of strings; cached per string. Returns array of Float64Array.
  async function embed(texts, onProgress) {
    const missing = texts.filter((t) => !cache.has(t));
    if (missing.length) {
      const extractor = await getExtractor(onProgress);
      const out = await extractor(missing, { pooling: "mean", normalize: true });
      const vecs = out.tolist();
      missing.forEach((t, i) => cache.set(t, Float64Array.from(vecs[i])));
    }
    return texts.map((t) => cache.get(t));
  }

  // Cosine similarity for normalized vectors == dot product.
  function cosine(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  return { getExtractor, embed, cosine };
})();

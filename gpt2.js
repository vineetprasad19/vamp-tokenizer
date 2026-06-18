// Shared in-browser distilGPT-2 (Transformers.js). Loaded once and reused by the
// next-token predictor and the greedy-vs-sampling demo so the model is only
// downloaded a single time.

window.GPT2 = (function () {
  const URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3";
  const MODEL = "Xenova/distilgpt2";

  let _tok = null;
  let _model = null;
  let _loading = null;

  async function load(onProgress) {
    if (_tok && _model) return { tok: _tok, model: _model };
    if (_loading) return _loading;
    _loading = (async () => {
      const { AutoTokenizer, AutoModelForCausalLM, env } = await import(URL);
      env.allowLocalModels = false;
      const opts = {
        progress_callback: (p) => {
          if (onProgress && p.status === "progress" && p.total) {
            onProgress(Math.round((p.loaded / p.total) * 100));
          }
        },
      };
      _tok = await AutoTokenizer.from_pretrained(MODEL, opts);
      _model = await AutoModelForCausalLM.from_pretrained(MODEL, opts);
      return { tok: _tok, model: _model };
    })();
    return _loading;
  }

  return { load };
})();

// Greedy vs. sampling — same prompt, two decoding strategies, side by side.
// Greedy always takes the single most likely next token (safe, repetitive).
// Sampling rolls the dice weighted by probability and reshaped by temperature
// (varied, sometimes wild). Reuses the shared distilGPT-2 model.

(function () {
  const input = document.getElementById("genInput");
  const btn = document.getElementById("genBtn");
  const temp = document.getElementById("genTemp");
  const tempVal = document.getElementById("genTempVal");
  const status = document.getElementById("genStatus");
  const greedyOut = document.getElementById("genGreedy");
  const sampleOut = document.getElementById("genSample");
  if (!input) return;

  const NEW_TOKENS = 30;
  let busy = false;

  function gStatus(msg, kind) {
    status.textContent = msg;
    status.className = "cluster-status" + (kind ? " " + kind : "");
  }

  temp.addEventListener("input", () => { tempVal.textContent = parseFloat(temp.value).toFixed(1); });

  async function run() {
    if (busy) return;
    const prompt = input.value.trim();
    if (!prompt) { gStatus("Type a prompt first.", "error"); return; }
    busy = true; btn.disabled = true;
    try {
      gStatus("Loading model…", "busy");
      const { tok, model } = await GPT2.load((p) =>
        gStatus(`Downloading model… ${p}% (one-time, then cached)`, "busy"));

      gStatus("Generating…", "busy");
      const inputs = await tok(prompt);

      const greedy = await model.generate({ ...inputs, max_new_tokens: NEW_TOKENS, do_sample: false });
      const greedyText = tok.batch_decode(greedy, { skip_special_tokens: true })[0];

      const sampled = await model.generate({
        ...inputs, max_new_tokens: NEW_TOKENS, do_sample: true,
        temperature: parseFloat(temp.value),
      });
      const sampleText = tok.batch_decode(sampled, { skip_special_tokens: true })[0];

      renderColumn(greedyOut, prompt, greedyText);
      renderColumn(sampleOut, prompt, sampleText);
      gStatus("", "hidden");
    } catch (err) {
      console.error(err);
      gStatus("Error: " + err.message, "error");
    } finally {
      busy = false; btn.disabled = false;
    }
  }

  // Show the prompt dimmed and the generated continuation highlighted.
  function renderColumn(elm, prompt, full) {
    elm.replaceChildren();
    const cont = full.startsWith(prompt) ? full.slice(prompt.length) : full;
    const p = document.createElement("span");
    p.className = "gen-prompt"; p.textContent = prompt;
    const c = document.createElement("span");
    c.className = "gen-new"; c.textContent = cont;
    elm.append(p, c);
  }

  btn.addEventListener("click", run);
})();

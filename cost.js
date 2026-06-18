// Token cost calculator — estimate what one prompt+response costs across popular
// LLMs, with Claude front and center. Enter (or estimate) token counts and see
// per-call and total cost ranked cheapest-first.
//
// Claude prices are the official Anthropic list prices (USD per 1M tokens).
// Non-Claude prices are approximate public list prices as of mid-2026 and are
// easy to edit here — providers change them often.

(function () {
  const MODELS = [
    { name: "Claude Haiku 4.5", vendor: "Anthropic", in: 1.0, out: 5.0, claude: true },
    { name: "Claude Sonnet 4.6", vendor: "Anthropic", in: 3.0, out: 15.0, claude: true },
    { name: "Claude Opus 4.8", vendor: "Anthropic", in: 5.0, out: 25.0, claude: true },
    { name: "Claude Fable 5", vendor: "Anthropic", in: 10.0, out: 50.0, claude: true },
    { name: "GPT-4o mini", vendor: "OpenAI", in: 0.15, out: 0.6, approx: true },
    { name: "GPT-4o", vendor: "OpenAI", in: 2.5, out: 10.0, approx: true },
    { name: "Gemini 2.5 Flash", vendor: "Google", in: 0.3, out: 2.5, approx: true },
    { name: "Gemini 2.5 Pro", vendor: "Google", in: 1.25, out: 10.0, approx: true },
  ];

  const text = document.getElementById("costText");
  const inTok = document.getElementById("costIn");
  const outTok = document.getElementById("costOut");
  const calls = document.getElementById("costCalls");
  const tbody = document.getElementById("costBody");
  if (!tbody) return;

  // Rough token estimate from text (~4 chars/token). Just a starting point.
  text.addEventListener("input", () => {
    const est = Math.max(0, Math.round(text.value.length / 4));
    inTok.value = String(est);
    render();
  });

  const money = (n) =>
    n >= 1 ? "$" + n.toFixed(2)
      : n >= 0.01 ? "$" + n.toFixed(4)
        : "$" + n.toFixed(6);

  function render() {
    const i = Math.max(0, parseFloat(inTok.value) || 0);
    const o = Math.max(0, parseFloat(outTok.value) || 0);
    const n = Math.max(1, parseFloat(calls.value) || 1);

    const rows = MODELS.map((m) => {
      const per = (i / 1e6) * m.in + (o / 1e6) * m.out;
      return { ...m, per, total: per * n };
    }).sort((a, b) => a.total - b.total);

    const cheapest = rows[0].total;
    tbody.replaceChildren();
    for (const r of rows) {
      const tr = document.createElement("tr");
      if (r.claude) tr.className = "cost-claude";
      const mult = cheapest > 0 ? (r.total / cheapest) : 1;
      tr.innerHTML =
        `<td>${r.name}${r.approx ? ' <span class="cost-approx" title="approximate public list price">≈</span>' : ""}</td>` +
        `<td class="cost-num">$${r.in.toFixed(2)}</td>` +
        `<td class="cost-num">$${r.out.toFixed(2)}</td>` +
        `<td class="cost-num">${money(r.per)}</td>` +
        `<td class="cost-num"><b>${money(r.total)}</b><span class="cost-mult">${mult.toFixed(1)}×</span></td>`;
      tbody.appendChild(tr);
    }
  }

  [inTok, outTok, calls].forEach((el) => el.addEventListener("input", render));
  render();
})();

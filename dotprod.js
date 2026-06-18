// Dot-product playground — the math behind a single attention score.
// Drag the Query and Key vectors; the dot product Q·K is the raw attention score
// (before softmax). Aligned vectors → big positive score → strong attention;
// perpendicular → ~0; opposite → negative. No model, pure geometry.

(function () {
  const svg = document.getElementById("dotSvg");
  const out = document.getElementById("dotOut");
  if (!svg) return;

  const W = 320, H = 320, CX = W / 2, CY = H / 2, SCALE = 70; // px per unit
  const NS = "http://www.w3.org/2000/svg";
  // Vectors in math coords (y up). Start aligned-ish.
  const vecs = { q: { x: 1.4, y: 0.9 }, k: { x: 1.7, y: 0.4 } };
  let dragging = null;

  const toPx = (v) => ({ x: CX + v.x * SCALE, y: CY - v.y * SCALE });
  const toMath = (px, py) => ({ x: (px - CX) / SCALE, y: (CY - py) / SCALE });

  function el(name, attrs) {
    const e = document.createElementNS(NS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function draw() {
    svg.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    // grid
    for (let i = -2; i <= 2; i++) {
      svg.appendChild(el("line", { x1: CX + i * SCALE, y1: 0, x2: CX + i * SCALE, y2: H, stroke: "#1d2733", "stroke-width": 1 }));
      svg.appendChild(el("line", { x1: 0, y1: CY + i * SCALE, x2: W, y2: CY + i * SCALE, stroke: "#1d2733", "stroke-width": 1 }));
    }
    svg.appendChild(el("line", { x1: 0, y1: CY, x2: W, y2: CY, stroke: "#33414f", "stroke-width": 1.5 }));
    svg.appendChild(el("line", { x1: CX, y1: 0, x2: CX, y2: H, stroke: "#33414f", "stroke-width": 1.5 }));

    drawVec("k", "#56d4dd", "Key");
    drawVec("q", "#3fd3a6", "Query");
    update();
  }

  function drawVec(id, color, label) {
    const p = toPx(vecs[id]);
    svg.appendChild(el("line", { x1: CX, y1: CY, x2: p.x, y2: p.y, stroke: color, "stroke-width": 3, "stroke-linecap": "round" }));
    const handle = el("circle", { cx: p.x, cy: p.y, r: 9, fill: color, cursor: "grab", "data-id": id });
    svg.appendChild(handle);
    const t = el("text", { x: p.x + 10, y: p.y - 8, fill: color, "font-size": 12, "font-weight": 700 });
    t.textContent = label;
    svg.appendChild(t);
  }

  function update() {
    const q = vecs.q, k = vecs.k;
    const dot = q.x * k.x + q.y * k.y;
    const mq = Math.hypot(q.x, q.y), mk = Math.hypot(k.x, k.y);
    const cos = dot / ((mq * mk) || 1);
    const ang = Math.round((Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI);
    let verdict = "weak / no attention";
    if (dot > 1.2) verdict = "strong attention 🔥";
    else if (dot > 0.3) verdict = "some attention";
    else if (dot < -0.3) verdict = "actively ignored (negative)";
    out.innerHTML =
      `<b>Q · K = ${dot.toFixed(2)}</b> &nbsp;→&nbsp; ${verdict}<br>` +
      `<span class="dot-sub">|Q| = ${mq.toFixed(2)} · |K| = ${mk.toFixed(2)} · angle = ${ang}° · cosine = ${cos.toFixed(2)}</span>`;
  }

  function pointFromEvent(e) {
    const r = svg.getBoundingClientRect();
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
    return { x: (cx / r.width) * W, y: (cy / r.height) * H };
  }

  svg.addEventListener("pointerdown", (e) => {
    const id = e.target.getAttribute && e.target.getAttribute("data-id");
    if (id) { dragging = id; svg.setPointerCapture(e.pointerId); }
  });
  svg.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const p = pointFromEvent(e);
    const m = toMath(p.x, p.y);
    vecs[dragging] = { x: Math.max(-2.4, Math.min(2.4, m.x)), y: Math.max(-2.4, Math.min(2.4, m.y)) };
    draw();
  });
  svg.addEventListener("pointerup", () => { dragging = null; });

  draw();
})();

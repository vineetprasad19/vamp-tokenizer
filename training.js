// Training visualizer — gradient descent as a ball rolling downhill on a loss
// curve. Each step nudges the ball downhill by (learning rate × slope). Too small
// a rate = crawls; too big = overshoots and diverges. No model, pure JS.

(function () {
  const canvas = document.getElementById("trainCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const lr = document.getElementById("trainLr");
  const lrVal = document.getElementById("trainLrVal");
  const stepBtn = document.getElementById("trainStep");
  const runBtn = document.getElementById("trainRun");
  const resetBtn = document.getElementById("trainReset");
  const out = document.getElementById("trainOut");

  const W = canvas.width, H = canvas.height;
  const XMIN = -4, XMAX = 4;
  // Non-convex loss with two valleys, so a big learning rate can bounce around.
  const loss = (x) => 0.06 * x ** 4 - 0.5 * x ** 2 + 0.15 * x + 1.6;
  const grad = (x) => 0.24 * x ** 3 - x + 0.15;

  let x = 3.3, step = 0, timer = null;

  // Map loss-space to canvas pixels.
  const sx = (x) => ((x - XMIN) / (XMAX - XMIN)) * (W - 40) + 20;
  let LMAX = 5;
  const sy = (l) => H - 24 - (l / LMAX) * (H - 48);

  function drawCurve() {
    ctx.clearRect(0, 0, W, H);
    // axis
    ctx.strokeStyle = "#33414f"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(20, H - 24); ctx.lineTo(W - 20, H - 24); ctx.stroke();
    // curve
    ctx.strokeStyle = "#56d4dd"; ctx.lineWidth = 2.5; ctx.beginPath();
    let first = true;
    for (let px = 20; px <= W - 20; px++) {
      const xv = XMIN + ((px - 20) / (W - 40)) * (XMAX - XMIN);
      const py = sy(loss(xv));
      if (first) { ctx.moveTo(px, py); first = false; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
    // labels
    ctx.fillStyle = "#8b949e"; ctx.font = "11px system-ui";
    ctx.fillText("loss ↑", 22, 16);
    ctx.fillText("parameter →", W - 92, H - 8);
  }

  function drawBall() {
    drawCurve();
    const px = sx(x), py = sy(loss(x));
    const g = grad(x);
    ctx.fillStyle = "#3fd3a6";
    ctx.beginPath(); ctx.arc(px, Math.min(py, H - 26), 8, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#06281e"; ctx.lineWidth = 2; ctx.stroke();
    const l = loss(x);
    const diverged = Math.abs(x) > 4.2 || !isFinite(l);
    out.innerHTML = diverged
      ? `<b style="color:#f87171">Diverged! 💥</b> learning rate too high — the ball overshot the valley. step ${step}`
      : `step <b>${step}</b> · parameter = ${x.toFixed(2)} · loss = ${l.toFixed(3)} · slope = ${g.toFixed(2)}`;
    return diverged;
  }

  function doStep() {
    const rate = parseFloat(lr.value);
    x = x - rate * grad(x);
    step++;
    const diverged = drawBall();
    if (diverged) stopRun();
  }

  function startRun() {
    if (timer) return stopRun();
    runBtn.textContent = "⏸ Pause";
    timer = setInterval(() => {
      doStep();
      if (Math.abs(grad(x)) < 0.01 || step > 200) stopRun();
    }, 120);
  }
  function stopRun() { clearInterval(timer); timer = null; runBtn.textContent = "▶ Run"; }

  function reset() { stopRun(); x = 3.3; step = 0; drawBall(); }

  lr.addEventListener("input", () => { lrVal.textContent = parseFloat(lr.value).toFixed(2); });
  stepBtn.addEventListener("click", () => { stopRun(); doStep(); });
  runBtn.addEventListener("click", startRun);
  resetBtn.addEventListener("click", reset);

  drawBall();
})();

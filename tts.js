// Text-to-speech — the mirror image of the Whisper demo. Uses the browser's
// built-in SpeechSynthesis API, so it's free and needs no download.
//
// To show "words becoming sound", each word lights up as it's spoken (driven by
// real `boundary` events) and an oscilloscope canvas pulses on every word. The
// browser doesn't expose the voice's raw audio, so the waveform is illustrative —
// but its timing is tied to the actual speech.

(function () {
  const input = document.getElementById("ttsInput");
  const voiceSel = document.getElementById("ttsVoice");
  const rate = document.getElementById("ttsRate");
  const rateVal = document.getElementById("ttsRateVal");
  const pitch = document.getElementById("ttsPitch");
  const pitchVal = document.getElementById("ttsPitchVal");
  const speakBtn = document.getElementById("ttsSpeak");
  const stopBtn = document.getElementById("ttsStop");
  const status = document.getElementById("ttsStatus");
  const wordsEl = document.getElementById("ttsWords");
  const wave = document.getElementById("ttsWave");
  if (!input) return;

  const synth = window.speechSynthesis;
  if (!synth) {
    status.textContent = "Your browser doesn't support speech synthesis.";
    status.className = "cluster-status error";
    speakBtn.disabled = true;
    return;
  }

  const ctx = wave.getContext("2d");
  let spans = [];
  let env = 0;     // amplitude envelope, pulses on each word
  let phase = 0;
  let raf = null;
  let speaking = false;

  function loadVoices() {
    const voices = synth.getVoices();
    if (!voices.length) return;
    voiceSel.replaceChildren();
    voices.forEach((v, i) => {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = `${v.name} (${v.lang})${v.default ? " — default" : ""}`;
      voiceSel.appendChild(o);
    });
    const en = voices.findIndex((v) => v.default || /^en/i.test(v.lang));
    voiceSel.value = String(en >= 0 ? en : 0);
  }
  loadVoices();
  synth.onvoiceschanged = loadVoices;

  rate.addEventListener("input", () => { rateVal.textContent = parseFloat(rate.value).toFixed(1) + "×"; });
  pitch.addEventListener("input", () => { pitchVal.textContent = parseFloat(pitch.value).toFixed(1); });

  // Split text into word spans, remembering each word's character offset so we can
  // match it against the boundary event's charIndex.
  function buildWords(text) {
    wordsEl.replaceChildren();
    spans = [];
    const re = /\S+/g;
    let m, last = 0;
    while ((m = re.exec(text))) {
      if (m.index > last) wordsEl.appendChild(document.createTextNode(text.slice(last, m.index)));
      const s = document.createElement("span");
      s.className = "tts-word";
      s.textContent = m[0];
      s.dataset.start = String(m.index);
      s.dataset.end = String(m.index + m[0].length);
      wordsEl.appendChild(s);
      spans.push(s);
      last = m.index + m[0].length;
    }
    if (last < text.length) wordsEl.appendChild(document.createTextNode(text.slice(last)));
  }

  function highlight(charIndex) {
    for (const sp of spans) {
      const on = charIndex >= +sp.dataset.start && charIndex < +sp.dataset.end;
      sp.classList.toggle("active", on);
    }
  }
  function clearHighlight() { for (const sp of spans) sp.classList.remove("active"); }

  // Oscilloscope: composite of a few sine waves + a little noise, scaled by the
  // envelope. The envelope spikes on each word boundary, then decays — so the wave
  // visibly "speaks".
  function drawFrame() {
    const W = wave.width, H = wave.height, mid = H / 2;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "#1d2733"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();

    phase += 0.35;
    env *= 0.94;                       // decay between words
    const amp = (0.08 + env * 0.92) * (mid - 8);

    ctx.strokeStyle = "#3fd3a6"; ctx.lineWidth = 2; ctx.beginPath();
    for (let x = 0; x <= W; x += 2) {
      const v =
        Math.sin(x * 0.045 + phase) * 0.55 +
        Math.sin(x * 0.13 + phase * 1.7) * 0.3 +
        (Math.random() - 0.5) * 0.18;
      const y = mid + v * amp;
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    if (speaking) raf = requestAnimationFrame(drawFrame);
  }

  function startWave() {
    speaking = true;
    wave.hidden = false;
    if (!raf) raf = requestAnimationFrame(drawFrame);
  }
  function stopWave() {
    speaking = false;
    if (raf) { cancelAnimationFrame(raf); raf = null; }
    ctx.clearRect(0, 0, wave.width, wave.height);
    wave.hidden = true;
  }

  speakBtn.addEventListener("click", () => {
    const text = input.value.trim();
    if (!text) { status.textContent = "Type something first."; status.className = "cluster-status error"; return; }
    synth.cancel();
    buildWords(input.value);

    const u = new SpeechSynthesisUtterance(text);
    const voices = synth.getVoices();
    const idx = parseInt(voiceSel.value, 10);
    if (voices[idx]) u.voice = voices[idx];
    u.rate = parseFloat(rate.value);
    u.pitch = parseFloat(pitch.value);

    u.onstart = () => { status.textContent = "🔊 Speaking…"; status.className = "cluster-status busy"; startWave(); };
    u.onboundary = (e) => {
      if (e.name && e.name !== "word") return;
      env = 1;                          // pulse the waveform on each word
      highlight(e.charIndex);
    };
    u.onend = () => { status.className = "cluster-status hidden"; stopWave(); clearHighlight(); };
    u.onerror = () => { status.textContent = "Playback error."; status.className = "cluster-status error"; stopWave(); };
    synth.speak(u);
  });

  stopBtn.addEventListener("click", () => {
    synth.cancel();
    status.className = "cluster-status hidden";
    stopWave();
    clearHighlight();
  });
})();

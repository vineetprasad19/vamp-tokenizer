// Text-to-speech — the mirror image of the Whisper demo. Uses the browser's
// built-in SpeechSynthesis API, so it's free and needs no download.

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
  if (!input) return;

  const synth = window.speechSynthesis;
  if (!synth) {
    status.textContent = "Your browser doesn't support speech synthesis.";
    status.className = "cluster-status error";
    speakBtn.disabled = true;
    return;
  }

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

  speakBtn.addEventListener("click", () => {
    const text = input.value.trim();
    if (!text) { status.textContent = "Type something first."; status.className = "cluster-status error"; return; }
    synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    const voices = synth.getVoices();
    const idx = parseInt(voiceSel.value, 10);
    if (voices[idx]) u.voice = voices[idx];
    u.rate = parseFloat(rate.value);
    u.pitch = parseFloat(pitch.value);
    u.onstart = () => { status.textContent = "🔊 Speaking…"; status.className = "cluster-status busy"; };
    u.onend = () => { status.className = "cluster-status hidden"; };
    u.onerror = () => { status.textContent = "Playback error."; status.className = "cluster-status error"; };
    synth.speak(u);
  });

  stopBtn.addEventListener("click", () => { synth.cancel(); status.className = "cluster-status hidden"; });
})();

// Speech-to-text — records mic audio and transcribes it with Whisper (tiny) in
// the browser. Demonstrates audio AI / multimodality: sound -> numbers -> tokens.

const ASR_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3";
const ASR_MODEL = "Xenova/whisper-tiny.en";

const spEls = {
  btn: document.getElementById("micBtn"),
  status: document.getElementById("micStatus"),
  wave: document.getElementById("micWave"),
  numbers: document.getElementById("micNumbers"),
  result: document.getElementById("micResult"),
};

// Show that "sound is just a stream of numbers": draw the waveform and print the
// first raw sample values the model receives.
function renderAudioNumbers(pcm) {
  const canvas = spEls.wave;
  canvas.hidden = false;
  const W = canvas.width, H = canvas.height, mid = H / 2;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#3fd3a6";
  ctx.beginPath();
  const step = Math.max(1, Math.floor(pcm.length / W));
  for (let x = 0; x < W; x++) {
    let min = 1, max = -1;
    for (let j = 0; j < step; j++) {
      const s = pcm[x * step + j] || 0;
      if (s < min) min = s;
      if (s > max) max = s;
    }
    ctx.moveTo(x, mid + min * mid);
    ctx.lineTo(x, mid + max * mid);
  }
  ctx.stroke();

  const cap = document.createElement("div");
  cap.className = "num-cap";
  cap.textContent = `Your audio = ${pcm.length.toLocaleString()} samples at 16,000 Hz ` +
    `(${(pcm.length / 16000).toFixed(1)} s). Each sample is a number from −1 to 1 — here are the first 16:`;
  const samp = document.createElement("div");
  samp.className = "num-samples";
  samp.textContent = "[" + Array.from(pcm.slice(0, 16)).map((v) => v.toFixed(3)).join(", ") + ", …]";
  spEls.numbers.replaceChildren(cap, samp);
}

let _asr = null, _asrLoading = null;
let _recorder = null, _chunks = [], _recording = false, _stream = null;

function spStatus(msg, kind) {
  spEls.status.textContent = msg;
  spEls.status.className = "cluster-status" + (kind ? " " + kind : "");
}

async function getAsr() {
  if (_asr) return _asr;
  if (_asrLoading) return _asrLoading;
  _asrLoading = (async () => {
    const { pipeline, env } = await import(ASR_URL);
    env.allowLocalModels = false;
    _asr = await pipeline("automatic-speech-recognition", ASR_MODEL, {
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          spStatus(`Downloading model… ${Math.round((p.loaded / p.total) * 100)}% (one-time, then cached)`, "busy");
        }
      },
    });
    return _asr;
  })();
  return _asrLoading;
}

// Decode a recorded audio blob to mono 16 kHz Float32 samples (what Whisper wants).
async function toPcm16k(blob) {
  const ab = await blob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ctx.decodeAudioData(ab);
  ctx.close();
  const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

async function startRecording() {
  try {
    _stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    spStatus("Microphone access denied or unavailable.", "error");
    return;
  }
  _chunks = [];
  _recorder = new MediaRecorder(_stream);
  _recorder.ondataavailable = (e) => { if (e.data.size) _chunks.push(e.data); };
  _recorder.onstop = transcribe;
  _recorder.start();
  _recording = true;
  spEls.btn.textContent = "⏹ Stop & transcribe";
  spEls.btn.classList.add("recording");
  spStatus("Recording… speak now, then click stop.", "busy");
}

function stopRecording() {
  _recording = false;
  spEls.btn.textContent = "🎤 Start recording";
  spEls.btn.classList.remove("recording");
  if (_recorder && _recorder.state !== "inactive") _recorder.stop();
  if (_stream) _stream.getTracks().forEach((t) => t.stop());
}

async function transcribe() {
  try {
    spStatus("Loading model…", "busy");
    const model = await getAsr();
    spStatus("Transcribing…", "busy");
    const blob = new Blob(_chunks);
    const pcm = await toPcm16k(blob);
    renderAudioNumbers(pcm);
    const out = await model(pcm);
    spEls.result.textContent = (out.text || "").trim() || "(nothing heard — try again)";
    spStatus("", "hidden");
  } catch (err) {
    console.error(err);
    spStatus("Error: " + err.message, "error");
  }
}

spEls.btn.addEventListener("click", () => {
  if (_recording) stopRecording();
  else startRecording();
});

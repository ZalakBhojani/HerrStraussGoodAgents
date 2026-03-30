// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const borrowerId = (crypto.randomUUID
  ? crypto.randomUUID()
  : Math.random().toString(36).slice(2) + Date.now().toString(36));

let chatWs = null;
let voiceWs = null;
let audioContext = null;
let micStream = null;
let scriptProcessor = null;
let isCallActive = false;

const messagesEl     = document.getElementById('messages');
const inputEl        = document.getElementById('input');
const sendBtn        = document.getElementById('send');
const statusEl       = document.getElementById('status');
const voiceBanner    = document.getElementById('voiceBanner');
const callBtn        = document.getElementById('callBtn');
const callStatus     = document.getElementById('callStatus');
const callStatusText = document.getElementById('callStatusText');
const callSummary    = document.getElementById('callSummary');

// ---------------------------------------------------------------------------
// Chat helpers
// ---------------------------------------------------------------------------
function appendMessage(text, role) {
  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendSystemNotice(text) {
  const el = document.createElement('div');
  el.className = 'message system-notice';
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ---------------------------------------------------------------------------
// Chat WebSocket
// ---------------------------------------------------------------------------
function connectChat() {
  console.log(`BorrowerId is ${borrowerId}`)
  chatWs = new WebSocket(`ws://${location.host}/ws/${borrowerId}`);

  chatWs.onopen = () => {
    statusEl.textContent = 'Connected · ' + borrowerId.slice(0, 8);
    inputEl.disabled = false;
    sendBtn.disabled = false;
    inputEl.focus();
  };

  chatWs.onclose = () => {
    statusEl.textContent = 'Disconnected';
    inputEl.disabled = true;
    sendBtn.disabled = true;
  };

  chatWs.onerror = () => { statusEl.textContent = 'Connection error'; };

  chatWs.onmessage = (e) => {
    const text = e.data;
    if (text === '__VOICE_CALL_READY__') {
      voiceBanner.classList.add('visible');
      appendSystemNotice('A voice call is available. Click "Start Call" when ready.');
      inputEl.disabled = true;
      sendBtn.disabled = true;
      return;
    }
    appendMessage(text, 'agent');
  };
}

function sendChat() {
  const text = inputEl.value.trim();
  if (!text || !chatWs || chatWs.readyState !== WebSocket.OPEN) return;
  appendMessage(text, 'user');
  chatWs.send(text);
  inputEl.value = '';
}

sendBtn.addEventListener('click', sendChat);
inputEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendChat(); });

// ---------------------------------------------------------------------------
// Voice call
// ---------------------------------------------------------------------------
callBtn.addEventListener('click', async () => {
  if (isCallActive) {
    endCall();
  } else {
    await startCall();
  }
});

async function startCall() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (err) {
    appendSystemNotice('Microphone access denied. Please allow microphone access and try again.');
    return;
  }

  voiceBanner.classList.remove('visible');
  callStatus.classList.add('visible');
  callStatusText.textContent = 'Connecting voice call…';
  callBtn.textContent = 'End Call';
  callBtn.classList.add('active');
  isCallActive = true;

  voiceWs = new WebSocket(`ws://${location.host}/voice/${borrowerId}`);
  voiceWs.binaryType = 'arraybuffer';

  voiceWs.onopen = () => {
    callStatusText.textContent = 'Call in progress…';
    startAudioCapture();
  };

  voiceWs.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) {
      playAudio(e.data);
    }
  };

  voiceWs.onclose = () => {
    if (isCallActive) endCall();
  };

  voiceWs.onerror = () => {
    appendSystemNotice('Voice connection error.');
    if (isCallActive) endCall();
  };
}

function endCall() {
  isCallActive = false;
  stopAudioCapture();
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) voiceWs.close();
  voiceWs = null;
  callStatus.classList.remove('visible');
  callBtn.textContent = 'Start Call';
  callBtn.classList.remove('active');
  callBtn.disabled = true;
  appendSystemNotice('Voice call ended.');
  showCallSummary();
}

function showCallSummary() {
  callSummary.classList.add('visible');
  document.getElementById('summaryStatus').textContent = 'Call completed';
  document.getElementById('summaryResolution').textContent = 'See chat for next steps';
  document.getElementById('summarySettlement').textContent = '—';
  // Re-enable chat for Final Notice stage
  if (chatWs && chatWs.readyState === WebSocket.OPEN) {
    inputEl.disabled = false;
    sendBtn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Audio capture (mic → voice WebSocket)
// ---------------------------------------------------------------------------
function startAudioCapture() {
  const sampleRate = 16000;
  audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
  const source = audioContext.createMediaStreamSource(micStream);
  // Buffer size: 4096 samples ≈ 256ms at 16 kHz
  scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

  scriptProcessor.onaudioprocess = (e) => {
    if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) return;
    const float32 = e.inputBuffer.getChannelData(0);
    const int16 = float32ToInt16(float32);
    voiceWs.send(int16.buffer);
  };

  source.connect(scriptProcessor);
  scriptProcessor.connect(audioContext.destination);
}

function stopAudioCapture() {
  if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
}

function float32ToInt16(float32) {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const clamped = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = clamped < 0 ? clamped * 32768 : clamped * 32767;
  }
  return int16;
}

// ---------------------------------------------------------------------------
// Audio playback (voice WebSocket → speaker)
// ---------------------------------------------------------------------------
const playbackQueue = [];
let isPlaying = false;
let playbackCtx = null;

function playAudio(arrayBuffer) {
  if (!playbackCtx) {
    playbackCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  }
  playbackQueue.push(arrayBuffer);
  if (!isPlaying) drainPlaybackQueue();
}

function drainPlaybackQueue() {
  if (playbackQueue.length === 0) { isPlaying = false; return; }
  isPlaying = true;
  const buffer = playbackQueue.shift();
  // Raw 16-bit PCM mono 16 kHz
  const int16 = new Int16Array(buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
  const audioBuffer = playbackCtx.createBuffer(1, float32.length, 16000);
  audioBuffer.copyToChannel(float32, 0);
  const source = playbackCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(playbackCtx.destination);
  source.onended = drainPlaybackQueue;
  source.start();
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
connectChat();

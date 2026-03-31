import { PipecatClient, RTVIEvent } from '@pipecat-ai/client-js';
import { WebSocketTransport } from '@pipecat-ai/websocket-transport';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const borrowerId = (crypto.randomUUID
  ? crypto.randomUUID()
  : Math.random().toString(36).slice(2) + Date.now().toString(36));

let chatWs = null;
let rtviClient = null;
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
  voiceBanner.classList.remove('visible');
  callStatus.classList.add('visible');
  callStatusText.textContent = 'Connecting voice call…';
  callBtn.textContent = 'End Call';
  callBtn.classList.add('active');
  isCallActive = true;

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${proto}//${location.host}/voice/${borrowerId}`;

  rtviClient = new PipecatClient({
    transport: new WebSocketTransport({ wsUrl }),
    enableMic: true,
    enableCam: false,
  });

  rtviClient.on(RTVIEvent.Connected, () => {
    callStatusText.textContent = 'Call in progress…';
  });

  rtviClient.on(RTVIEvent.Disconnected, () => {
    if (isCallActive) endCall();
  });

  rtviClient.on(RTVIEvent.Error, (err) => {
    appendSystemNotice(`Voice connection error: ${err?.message ?? err}`);
    if (isCallActive) endCall();
  });

  try {
    await rtviClient.initDevices();
    await rtviClient.connect();
  } catch (err) {
    appendSystemNotice('Could not start voice call. Check microphone permissions.');
    endCall();
  }
}

function endCall() {
  isCallActive = false;  // guard against re-entry from Disconnected event
  if (rtviClient) {
    rtviClient.disconnect().catch(() => {});
    rtviClient = null;
  }
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
// Boot
// ---------------------------------------------------------------------------
connectChat();

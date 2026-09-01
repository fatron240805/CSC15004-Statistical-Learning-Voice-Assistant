// Chat flow: ghi âm 1 lệnh -> POST /chat -> phát audio trả lời + hiển thị data usecase-specific.
// SV retry (mục 5.2 spec) phải ghi âm LẠI mỗi lần — audio giống hệt sẽ luôn cho cùng 1 score.
let mediaRecorder = null;
let chunks = [];
let svAttempt = 1;

const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const retryBtn = document.getElementById("retryBtn");
const statusEl = document.getElementById("status");

async function startRecording(attempt) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    recordBtn.classList.remove("recording");
    sendChat(new Blob(chunks, { type: "audio/webm" }), attempt);
  };
  mediaRecorder.start();
  recordBtn.disabled = true;
  recordBtn.classList.add("recording");
  stopBtn.disabled = false;
  retryBtn.style.display = "none";
  statusEl.textContent = "Đang ghi âm...";
}

recordBtn.onclick = () => {
  svAttempt = 1;
  startRecording(svAttempt);
};

stopBtn.onclick = () => {
  mediaRecorder.stop();
  stopBtn.disabled = true;
};

retryBtn.onclick = () => {
  svAttempt++;
  startRecording(svAttempt);
};

async function sendChat(blob, attempt) {
  statusEl.textContent = "Đang xử lý...";
  recordBtn.disabled = true;
  retryBtn.style.display = "none";

  const form = new FormData();
  form.append("audio", blob, "command.webm");
  form.append("sv_attempt", attempt);

  try {
    const res = await fetch(`${API_BASE}/chat`, { method: "POST", body: form });
    const data = await res.json();
    renderResult(data);
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = "Lỗi gọi /chat: " + err;
  } finally {
    recordBtn.disabled = false;
  }
}

function renderResult(data) {
  const resultCard = document.getElementById("resultCard");
  resultCard.style.display = "block";
  resultCard.className = "card"; // reset tint trước khi gán lại theo usecase
  document.getElementById("transcript").textContent = data.transcript;
  document.getElementById("intent").textContent = data.intent;
  document.getElementById("usecase").textContent = data.usecase;
  document.getElementById("text").textContent = data.text;

  const audioEl = document.getElementById("ttsAudio");
  audioEl.src = data.audio_base64 ? `data:audio/mpeg;base64,${data.audio_base64}` : "";
  if (data.audio_base64) audioEl.play().catch(() => {});

  const svPanel = document.getElementById("svPanel");
  const sidPanel = document.getElementById("sidPanel");
  svPanel.style.display = "none";
  sidPanel.style.display = "none";
  retryBtn.style.display = "none";

  if (data.usecase === "general") {
    resultCard.classList.add("result-general");
  }

  if (data.usecase === "sv" && data.data) {
    svPanel.style.display = "block";
    resultCard.classList.add(data.data.verified ? "result-sv-success" : "result-sv-fail");
    document.getElementById("svMessage").textContent =
      `Xác thực: ${data.data.verified ? "thành công" : "thất bại"}` +
      (data.data.score != null ? ` (score ${data.data.score.toFixed(3)})` : "");
    if (data.data.can_retry) {
      retryBtn.style.display = "inline-block";
      retryBtn.textContent = "Ghi âm lại để xác thực";
    }
  }

  if (data.usecase === "sid" && data.data) {
    sidPanel.style.display = "block";
    resultCard.classList.add("result-sid");
    const container = document.getElementById("playlist");
    container.innerHTML = "";
    (data.data.playlist || []).forEach((track) => {
      const div = document.createElement("div");
      div.className = "track";
      div.innerHTML = `<b>${track.title}</b> — ${track.artist}<br>` +
        `<audio controls src="${track.preview_url}"></audio>`;
      container.appendChild(div);
    });
    if ((data.data.playlist || []).length === 0) {
      container.textContent = "Không có bài hát nào.";
    }
  }
}

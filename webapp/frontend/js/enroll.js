// Enroll flow: /enroll/start -> đọc 7 câu (mỗi câu ghi âm -> /enroll/sentence/{idx}) -> /enroll/finish
let userId = null;
let sentences = [];
let idx = 0;
let mediaRecorder = null;
let chunks = [];
let sentenceStatus = []; // "upcoming" | "current" | "retry" | "done", theo dữ liệu thật của luồng enroll

function renderStepDots() {
  const el = document.getElementById("stepDots");
  el.innerHTML = "";
  sentenceStatus.forEach((s, i) => {
    const span = document.createElement("span");
    span.className = s === "done" ? "done" : s === "retry" ? "retry" : s === "current" ? "current" : "";
    // fix #6: không chỉ dựa vào màu để phân biệt trạng thái (rủi ro mù màu đỏ-lục)
    span.textContent = s === "done" ? "✓" : s === "retry" ? "!" : String(i + 1);
    span.title = s === "done" ? "Đạt" : s === "retry" ? "Cần đọc lại" : s === "current" ? "Đang đọc" : "Chưa tới";
    el.appendChild(span);
  });
}

const startBtn = document.getElementById("startBtn");
const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const nextBtn = document.getElementById("nextBtn");
const finishBtn = document.getElementById("finishBtn");
const statusEl = document.getElementById("status");

startBtn.onclick = async () => {
  const name = document.getElementById("nameInput").value.trim();
  if (!name) { alert("Nhập tên"); return; }
  const favorite_tracks = document.getElementById("tracksInput").value.trim();

  const form = new FormData();
  form.append("name", name);
  form.append("favorite_tracks", favorite_tracks);
  const res = await fetch(`${API_BASE}/enroll/start`, { method: "POST", body: form });
  const data = await res.json();
  userId = data.user_id;
  sentences = data.sentences;
  sentenceStatus = sentences.map(() => "upcoming");
  localStorage.setItem("sva_user_id", userId);

  document.getElementById("startCard").style.display = "none";
  document.getElementById("sentenceCard").style.display = "block";
  document.getElementById("userIdLabel").textContent = userId;
  showSentence();
};

function showSentence() {
  document.getElementById("idxLabel").textContent = idx + 1;
  document.getElementById("sentenceText").textContent = sentences[idx];
  statusEl.textContent = "";
  statusEl.className = "";
  nextBtn.style.display = "none";
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  if (sentenceStatus[idx] !== "done") sentenceStatus[idx] = "current";
  renderStepDots();
}

recordBtn.onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    recordBtn.classList.remove("recording");
    uploadSentence(new Blob(chunks, { type: "audio/webm" }));
  };
  mediaRecorder.start();
  recordBtn.disabled = true;
  recordBtn.classList.add("recording");
  stopBtn.disabled = false;
  statusEl.textContent = "Đang ghi âm...";
};

stopBtn.onclick = () => {
  mediaRecorder.stop();
  stopBtn.disabled = true;
};

async function uploadSentence(blob) {
  statusEl.textContent = "Đang xử lý...";
  const form = new FormData();
  form.append("user_id", userId);
  form.append("audio", blob, "sentence.webm");
  const res = await fetch(`${API_BASE}/enroll/sentence/${idx}`, { method: "POST", body: form });
  const data = await res.json();

  if (data.pass) {
    statusEl.textContent = `Đạt. Nhận dạng: "${data.transcript}" (WER ${data.wer})`;
    statusEl.className = "pass";
    nextBtn.style.display = "inline-block";
    recordBtn.disabled = true;
    sentenceStatus[idx] = "done";
  } else {
    statusEl.textContent = `Chưa đạt (WER ${data.wer}, ${data.duration_sec}s). Nhận dạng: "${data.transcript}". Ghi âm lại.`;
    statusEl.className = "fail";
    recordBtn.disabled = false;
    sentenceStatus[idx] = "retry";
  }
  renderStepDots();
}

nextBtn.onclick = () => {
  idx++;
  if (idx >= sentences.length) {
    document.getElementById("sentenceCard").style.display = "none";
    document.getElementById("finishCard").style.display = "block";
  } else {
    showSentence();
  }
};

finishBtn.onclick = async () => {
  const form = new FormData();
  form.append("user_id", userId);
  const res = await fetch(`${API_BASE}/enroll/finish`, { method: "POST", body: form });
  const data = await res.json();
  const el = document.getElementById("finishStatus");
  if (res.ok) {
    el.textContent = `Enroll thành công. user_id=${data.user_id}, embedding_dim=${data.embedding_dim}.`;
    el.className = "pass";
  } else {
    el.textContent = `Lỗi: ${data.detail || "không rõ"}`;
    el.className = "fail";
  }
};

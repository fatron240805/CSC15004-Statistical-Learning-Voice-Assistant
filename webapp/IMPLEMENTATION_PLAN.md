# Kế hoạch Implement — Module B Web App

Dựa trên `spec_ModuleB_web_app.md`. Thứ tự các milestone dưới đây build từ trong ra ngoài: model → data → pipeline lõi → usecase → UI → deploy, để mỗi bước có thể test độc lập trước khi ráp tiếp.

## 0. Cấu trúc thư mục dự kiến

```
webapp/
├── spec_ModuleB_web_app.md
├── IMPLEMENTATION_PLAN.md
├── backend/
│   ├── main.py                    # FastAPI app, mount static frontend build
│   ├── requirements.txt
│   ├── config.py                  # đọc .env: GEMINI_API_KEY, OPENWEATHER_KEY...
│   ├── db/
│   │   ├── database.py            # engine + session SQLite
│   │   └── models.py              # User, Preference, ActionLog
│   ├── services/
│   │   ├── speaker_service.py     # wrap speaker_model.py (copy nguyên file vào đây)
│   │   ├── audio_convert.py       # webm -> wav 16kHz mono (pydub/ffmpeg)
│   │   ├── asr_service.py         # PhoWhisper wrapper
│   │   ├── orchestrator_service.py# gọi Gemini, ép JSON intent
│   │   ├── tts_service.py         # gTTS wrapper
│   │   ├── weather_service.py     # OpenWeatherMap
│   │   └── music_service.py       # Deezer API
│   ├── api/
│   │   ├── enrollment.py          # POST /enroll/*, GET /users
│   │   └── chat.py                # POST /chat (audio in -> audio+text out)
│   ├── gating.py                  # bảng mapping intent -> usecase -> cần SV?
│   └── speaker_model.py           # copy nguyên từ model/moduleA_artifacts_final/
├── checkpoints/best.pt            # copy checkpoint vào đây lúc build
└── frontend/                      # React (Vite) hoặc HTML/JS thuần
    ├── src/pages/Enroll.jsx
    ├── src/pages/Chat.jsx
    └── ...
```

## Milestone 1 — Skeleton + môi trường

- [ ] Tạo `backend/requirements.txt`: `fastapi`, `uvicorn`, `torch==2.4.1`, `torchaudio`, `numpy<2` (`1.26.4`), `speechbrain`, `transformers`, `pydub`, `gTTS`, `sqlalchemy`, `python-dotenv`, `google-generativeai`, `requests`.
- [ ] `main.py` chạy được `uvicorn backend.main:app`, trả `{"status": "ok"}` ở `/health`.
- [ ] `config.py` đọc `.env` (đã có `GEMINI_API_KEY`), thêm placeholder `OPENWEATHER_KEY`.
- [ ] **Tiêu chí xong:** server chạy local, `/health` trả 200.

## Milestone 2 — Tích hợp speaker_model.py (theo mục 6 README Module A)

- [ ] Copy `model/moduleA_artifacts_final/speaker_model.py` + `checkpoints/best.pt` vào `webapp/backend/` và `webapp/checkpoints/`.
- [ ] `speaker_service.py`: load `SpeakerModel` 1 lần lúc startup (singleton), expose `get_embedding()`, `verify()`, `identify()` — không đổi tên hàm/threshold.
- [ ] Test tay: chạy `get_embedding()` trên 1 file `.wav` bất kỳ, xác nhận không lỗi version torch/numpy (đây là việc #1 Khoa yêu cầu xác nhận trong README mục 6).
- [ ] Test tay: `verify()` với 1 cặp audio enrollment/runtime giả lập (README mục 6, việc #2).
- [ ] **Tiêu chí xong:** embedding ra đúng 192-dim, `verify()`/`identify()` chạy không lỗi.

## Milestone 3 — Audio pipeline + ASR

- [ ] `audio_convert.py`: nhận `.webm` (bytes), convert sang `.wav` 16kHz mono bằng `pydub` (cần `ffmpeg` cài trong môi trường/Docker image).
- [ ] `asr_service.py`: load PhoWhisper (`transformers`, model nhỏ vd `vinai/PhoWhisper-base`), hàm `transcribe(wav_path) -> str`.
- [ ] **Tiêu chí xong:** ghi 1 file test, convert + transcribe ra text tiếng Việt đọc được.

## Milestone 4 — Database

- [ ] `db/models.py`: `User(user_id, name, embedding: JSON, enrollment_date)`, `Preference(user_id, favorite_tracks: JSON)`, `ActionLog(id, timestamp, user_id, action, verified, score)`.
- [ ] `db/database.py`: SQLite file `webapp/backend/app.db`, tạo bảng lúc startup nếu chưa có.
- [ ] **Tiêu chí xong:** insert/query thử qua script hoặc `/health` mở rộng kiểm tra kết nối DB.

## Milestone 5 — Enrollment flow

- [ ] Copy nguyên văn 7 câu từ `enrollment_sentences.md` vào constant trong code (không code cứng khác đi).
- [ ] `POST /enroll/start` — tạo user mới (name, favorite_tracks ban đầu).
- [ ] `POST /enroll/sentence/{idx}` — nhận audio 1 câu, convert, chạy ASR, tính WER so với câu mẫu, kiểm tra độ dài audio tối thiểu → trả `pass/fail` để frontend cho đọc lại nếu fail.
- [ ] `POST /enroll/finish` — sau 7 câu đạt: tính `get_embedding()` từng câu, gộp (trung bình + L2-normalize lại), lưu vào `users.embedding`.
- [ ] Frontend `Enroll.jsx`: UI đọc lần lượt 7 câu, hiện trạng thái pass/fail từng câu, nút ghi lại.
- [ ] **Tiêu chí xong:** enroll xong 1 user thật, `verify()`/`identify()` sau đó nhận đúng người này.

## Milestone 6 — Orchestrator (Gemini) + gating cứng

- [ ] `orchestrator_service.py`: gọi Gemini API, prompt ép trả JSON đúng schema `{intent, entities}` (dùng response schema / JSON mode của Gemini SDK).
- [ ] `gating.py`: dict tĩnh `INTENT_MAP = {intent: (usecase, need_sv: bool)}` — liệt kê đủ các intent trong mục 4 spec.
- [ ] **Tiêu chí xong:** đưa vài câu test tiếng Việt (vd "mở khoang xe giúp tôi", "thời tiết hôm nay thế nào") ra đúng intent JSON, và route đúng usecase qua `gating.py`.

## Milestone 7 — 3 usecase handlers

- [ ] `weather_service.py` + handler General: gọi OpenWeatherMap với toạ độ cố định, trả câu trả lời text.
- [ ] Handler SV: nhận `user_id` (từ UI chọn, xem mục "việc còn mở" trong spec) → `verify()` → nếu `False` cho phép FE gọi lại tối đa 2 lần → ghi `ActionLog` mỗi lần thử → action mock (đổi trạng thái "đã mở khoang"/"đã khởi động").
- [ ] `music_service.py` + Handler SID: `identify()` → lấy `favorite_tracks` từ `Preference` → gọi Deezer search từng bài → trả list `{title, artist, preview_url}`.
- [ ] **Tiêu chí xong:** gọi `/chat` end-to-end với audio thật cho từng usecase, nhận đúng phản hồi.

## Milestone 8 — TTS + endpoint `/chat` hoàn chỉnh

- [ ] `tts_service.py`: text → file mp3 (gTTS), trả về base64 hoặc URL tạm.
- [ ] `api/chat.py`: gộp toàn bộ pipeline (mục 3 spec) thành 1 endpoint `POST /chat` (input: audio + user_id nếu có; output: text phản hồi + audio TTS + data usecase-specific như playlist).
- [ ] **Tiêu chí xong:** 1 request audio → nhận về đầy đủ text + audio trả lời, đúng luồng gate.

## Milestone 9 — Frontend Chat UI

- [ ] `Chat.jsx`: nút ghi âm (MediaRecorder), gửi lên `/chat`, phát audio trả lời, hiển thị playlist (SID) hoặc trạng thái hành động (SV) hoặc text (General).
- [ ] Chọn "tôi là ai" trước khi nói (theo quyết định mục "việc còn mở" trong spec) hoặc để hệ thống tự `identify()` trước — chốt lúc code milestone này.
- [ ] **Tiêu chí xong:** demo tay được cả 3 usecase qua UI, không cần Postman.

## Milestone 10 — Deploy Hugging Face Spaces

- [ ] `Dockerfile`: cài `ffmpeg`, Python deps, copy backend + frontend build, `CMD uvicorn`.
- [ ] Tạo HF Space (Docker SDK, free CPU), set secret `GEMINI_API_KEY`, `OPENWEATHER_KEY` qua Space Settings (không commit `.env`).
- [ ] Test lại toàn bộ 3 usecase trên URL public.
- [ ] **Tiêu chí xong:** demo chạy được từ URL HF Space, không phụ thuộc máy local.

## Milestone 11 — Chuẩn bị báo cáo (Yêu cầu 2 — 5 điểm)

- [ ] Chụp/ghi lại sơ đồ kiến trúc + processing flow (dùng lại sơ đồ mục 3 spec).
- [ ] Mô tả enrollment procedure (mục 6 spec) trong báo cáo.
- [ ] Liệt kê rõ giới hạn đã biết (mục 8 spec: DB reset, Deezer 30s, toạ độ demo cố định, chưa test noise cabin xe thật — kế thừa từ Module A).

## Việc cần chốt khi bắt đầu code (không block viết plan, nhưng phải quyết định ở milestone tương ứng)

- Cách xác định "ai đang nói" trước `verify()` (M9).
- Cách gộp 7 embedding enrollment (M5).

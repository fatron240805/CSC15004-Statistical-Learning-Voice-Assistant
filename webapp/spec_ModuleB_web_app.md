# Spec — Module B: Web App Trợ lý ảo Xe thông minh
**Người phụ trách:** Hiếu | **Đề 3 — Secure Virtual Assistant with Speaker Recognition**
**Nhận bàn giao từ:** Module A (Khoa) — `model/moduleA_artifacts_final/`

---

## 1. Phạm vi & ranh giới với Module A

Module B **kế thừa nguyên** `speaker_model.py` và `checkpoints/best.pt` từ Module A, không sửa code hay threshold mặc định (`DEFAULT_THRESHOLD_SV = 0.35`, `DEFAULT_THRESHOLD_SID = 0.20`). Việc của Module B:

- Xây web app hoàn chỉnh: ghi âm → ASR → orchestrator → 3 usecase (general/SV/SID) → TTS.
- Component enrollment & quản lý thành viên trong xe (thu giọng, lưu DB).
- `check_audio_quality()` cho luồng enrollment (Module A chỉ định hướng, không code).
- Convert audio `.webm` (MediaRecorder trình duyệt) → `.wav` 16kHz mono trước khi gọi `get_embedding()` — **trách nhiệm của Module B**, dùng `pydub`/`ffmpeg`.
- Đảm bảo logic gating SV/SID nằm cứng trong code Python (mục 3), không giao cho LLM quyết định.

## 2. Tech stack đã chốt

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Backend | FastAPI (Python) | Chạy trực tiếp `speaker_model.py` (torch/speechbrain) trong cùng process, load model 1 lần lúc startup |
| Frontend | React (Vite) hoặc HTML/JS thuần | Gộp chung 1 Space với backend — FastAPI serve luôn static build, tránh CORS/multi-deploy |
| ASR | PhoWhisper (`transformers`) | Chạy CPU trong backend, đủ cho câu ngắn 2-6 giây |
| Orchestrator | Gemini API (key đã có trong `.env.txt`) | Ép **structured/JSON output** `{intent, entities}` — Gemini chỉ classify, KHÔNG tự quyết định có cần SV hay không |
| TTS | gTTS | Free, không cần key riêng |
| Speaker model | `speaker_model.py` (Module A) | Không đổi interface/threshold |
| Database | SQLite (SQLAlchemy) | Local trong container HF Space — **chấp nhận reset khi Space sleep/restart**, đủ cho demo, không cần vector DB (identify() đã dot-product thuần Python) |
| Nhạc (SID) | Deezer API | Free, không cần key cho search; **giới hạn: chỉ preview 30s/bài**, chấp nhận cho demo |
| Thời tiết/định vị (General) | OpenWeatherMap | Free tier; vị trí dùng **toạ độ cố định demo** (không có GPS xe thật) |
| Deploy | Hugging Face Spaces (Docker SDK, free CPU tier) | ~16GB RAM free, đủ chạy đồng thời PhoWhisper + speechbrain |

**Rủi ro đã biết, chấp nhận:** DB mất dữ liệu khi HF Space sleep (free tier không có persistent disk) → cần re-enroll sau mỗi lần Space ngủ dậy nếu demo cách nhau lâu. Nếu cần bền hơn, cân nhắc chuyển sang Supabase (free Postgres) sau — không bắt buộc cho demo báo cáo.

## 3. Kiến trúc pipeline tổng thể

```
[Browser: ghi âm .webm]
        │
        ▼
  convert .webm → .wav 16kHz mono (pydub/ffmpeg)
        │
        ▼
  ASR: PhoWhisper  →  text
        │
        ▼
  Orchestrator: Gemini API (JSON mode)
  input: text          output: { "intent": "...", "entities": {...} }
        │
        ▼
  [GATE CỨNG - code Python, KHÔNG phải LLM]
  bảng mapping intent → usecase → có cần SV không
        │
   ┌────┼─────────────┬─────────────┐
   ▼                  ▼             ▼
General            SV            SID
(không auth)   (verify() trước)  (identify())
   │                  │             │
   ▼                  ▼             ▼
Action Handler tương ứng (mục 4)
        │
        ▼
  TTS: gTTS → audio response → trả về Browser
```

## 4. Orchestrator (Gemini) — hợp đồng I/O

- **Input:** text đã ASR (tiếng Việt).
- **Output bắt buộc dạng JSON có schema cố định**, ví dụ:
  ```json
  { "intent": "sv_unlock_door" | "sv_start_engine" | "sid_play_playlist" | "general_weather" | "general_location" | "unknown", "entities": {} }
  ```
- Gemini **chỉ làm nhiệm vụ hiểu intent/entity**, không quyết định pass/fail xác thực.
- Code Python có 1 bảng mapping tĩnh (hardcode), ví dụ:

  | intent | usecase | cần SV? |
  |---|---|---|
  | `general_weather`, `general_location` | General | Không |
  | `sv_unlock_door`, `sv_start_engine` | SV | **Có** |
  | `sid_play_playlist` | SID | Không (nhưng cần `identify()`) |

  Bảng này quyết định route, **không phải Gemini quyết định**.

## 5. Ba usecase — chi tiết

### 5.1 General — thời tiết / nhiệt độ / định vị
- Không cần xác thực giọng nói.
- Gọi OpenWeatherMap current weather API với toạ độ cố định demo (vd TP.HCM).
- Trả lời qua TTS: "Hiện tại trời [mô tả], nhiệt độ [x]°C."
- Định vị: trả toạ độ/tên địa điểm demo cố định (do không có GPS xe thật) — cần nêu rõ giới hạn này trong báo cáo.

### 5.2 SV — mở khoá cửa / khởi động xe
1. Ghi âm runtime → `get_embedding()`.
2. So khớp với embedding đã enroll của user hiện tại (hoặc quét toàn bộ DB nếu chưa xác định ai đang nói — cần chốt thêm ở bước code: có yêu cầu chọn "tôi là ai" trước hay auto quét).
3. `verify(emb_enrolled, emb_runtime)` — threshold mặc định 0.35.
4. Nếu `True`: thực hiện action (mock: đổi trạng thái "đã mở khoang"/"đã khởi động"), ghi vào bảng `action_log` (timestamp, user_id, action, verified=True).
5. Nếu `False`: cho phép **retry 1-2 lần** trước khi báo "không xác thực được" (theo khuyến nghị README Module A, vì FRR thực nghiệm ~37.5%) — không chặn ngay từ lần đầu.
6. Log cả các lần verify thất bại (verified=False) để phục vụ báo cáo/demo.

### 5.3 SID — phát playlist cá nhân hoá
1. Ghi âm runtime → `get_embedding()`.
2. `identify(emb, db_embeddings)` — threshold mặc định 0.20 — trả về `user_id` hoặc `None`.
3. Nếu có `user_id`: lấy danh sách bài hát yêu thích của user từ bảng `preferences` → gọi Deezer API lấy preview URL từng bài → trả về frontend hiển thị playlist thật + phát qua `<audio>`.
4. Nếu `None`: phản hồi không nhận diện được, có thể fallback playlist mặc định hoặc yêu cầu enroll.

## 6. Enrollment flow

1. User nhập thông tin cơ bản (tên) + chọn bài hát yêu thích ban đầu (lưu vào `preferences`).
2. Đọc lần lượt 7 câu mẫu trong `enrollment_sentences.md` (nguyên văn, không đổi).
3. Với mỗi câu ghi âm:
   - Convert `.webm` → `.wav` 16kHz mono.
   - **Check chất lượng:** chạy qua PhoWhisper ASR, so khớp WER với câu mẫu gốc; kiểm tra độ dài audio tối thiểu (khuyến nghị ≥ vài giây, khớp câu mẫu ước tính 4-6s). Nếu WER quá cao hoặc audio quá ngắn/nhiễu → yêu cầu đọc lại câu đó.
4. Sau khi đủ 7 câu đạt chất lượng: tính `get_embedding()` cho từng câu, **gộp thành 1 embedding đại diện** (trung bình rồi L2-normalize lại, hoặc lưu cả 7 và dùng câu tốt nhất — quyết định lúc code, ghi rõ vào báo cáo).
5. Lưu embedding + metadata vào bảng `users`.

## 7. Database — schema đề xuất

```
users
  user_id (PK)
  name
  embedding        -- JSON array 192-dim, đã L2-normalize
  enrollment_date

preferences
  user_id (FK -> users)
  favorite_tracks   -- JSON list track_id/tên bài (khớp Deezer)

action_log
  id (PK)
  timestamp
  user_id (nullable nếu chưa xác định được ai)
  action            -- "unlock_door" | "start_engine" | "play_playlist" | ...
  verified          -- bool, kết quả verify()/identify()
  score             -- điểm cosine similarity thực tế (để debug/báo cáo)
```

## 8. Việc còn mở / cần xác nhận thêm

- **Cách xác định "ai đang nói" trước khi gọi `verify()`:** chọn user từ UI trước, hay auto quét toàn bộ DB bằng `identify()` trước rồi mới `verify()` với đúng người đó? Ảnh hưởng UX, cần chốt lúc code.
- **Cách gộp 7 embedding enrollment thành 1** (trung bình vs giữ nhiều mẫu) — chưa chốt, ảnh hưởng độ chính xác `verify()`/`identify()`.
- DB reset khi HF Space sleep — chấp nhận cho demo, nêu rõ trong báo cáo là giới hạn hạ tầng free tier, không phải lỗi thiết kế.
- Deezer preview 30s — nêu rõ trong báo cáo là giới hạn nguồn nhạc free, không phải giới hạn hệ thống SID.
- Chưa quét lại threshold SID (0.20) ở nhiều mức khác — Module A đã note trong README, có thể thử nếu còn thời gian bằng `evaluate_identification.py --threshold <giá_trị>`.

## 9. Tài liệu liên quan

- `model/moduleA_artifacts_final/README_ban_giao.md` — interface, threshold, giới hạn model.
- `model/moduleA_artifacts_final/enrollment_sentences.md` — 7 câu mẫu, dùng nguyên văn.
- `Secure-Virtual-Assistant-with-Speaker-Recognition.pdf` — yêu cầu đề bài gốc.

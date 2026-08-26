# README Bàn giao — Module A (Khoa) → Module B (Hiếu)

## 1. Artefact bàn giao

| File | Mô tả |
|---|---|
| `speaker_model.py` | Class `SpeakerModel` độc lập, copy nguyên file vào backend, không cần mang theo code Module A khác |
| `checkpoints/best.pt` | Checkpoint ECAPA-TDNN đã fine-tune trên VoxVietnam (150 speaker, streaming subsample) |
| `metrics/clean_test_metrics.md` | EER/minDCF đo trên VoxVietnam |
| `enrollment_sentences.md` | 7 câu mẫu enrollment, dùng nguyên văn trong luồng đăng ký (M5) |
| `spec_ModuleA_speaker_training.md` | Spec gốc, tham khảo thêm bối cảnh/quyết định kỹ thuật |

## 2. Interface — dùng đúng như sau, không đổi tên hàm

```python
from speaker_model import SpeakerModel

model = SpeakerModel(checkpoint_path="checkpoints/best.pt")

# Trích embedding từ 1 file audio (bắt buộc .wav, 16kHz mono)
emb = model.get_embedding("path/to/audio.wav")  # trả về np.ndarray, đã L2-normalize

# Speaker Verification — dùng cho lệnh nhạy cảm (threshold mặc định = 0.35, không cần truyền)
is_same_person = model.verify(emb_enrolled, emb_runtime)

# Speaker Identification — dùng để cá nhân hoá (threshold mặc định = 0.20, không cần truyền)
db = {"user_001": emb_user1, "user_002": emb_user2}
matched_id = model.identify(emb_runtime, db)  # None nếu không khớp ai

# Nếu muốn truyền tường minh thay vì dùng mặc định:
is_same_person = model.verify(emb_enrolled, emb_runtime, threshold=0.35)
matched_id = model.identify(emb_runtime, db, threshold=0.20)
```

**Yêu cầu môi trường:** `speechbrain`, `torch`, `torchaudio`, `numpy` — khuyến nghị pin version `torch==2.4.1`, `numpy<2` (`1.26.4`), vì nhóm đã gặp lỗi tương thích thật giữa các bản mới hơn khi test local (xem lịch sử debug nếu cần đối chiếu).

**Audio input:** phải là `.wav` 16kHz mono. Nếu backend nhận `.webm` từ trình duyệt (MediaRecorder), **cần convert bằng `pydub`/`ffmpeg` trước khi gọi `get_embedding()`** — việc convert này thuộc trách nhiệm Module B, `speaker_model.py` không tự xử lý.

## 3. ⚠️ QUAN TRỌNG — 2 threshold TÁCH RIÊNG cho SV và SID

`spec_ModuleA_speaker_training.md` (viết trước khi có kết quả thật) chưa ghi con số cuối cùng, và bản đầu tiên của README này từng dùng chung 1 threshold cho cả 2 hàm — **đã sửa lại sau khi phát hiện vấn đề qua thực nghiệm**, xem lý do bên dưới.

| Hàm | Use case | Threshold mặc định | Vì sao |
|---|---|---|---|
| `verify()` (SV) | Lệnh nhạy cảm (mở khoang, khởi động xe) | **`0.35`** | Ưu tiên tuyệt đối an toàn — đo được FAR=0% trên thực nghiệm nhỏ |
| `identify()` (SID) | Cá nhân hoá (gợi ý nhạc, chỉnh ghế) | **`0.20`** | Ưu tiên trải nghiệm — threshold của SV áp cho SID làm giảm accuracy nghiêm trọng |

**Không phải `0.1443`** cho cả 2 — đây là threshold đo tại điểm cân bằng EER trên VoxVietnam (dataset gốc), nhưng khi test với giọng thu qua mic điện thoại thông thường (ngoài phạm vi VoxVietnam), threshold này gây **False Accept Rate cao (10-24%)** cho SV — nhận nhầm người lạ là người quen, rất nguy hiểm cho lệnh nhạy cảm. Threshold `0.35` được chọn qua thực nghiệm nhỏ, đạt FAR = 0%.

**Vì sao SID cần threshold riêng, thấp hơn:** đo trên test set VoxVietnam (105 truy vấn, 15 speaker), nếu dùng chung threshold `0.35` của SV cho `identify()`:
- Top-1 accuracy thật của model (không áp threshold): **84.76%**
- Accuracy khi áp threshold `0.35`: tụt xuống **73.33%** — vì **26.67% truy vấn bị từ chối oan** thành "unknown" dù model đã chọn đúng người, chỉ vì điểm số chưa đủ cao để vượt ngưỡng an toàn của SV.

Hạ threshold xuống `0.20` cho riêng `identify()` giúp giảm tỉ lệ từ chối oan này, chấp nhận đánh đổi độ chọn lọc thấp hơn — hợp lý vì hậu quả nhận nhầm ở SID (gợi ý nhạc sai gu) nhẹ hơn nhiều so với SV (truy cập trái phép). Chi tiết xem `metrics/identification_report.md`.

## 4. ⚠️ Đánh đổi cần biết trước khi thiết kế UX

**Với `verify()` (threshold `0.35`):** FRR (False Reject Rate) đo được trong thực nghiệm nhỏ là ~37.5% — nghĩa là hơn 1/3 khả năng người dùng thật sẽ bị từ chối oan ở lần thử đầu tiên khi dùng SV.

**Khuyến nghị bắt buộc cho Module B:** thiết kế UX phải cho phép **thử lại (retry)** khi `verify()` trả về `False`, thay vì chặn hẳn ngay lần đầu. Ví dụ: cho phép nói lại lệnh 1-2 lần trước khi báo "không xác thực được". Đây là cách xử lý đúng chuẩn thực tế cho hệ thống sinh trắc học có FRR cao — không nên coi `False` ở lần đầu là quyết định cuối cùng.

**Với `identify()` (threshold `0.20`):** tỉ lệ từ chối oan thấp hơn nhiều (do threshold đã hạ), nhưng đổi lại độ chọn lọc kém hơn — nếu 2 người dùng có giọng khá giống nhau, khả năng nhận nhầm nhân dạng khi cá nhân hoá cao hơn so với khi dùng threshold `0.35`. Chấp nhận được vì hậu quả nhẹ (gợi ý sai gu nhạc), không cần retry logic như SV.

## 5. Giới hạn đã biết — nêu trong báo cáo, không phải lỗi cần Hiếu sửa

- Model fine-tune trên **150/1000+ speaker** của VoxVietnam (subsample qua streaming, do giới hạn dung lượng đĩa Kaggle), EER trên VoxVietnam: **6.58-7.25%**.
- Chưa thực nghiệm robustness với noise cabin xe thật (quyết định bỏ qua do giới hạn thời gian, xem spec gốc mục 5).
- Threshold `0.35` (SV) dựa trên cỡ mẫu thực nghiệm nhỏ (dưới 50 cặp) — đủ dùng cho demo, không phải con số đã kiểm định thống kê chặt chẽ. Threshold `0.20` (SID) là giá trị đề xuất ban đầu dựa trên 1 mức thử duy nhất, chưa quét nhiều mức để tìm điểm tối ưu — nếu có thời gian, nên thử thêm vài threshold khác (0.15, 0.25...) bằng `evaluate_identification.py --threshold <giá_trị>` để so sánh.
- Chỉ xác thực bằng giọng nói — không chống được replay attack (phát lại bản ghi âm) hoặc deepfake giọng nói, cần nêu rõ đây là giới hạn hệ thống trong báo cáo nếu bị hỏi.

## 6. Việc cần Hiếu xác nhận trước khi coi là tích hợp xong

1. Chạy thử `get_embedding()` với 1 file `.wav` bất kỳ trên máy Hiếu, xác nhận môi trường (`speechbrain`/`torch`/`numpy`) không xung đột version.
2. Test `verify()` với 1 cặp audio enrollment/runtime giả lập, xác nhận `True`/`False` trả về đúng logic mong đợi trước khi ráp vào `speaker_service.py`.
3. Đảm bảo logic gating SV/SID vẫn nằm ở code Python cứng trong Module B (M4), không giao cho LLM quyết định — đúng nguyên tắc đã thống nhất trong `00_Workflow_And_TechStack.pdf` mục 2.

Có vấn đề gì khi tích hợp, báo lại Khoa trực tiếp thay vì tự đoán hành vi của `speaker_model.py`.

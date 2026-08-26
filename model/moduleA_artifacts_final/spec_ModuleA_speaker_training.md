# Spec — Module A: Speaker Verification / Identification
**Người phụ trách:** Khoa | **Đề 3 — bối cảnh: Trợ lý ảo xe thông minh**
**Model:** ECAPA-TDNN (fine-tune từ pretrained SpeechBrain) | **Dataset:** VoxVietnam
**Nguồn quyết định:** ghi đè lựa chọn WavLM+CAM++ trong `00_Workflow_And_TechStack.pdf` bằng chốt mới nhất trong `2026_08_21.txt`.

---

## 0. Đổi bối cảnh: nhà thông minh → xe thông minh

Ảnh minh hoạ nhóm dùng làm ví dụ là *nhà thông minh*. Ánh xạ sang *xe thông minh* như sau — việc này ảnh hưởng đến cách bạn thiết kế thực nghiệm robustness (mục 3), không ảnh hưởng đến kiến trúc model:

| Nhóm chức năng | Nhà thông minh (ví dụ gốc) | Xe thông minh (bản của nhóm) |
|---|---|---|
| General (không xác thực) | Phát nhạc | Phát nhạc, hỏi thời tiết/giao thông, bật điều hoà mức thường |
| SID (cá nhân hoá) | Gợi ý nhạc theo giới tính/độ tuổi chủ nhà | Tự chỉnh ghế lái/gương/nhiệt độ theo tài xế đã đăng ký, gợi ý nhạc/tuyến đường quen thuộc |
| SV (xác thực thẩm quyền) | Mở cửa, mở app thanh toán | Mở khoang chứa đồ, khởi động động cơ, thanh toán qua ví liên kết trên xe |

**Điểm khác biệt quan trọng nhất cho Module A:** môi trường âm thanh trong xe (tiếng động cơ, tiếng gió, tiếng lốp, loa phát nhạc) khắc nghiệt hơn phòng khách rất nhiều → SNR thấp hơn thực tế đáng kể so với dataset gốc thu trong điều kiện sạch. Đây là lý do bước thực nghiệm robustness (mục 3, bước 5 trong sơ đồ) là phần bạn cần đầu tư kỹ, không phải chỉ làm cho có.

## 1. Phạm vi công việc (ranh giới rõ với Module B)

Bạn (Module A) **chỉ** chịu trách nhiệm phần model, chạy trên Kaggle, độc lập với web app:

- Chọn, phân tích, chia dataset
- Fine-tune ECAPA-TDNN
- Đánh giá bằng metric chuẩn (EER, minDCF)
- Thực nghiệm robustness theo noise xe hơi → đề xuất threshold (số liệu, **không code** phần dùng threshold)
- Thiết kế nội dung câu mẫu enrollment (nghiên cứu ngôn ngữ học, **không code**)
- Xuất checkpoint + 1 file `speaker_model.py` (class wrapper) cho Hiếu (Module B) import và code phần `check_audio_quality()`, API, DB thật trong FastAPI.

Bạn **không** phụ trách: ASR/TTS, orchestrator LLM, backend FastAPI, frontend, database — đó là việc của Hiếu/Module B.

## 2. Checklist chuẩn bị TRƯỚC KHI CODE

### 2.1 Môi trường
- [ ] Tài khoản Kaggle, xác nhận số giờ GPU còn lại (P100/T4 x2), ước tính đủ cho: 1 vòng fine-tune + vài lần eval + thực nghiệm robustness (nên dành ≥ 15-20 GPU-hour dự phòng).
- [ ] Tài khoản HuggingFace (để tải pretrained `speechbrain/spkrec-ecapa-voxceleb` và dataset VoxVietnam nếu host trên HF Hub) — kiểm tra license/điều khoản sử dụng của VoxVietnam trước khi tải.
- [ ] Cài đặt trong Kaggle notebook: `speechbrain`, `torchaudio`, `soundfile`, `scikit-learn`, `pandas`, `numpy`.

### 2.2 Quyết định cần chốt trước khi viết code (ghi vào báo cáo)
- [ ] Sample rate chuẩn hoá: **16kHz mono** (bắt buộc, vì cả VoxVietnam và ECAPA-TDNN pretrained đều ở 16kHz).
- [ ] Tỉ lệ chia train/val/test — đề xuất: 80/10/10 theo **speaker-disjoint split** (một người nói không được xuất hiện ở cả train và test, nếu không sẽ đánh giá sai bản chất bài toán verification).
- [ ] Định dạng eval trial pairs: cách sinh cặp (audio1, audio2, label same/different) cho test set — cần cân bằng số cặp positive/negative.
- [ ] Loss function: AAM-softmax (Additive Angular Margin) — chuẩn cho speaker embedding, không dùng cross-entropy thường.
- [ ] Optimizer/scheduler: Adam, learning rate warmup + cosine decay (mặc định của SpeechBrain recipe ECAPA, không cần tự thiết kế lại).

### 2.3 Thực nghiệm robustness (đặc thù bối cảnh xe — quan trọng)
- [ ] Chuẩn bị bộ noise mô phỏng cabin xe: có thể dùng noise segment từ MUSAN (traffic/car noise nếu có) hoặc tự thu/tải noise động cơ, tiếng gió trên YouTube (ghi nguồn rõ trong báo cáo).
- [ ] Xác định các mức SNR sẽ test: ví dụ 20dB (xe yên tĩnh, đỗ), 10dB (chạy tốc độ trung bình), 5dB (cao tốc/mở cửa sổ) — augment test set ở các mức này rồi đo lại EER.
- [ ] Xác định các mức duration audio ngắn sẽ test (vì lệnh thoại trong xe thường ngắn, 2-5 giây) — đo ảnh hưởng lên EER khi cắt ngắn.
- [ ] Output: bảng EER theo từng điều kiện (SNR × duration) → từ đó đề xuất threshold vận hành thực tế + khuyến nghị enrollment tối thiểu bao nhiêu giây audio.

### 2.4 Thiết kế câu mẫu enrollment (chỉ nội dung, không code)
- [ ] Soạn 7 câu tiếng Việt phủ đa dạng thanh điệu (6 thanh) và nhóm âm vị chính, nên lồng ngữ cảnh xe hơi để tự nhiên khi enroll trong app (ví dụ câu liên quan đến lái xe, không bắt buộc nhưng hợp lý về mặt trải nghiệm người dùng) — Module B sẽ dùng nguyên văn các câu này trong luồng đăng ký.
- [ ] Ghi rõ rationale chọn câu (âm vị nào được phủ, vì sao cần ít nhất 7 câu) để đưa vào báo cáo.

### 2.5 Output contract phải bàn giao cho Module B
- [ ] `speaker_model.py`: 1 class duy nhất, interface cố định gồm `load_checkpoint()`, `get_embedding(audio_path) -> np.ndarray`, `verify(emb1, emb2, threshold) -> bool`, `identify(emb, db_embeddings, threshold) -> speaker_id | None`.
- [ ] File checkpoint model (`.ckpt`/`.pt`).
- [ ] `threshold_report.md`/`.json`: threshold đề xuất kèm điều kiện đo (SNR, duration).
- [ ] `enrollment_sentences.json`: 7 câu mẫu + rationale.
- [ ] `metrics_report.md`: EER, minDCF trên test set sạch và test set nhiễu.

## 3. Cấu trúc thư mục

```
moduleA_speaker_training/
├── spec_ModuleA_speaker_training.md   # file này
├── configs/
│   └── ecapa_voxvietnam.yaml          # toàn bộ hyperparameter, path, split ratio
├── data/
│   └── prepare_dataset.py             # tải, kiểm tra, sinh manifest + split
├── src/
│   ├── dataset.py                     # Dataset/Dataloader cho fine-tune
│   ├── model.py                       # Load ECAPA-TDNN pretrained + head AAM-softmax
│   ├── train.py                       # Vòng lặp fine-tune
│   ├── evaluate.py                    # Tính EER, minDCF trên trial pairs
│   ├── robustness_experiment.py       # Augment SNR/duration -> đo EER theo điều kiện
│   └── export.py                      # Xuất speaker_model.py + checkpoint cho Module B
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   └── speaker_model.py               # artefact cuối cùng bàn giao Module B
└── requirements.txt
```

## 4. Các bước tiếp theo (sau khi hoàn tất checklist ở mục 2)

1. Chạy `data/prepare_dataset.py` trên Kaggle để tải VoxVietnam, sinh `splits/{train,val,test}.csv` và một file phân tích chất lượng dataset (class/speaker imbalance, số utterance/người, phân bố duration).
2. Chạy `src/train.py` với `configs/ecapa_voxvietnam.yaml` để fine-tune.
3. Chạy `src/evaluate.py` trên test set sạch → ghi EER/minDCF baseline.
4. Chạy `src/robustness_experiment.py` → sinh bảng EER theo SNR/duration, chốt threshold.
5. Hoàn thiện nội dung 7 câu enrollment (`enrollment_sentences.json`).
6. Chạy `src/export.py` để đóng gói `speaker_model.py` + checkpoint + report, gửi cho Hiếu.
7. Viết phần báo cáo Yêu cầu 1 (5 điểm) dựa trên các report đã sinh ra.

## 5. Rủi ro cần lưu ý riêng cho bối cảnh xe

- Nếu không tìm được noise cabin xe chất lượng tốt, thực nghiệm robustness sẽ kém thuyết phục — nên ưu tiên tìm noise thật (dashcam audio, MUSAN) hơn là noise trắng tổng hợp thuần tuý.
- Lệnh thoại trong xe thường rất ngắn (2-3 giây) — nếu EER tăng mạnh ở duration ngắn, cần nêu rõ trong báo cáo đây là giới hạn thực tế, đề xuất fallback (ví dụ yêu cầu người dùng nói câu dài hơn cho lệnh nhạy cảm).
- Tránh chọn threshold "đẹp" theo cảm tính — phải bám vào EER đo được, giải thích trade-off FAR/FRR khi vấn đáp (đặc biệt vì lệnh SV trong xe liên quan đến an toàn/tài sản, nên thiên về giảm FAR — chấp nhận FRR cao hơn một chút).

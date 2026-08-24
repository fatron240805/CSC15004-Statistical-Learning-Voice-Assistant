# Module A — Speaker Verification / Identification cho Trợ lý ảo Xe thông minh

## Chương trình này dùng để làm gì

Đây là toàn bộ pipeline huấn luyện và đánh giá model nhận diện giọng nói (ECAPA-TDNN) cho **Đề 3 — Secure Virtual Assistant with Speaker Recognition**, phần việc của **Khoa (Module A)**. Model được train ở đây phục vụ 2 bài toán:

- **Speaker Verification (SV):** xác thực đúng người trước khi thực hiện lệnh nhạy cảm (mở khoang xe, khởi động động cơ, thanh toán qua ví liên kết trên xe).
- **Speaker Identification (SID):** nhận diện ai đang nói để cá nhân hoá trải nghiệm (tự chỉnh ghế/gương theo tài xế, gợi ý nhạc quen thuộc).

Toàn bộ pipeline chạy trên **Kaggle** (cần GPU để train hợp lý), độc lập hoàn toàn với web app (Module B) — kết quả cuối cùng là 1 file `speaker_model.py` + checkpoint, bàn giao cho Module B tích hợp vào backend.

## Vì sao chọn ECAPA-TDNN + VoxVietnam

- **ECAPA-TDNN:** kiến trúc chuẩn, phổ biến trong lĩnh vực speaker recognition, có pretrained chất lượng tốt từ SpeechBrain — chỉ cần fine-tune thay vì train từ đầu, phù hợp giới hạn thời gian đồ án.
- **VoxVietnam:** dataset tiếng Việt có quy mô lớn (1000+ speaker), có paper tham chiếu, phù hợp với bối cảnh sản phẩm hướng đến người dùng Việt Nam — quan trọng vì model speaker recognition train trên tiếng Anh thường không đủ tốt cho tiếng Việt.
- Nhóm từng thử baseline khác (WavLM + CAM++ trên Vietnam-Celeb) nhưng kết quả kém (EER ~28%), nên chuyển hẳn sang hướng này.

## Luồng xử lý tổng thể

```
prepare_dataset.py  →  train.py  →  evaluate.py  →  export.py
   (chuẩn bị data)      (fine-tune)   (đo EER/AUC)   (đóng gói bàn giao)
```

### 1. `data/prepare_dataset.py` — Chuẩn bị dữ liệu

VoxVietnam nặng 44.2GB, vượt quota đĩa mặc định của Kaggle (~20GB). Script này **không tải nguyên dataset** mà dùng cơ chế **streaming** của HuggingFace: đọc từng sample một qua mạng, chỉ giữ lại đúng số speaker/utterance cần thiết (mặc định cấu hình trong `configs/ecapa_voxvietnam.yaml`), ghi ra file `.wav` thật, rồi chia thành:

- **Test set:** tách theo speaker-disjoint (người trong test hoàn toàn chưa từng xuất hiện lúc train) — đúng chuẩn đánh giá open-set verification.
- **Train/Val:** chia theo utterance trên các speaker còn lại (cùng speaker pool, chỉ khác câu nói) — vì val dùng để theo dõi việc học phân loại lúc train (closed-set), không phải đánh giá verification.

### 2. `src/train.py` — Fine-tune model

Load pretrained `speechbrain/spkrec-ecapa-voxceleb`, gắn thêm lớp phân loại **AAM-Softmax** (chuẩn cho speaker embedding, ép các vector embedding tách xa nhau rõ rệt theo từng người), fine-tune trên dữ liệu VoxVietnam đã chuẩn bị. Model liên quan: `src/model.py` (định nghĩa kiến trúc), `src/dataset.py` (đọc và tiền xử lý audio).

### 3. `src/evaluate.py` — Đánh giá SV

Đo 3 chỉ số chuẩn cho speaker verification trên tập test:
- **EER** (Equal Error Rate) — sai số tại điểm cân bằng giữa nhận nhầm người lạ và từ chối nhầm người quen.
- **AUC** — khả năng phân tách tổng thể của model, không phụ thuộc 1 threshold cụ thể.
- **minDCF** — chi phí sai số theo chuẩn NIST SRE.

### 4. `src/export.py` — Đóng gói bàn giao

Sinh ra file `speaker_model.py` độc lập (copy thẳng vào backend, không cần mang theo code khác) với interface cố định: `get_embedding()`, `verify()`, `identify()`. Threshold cho `verify()` (SV) và `identify()` (SID) được **tách riêng**, vì thực nghiệm cho thấy 1 threshold tối ưu cho SV (ưu tiên an toàn tuyệt đối) sẽ làm giảm mạnh độ chính xác nếu dùng chung cho SID (ưu tiên trải nghiệm).

### 5. `src/robustness_experiment.py` — Thực nghiệm mở rộng (tuỳ chọn)

Mô phỏng điều kiện noise trong cabin xe (tiếng động cơ, gió) ở nhiều mức SNR/duration khác nhau, đo lại EER để đề xuất threshold sát thực tế hơn. **Không bắt buộc theo đề bài** — nhóm quyết định bỏ qua bước này do giới hạn thời gian, dùng thay bằng cách kiểm chứng threshold bằng dữ liệu giọng nói thật (kết quả nằm trong `outputs/metrics/`, không sinh ra từ script này).

## Cách chạy trên Kaggle

```bash
pip install -r requirements.txt

# Cần HF_TOKEN đã lấy từ Kaggle Secrets, set biến môi trường trước:
# os.environ["HF_TOKEN"] = hf_token

python data/prepare_dataset.py --config configs/ecapa_voxvietnam.yaml
python src/train.py --config configs/ecapa_voxvietnam.yaml
python src/evaluate.py --config configs/ecapa_voxvietnam.yaml --checkpoint outputs/checkpoints/best.pt
python src/export.py --config configs/ecapa_voxvietnam.yaml --checkpoint outputs/checkpoints/best.pt --threshold_sv <giá_trị> --threshold_sid <giá_trị>
```

## Giới hạn đã biết

- Model fine-tune trên **subsample** của VoxVietnam (150-300 speaker, tuỳ cấu hình), không phải toàn bộ dataset — do giới hạn dung lượng đĩa Kaggle, không phải sơ suất.
- Threshold cuối cùng được hiệu chỉnh dựa trên thực nghiệm nhỏ với giọng nói thật ngoài phạm vi VoxVietnam (phát hiện vấn đề domain mismatch — mic điện thoại khác điều kiện thu âm gốc của dataset).
- Chỉ xác thực bằng giọng nói — không chống được replay attack hay giọng nói giả (deepfake), là giới hạn hệ thống cần nêu rõ trong báo cáo.

## Tài liệu liên quan

- `spec_ModuleA_speaker_training.md` — spec chi tiết, checklist chuẩn bị, phân tích rủi ro theo bối cảnh xe.
- `outputs/metrics/` — kết quả EER/AUC/minDCF và độ chính xác SID sau khi chạy `evaluate.py`.

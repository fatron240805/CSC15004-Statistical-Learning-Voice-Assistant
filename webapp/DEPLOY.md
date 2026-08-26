# Deploy lên Hugging Face Spaces (Milestone 10)

**Docker SDK bị khoá (paid) trên tài khoản hiện tại** — dùng **Gradio SDK** thay thế: Space chỉ chạy `python app.py`, không yêu cầu code Gradio thật, nên `app.py` ở đây chỉ khởi động thẳng FastAPI qua uvicorn. `packages.txt` thay cho `Dockerfile` để cài `ffmpeg` (Gradio/Streamlit SDK có hỗ trợ apt package qua file này). Nếu sau này tài khoản có Docker (nâng cấp/xác minh), `Dockerfile` vẫn còn trong repo, dùng lại được ngay — xem mục "Phương án Docker (khi có quyền)" cuối file.

## 1. Tạo Space

1. Vào https://huggingface.co/new-space
2. Chọn **Gradio** làm SDK (Docker đang bị khoá/paid) → template **Blank** (không dùng giao diện Gradio thật, chỉ mượn SDK để chạy `app.py` riêng).
3. Hardware: chỉ **ZeroGPU (Free)** khả dụng trên tài khoản này (CPU Basic bị khoá) — chọn ZeroGPU, không sao vì app chỉ dùng CPU (`torch` bản CPU-only), phần GPU on-demand của ZeroGPU không dùng tới, không ảnh hưởng.
4. **Storage Bucket:** bật, tạo bucket mới (vd `SL-storage`), **mount path `/data`**, access **Read & Write**. Code đã tự động phát hiện `/data` và dùng làm nơi lưu bền cho SQLite DB + cache pretrained model (xem `backend/config.py`) — giải quyết luôn nhược điểm "mất dữ liệu khi Space sleep" đã nêu trong spec mục 8.
5. Sau khi tạo xong, Space có 1 git remote riêng, ví dụ `https://huggingface.co/spaces/<username>/<space-name>`.

## 2. Set Secrets (KHÔNG commit .env)

Vào Space → Settings → **Variables and secrets** → thêm:
- `GEMINI_API_KEY`
- `OPENWEATHER_KEY`

(`HF_KEY` không bắt buộc — chỉ giúp tăng rate limit khi tải pretrained model từ HuggingFace Hub.)

## 3. Đẩy code lên Space (git repo riêng, khác GitHub)

Space yêu cầu file `README.md` ở root với YAML frontmatter để biết cấu hình. **Tự tạo file này TRỰC TIẾP trong git repo của Space** (không phải trong repo GitHub hiện tại), nội dung:

```yaml
---
title: Secure Virtual Assistant
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.11"
app_file: app.py
---
```

Các bước đẩy code (chạy trong 1 thư mục riêng, KHÔNG phải trong repo GitHub này):

```bash
git clone https://huggingface.co/spaces/<username>/<space-name>
cd <space-name>
# copy toàn bộ nội dung webapp/ vào đây: app.py, requirements.txt, packages.txt,
# backend/, frontend/, checkpoints/ (Dockerfile/.dockerignore không cần cho Gradio SDK, giữ lại cũng không sao)
# tạo README.md như trên
git lfs install
git lfs track "*.pt"          # checkpoint 89MB cần git-lfs, không push thẳng
git add .
git commit -m "Deploy Module B"
git push
```

Lưu ý: `requirements.txt` dùng ở đây là **`webapp/requirements.txt`** (ở gốc, có dòng `--extra-index-url .../whl/cpu` để lấy bản torch CPU nhẹ) — không phải `webapp/backend/requirements.txt` (bản đó dùng cho venv dev local, không có dòng index CPU vì lúc dev local dùng bản GPU/CPU tuỳ máy đều được).

**Lưu ý về checkpoint (`checkpoints/best.pt`, 89MB):** file này đang bị `.gitignore` chặn trong repo GitHub chính (đúng ý — repo bài nộp không nên chứa file nặng, theo hướng dẫn đề bài mục 4 "upload Drive nếu quá lớn"). Khi đẩy sang HF Space, đây là git repo khác — cần add + `git lfs track` riêng cho nó ở đó, không ảnh hưởng `.gitignore` của repo GitHub.

## 4. Kiểm tra sau khi deploy

- Space build xong (theo dõi tab "Logs"), mở URL `https://<username>-<space-name>.hf.space`.
- Test `/health` → phải thấy `speaker_model_ready: true` sau khi model load xong (lần đầu có thể chậm vì tải pretrained model từ HF Hub).
- Test enrollment + chat qua giao diện web thật.

## Giới hạn đã biết (nêu trong báo cáo)

- **Đã nâng cấp so với bản trước:** nhờ Storage Bucket mount ở `/data`, SQLite DB và cache pretrained model giờ **sống sót qua Space sleep/restart** — không còn phải enroll lại mỗi lần. `backend/config.py` tự phát hiện `/data` (`DATA_DIR`), fallback về thư mục local khi dev trên máy không có bucket.
- Lần đầu khởi động (bucket rỗng) vẫn phải tải `speechbrain/spkrec-ecapa-voxceleb` (~90MB) + `vinai/PhoWhisper-base` từ HuggingFace Hub — mất vài phút; các lần sau đọc từ `/data/hf_cache` nên nhanh, kể cả sau khi Space sleep/restart.
- Chạy qua "Gradio SDK" chỉ là cách lách hạ tầng (không dùng Gradio thật) — không ảnh hưởng kiến trúc/pipeline đã thiết kế, nên nêu trong báo cáo là quyết định hạ tầng do giới hạn tài khoản, không phải thay đổi thiết kế hệ thống.
- Hardware ZeroGPU thay vì CPU Basic — cũng là giới hạn tài khoản, app không dùng GPU nên không ảnh hưởng chức năng, chỉ nêu cho đầy đủ trong báo cáo.

## Phương án Docker (khi có quyền)

Nếu sau này Docker SDK khả dụng: tạo Space mới chọn SDK **Docker**, dùng `webapp/Dockerfile` (đã build + test thật thành công trên máy local, image 708MB, health check pass) thay vì `app.py`/`packages.txt`/`requirements.txt` ở root. `README.md` frontmatter đổi `sdk: docker`, `app_port: 7860`, bỏ `python_version`/`app_file`.

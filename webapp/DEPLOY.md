# Deploy lên Hugging Face Spaces (Milestone 10) — ĐÃ LIVE

**URL:** https://thales1020-secure-virtual-assistant.hf.space
**Space:** https://huggingface.co/spaces/thales1020/secure-virtual-assistant

Đã deploy thành công qua `huggingface_hub` API (không phải git clone/push tay) + GitHub Actions tự động đồng bộ mỗi khi push. Tài khoản `thales1020` bị khoá cả **Docker SDK** lẫn **CPU Basic hardware** (yêu cầu PRO, xác nhận qua lỗi `402 Payment Required` khi thử qua API) — chỉ **Gradio SDK + ZeroGPU (Free)** khả dụng. 3 vấn đề riêng của ZeroGPU đã gặp và sửa, liệt kê ở mục 4 — nên đọc kỹ nếu redeploy hoặc đổi hardware.

## 1. Cấu hình hiện tại

| Thành phần | Giá trị |
|---|---|
| SDK | Gradio (mượn, không dùng giao diện Gradio thật — `app.py` chạy thẳng FastAPI qua uvicorn) |
| Hardware | ZeroGPU (Free) — app chỉ dùng CPU, không cần GPU thật |
| Storage | Bucket `thales1020/sl-storage`, mount `/data`, Read & Write — DB + cache pretrained model sống sót qua Space sleep/restart |
| Secrets | `GEMINI_API_KEY`, `OPENWEATHER_KEY` (set qua `add_space_secret`, không commit `.env`) |
| Deploy | `scripts/deploy_hf_space.py` (đồng bộ `webapp/` qua `HfApi.upload_folder`), chạy tay hoặc qua CI |

## 2. CI/CD — GitHub Actions

`.github/workflows/deploy-hf-space.yml`: push lên nhánh `khoa` có đổi trong `webapp/**` (trừ các file `.md`) → tự chạy `scripts/deploy_hf_space.py` → đồng bộ sang Space → Space tự rebuild.

Cần secret `HF_TOKEN` (quyền Write) trong GitHub repo settings — đã set. Checkpoint (`checkpoints/best.pt`, 89MB) **không** đồng bộ qua CI (ignore_patterns loại trừ `checkpoints/*`) vì nó gitignored, hiếm khi đổi, và đã có sẵn trên Space từ lần deploy tay đầu tiên. Nếu train lại model, upload tay:

```bash
python -c "
from huggingface_hub import HfApi
HfApi(token='<HF_TOKEN write>').upload_file(
    path_or_fileobj='webapp/checkpoints/best.pt',
    path_in_repo='checkpoints/best.pt',
    repo_id='thales1020/secure-virtual-assistant',
    repo_type='space',
)"
```

Chạy tay full sync (bằng đúng script CI dùng):

```bash
HF_TOKEN=<write token> python scripts/deploy_hf_space.py
```

## 3. Tạo lại từ đầu (nếu cần Space mới)

Dùng `huggingface_hub` API thay vì UI/git clone thủ công — nhanh và có thể lặp lại:

```python
from huggingface_hub import HfApi, SpaceHardware
from huggingface_hub.hf_api import Volume

api = HfApi(token="<write token>")
repo_id = "thales1020/secure-virtual-assistant"

api.create_repo(repo_id, repo_type="space", space_sdk="gradio",
                 space_hardware=SpaceHardware.ZERO_A10G, private=False, exist_ok=True)
api.create_bucket("thales1020/sl-storage", private=False, exist_ok=True)
api.set_space_volumes(repo_id, volumes=[
    Volume(type="bucket", source="thales1020/sl-storage", mount_path="/data", read_only=False)
])
api.add_space_secret(repo_id, "GEMINI_API_KEY", "<value>")
api.add_space_secret(repo_id, "OPENWEATHER_KEY", "<value>")
api.upload_folder(repo_id=repo_id, repo_type="space", folder_path="webapp",
                   ignore_patterns=[".venv/*", "**/__pycache__/*", "*.pyc", "**/desktop.ini"])
```

`README.md` ở gốc `webapp/` (được upload lên) chứa YAML frontmatter Space đọc để biết SDK/entry point:

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

`requirements.txt` dùng ở gốc `webapp/` (khác `webapp/backend/requirements.txt` — bản đó cho venv dev local, torch 2.4.1 theo khuyến nghị Module A). Bản deploy dùng torch 2.11.0 vì lý do ở mục 4.

## 4. Ba vấn đề riêng của ZeroGPU đã gặp — QUAN TRỌNG nếu redeploy

**(a) torch version:** ZeroGPU chỉ chấp nhận torch 2.8.0/2.9.1/2.10.0/2.11.0 trong `requirements.txt` — `torch==2.4.1` (khuyến nghị Module A cho dev local) bị `CONFIG_ERROR` ngay khi build. `webapp/requirements.txt` dùng `torch==2.11.0` + `torchaudio==2.11.0` (đổi backend đọc audio sang `torchcodec`, đã thêm dependency) — **chỉ áp dụng cho bản deploy**, không đổi `webapp/backend/requirements.txt` (dev local Windows vẫn 2.4.1, đã verify chạy tốt).

**(b) `@spaces.GPU` bắt buộc:** ZeroGPU Space báo `RUNTIME_ERROR: No @spaces.GPU function detected during startup` nếu không có function nào decorate `@spaces.GPU` được **gọi** lúc khởi động (chỉ định nghĩa không đủ — phải gọi). `webapp/app.py` có hàm `_zerogpu_probe()` decorate `@spaces.GPU`, gọi 1 lần trước khi `uvicorn.run()` — không tốn GPU thật (không nằm trong luồng xử lý request nào), chỉ để platform nhận diện.

**(c) `torch.cuda.is_available()` bị patch:** gói `spaces` patch hàm này luôn trả `True` (giả lập có GPU cho cơ chế cấp phát theo yêu cầu). `speaker_model.py` (Module A, không được sửa) tự chọn `device="cuda"` dựa trên giá trị đó, rồi crash vì gọi CUDA thật ngoài phạm vi hàm `@spaces.GPU`. Sửa trong `webapp/backend/services/speaker_service.py` (file của Module B, không phải `speaker_model.py`): patch chồng `torch.cuda.is_available = lambda: False` ngay trước khi khởi tạo `SpeakerModel`, chạy sau patch của `spaces` nên thắng, buộc chọn đúng CPU.

## 5. Kiểm tra sau khi deploy

```bash
curl https://thales1020-secure-virtual-assistant.hf.space/health
# {"status":"ok","speaker_model_ready":true}
```

Lần đầu bucket rỗng: tải `speechbrain/spkrec-ecapa-voxceleb` (~90MB) + `vinai/PhoWhisper-base` từ HF Hub, mất khoảng 1-2 phút. Các lần sau đọc từ `/data/hf_cache` nên nhanh, kể cả sau khi Space sleep/restart.

## Giới hạn đã biết (nêu trong báo cáo)

- Docker SDK và CPU Basic hardware bị khoá do giới hạn tài khoản (xác nhận qua lỗi `402 Payment Required`), không phải lựa chọn thiết kế — dùng Gradio SDK (chạy FastAPI thuần, không dùng UI Gradio) + ZeroGPU thay thế.
- App 100% dùng CPU, không có nhu cầu GPU thật — 3 vấn đề ở mục 4 đều là workaround để platform ZeroGPU chấp nhận một app không thật sự cần GPU, không phải thay đổi kiến trúc hệ thống.
- Storage Bucket là tính năng có thể tính phí trên tài khoản Enterprise/PRO; với tài khoản free hiện tại việc gắn qua API không báo lỗi thanh toán — nên kiểm tra huggingface.co/settings/billing định kỳ.

## Phương án Docker (khi có quyền/nâng cấp)

Nếu sau này Docker SDK khả dụng: tạo Space mới chọn SDK **Docker**, dùng `webapp/Dockerfile` (đã build + test thật thành công trên máy local, image 708MB, health check pass, dùng `torch==2.4.1` như dev local — không cần 3 workaround ZeroGPU ở mục 4) thay vì `app.py`/`packages.txt`/`requirements.txt` ở root. `README.md` frontmatter đổi `sdk: docker`, `app_port: 7860`, bỏ `python_version`/`app_file`.

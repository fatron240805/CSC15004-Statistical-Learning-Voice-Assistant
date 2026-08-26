"""Config — đọc .env ở repo root và định nghĩa các path dùng chung."""

from pathlib import Path

from dotenv import load_dotenv
import os

BACKEND_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = BACKEND_DIR.parent
REPO_ROOT = WEBAPP_DIR.parent

load_dotenv(REPO_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "").strip()

CHECKPOINT_PATH = str(WEBAPP_DIR / "checkpoints" / "best.pt")

# Set LOAD_SPEAKER_MODEL=0 để bỏ qua load model lúc startup (dev/test không có torch).
LOAD_SPEAKER_MODEL = os.getenv("LOAD_SPEAKER_MODEL", "1") != "0"

# HF Space storage bucket (nếu có) mount ở /data — dùng làm nơi lưu bền cho DB
# và cache pretrained model, sống sót qua việc Space sleep/restart. Không có
# bucket (dev local) -> fallback về thư mục backend/ như cũ.
DATA_DIR = Path("/data") if Path("/data").is_dir() else BACKEND_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Đặt SỚM, trước khi bất kỳ chỗ nào import torch/huggingface_hub/speechbrain
# (chúng lazy-import sau config.py), để cache tải pretrained model nằm trong
# DATA_DIR bền vững thay vì mất mỗi lần container khởi động lại.
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_cache"))

# App này chỉ dùng CPU. Trên hardware ZeroGPU (HF Space), torch.cuda.is_available()
# trả về True (GPU được cấp phát theo yêu cầu qua @spaces.GPU), khiến
# speaker_model.py (không được sửa) tự chọn device="cuda" rồi crash vì gọi CUDA
# thật ngoài phạm vi hàm @spaces.GPU. Ép CUDA_VISIBLE_DEVICES rỗng để
# torch.cuda.is_available() trả về False thật, buộc chọn đúng CPU — không cần
# đụng speaker_model.py. Không ảnh hưởng máy có GPU thật/không có GPU (no-op).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

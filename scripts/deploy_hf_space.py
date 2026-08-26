"""Đồng bộ webapp/ lên HF Space thales1020/secure-virtual-assistant.
Dùng trong GitHub Actions (.github/workflows/deploy-hf-space.yml) và có thể
chạy tay: HF_TOKEN=... python scripts/deploy_hf_space.py

Không upload checkpoints/best.pt (89MB, gitignored, hiếm khi đổi) — file này
đã được đẩy lên Space thủ công 1 lần, chỉ cần re-upload nếu train lại model
(chạy tay, không qua CI).
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "thales1020/secure-virtual-assistant"
WEBAPP_DIR = Path(__file__).resolve().parent.parent / "webapp"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Thiếu biến môi trường HF_TOKEN", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    result = api.upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=str(WEBAPP_DIR),
        ignore_patterns=[
            ".venv/*",
            "**/__pycache__/*",
            "*.pyc",
            "**/desktop.ini",
            "checkpoints/*",  # đã có sẵn trên Space, không re-upload qua CI
        ],
        delete_patterns=[
            "backend/app.db",
            "backend/hf_cache/**",
            "backend/chat_tmp/**",
            "backend/enroll_tmp/**",
            "backend/pretrained_models/**",
            "**/desktop.ini",
        ],
        commit_message="CI: sync webapp/ to HF Space",
    )
    print("Deployed:", result)


if __name__ == "__main__":
    main()

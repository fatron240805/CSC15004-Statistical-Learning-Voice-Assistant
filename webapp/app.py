"""Entry point cho HF Spaces (Gradio SDK) — Docker SDK bị khoá (paid) trên tài
khoản hiện tại, nên chạy thẳng FastAPI qua uvicorn thay vì dùng giao diện Gradio.
Không đổi logic backend, chỉ là điểm khởi động mà HF Spaces gọi (`python app.py`).
"""

import os

import uvicorn

from backend.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

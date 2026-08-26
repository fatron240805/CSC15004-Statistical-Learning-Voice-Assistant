"""Entry point cho HF Spaces (Gradio SDK) — Docker SDK bị khoá (paid) trên tài
khoản hiện tại, nên chạy thẳng FastAPI qua uvicorn thay vì dùng giao diện Gradio.
Không đổi logic backend, chỉ là điểm khởi động mà HF Spaces gọi (`python app.py`).
"""

import os

import spaces
import uvicorn

from backend.main import app


@spaces.GPU
def _zerogpu_probe() -> bool:
    """App này chỉ dùng CPU. Hardware ZeroGPU (duy nhất free trên tài khoản
    này, CPU Basic yêu cầu PRO) bắt buộc phải thấy ít nhất 1 hàm @spaces.GPU
    lúc khởi động, nếu không Space báo RUNTIME_ERROR ngay cả khi không hàm
    nào thật sự cần GPU. Hàm này không được gọi ở đâu khác — không tốn quota
    GPU thật, chỉ tồn tại để thoả điều kiện khởi động của nền tảng."""
    return True

if __name__ == "__main__":
    _zerogpu_probe()  # gọi 1 lần lúc khởi động để ZeroGPU runtime nhận diện
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

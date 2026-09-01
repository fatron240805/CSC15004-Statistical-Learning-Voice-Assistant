"""gTTS wrapper — text -> audio mp3 (base64), Milestone 8.

Không cần key riêng. Lazy import gTTS để /health vẫn chạy khi chưa cài gTTS.
"""

from __future__ import annotations

import base64
import io
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)

_FALLBACK_TEXT = "Xin lỗi, hiện chưa thể tạo giọng nói trả lời."
_TIMEOUT_SEC = 15  # gTTS không có tham số timeout riêng -> chặn bằng thread watchdog


def _synthesize(text: str, lang: str) -> str:
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def synthesize_base64(text: str, lang: str = "vi") -> str:
    """Trả về mp3 audio dạng base64 (không kèm prefix data:). Không raise, không treo
    vô thời hạn — trả rỗng nếu lỗi hoặc quá _TIMEOUT_SEC giây (gTTS gọi HTTP ra ngoài,
    không có tham số timeout, nên bọc bằng thread có giới hạn thời gian)."""
    if not text:
        return ""
    # Không dùng `with` quanh executor: __exit__ gọi shutdown(wait=True) và sẽ đợi
    # thread hung xong xuôi, phá mất tác dụng của timeout bên dưới.
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_synthesize, text, lang)
        return future.result(timeout=_TIMEOUT_SEC)
    except FutureTimeoutError:
        logger.error("gTTS synthesize vượt quá %ss — bỏ qua audio.", _TIMEOUT_SEC)
        executor.shutdown(wait=False)
        return ""
    except Exception:
        logger.exception("gTTS synthesize thất bại.")
        executor.shutdown(wait=False)
        return ""
    else:
        executor.shutdown(wait=False)

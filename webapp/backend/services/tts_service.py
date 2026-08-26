"""gTTS wrapper — text -> audio mp3 (base64), Milestone 8.

Không cần key riêng. Lazy import gTTS để /health vẫn chạy khi chưa cài gTTS.
"""

from __future__ import annotations

import base64
import io
import logging

logger = logging.getLogger(__name__)

_FALLBACK_TEXT = "Xin lỗi, hiện chưa thể tạo giọng nói trả lời."


def synthesize_base64(text: str, lang: str = "vi") -> str:
    """Trả về mp3 audio dạng base64 (không kèm prefix data:). Không raise — trả rỗng nếu lỗi."""
    if not text:
        return ""
    try:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        logger.exception("gTTS synthesize thất bại.")
        return ""

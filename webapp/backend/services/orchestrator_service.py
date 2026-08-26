"""Orchestrator — gọi Gemini API, ép JSON output {intent, entities} (mục 4 spec).

Gemini chỉ làm nhiệm vụ hiểu intent/entity từ text ASR tiếng Việt, KHÔNG quyết
định có cần xác thực giọng nói hay không — việc đó do backend.gating.route() quyết
định cứng trong code Python.
"""

from __future__ import annotations

import json
import logging

from backend.config import GEMINI_API_KEY
from backend.gating import INTENT_MAP

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.1-flash-lite"

_UNKNOWN_RESULT = {"intent": "unknown", "entities": {}}

_VALID_INTENTS = list(INTENT_MAP.keys())

_PROMPT_TEMPLATE = """Bạn là bộ phân loại ý định cho trợ lý ảo trên xe hơi.
Cho câu nói tiếng Việt của người dùng, hãy trả về intent và entities liên quan.

Các intent hợp lệ: {intents}
- general_weather: hỏi thời tiết, nhiệt độ
- general_location: hỏi vị trí, định vị hiện tại
- sv_unlock_door: yêu cầu mở khoá cửa/khoang xe
- sv_start_engine: yêu cầu khởi động xe
- sid_play_playlist: yêu cầu phát nhạc/danh sách yêu thích
- unknown: không khớp intent nào ở trên

Chỉ trả JSON đúng schema, không giải thích thêm:
{{"intent": "<một trong các intent trên>", "entities": {{}}}}

Câu nói: "{text}"
"""


def classify(text: str) -> dict:
    """Input: text ASR. Output: {"intent": str, "entities": dict}. Không bao giờ raise."""
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY rỗng — trả unknown.")
        return dict(_UNKNOWN_RESULT)

    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = _PROMPT_TEMPLATE.format(intents=", ".join(_VALID_INTENTS), text=text)
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        raw = json.loads(response.text)
        if not isinstance(raw, dict):
            raise ValueError(f"Gemini trả JSON không phải object: {raw!r}")
    except Exception:
        logger.exception("Gemini classify thất bại — trả unknown.")
        return dict(_UNKNOWN_RESULT)

    intent = raw.get("intent")
    if intent not in INTENT_MAP:
        logger.warning("Gemini trả intent lạ %r — coerce về unknown.", intent)
        intent = "unknown"
    entities = raw.get("entities")
    if not isinstance(entities, dict):
        entities = {}
    return {"intent": intent, "entities": entities}

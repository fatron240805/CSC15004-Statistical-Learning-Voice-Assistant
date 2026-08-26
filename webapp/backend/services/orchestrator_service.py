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
- sid_personal_query: hỏi thông tin cá nhân đã đăng ký (tên tôi là gì, tôi thích bài hát nào...)
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


_ANSWER_PROMPT_TEMPLATE = """Bạn là trợ lý ảo trên xe hơi. Người dùng vừa được NHẬN DIỆN GIỌNG NÓI
thành công (identify() đã xác nhận), đây là hồ sơ THẬT của họ trong hệ thống — không phải suy đoán:

{context}

Người dùng vừa hỏi: "{question}"

Trả lời NGẮN GỌN (1-2 câu), tự nhiên bằng tiếng Việt, dựa ĐÚNG vào thông tin trên.
Không bịa thêm thông tin nào ngoài dữ liệu đã cho. Nếu câu hỏi không liên quan gì đến
hồ sơ trên, trả lời rằng bạn chỉ biết những thông tin đã đăng ký."""


def answer_with_context(question: str, context: dict) -> str:
    """RAG đơn giản: đưa dữ liệu đã identify() thật cho Gemini trả lời tự nhiên.

    Gemini chỉ diễn đạt lại dữ liệu có sẵn (context lấy từ DB sau identify() thành
    công), không tự quyết định danh tính hay tự bịa thông tin — khác với classify(),
    hàm này KHÔNG quyết định route/quyền hạn, chỉ soạn câu trả lời hiển thị cho user.
    """
    if not GEMINI_API_KEY:
        return f"Bạn là {context.get('ten', 'người dùng đã đăng ký')}."

    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = _ANSWER_PROMPT_TEMPLATE.format(
            context=json.dumps(context, ensure_ascii=False), question=question
        )
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        return text or f"Bạn là {context.get('ten', 'người dùng đã đăng ký')}."
    except Exception:
        logger.exception("Gemini answer_with_context thất bại — fallback trả trực tiếp dữ liệu.")
        return f"Bạn là {context.get('ten', 'người dùng đã đăng ký')}."

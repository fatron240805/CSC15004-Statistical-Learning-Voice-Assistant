"""Bảng gate cứng intent -> usecase -> cần SV hay không (mục 4 spec).

Đây là điểm quyết định route DUY NHẤT. Gemini (orchestrator_service) chỉ trả
về intent/entities, KHÔNG được tự quyết định có cần xác thực giọng nói hay không.
"""

from __future__ import annotations

# intent -> (usecase, need_sv)
INTENT_MAP: dict[str, tuple[str, bool]] = {
    "general_weather": ("general", False),
    "general_location": ("general", False),
    "sv_unlock_door": ("sv", True),
    "sv_start_engine": ("sv", True),
    "sid_play_playlist": ("sid", False),  # không cần verify(), nhưng cần identify()
    "unknown": ("unknown", False),
}

# intent -> action ghi vào action_log.action (mục 7 spec)
INTENT_ACTION: dict[str, str] = {
    "sv_unlock_door": "unlock_door",
    "sv_start_engine": "start_engine",
    "sid_play_playlist": "play_playlist",
}


def route(intent: str) -> tuple[str, bool]:
    """Trả về (usecase, need_sv) cho 1 intent. Intent lạ -> ("unknown", False)."""
    return INTENT_MAP.get(intent, INTENT_MAP["unknown"])

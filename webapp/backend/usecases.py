"""Usecase handlers: General / SV / SID (mục 5 spec).

Gọi từ api/chat.py (Milestone 8) sau khi gating.route() đã quyết định usecase.
verify()/identify() là nguồn xác thực duy nhất — score chỉ để log/debug, KHÔNG
dùng để tự so sánh threshold trong file này.
"""

from __future__ import annotations

import logging

import numpy as np

from backend.db.database import get_session
from backend.db.models import ActionLog, Preference, User
from backend.gating import INTENT_ACTION
from backend.services import music_service, orchestrator_service, speaker_service, weather_service
from backend.speaker_model import DEFAULT_THRESHOLD_SV

logger = logging.getLogger(__name__)

MAX_SV_ATTEMPTS = 3  # lần đầu + tối đa 2 lần retry (mục 5.2 spec)

_ACTION_TEXT = {
    "unlock_door": "đã mở khoang",
    "start_engine": "đã khởi động xe",
}


def handle_general(intent: str, entities: dict) -> str:
    if intent == "general_weather":
        return weather_service.get_weather_text()
    if intent == "general_location":
        return weather_service.get_location_text()
    return "Xin lỗi, tôi chưa hiểu yêu cầu này."


def _identify_speaker(session, wav_path: str, threshold: float | None = None):
    """Nhận diện giọng nói qua toàn bộ hồ sơ đã enroll. Trả (user_id|None, score|None,
    emb_runtime). threshold=None dùng mặc định SID (0.20); truyền tường minh
    DEFAULT_THRESHOLD_SV cho các usecase cần ngưỡng chặt hơn (SV)."""
    users = session.query(User).filter(User.embedding.isnot(None)).all()
    db_embeddings = {u.user_id: np.array(u.embedding) for u in users}
    emb_runtime = speaker_service.get_embedding(wav_path)
    user_id = speaker_service.identify(emb_runtime, db_embeddings, threshold=threshold)
    score = float(np.dot(emb_runtime, db_embeddings[user_id])) if user_id is not None else None
    return user_id, score, emb_runtime


def handle_sv(intent: str, wav_path: str, attempt: int = 1) -> dict:
    """SV: KHÔNG yêu cầu người dùng tự khai báo danh tính (không thực tế trong xe) —
    tự nhận diện qua toàn bộ hồ sơ đã enroll bằng ngưỡng SV (0.35, chặt hơn ngưỡng
    SID 0.20) để đảm bảo an toàn cho lệnh nhạy cảm. Ghi ActionLog mỗi lần gọi."""
    action = INTENT_ACTION.get(intent, intent)
    session = get_session()
    try:
        user_id, score, _ = _identify_speaker(session, wav_path, threshold=DEFAULT_THRESHOLD_SV)
        verified = user_id is not None

        session.add(ActionLog(user_id=user_id, action=action, verified=verified, score=score))
        session.commit()

        can_retry = (not verified) and attempt < MAX_SV_ATTEMPTS
        if verified:
            message = f"{_ACTION_TEXT.get(action, action)}."
        elif can_retry:
            message = "Không xác thực được giọng nói, vui lòng thử lại."
        else:
            message = "Không xác thực được giọng nói."

        return {"verified": verified, "can_retry": can_retry, "score": score, "message": message}
    finally:
        session.close()


def handle_sid(wav_path: str) -> dict:
    """SID: identify() -> lấy favorite_tracks -> tìm preview trên Deezer."""
    session = get_session()
    try:
        user_id, score, _ = _identify_speaker(session, wav_path)

        if user_id is None:
            session.add(ActionLog(user_id=None, action="play_playlist", verified=False, score=None))
            session.commit()
            return {
                "user_id": None,
                "verified": False,
                "playlist": [],
                "message": "Không nhận diện được giọng nói của bạn.",
            }

        pref = session.get(Preference, user_id)
        tracks = pref.favorite_tracks if pref and pref.favorite_tracks else []
        playlist = music_service.get_playlist(tracks)

        session.add(ActionLog(user_id=user_id, action="play_playlist", verified=True, score=score))
        session.commit()

        return {
            "user_id": user_id,
            "verified": True,
            "playlist": playlist,
            "message": "Đang phát danh sách yêu thích của bạn.",
        }
    finally:
        session.close()


def handle_personal_query(wav_path: str, question: str) -> dict:
    """SID mở rộng: identify() -> lấy hồ sơ thật (tên, sở thích) -> đưa cho Gemini
    trả lời tự nhiên (orchestrator_service.answer_with_context). Chỉ dữ liệu đã
    identify() thành công mới được đưa vào prompt — Gemini không tự bịa danh tính."""
    session = get_session()
    try:
        user_id, score, _ = _identify_speaker(session, wav_path)

        if user_id is None:
            session.add(ActionLog(user_id=None, action="personal_query", verified=False, score=None))
            session.commit()
            return {
                "user_id": None,
                "verified": False,
                "message": "Không nhận diện được giọng nói của bạn, chưa thể trả lời.",
            }

        user = session.get(User, user_id)
        pref = session.get(Preference, user_id)
        context = {
            "ten": user.name,
            "bai_hat_yeu_thich": pref.favorite_tracks if pref and pref.favorite_tracks else [],
        }
        message = orchestrator_service.answer_with_context(question, context)

        session.add(ActionLog(user_id=user_id, action="personal_query", verified=True, score=score))
        session.commit()

        return {"user_id": user_id, "verified": True, "message": message}
    finally:
        session.close()

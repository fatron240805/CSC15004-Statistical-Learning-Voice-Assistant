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
from backend.services import music_service, speaker_service, weather_service

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


def handle_sv(intent: str, user_id: int | None, wav_path: str, attempt: int = 1) -> dict:
    """SV: verify() trước khi thực hiện action nhạy cảm. Ghi ActionLog mỗi lần gọi."""
    action = INTENT_ACTION.get(intent, intent)
    session = get_session()
    try:
        user = session.get(User, user_id) if user_id else None
        if user is None or user.embedding is None:
            session.add(ActionLog(user_id=user_id, action=action, verified=False, score=None))
            session.commit()
            return {
                "verified": False,
                "can_retry": False,
                "message": "Chưa xác định được bạn là ai, vui lòng chọn người dùng hoặc enroll trước.",
            }

        emb_runtime = speaker_service.get_embedding(wav_path)
        emb_enrolled = np.array(user.embedding)
        score = float(np.dot(emb_runtime, emb_enrolled))
        verified = speaker_service.verify(emb_runtime, emb_enrolled)

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
        users = session.query(User).filter(User.embedding.isnot(None)).all()
        db_embeddings = {u.user_id: np.array(u.embedding) for u in users}

        emb_runtime = speaker_service.get_embedding(wav_path)
        user_id = speaker_service.identify(emb_runtime, db_embeddings)

        if user_id is None:
            session.add(ActionLog(user_id=None, action="play_playlist", verified=False, score=None))
            session.commit()
            return {
                "user_id": None,
                "verified": False,
                "playlist": [],
                "message": "Không nhận diện được giọng nói của bạn.",
            }

        score = float(np.dot(emb_runtime, db_embeddings[user_id]))
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

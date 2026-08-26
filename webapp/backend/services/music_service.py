"""Deezer API — search bài hát cho playlist SID (mục 5.3 spec).

Free, không cần key cho search. Giới hạn: chỉ preview 30s/bài (chấp nhận cho demo).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.deezer.com/search"


def search_track(query: str) -> dict | None:
    """Tìm 1 bài trên Deezer. Trả {"title", "artist", "preview_url"} hoặc None nếu không thấy."""
    try:
        import requests

        resp = requests.get(SEARCH_URL, params={"q": query, "limit": 1}, timeout=5)
        resp.raise_for_status()
        results = resp.json().get("data", [])
        if not results:
            return None
        track = results[0]
        return {
            "title": track["title"],
            "artist": track["artist"]["name"],
            "preview_url": track["preview"],
        }
    except Exception:
        logger.exception("Deezer search thất bại cho query=%r", query)
        return None


def get_playlist(track_names: list[str]) -> list[dict]:
    """Tìm từng bài trong danh sách yêu thích, bỏ qua bài không tìm thấy."""
    playlist = []
    for name in track_names:
        track = search_track(name)
        if track is not None:
            playlist.append(track)
    return playlist

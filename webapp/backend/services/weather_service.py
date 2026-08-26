"""OpenWeatherMap — thời tiết/định vị với toạ độ cố định demo (mục 5.1 spec).

Không có GPS xe thật -> dùng toạ độ TP.HCM cố định cho cả weather lẫn location.
Giới hạn này cần nêu rõ trong báo cáo (mục 8 spec).
"""

from __future__ import annotations

import logging

from backend.config import OPENWEATHER_KEY

logger = logging.getLogger(__name__)

DEMO_LAT = 10.7769
DEMO_LON = 106.7009
DEMO_LOCATION_NAME = "Thành phố Hồ Chí Minh"

_FALLBACK_TEXT = "Xin lỗi, hiện chưa lấy được dữ liệu thời tiết."


def get_weather_text() -> str:
    """Trả câu trả lời TTS: "Hiện tại trời [mô tả], nhiệt độ [x]°C."."""
    if not OPENWEATHER_KEY:
        logger.warning("OPENWEATHER_KEY rỗng — trả fallback.")
        return _FALLBACK_TEXT

    try:
        import requests

        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": DEMO_LAT,
                "lon": DEMO_LON,
                "appid": OPENWEATHER_KEY,
                "units": "metric",
                "lang": "vi",
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        description = data["weather"][0]["description"]
        temp = round(data["main"]["temp"])
        return f"Hiện tại trời {description}, nhiệt độ {temp}°C."
    except Exception:
        logger.exception("Gọi OpenWeatherMap thất bại.")
        return _FALLBACK_TEXT


def get_location_text() -> str:
    """Trả câu trả lời định vị — toạ độ/tên địa điểm demo cố định (không có GPS xe thật)."""
    return f"Vị trí hiện tại (demo): {DEMO_LOCATION_NAME} ({DEMO_LAT}, {DEMO_LON})."

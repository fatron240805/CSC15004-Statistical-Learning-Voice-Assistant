"""FastAPI app — Module B. Chạy: uvicorn backend.main:app (cwd = webapp/)."""

import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.api.chat import router as chat_router
from backend.api.enrollment import router as enrollment_router
from backend.config import LOAD_SPEAKER_MODEL, WEBAPP_DIR
from backend.db.database import init_db
from backend.services import speaker_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIR = WEBAPP_DIR / "frontend"

app = FastAPI(title="Secure Virtual Assistant — Module B")
app.include_router(enrollment_router)
app.include_router(chat_router)


@app.on_event("startup")
def startup() -> None:
    init_db()
    if not LOAD_SPEAKER_MODEL:
        logger.warning("LOAD_SPEAKER_MODEL=0 — bỏ qua load SpeakerModel.")
        return
    try:
        speaker_service.load_model()
    except Exception:
        logger.exception("Load SpeakerModel thất bại — server vẫn chạy, các usecase SV/SID sẽ lỗi.")


@app.get("/health")
def health():
    return {"status": "ok", "speaker_model_ready": speaker_service.is_ready()}


# Mount cuối cùng: route API ở trên khớp trước, phần còn lại serve static frontend
# (mục 2 spec — gộp chung 1 Space với backend, tránh CORS/multi-deploy).
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

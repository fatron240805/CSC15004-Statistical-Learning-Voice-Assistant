"""POST /chat — pipeline đầy đủ (mục 3 spec, Milestone 8).

audio in -> audio_convert -> asr -> orchestrator -> gating (cứng) ->
usecase handler (General/SV/SID) -> tts -> trả text + audio + data.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile

import backend.usecases as usecases
from backend.config import WEBAPP_DIR
from backend.gating import route
from backend.services import asr_service, orchestrator_service, tts_service
from backend.services.audio_convert import webm_bytes_to_wav_file

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

TMP_DIR = WEBAPP_DIR / "backend" / "chat_tmp"


@router.post("/chat")
def chat(
    audio: UploadFile,
    sv_attempt: int = Form(1),
):
    """def thường (không phải async def): toàn bộ xử lý bên dưới là các lệnh gọi
    đồng bộ/chặn (ASR, Gemini, TTS) — nếu khai async def mà không await, chúng sẽ
    chặn nguyên event loop, treo luôn cả các request khác đang chờ (đã từng xảy ra
    với /enroll/sentence, sửa tương tự ở đây). def thường để FastAPI tự chạy trong
    threadpool, tách biệt với các request song song khác."""
    if audio is None:
        raise HTTPException(400, "thiếu file audio")

    # Đo thời gian từng bước — phục vụ bảng latency trong báo cáo (mục 5.2 Chap5).
    # perf_counter, không phải profiling đầy đủ, nhưng đủ để so sánh tương đối
    # giữa các bước và tổng thời gian cảm nhận được của người dùng.
    t = {}
    t_start = time.perf_counter()

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = str(TMP_DIR / f"{uuid.uuid4().hex}.wav")
    try:
        t0 = time.perf_counter()
        webm_bytes = audio.file.read()
        webm_bytes_to_wav_file(webm_bytes, wav_path)
        t["preprocess"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        transcript = asr_service.transcribe(wav_path)
        t["asr"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        classification = orchestrator_service.classify(transcript)
        t["gemini_classify"] = time.perf_counter() - t0
        intent = classification["intent"]
        entities = classification["entities"]

        usecase, _need_sv = route(intent)

        t0 = time.perf_counter()
        data = None
        if usecase == "general":
            text = usecases.handle_general(intent, entities)
        elif usecase == "sv":
            data = usecases.handle_sv(intent, wav_path, attempt=sv_attempt)
            text = data["message"]
        elif usecase == "sid":
            if intent == "sid_personal_query":
                data = usecases.handle_personal_query(wav_path, transcript)
            else:
                data = usecases.handle_sid(wav_path)
            text = data["message"]
        else:
            text = "Xin lỗi, tôi chưa hiểu yêu cầu này."
        t["usecase"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        audio_base64 = tts_service.synthesize_base64(text)
        t["tts"] = time.perf_counter() - t0

        t["total"] = time.perf_counter() - t_start
        logger.info("chat timing (s): %s", {k: round(v, 3) for k, v in t.items()})

        return {
            "transcript": transcript,
            "intent": intent,
            "entities": entities,
            "usecase": usecase,
            "text": text,
            "audio_base64": audio_base64,
            "data": data,
            "timing": {k: round(v, 3) for k, v in t.items()},
        }
    finally:
        Path(wav_path).unlink(missing_ok=True)

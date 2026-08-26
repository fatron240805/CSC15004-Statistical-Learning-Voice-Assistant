"""Enrollment flow (mục 6 spec, Milestone 5 plan).

POST /enroll/start        - tạo user mới + preferences ban đầu
POST /enroll/sentence/{idx} - nhận audio 1 trong 7 câu mẫu, check WER + độ dài
POST /enroll/finish       - gộp 7 embedding (trung bình + L2-normalize), lưu vào users.embedding
"""

from __future__ import annotations

import re
import shutil
import wave
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile

from backend.config import WEBAPP_DIR
from backend.db.database import get_session
from backend.db.models import Preference, User
from backend.services import asr_service, speaker_service
from backend.services.audio_convert import webm_bytes_to_wav_file

router = APIRouter(prefix="/enroll", tags=["enrollment"])

# 7 câu mẫu — copy nguyên văn từ model/moduleA_artifacts_final/enrollment_sentences.md
ENROLLMENT_SENTENCES = [
    "Xin chào, tôi tên là Khánh, đây là giọng nói của tôi.",
    "Hôm nay trời nắng đẹp, tôi muốn nghe một bản nhạc nhẹ nhàng.",
    "Làm ơn mở cửa sổ bên trái và bật điều hòa mát mẻ.",
    "Chỉ đường cho tôi đến quán cà phê gần nhất được không?",
    "Tôi thường lái xe vào buổi sáng sớm để tránh kẹt xe.",
    "Xin vui lòng ghi nhớ giọng nói này để nhận diện tôi sau này.",
    "Bây giờ hãy phát danh sách nhạc yêu thích của tôi.",
]

MIN_DURATION_SEC = 2.0  # câu mẫu ước tính 4-6s, chấp nhận tối thiểu vài giây
MAX_WER = 0.5  # WER quá cao -> yêu cầu đọc lại

TMP_DIR = WEBAPP_DIR / "backend" / "enroll_tmp"


def _normalize_words(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^\w\sÀ-ỹ]", "", text, flags=re.UNICODE)
    return text.split()


def _word_edit_distance(ref: list[str], hyp: list[str]) -> int:
    dp = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(hyp) + 1):
            cur = dp[j]
            dp[j] = prev if ref[i - 1] == hyp[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[len(hyp)]


def _wer(reference: str, hypothesis: str) -> float:
    ref_words = _normalize_words(reference)
    hyp_words = _normalize_words(hypothesis)
    if not ref_words:
        return 0.0
    return _word_edit_distance(ref_words, hyp_words) / len(ref_words)


def _wav_duration_sec(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as f:
        return f.getnframes() / float(f.getframerate())


@router.post("/start")
def enroll_start(name: str = Form(...), favorite_tracks: str = Form("")):
    """Tạo user mới. favorite_tracks: chuỗi tên bài, phân cách bởi dấu phẩy."""
    session = get_session()
    try:
        user = User(name=name)
        session.add(user)
        session.flush()
        tracks = [t.strip() for t in favorite_tracks.split(",") if t.strip()]
        session.add(Preference(user_id=user.user_id, favorite_tracks=tracks))
        session.commit()
        return {"user_id": user.user_id, "name": user.name, "sentences": ENROLLMENT_SENTENCES}
    finally:
        session.close()


@router.post("/sentence/{idx}")
def enroll_sentence(idx: int, user_id: int = Form(...), audio: UploadFile = None):
    if not (0 <= idx < len(ENROLLMENT_SENTENCES)):
        raise HTTPException(400, "idx ngoài phạm vi 0-6")
    if audio is None:
        raise HTTPException(400, "thiếu file audio")

    user_dir = TMP_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    wav_path = str(user_dir / f"{idx}.wav")

    webm_bytes = audio.file.read()
    try:
        webm_bytes_to_wav_file(webm_bytes, wav_path)
    except Exception as exc:
        raise HTTPException(400, f"audio không hợp lệ, hãy ghi lại: {exc}") from exc

    duration = _wav_duration_sec(wav_path)
    transcript = asr_service.transcribe(wav_path)
    wer = _wer(ENROLLMENT_SENTENCES[idx], transcript)

    passed = duration >= MIN_DURATION_SEC and wer <= MAX_WER
    if not passed:
        Path(wav_path).unlink(missing_ok=True)

    return {
        "idx": idx,
        "pass": passed,
        "transcript": transcript,
        "wer": round(wer, 3),
        "duration_sec": round(duration, 2),
    }


@router.post("/finish")
def enroll_finish(user_id: int = Form(...)):
    user_dir = TMP_DIR / str(user_id)
    wav_paths = [str(user_dir / f"{i}.wav") for i in range(len(ENROLLMENT_SENTENCES))]
    missing = [i for i, p in enumerate(wav_paths) if not Path(p).exists()]
    if missing:
        raise HTTPException(400, f"chưa đủ audio đạt chất lượng cho câu: {missing}")

    embeddings = [speaker_service.get_embedding(p) for p in wav_paths]
    avg = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(avg)
    final_embedding = (avg / norm if norm > 0 else avg).tolist()

    session = get_session()
    try:
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(404, "user không tồn tại")
        user.embedding = final_embedding
        session.commit()
    finally:
        session.close()

    shutil.rmtree(user_dir, ignore_errors=True)
    return {"user_id": user_id, "embedding_dim": len(final_embedding)}

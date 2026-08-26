"""Wrap PhoWhisper (transformers) — ASR tiếng Việt cho câu ngắn 2-6s.

Load model lazy (singleton) khi transcribe() được gọi lần đầu, để /health và
các endpoint chưa cần ASR vẫn chạy được khi chưa có transformers/model tải về.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "vinai/PhoWhisper-base"

_pipeline = None  # singleton transformers pipeline


def load_model() -> None:
    global _pipeline
    if _pipeline is not None:
        return
    from transformers import pipeline

    _pipeline = pipeline("automatic-speech-recognition", model=MODEL_NAME)
    logger.info("PhoWhisper loaded: %s", MODEL_NAME)


def is_ready() -> bool:
    return _pipeline is not None


def transcribe(wav_path: str) -> str:
    """Input: đường dẫn file .wav 16kHz mono. Output: text tiếng Việt."""
    if _pipeline is None:
        load_model()
    result = _pipeline(wav_path)
    return result["text"].strip()

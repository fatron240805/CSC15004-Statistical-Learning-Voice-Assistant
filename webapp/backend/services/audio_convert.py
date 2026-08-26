"""Convert audio .webm (bytes, từ MediaRecorder trình duyệt) -> .wav 16kHz mono.

Dùng pydub (cần ffmpeg cài trong môi trường/Docker image). SpeakerModel và
PhoWhisper đều yêu cầu input .wav 16kHz mono.
"""

from __future__ import annotations

import io

SAMPLE_RATE = 16000


def webm_bytes_to_wav_file(webm_bytes: bytes, out_path: str) -> str:
    """Convert webm bytes -> wav 16kHz mono, ghi ra out_path. Trả về out_path."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(io.BytesIO(webm_bytes), format="webm")
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
    audio.export(out_path, format="wav")
    return out_path

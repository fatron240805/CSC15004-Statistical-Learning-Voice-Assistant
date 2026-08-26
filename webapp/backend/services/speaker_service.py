"""Wrap speaker_model.SpeakerModel — singleton load lúc startup.

Không đổi tên hàm/threshold của Module A: get_embedding/verify/identify
delegate thẳng xuống SpeakerModel, không tự khai báo lại threshold.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil

from backend.config import CHECKPOINT_PATH

logger = logging.getLogger(__name__)

_model = None  # singleton SpeakerModel, load lúc startup qua load_model()


def _patch_symlink_fallback_on_windows() -> None:
    """speechbrain/huggingface_hub cache pretrained files bằng symlink, nhưng
    Windows chặn symlink nếu chưa bật Developer Mode / chạy admin -> load model
    lỗi mỗi lần dev local. Deploy thật (Linux) không bị ảnh hưởng (symlink hoạt
    động bình thường nên fallback này không bao giờ kích hoạt ở đó). Không đụng
    speaker_model.py — chỉ vá hành vi symlink ở tầng OS trước khi nó được gọi.
    """
    if platform.system() != "Windows":
        return
    orig_symlink = os.symlink

    def _symlink_or_copy(src, dst, target_is_directory=False, *, dir_fd=None):
        try:
            return orig_symlink(src, dst, target_is_directory, dir_fd=dir_fd)
        except OSError:
            shutil.copy2(src, dst)

    os.symlink = _symlink_or_copy


def _patch_force_cpu_on_zerogpu() -> None:
    """Trên ZeroGPU (HF Space), gói `spaces` patch torch.cuda.is_available()
    luôn trả True để giả lập có GPU cho cơ chế cấp phát theo yêu cầu — nhưng
    speaker_model.py (không được sửa) tự chọn device="cuda" theo giá trị đó,
    rồi crash vì gọi CUDA thật ngoài phạm vi hàm @spaces.GPU. App này chỉ cần
    CPU, nên patch chồng lên sau `spaces` để is_available() trả False thật —
    patch của mình chạy sau nên thắng. Không ảnh hưởng máy không có `spaces`
    (import lỗi -> bỏ qua lặng lẽ) hay máy CPU/GPU thường (torch.cuda vẫn hoạt
    động đúng, chỉ thêm 1 layer patch không đổi hành vi thật).
    """
    try:
        import torch
    except ImportError:
        return
    torch.cuda.is_available = lambda: False


def load_model() -> None:
    """Load SpeakerModel 1 lần. Import speaker_model bên trong hàm này để
    /health và các endpoint chưa cần model vẫn chạy được khi chưa có torch."""
    global _model
    if _model is not None:
        return
    _patch_symlink_fallback_on_windows()
    _patch_force_cpu_on_zerogpu()
    from backend.speaker_model import SpeakerModel

    _model = SpeakerModel(checkpoint_path=CHECKPOINT_PATH)
    logger.info("SpeakerModel loaded from %s", CHECKPOINT_PATH)


def is_ready() -> bool:
    return _model is not None


def get_embedding(audio_path: str):
    if _model is None:
        raise RuntimeError("SpeakerModel chưa được load — gọi load_model() trước.")
    return _model.get_embedding(audio_path)


def verify(emb1, emb2) -> bool:
    if _model is None:
        raise RuntimeError("SpeakerModel chưa được load — gọi load_model() trước.")
    return _model.verify(emb1, emb2)


def identify(emb, db_embeddings: dict):
    if _model is None:
        raise RuntimeError("SpeakerModel chưa được load — gọi load_model() trước.")
    return _model.identify(emb, db_embeddings)

"""
speaker_model.py — Artefact bàn giao từ Module A (Khoa) cho Module B (Hiếu).

Interface cố định, KHÔNG đổi tên hàm khi tích hợp vào speaker_service.py:
- load_checkpoint()
- get_embedding(audio_path) -> np.ndarray (192-dim, đã L2-normalize)
- verify(emb1, emb2, threshold=DEFAULT_THRESHOLD_SV) -> bool       # dùng cho lệnh nhạy cảm (SV)
- identify(emb, db_embeddings, threshold=DEFAULT_THRESHOLD_SID)     # dùng cho cá nhân hoá (SID)

LƯU Ý QUAN TRỌNG: verify() và identify() dùng 2 threshold MẶC ĐỊNH KHÁC NHAU.
Đã xác nhận bằng thực nghiệm: threshold tối ưu cho SV (ưu tiên an toàn) làm
giảm mạnh accuracy khi dùng chung cho SID (ưu tiên trải nghiệm). Không gọi
identify() với threshold của verify() trừ khi có lý do rõ ràng.

Yêu cầu môi trường: speechbrain, torch, torchaudio, numpy (xem requirements.txt
đi kèm). Model expect audio 16kHz mono — nếu backend nhận .webm, convert sang
.wav 16kHz mono TRƯỚC khi gọi get_embedding (dùng pydub, việc này thuộc Module B).
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

SAMPLE_RATE = 16000
EMBEDDING_DIM = 192

# Threshold TÁCH RIÊNG cho SV và SID — không dùng chung 1 giá trị.
# Lý do: threshold tối ưu cho SV (ưu tiên tuyệt đối giảm False Accept cho lệnh
# nhạy cảm) khi áp cho SID lại gây tỉ lệ từ chối oan quá cao, làm giảm mạnh
# accuracy nhận diện (đo thực nghiệm: identify() dùng threshold SV làm giảm
# top-1 accuracy từ ~85% xuống ~73%, với ~27% truy vấn bị từ chối oan thành
# "unknown" dù model đã chọn đúng người). Xem outputs/metrics/identification_report.md.
DEFAULT_THRESHOLD_SV = 0.35    # dùng cho verify() — lệnh nhạy cảm, ưu tiên an toàn
DEFAULT_THRESHOLD_SID = 0.2   # dùng cho identify() — cá nhân hoá, ưu tiên trải nghiệm


class SpeakerModel:
    def __init__(self, checkpoint_path: str, pretrained_source: str = "speechbrain/spkrec-ecapa-voxceleb"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = EncoderClassifier.from_hparams(
            source=pretrained_source, savedir="./pretrained_models/ecapa"
        )
        self.embedding_model = self.classifier.mods["embedding_model"]
        self.load_checkpoint(checkpoint_path)
        self.embedding_model.to(self.device)
        self.embedding_model.eval()

    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        # Chỉ lấy phần weight của encoder, bỏ qua head AAM-softmax (chỉ dùng lúc train)
        encoder_state = {
            k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")
        }
        self.classifier.mods.load_state_dict(encoder_state, strict=False)

    def _load_waveform(self, audio_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        return waveform  # (1, samples), yêu cầu file .wav 16kHz mono

    @torch.no_grad()
    def get_embedding(self, audio_path: str) -> np.ndarray:
        """Input: đường dẫn file .wav 16kHz mono. Output: vector 192-dim đã L2-normalize."""
        waveform = self._load_waveform(audio_path).to(self.device)
        feats = self.classifier.mods["compute_features"](waveform)
        feats = self.classifier.mods["mean_var_norm"](
            feats, torch.ones(feats.shape[0]).to(self.device)
        )
        embedding = self.embedding_model(feats).squeeze(0).squeeze(0)
        embedding = torch.nn.functional.normalize(embedding, dim=0)
        return embedding.cpu().numpy()

    def verify(self, emb1: np.ndarray, emb2: np.ndarray, threshold: float = DEFAULT_THRESHOLD_SV) -> bool:
        """Speaker Verification: đúng 1 người đã đăng ký hay không.

        Dùng cho lệnh NHẠY CẢM (mở khoang, khởi động xe...). Mặc định threshold
        cao (ưu tiên an toàn) — KHÔNG hạ threshold này để "dễ dùng hơn", vì
        đây là hàng rào bảo mật, không phải tiện ích.
        """
        score = float(np.dot(emb1, emb2))
        return score >= threshold

    def identify(self, emb: np.ndarray, db_embeddings: dict, threshold: float = DEFAULT_THRESHOLD_SID):
        """Speaker Identification: tìm người khớp nhất trong database enrollment.

        Dùng cho CÁ NHÂN HOÁ (gợi ý nhạc, chỉnh ghế theo người dùng...), KHÔNG
        dùng cho lệnh nhạy cảm. Mặc định threshold thấp hơn verify() vì hậu quả
        nhận nhầm ở đây nhẹ hơn nhiều so với SV — ưu tiên trải nghiệm mượt mà.

        db_embeddings: dict {speaker_id: np.ndarray}. Trả về speaker_id có
        điểm cao nhất nếu vượt threshold, ngược lại trả về None (không nhận
        diện được ai, tránh gán nhầm người).
        """
        best_id, best_score = None, -1.0
        for speaker_id, ref_emb in db_embeddings.items():
            score = float(np.dot(emb, ref_emb))
            if score > best_score:
                best_id, best_score = speaker_id, score
        if best_score >= threshold:
            return best_id
        return None

"""
Đóng gói artefact cuối cùng để bàn giao cho Module B (Hiếu):
- outputs/speaker_model.py : class wrapper, interface cố định
- outputs/checkpoints/best.pt : trọng số đã fine-tune
- outputs/metrics/*.md : báo cáo EER/minDCF/robustness (đã sinh ở bước trước)

speaker_model.py được sinh ra là 1 file ĐỘC LẬP (không import từ src/ khác)
để Module B có thể copy nguyên file vào FastAPI backend mà không phải mang
theo toàn bộ repo Module A.

Chạy: python export.py --config ../configs/ecapa_voxvietnam.yaml --checkpoint ../outputs/checkpoints/best.pt
"""

import argparse
import shutil
from pathlib import Path

import yaml

SPEAKER_MODEL_TEMPLATE = '''"""
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

SAMPLE_RATE = {sample_rate}
EMBEDDING_DIM = {embedding_dim}

# Threshold TÁCH RIÊNG cho SV và SID — không dùng chung 1 giá trị.
# Lý do: threshold tối ưu cho SV (ưu tiên tuyệt đối giảm False Accept cho lệnh
# nhạy cảm) khi áp cho SID lại gây tỉ lệ từ chối oan quá cao, làm giảm mạnh
# accuracy nhận diện (đo thực nghiệm: identify() dùng threshold SV làm giảm
# top-1 accuracy từ ~85% xuống ~73%, với ~27% truy vấn bị từ chối oan thành
# "unknown" dù model đã chọn đúng người). Xem outputs/metrics/identification_report.md.
DEFAULT_THRESHOLD_SV = {default_threshold_sv}    # dùng cho verify() — lệnh nhạy cảm, ưu tiên an toàn
DEFAULT_THRESHOLD_SID = {default_threshold_sid}   # dùng cho identify() — cá nhân hoá, ưu tiên trải nghiệm


class SpeakerModel:
    def __init__(self, checkpoint_path: str, pretrained_source: str = "{pretrained_source}"):
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
        encoder_state = {{
            k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")
        }}
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

        db_embeddings: dict {{speaker_id: np.ndarray}}. Trả về speaker_id có
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
'''


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def confirm_thresholds_or_exit(threshold_sv: float | None, threshold_sid: float | None,
                                robustness_report_path: Path) -> tuple[float, float]:
    """Bắt buộc cả 2 threshold phải được truyền tường minh qua CLI.

    Tách riêng threshold_sv (verify, ưu tiên an toàn) và threshold_sid
    (identify, ưu tiên trải nghiệm) — không dùng chung 1 giá trị nữa, vì
    thực nghiệm cho thấy threshold tối ưu cho SV làm giảm mạnh accuracy SID
    (xem outputs/metrics/identification_report.md).
    """
    missing = []
    if threshold_sv is None:
        missing.append("--threshold_sv")
    if threshold_sid is None:
        missing.append("--threshold_sid")

    if missing:
        print(f"\n[LỖI] Thiếu tham số bắt buộc: {', '.join(missing)}")
        if robustness_report_path.exists():
            print(f"  - Xem {robustness_report_path} để tham khảo threshold_sv (ưu tiên FAR=0%).")
        print("  - Xem outputs/metrics/identification_report.md để tham khảo threshold_sid "
              "(chọn giá trị giảm tỉ lệ 'unknown' oan mà vẫn chấp nhận được).")
        print("  - Chạy lại: python export.py --config ... --checkpoint ... "
              "--threshold_sv <giá_trị> --threshold_sid <giá_trị>")
        raise SystemExit(1)

    print(f"[OK] threshold_sv = {threshold_sv} (verify - ưu tiên an toàn)")
    print(f"[OK] threshold_sid = {threshold_sid} (identify - ưu tiên trải nghiệm)")
    return threshold_sv, threshold_sid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold_sv", type=float, default=None,
                         help="Threshold cho verify() (SV) — ưu tiên an toàn, dựa trên FAR=0%% thực nghiệm.")
    parser.add_argument("--threshold_sid", type=float, default=None,
                         help="Threshold cho identify() (SID) — ưu tiên trải nghiệm, dựa trên identification_report.md.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint vào outputs/checkpoints/ nếu chưa ở đó
    checkpoint_src = Path(args.checkpoint)
    checkpoint_dst = checkpoints_dir / "best.pt"
    if checkpoint_src.resolve() != checkpoint_dst.resolve():
        shutil.copy(checkpoint_src, checkpoint_dst)

    robustness_report = Path(cfg["robustness_experiment"]["output_report"])
    threshold_sv, threshold_sid = confirm_thresholds_or_exit(
        args.threshold_sv, args.threshold_sid, robustness_report
    )

    speaker_model_code = SPEAKER_MODEL_TEMPLATE.format(
        sample_rate=cfg["audio"]["sample_rate"],
        embedding_dim=cfg["model"]["embedding_dim"],
        default_threshold_sv=threshold_sv,
        default_threshold_sid=threshold_sid,
        pretrained_source=cfg["paths"]["pretrained_source"],
    )

    speaker_model_path = output_dir / "speaker_model.py"
    with open(speaker_model_path, "w", encoding="utf-8") as f:
        f.write(speaker_model_code)

    print(f"\nĐã xuất artefact bàn giao Module B:")
    print(f"  - {speaker_model_path}")
    print(f"  - {checkpoint_dst}")
    print(f"  - {output_dir / 'metrics'}/ (các report EER/minDCF/robustness)")
    print("\n=> Gửi cho Hiếu kèm spec_ModuleA_speaker_training.md mục 2.5 (output contract).")


if __name__ == "__main__":
    main()

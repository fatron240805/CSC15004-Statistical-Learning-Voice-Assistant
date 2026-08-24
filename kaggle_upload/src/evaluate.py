"""
Đánh giá model đã fine-tune bằng EER và minDCF trên tập test.

Quy trình:
1. Sinh trial pairs (positive: cùng speaker, negative: khác speaker) cân bằng 1:1.
2. Tính embedding cho từng utterance (xử lý từng file, không load hết vào RAM).
3. Tính cosine similarity cho từng cặp.
4. Từ điểm số + nhãn -> tính EER và minDCF.

Chạy: python evaluate.py --config ../configs/ecapa_voxvietnam.yaml --checkpoint ../outputs/checkpoints/best.pt
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from model import SpeakerFinetuneModel


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def generate_trial_pairs(rows: list, num_pairs: int, pos_neg_ratio: float, seed: int = 42):
    rng = random.Random(seed)
    by_speaker = {}
    for row in rows:
        by_speaker.setdefault(row["speaker_id"], []).append(row["utt_path"])

    speakers = [s for s, utts in by_speaker.items() if len(utts) >= 2]
    n_pos = int(num_pairs * pos_neg_ratio / (1 + pos_neg_ratio))
    n_neg = num_pairs - n_pos

    pairs = []
    # Positive pairs: cùng speaker, 2 utterance khác nhau
    for _ in range(n_pos):
        spk = rng.choice(speakers)
        a, b = rng.sample(by_speaker[spk], 2)
        pairs.append((a, b, 1))

    # Negative pairs: 2 speaker khác nhau
    for _ in range(n_neg):
        spk_a, spk_b = rng.sample(list(by_speaker.keys()), 2)
        a = rng.choice(by_speaker[spk_a])
        b = rng.choice(by_speaker[spk_b])
        pairs.append((a, b, 0))

    rng.shuffle(pairs)
    return pairs


def load_waveform(path: str, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform  # (1, samples)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Trả về (EER, threshold tại EER)."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC (Area Under ROC Curve) — đo khả năng phân tách tổng thể của model,
    không phụ thuộc vào 1 threshold cụ thể như EER/minDCF. Càng gần 1.0 càng tốt,
    0.5 nghĩa là model không phân biệt được gì (ngẫu nhiên)."""
    return float(roc_auc_score(labels, scores))


def compute_mindcf(scores: np.ndarray, labels: np.ndarray,
                    p_target: float, c_miss: float, c_fa: float) -> float:
    """minDCF theo công thức NIST SRE chuẩn."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    return float(np.min(dcf))


@torch.no_grad()
def score_pairs(model, pairs, sample_rate, device):
    model.eval()
    embedding_cache = {}
    scores, labels = [], []

    def get_emb(path):
        if path not in embedding_cache:
            waveform = load_waveform(path, sample_rate).to(device)
            emb = model.get_embedding(waveform)
            embedding_cache[path] = torch.nn.functional.normalize(emb, dim=1).squeeze(0)
        return embedding_cache[path]

    for a_path, b_path, label in pairs:
        emb_a = get_emb(a_path)
        emb_b = get_emb(b_path)
        score = torch.dot(emb_a, emb_b).item()
        scores.append(score)
        labels.append(label)

    return np.array(scores), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None,
                         help="Mặc định dùng test.csv trong manifest_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = args.manifest or str(Path(cfg["paths"]["manifest_dir"]) / "test.csv")
    rows = load_manifest(manifest_path)

    # num_speakers ở đây chỉ để khởi tạo head cho đúng kiểu dữ liệu, KHÔNG dùng
    # khi eval (eval chỉ lấy embedding từ encoder). Vì số speaker lúc train
    # (checkpoint) và số speaker trong tập đang eval thường khác nhau, head
    # trong checkpoint sẽ luôn lệch shape với head khởi tạo ở đây -> phải lọc
    # bỏ toàn bộ key "head.*" trước khi load, chỉ giữ lại "encoder.*".
    num_speakers_dummy = len({r["speaker_id"] for r in rows})
    model = SpeakerFinetuneModel(
        pretrained_source=cfg["paths"]["pretrained_source"],
        num_speakers=num_speakers_dummy,
        embedding_dim=cfg["model"]["embedding_dim"],
    ).to(device)

    full_state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder_only_state_dict = {
        k: v for k, v in full_state_dict.items() if k.startswith("encoder.")
    }
    missing, unexpected = model.load_state_dict(encoder_only_state_dict, strict=False)
    print(f"Đã load {len(encoder_only_state_dict)} tensor encoder từ checkpoint "
          f"(bỏ qua head vì không cần cho việc trích embedding).")
    if missing:
        print(f"[WARN] Thiếu {len(missing)} key khi load (thường là head.*, không đáng lo).")

    pairs = generate_trial_pairs(
        rows,
        num_pairs=cfg["eval"]["num_trial_pairs"],
        pos_neg_ratio=cfg["eval"]["positive_negative_ratio"],
        seed=cfg["seed"],
    )
    print(f"Đã sinh {len(pairs)} trial pairs từ {manifest_path}")

    scores, labels = score_pairs(model, pairs, cfg["audio"]["sample_rate"], device)

    eer, eer_threshold = compute_eer(scores, labels)
    auc = compute_auc(scores, labels)
    mindcf = compute_mindcf(
        scores, labels,
        p_target=cfg["eval"]["mindcf_p_target"],
        c_miss=cfg["eval"]["mindcf_c_miss"],
        c_fa=cfg["eval"]["mindcf_c_fa"],
    )

    print(f"\nAUC = {auc:.4f}")
    print(f"EER = {eer * 100:.2f}%  (threshold ~ {eer_threshold:.4f})")
    print(f"minDCF = {mindcf:.4f}")

    output_path = Path(cfg["paths"]["output_dir"]) / "metrics" / "clean_test_metrics.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Kết quả đánh giá trên test set sạch\n\n")
        f.write(f"- Số trial pairs: {len(pairs)}\n")
        f.write(f"- AUC: {auc:.4f}\n")
        f.write(f"- EER: {eer * 100:.2f}%\n")
        f.write(f"- Threshold tại EER: {eer_threshold:.4f}\n")
        f.write(f"- minDCF (p_target={cfg['eval']['mindcf_p_target']}): {mindcf:.4f}\n")

    print(f"\nĐã ghi báo cáo vào {output_path}")


if __name__ == "__main__":
    main()
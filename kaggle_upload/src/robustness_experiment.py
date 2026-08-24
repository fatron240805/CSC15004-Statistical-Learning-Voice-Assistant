"""
Thực nghiệm robustness: mô phỏng điều kiện thực tế trong xe hơi.

Đây là phần "đặc thù bối cảnh xe" so với bối cảnh nhà thông minh gốc — noise
trong cabin xe (động cơ, gió, lốp) và lệnh thoại ngắn khiến EER đo trên tập
sạch không phản ánh đúng hiệu năng thực tế. Script này:

1. Với mỗi mức SNR trong config: cộng noise cabin xe vào audio ở mức SNR đó.
2. Với mỗi mức duration trong config: cắt ngắn audio.
3. Đo lại EER cho từng điều kiện (SNR x duration).
4. Xuất bảng kết quả -> dùng để đề xuất threshold vận hành + khuyến nghị
   enrollment tối thiểu bao nhiêu giây.

Chạy: python robustness_experiment.py --config ../configs/ecapa_voxvietnam.yaml --checkpoint ../outputs/checkpoints/best.pt
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

from evaluate import (compute_eer, generate_trial_pairs, load_config,
                       load_manifest, score_pairs)
from model import SpeakerFinetuneModel


def load_random_noise_segment(noise_dir: Path, num_samples: int, sample_rate: int) -> torch.Tensor:
    noise_files = list(noise_dir.rglob("*.wav"))
    if not noise_files:
        raise FileNotFoundError(f"Không tìm thấy file noise nào trong {noise_dir}")

    noise_path = random.choice(noise_files)
    waveform, sr = torchaudio.load(str(noise_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    waveform = waveform.squeeze(0)

    if waveform.shape[0] < num_samples:
        repeats = num_samples // waveform.shape[0] + 1
        waveform = waveform.repeat(repeats)
    start = random.randint(0, waveform.shape[0] - num_samples)
    return waveform[start : start + num_samples]


def add_noise_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(clean_power / (snr_linear * noise_power + 1e-10))
    return clean + noise * scale


def load_and_augment(path: str, sample_rate: int, noise_dir: Path,
                      snr_db: float | None, duration_sec: float | None) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    waveform = waveform.squeeze(0)

    if duration_sec is not None:
        target_len = int(duration_sec * sample_rate)
        if waveform.shape[0] >= target_len:
            start = random.randint(0, waveform.shape[0] - target_len)
            waveform = waveform[start : start + target_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))

    if snr_db is not None:
        noise = load_random_noise_segment(noise_dir, waveform.shape[0], sample_rate)
        waveform = add_noise_at_snr(waveform, noise, snr_db)

    return waveform.unsqueeze(0)  # (1, samples)


@torch.no_grad()
def score_pairs_augmented(model, pairs, sample_rate, noise_dir, snr_db, duration_sec, device):
    model.eval()
    scores, labels = [], []

    for a_path, b_path, label in pairs:
        wa = load_and_augment(a_path, sample_rate, noise_dir, snr_db, duration_sec).to(device)
        wb = load_and_augment(b_path, sample_rate, noise_dir, snr_db, duration_sec).to(device)
        emb_a = torch.nn.functional.normalize(model.get_embedding(wa), dim=1).squeeze(0)
        emb_b = torch.nn.functional.normalize(model.get_embedding(wb), dim=1).squeeze(0)
        scores.append(torch.dot(emb_a, emb_b).item())
        labels.append(label)

    return np.array(scores), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = Path(cfg["paths"]["manifest_dir"]) / "test.csv"
    rows = load_manifest(str(manifest_path))
    num_speakers_dummy = len({r["speaker_id"] for r in rows})

    model = SpeakerFinetuneModel(
        pretrained_source=cfg["paths"]["pretrained_source"],
        num_speakers=num_speakers_dummy,
        embedding_dim=cfg["model"]["embedding_dim"],
    ).to(device)

    # Chỉ load phần encoder, bỏ head (head có số lớp = số speaker lúc train,
    # luôn lệch shape với head khởi tạo tạm ở đây) — cùng lý do đã sửa ở evaluate.py.
    full_state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder_only_state_dict = {
        k: v for k, v in full_state_dict.items() if k.startswith("encoder.")
    }
    model.load_state_dict(encoder_only_state_dict, strict=False)

    pairs = generate_trial_pairs(
        rows,
        num_pairs=min(cfg["eval"]["num_trial_pairs"], 5000),  # ít hơn để tiết kiệm compute cho nhiều điều kiện
        pos_neg_ratio=cfg["eval"]["positive_negative_ratio"],
        seed=cfg["seed"],
    )

    noise_dir = Path(cfg["paths"]["noise_dir"])
    snr_levels = cfg["robustness_experiment"]["snr_levels_db"]
    durations = cfg["robustness_experiment"]["duration_crops_sec"]

    results = []
    # Baseline: audio sạch, độ dài gốc
    scores, labels = score_pairs(model, pairs, cfg["audio"]["sample_rate"], device)
    eer, _ = compute_eer(scores, labels)
    results.append(("clean", "gốc", eer))
    print(f"[clean / gốc] EER = {eer * 100:.2f}%")

    for snr in snr_levels:
        for dur in durations:
            scores, labels = score_pairs_augmented(
                model, pairs, cfg["audio"]["sample_rate"], noise_dir, snr, dur, device
            )
            eer, _ = compute_eer(scores, labels)
            results.append((f"{snr}dB", f"{dur}s", eer))
            print(f"[SNR={snr}dB / dur={dur}s] EER = {eer * 100:.2f}%")

    output_path = Path(cfg["robustness_experiment"]["output_report"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Báo cáo robustness — mô phỏng điều kiện trong xe\n\n")
        f.write("| Điều kiện SNR | Duration | EER (%) |\n|---|---|---|\n")
        for snr_label, dur_label, eer in results:
            f.write(f"| {snr_label} | {dur_label} | {eer * 100:.2f} |\n")
        f.write(
            "\n**Khuyến nghị:** chọn threshold vận hành dựa trên điều kiện SNR/duration "
            "gần nhất với kịch bản demo thực tế (ví dụ 10dB, 3s cho lệnh thoại khi xe "
            "đang chạy), không dùng threshold đo trên tập sạch.\n"
        )

    print(f"\nĐã ghi báo cáo robustness vào {output_path}")


if __name__ == "__main__":
    main()

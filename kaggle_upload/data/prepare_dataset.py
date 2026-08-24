"""
Chuẩn bị dataset VoxVietnam cho fine-tune ECAPA-TDNN — BẢN DÙNG STREAMING.

Lý do đổi cách tiếp cận so với bản đầu: VoxVietnam nặng 44.2GB, vượt quota
disk mặc định của Kaggle notebook (~20GB /kaggle/working), kể cả khi chỉ tải
riêng split "train_small" (vì HuggingFace vẫn kéo về nguyên shard Parquet
chứa split đó chứ không tách nhỏ theo từng speaker). Streaming giải quyết cả
2 vấn đề cùng lúc: không tải file lớn về đĩa, và bản thân dataset ở dạng
Parquet (không phải thư mục audio thô) nên phải đọc qua `datasets` chứ không
quét thư mục như bản cũ.

Việc script này làm:
1. Mở dataset ở chế độ streaming=True (không tải file về đĩa).
2. Duyệt tuần tự từng sample, gom theo speaker_id.
3. Dừng lại NGAY KHI đã đủ `target_num_speakers` speaker, mỗi speaker có
   ít nhất `min_utts_per_speaker` utterance (không cần duyệt hết dataset —
   VoxVietnam có ~1000+ speaker, ta chỉ cần một phần đủ dùng).
4. Ghi các utterance đã gom ra file .wav thật vào extracted_dir/speaker_id/.
5. Từ các file .wav đã ghi, chia dữ liệu theo ĐÚNG logic bản trước:
   a. Tách TEST set theo speaker-disjoint.
   b. TRAIN/VAL chia theo UTTERANCE trên các speaker còn lại (chung speaker
      pool, chỉ khác câu nói) — bắt buộc vì val dùng closed-set classification
      lúc train, không phải open-set verification.
6. Xuất train.csv / val.csv / test.csv + thống kê dataset cho báo cáo.

Yêu cầu: biến môi trường HF_TOKEN phải được set trước khi chạy (trên Kaggle:
lấy từ Kaggle Secrets rồi os.environ["HF_TOKEN"] = hf_token trong 1 cell
trước khi gọi script này), hoặc truyền qua --hf_token.

Chạy: python prepare_dataset.py --config ../configs/ecapa_voxvietnam.yaml
"""

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from datasets import load_dataset


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gather_utterances_streaming(hf_dataset_repo: str, hf_split: str, hf_token: str,
                                 target_num_speakers: int, min_utts_per_speaker: int,
                                 max_utts_per_speaker: int, sample_rate: int,
                                 extracted_dir: Path, seed: int = 42):
    """Duyệt streaming, ghi wav ra đĩa ngay khi nhận từng sample (không giữ
    audio trong RAM lâu), dừng khi đủ điều kiện.

    Trả về dict: speaker_id -> list[(wav_path, duration_sec)], chỉ gồm các
    speaker đã đạt đủ min_utts_per_speaker.
    """
    print(f"Mở dataset streaming: {hf_dataset_repo} [{hf_split}] ...")
    ds_stream = load_dataset(
        hf_dataset_repo, split=hf_split, token=hf_token, streaming=True,
    )
    ds_stream = ds_stream.shuffle(seed=seed, buffer_size=2000)  # trộn nhẹ để không lấy toàn speaker liền kề nhau

    speaker_counts = defaultdict(int)
    completed_speakers = set()
    result = defaultdict(list)

    extracted_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(ds_stream):
        speaker_id = sample["speaker"]

        if speaker_id in completed_speakers:
            continue  # speaker này đã đủ M utterance, bỏ qua để tiết kiệm thời gian
        if speaker_counts[speaker_id] >= max_utts_per_speaker:
            continue

        audio_info = sample["audio"]
        array = np.asarray(audio_info["array"], dtype=np.float32)
        sr = audio_info["sampling_rate"]
        if sr != sample_rate:
            # Dataset đã ở đúng 16kHz theo kiểm tra thực tế, nhưng vẫn cảnh báo
            # phòng trường hợp một số split khác sample rate.
            print(f"[WARN] Sample {i} có sampling_rate={sr}, khác {sample_rate} trong config. Bỏ qua.")
            continue

        speaker_dir = extracted_dir / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        utt_idx = speaker_counts[speaker_id]
        wav_path = speaker_dir / f"utt_{utt_idx}.wav"
        sf.write(str(wav_path), array, sr)

        duration = len(array) / sr
        result[speaker_id].append((str(wav_path), duration))
        speaker_counts[speaker_id] += 1

        if speaker_counts[speaker_id] >= min_utts_per_speaker:
            completed_speakers.add(speaker_id)

        if i % 200 == 0:
            print(f"  Đã duyệt {i} sample streaming, "
                  f"{len(completed_speakers)}/{target_num_speakers} speaker đã đủ utterance.")

        if len(completed_speakers) >= target_num_speakers:
            print(f"Đã gom đủ {target_num_speakers} speaker sau {i + 1} sample streaming.")
            break
    else:
        print(f"[WARN] Duyệt hết dataset nhưng chỉ gom được {len(completed_speakers)}/"
              f"{target_num_speakers} speaker đủ điều kiện. Cân nhắc giảm target_num_speakers "
              f"hoặc min_utts_per_speaker trong config.")

    # Chỉ giữ lại các speaker đã đủ điều kiện (loại speaker dở dang lúc dừng giữa chừng)
    final_result = {spk: utts for spk, utts in result.items() if spk in completed_speakers}
    return final_result


def split_test_speakers(speakers: dict, test_ratio: float, seed: int = 42):
    speaker_ids = list(speakers.keys())
    random.Random(seed).shuffle(speaker_ids)

    n_test = int(len(speaker_ids) * test_ratio)
    test_speaker_ids = set(speaker_ids[:n_test])
    remaining_speaker_ids = speaker_ids[n_test:]

    remaining_speakers = {spk: speakers[spk] for spk in remaining_speaker_ids}
    return test_speaker_ids, remaining_speakers


def split_train_val_by_utterance(remaining_speakers: dict, train_ratio: float,
                                  val_ratio: float, seed: int = 42):
    rng = random.Random(seed)
    val_fraction_within_remaining = val_ratio / (train_ratio + val_ratio)

    train_rows, val_rows = [], []

    for speaker_id, utts in remaining_speakers.items():
        utts_shuffled = utts.copy()
        rng.shuffle(utts_shuffled)

        n_val = max(1, int(len(utts_shuffled) * val_fraction_within_remaining))
        n_val = min(n_val, len(utts_shuffled) - 1) if len(utts_shuffled) > 1 else 0

        val_utts = utts_shuffled[:n_val]
        train_utts = utts_shuffled[n_val:]

        for path, dur in train_utts:
            train_rows.append((path, speaker_id, dur))
        for path, dur in val_utts:
            val_rows.append((path, speaker_id, dur))

    return train_rows, val_rows


def write_manifest_from_rows(out_path: Path, rows: list):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_path", "speaker_id", "duration"])
        for utt_path, speaker_id, dur in rows:
            writer.writerow([utt_path, speaker_id, f"{dur:.3f}"])


def write_manifest_from_speaker_subset(out_path: Path, speakers: dict, speaker_subset: set):
    rows = []
    for speaker_id in speaker_subset:
        for utt_path, dur in speakers[speaker_id]:
            rows.append((utt_path, speaker_id, dur))
    write_manifest_from_rows(out_path, rows)


def print_stats_from_rows(name: str, rows: list):
    speakers_in_set = {r[1] for r in rows}
    utt_counts = defaultdict(int)
    for _, speaker_id, _ in rows:
        utt_counts[speaker_id] += 1
    counts = list(utt_counts.values())

    print(f"\n== {name} ==")
    print(f"  Số speaker: {len(speakers_in_set)}")
    print(f"  Tổng số utterance: {len(rows)}")
    if counts:
        print(f"  Utterance/speaker: min={min(counts)}, max={max(counts)}, "
              f"trung bình={sum(counts) / len(counts):.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=None,
                         help="HF token. Nếu không truyền, đọc từ biến môi trường HF_TOKEN.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit(
            "Thiếu HF token. Trên Kaggle: os.environ['HF_TOKEN'] = hf_token trước khi "
            "chạy script này, hoặc truyền --hf_token trực tiếp."
        )

    extracted_dir = Path(cfg["paths"]["extracted_dir"])
    manifest_dir = Path(cfg["paths"]["manifest_dir"])
    sample_rate = cfg["audio"]["sample_rate"]
    train_ratio = cfg["split"]["train_ratio"]
    val_ratio = cfg["split"]["val_ratio"]
    test_ratio = cfg["split"]["test_ratio"]
    seed = cfg["seed"]

    speakers = gather_utterances_streaming(
        hf_dataset_repo=cfg["paths"]["hf_dataset_repo"],
        hf_split=cfg["paths"]["hf_split"],
        hf_token=hf_token,
        target_num_speakers=cfg["dataset_sampling"]["target_num_speakers"],
        min_utts_per_speaker=cfg["dataset_sampling"]["min_utts_per_speaker"],
        max_utts_per_speaker=cfg["dataset_sampling"]["max_utts_per_speaker"],
        sample_rate=sample_rate,
        extracted_dir=extracted_dir,
        seed=seed,
    )
    print(f"\nTổng số speaker đã trích xuất: {len(speakers)}")

    test_speaker_ids, remaining_speakers = split_test_speakers(speakers, test_ratio, seed)
    train_rows, val_rows = split_train_val_by_utterance(remaining_speakers, train_ratio, val_ratio, seed)

    write_manifest_from_rows(manifest_dir / "train.csv", train_rows)
    write_manifest_from_rows(manifest_dir / "val.csv", val_rows)
    write_manifest_from_speaker_subset(manifest_dir / "test.csv", speakers, test_speaker_ids)

    print_stats_from_rows("Train", train_rows)
    print_stats_from_rows("Val", val_rows)
    print(f"\n== Test ==\n  Số speaker (speaker-disjoint, chưa từng thấy lúc train): {len(test_speaker_ids)}")

    print(f"\nĐã ghi manifest vào {manifest_dir}/")
    print(f"Audio thực tế đã lưu tại: {extracted_dir}/ (chỉ phần đã lấy từ streaming, không phải toàn bộ dataset)")
    print("=> Đưa các số liệu thống kê ở trên vào phần phân tích dataset trong báo cáo, "
          "kèm giải thích rõ đây là subsample có chủ đích từ VoxVietnam (streaming), "
          "không phải toàn bộ dataset, do giới hạn dung lượng đĩa trên Kaggle.")


if __name__ == "__main__":
    main()

"""
Dataset cho fine-tune ECAPA-TDNN trên VoxVietnam.

Nguyên tắc: mỗi __getitem__ chỉ đọc và xử lý 1 file audio (line-by-line theo
đúng CSV manifest), không load toàn bộ dataset vào RAM cùng lúc.
"""

import csv
import random

import torch
import torchaudio
from torch.utils.data import Dataset


class VoxVietnamDataset(Dataset):
    def __init__(self, manifest_csv: str, sample_rate: int = 16000,
                 crop_duration_sec: float = 3.0, speaker_to_idx: dict | None = None):
        self.sample_rate = sample_rate
        self.crop_samples = int(crop_duration_sec * sample_rate)
        self.rows = []

        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

        speakers = sorted({row["speaker_id"] for row in self.rows})
        self.speaker_to_idx = speaker_to_idx or {spk: i for i, spk in enumerate(speakers)}
        self.num_speakers = len(self.speaker_to_idx)

    def __len__(self):
        return len(self.rows)

    def _load_and_resample(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform.squeeze(0)

    def _random_crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        n = waveform.shape[0]
        if n >= self.crop_samples:
            start = random.randint(0, n - self.crop_samples)
            return waveform[start : start + self.crop_samples]
        pad_amount = self.crop_samples - n
        return torch.nn.functional.pad(waveform, (0, pad_amount))

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        waveform = self._load_and_resample(row["utt_path"])
        waveform = self._random_crop_or_pad(waveform)
        speaker_idx = self.speaker_to_idx[row["speaker_id"]]
        return waveform, speaker_idx


def collate_fn(batch):
    waveforms = torch.stack([item[0] for item in batch])
    speaker_idxs = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return waveforms, speaker_idxs

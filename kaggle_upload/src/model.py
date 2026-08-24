"""
Wrapper cho ECAPA-TDNN pretrained (SpeechBrain) + AAM-softmax head để fine-tune
speaker classification trên VoxVietnam.

Sau khi fine-tune xong, chỉ phần encoder (ECAPA-TDNN) được giữ lại để sinh
embedding — head AAM-softmax chỉ dùng trong lúc train, không xuất cho Module B.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier


class ECAPAEncoder(nn.Module):
    """Bọc lại speechbrain EncoderClassifier để lấy embedding 192-dim."""

    def __init__(self, pretrained_source: str, savedir: str = "./pretrained_models/ecapa"):
        super().__init__()
        self.classifier = EncoderClassifier.from_hparams(
            source=pretrained_source, savedir=savedir
        )
        self.embedding_model = self.classifier.mods["embedding_model"]

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: (batch, samples) đã ở 16kHz mono
        feats = self.classifier.mods["compute_features"](waveforms)
        feats = self.classifier.mods["mean_var_norm"](feats, torch.ones(feats.shape[0]).to(feats.device))
        embeddings = self.embedding_model(feats)
        return embeddings.squeeze(1)  # (batch, embedding_dim)


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax — chuẩn cho speaker embedding training."""

    def __init__(self, embedding_dim: int, num_speakers: int, margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_speakers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb_norm = F.normalize(embeddings, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        cosine = F.linear(emb_norm, w_norm)  # (batch, num_speakers)

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + margin)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.scale
        return logits


class SpeakerFinetuneModel(nn.Module):
    def __init__(self, pretrained_source: str, num_speakers: int,
                 embedding_dim: int = 192, margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.encoder = ECAPAEncoder(pretrained_source)
        self.head = AAMSoftmax(embedding_dim, num_speakers, margin, scale)

    def forward(self, waveforms: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(waveforms)
        return self.head(embeddings, labels)

    def get_embedding(self, waveforms: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(waveforms)

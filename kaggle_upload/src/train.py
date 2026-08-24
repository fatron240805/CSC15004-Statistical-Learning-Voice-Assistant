"""
Fine-tune ECAPA-TDNN trên VoxVietnam với AAM-softmax.

Chạy: python train.py --config ../configs/ecapa_voxvietnam.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import VoxVietnamDataset, collate_fn
from model import SpeakerFinetuneModel


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for waveforms, labels in dataloader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(waveforms, labels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(waveforms, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * waveforms.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for waveforms, labels in dataloader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        logits = model(waveforms, labels)
        loss = criterion(logits, labels)
        total_loss += loss.item() * waveforms.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng device: {device}")

    manifest_dir = Path(cfg["paths"]["manifest_dir"])
    train_ds = VoxVietnamDataset(
        str(manifest_dir / "train.csv"),
        sample_rate=cfg["audio"]["sample_rate"],
        crop_duration_sec=3.0,
    )
    val_ds = VoxVietnamDataset(
        str(manifest_dir / "val.csv"),
        sample_rate=cfg["audio"]["sample_rate"],
        crop_duration_sec=3.0,
        speaker_to_idx=train_ds.speaker_to_idx,  # dùng chung mapping với train
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn,
    )

    model = SpeakerFinetuneModel(
        pretrained_source=cfg["paths"]["pretrained_source"],
        num_speakers=train_ds.num_speakers,
        embedding_dim=cfg["model"]["embedding_dim"],
        margin=cfg["loss"]["margin"],
        scale=cfg["loss"]["scale"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg["train"]["lr"],
                      weight_decay=cfg["train"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["num_epochs"])
    scaler = torch.cuda.amp.GradScaler() if cfg["train"]["mixed_precision"] and device.type == "cuda" else None

    output_dir = Path(cfg["paths"]["output_dir"]) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, cfg["train"]["num_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()

        print(f"[Epoch {epoch}/{cfg['train']['num_epochs']}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if cfg["train"]["checkpoint_every_epoch"]:
            torch.save(model.state_dict(), output_dir / f"epoch_{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"  -> Lưu best checkpoint (val_loss={val_loss:.4f})")

    print(f"\nHoàn tất fine-tune. Checkpoint tốt nhất: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

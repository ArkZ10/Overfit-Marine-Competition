#!/usr/bin/env python3
"""Train the 35-class crop rescorer (34 NAMR33 classes + background/neighbor).

  python3 -m rescorer.train_rescorer            # full run (~1-2 h)
  python3 -m rescorer.train_rescorer --spike    # 3 epochs on <=2k crops
"""
import argparse
import csv
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ENS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENS_DIR))

from paths12 import CROPS_DIR, RUNS_DIR, SEED  # noqa: E402

N_CLASSES = 35
MODEL = "convnext_tiny.fb_in22k_ft_in1k"
FALLBACK = "efficientnet_b2"
EPOCHS = 20
BATCH = 256
LR = 3e-4
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class CropDataset(Dataset):
    def __init__(self, manifest_csv, train=False, limit=None):
        rows = list(csv.DictReader(open(manifest_csv)))
        if limit:
            rng = random.Random(SEED)
            rng.shuffle(rows)
            rows = rows[:limit]
        self.items = [(CROPS_DIR / r["path"], int(r["label"])) for r in rows]
        self.train = train

    def __len__(self):
        return len(self.items)

    def labels(self):
        return [lb for _, lb in self.items]

    def __getitem__(self, idx):
        path, label = self.items[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
            t = t.view(im.size[1], im.size[0], 3).permute(2, 0, 1).float() / 255.0
        if self.train and torch.rand(1).item() < 0.5:
            t = t.flip(-1)
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return t, label


def build_model(device, pretrained=True, model_name=MODEL):
    import timm

    try:
        m = timm.create_model(model_name, pretrained=pretrained, num_classes=N_CLASSES)
    except Exception as e:  # noqa: BLE001
        print(f"{model_name} failed ({e}); falling back to {FALLBACK}")
        m = timm.create_model(FALLBACK, pretrained=pretrained, num_classes=N_CLASSES)
    return m.to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spike", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(a.device)
    limit = 2000 if a.spike else None
    epochs = 3 if a.spike else EPOCHS
    batch = 64 if a.spike else BATCH  # small batch so a 2k-crop spike still gets ~90 optimizer steps
    out_dir = RUNS_DIR / ("_rescorer_spike" if a.spike else "rescorer")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CropDataset(CROPS_DIR / "train_manifest.csv", train=True, limit=limit)
    val_ds = CropDataset(CROPS_DIR / "val_manifest.csv", train=False, limit=limit)

    counts = Counter(train_ds.labels())
    weights = [1.0 / counts[lb] for lb in train_ds.labels()]  # class-balanced sampling
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True,
                                    generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=8,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=8,
                            pin_memory=True)

    model = build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler()

    majority = max(Counter(val_ds.labels()).values()) / len(val_ds)
    print(f"train {len(train_ds)} crops / val {len(val_ds)}; majority-class baseline acc = {majority:.4f}")

    best_acc, t0 = 0.0, time.time()
    for ep in range(epochs):
        model.train()
        run = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast("cuda"):
                    pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += len(y)
        acc = correct / max(1, total)
        print(f"epoch {ep}: loss={run / max(1, len(train_loader)):.4f}  val_acc={acc:.4f}")
        torch.save({"model": model.state_dict(), "epoch": ep, "acc": acc}, out_dir / "last.pth")
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "epoch": ep, "acc": acc}, out_dir / "best.pth")

    print(f"done in {(time.time() - t0) / 60:.1f} min; best val_acc={best_acc:.4f} "
          f"(majority baseline {majority:.4f}); weights: {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()

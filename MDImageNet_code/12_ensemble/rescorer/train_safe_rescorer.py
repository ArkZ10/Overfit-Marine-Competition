#!/usr/bin/env python3
"""Train a crop rescorer without class rebalancing or ambiguous negatives."""

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from rescorer.train_rescorer import (IMAGENET_MEAN, IMAGENET_STD, N_CLASSES,
                                      build_model)

SEED = 42


class SafeCrops(Dataset):
    def __init__(self, root, manifest, train=False):
        self.root = Path(root)
        self.rows = list(csv.DictReader(open(manifest)))
        self.train = train

    def __len__(self): return len(self.rows)
    def labels(self): return [int(r['label']) for r in self.rows]

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as image:
            image = image.convert('RGB')
            tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            tensor = tensor.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255
        if self.train and torch.rand(()) < 0.5:
            tensor = tensor.flip(-1)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD, int(row['label'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--crops-root', type=Path, required=True)
    ap.add_argument('--train-manifest', type=Path, required=True)
    ap.add_argument('--val-manifest', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--model', default='convnext_tiny.fb_in22k_ft_in1k')
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    train = SafeCrops(args.crops_root / 'train', args.train_manifest, train=True)
    val = SafeCrops(args.crops_root / 'val_fit', args.val_manifest)
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=8,
                              pin_memory=True, drop_last=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=8,
                            pin_memory=True)
    device = torch.device(args.device)
    model = build_model(device, model_name=args.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {'train_manifest': str(args.train_manifest), 'val_manifest': str(args.val_manifest),
                  'train_counts': Counter(train.labels()), 'val_counts': Counter(val.labels()),
                  'sampling': 'natural/shuffle; no class balancing', 'seed': SEED,
                  'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr,
                  'model': args.model}
    (args.out_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2, default=dict))

    best_loss = float('inf'); started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_loss += loss.item()
        scheduler.step()
        model.eval(); val_loss = 0.0; correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.autocast('cuda'):
                    logits = model(images); loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item(); total += len(labels)
        val_loss /= max(total, 1); accuracy = correct / max(total, 1)
        checkpoint = {'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss,
                      'acc': accuracy, 'provenance': provenance}
        torch.save(checkpoint, args.out_dir / 'last.pth')
        if val_loss < best_loss:
            best_loss = val_loss; torch.save(checkpoint, args.out_dir / 'best.pth')
        print(f'epoch {epoch}: train_loss={train_loss/max(len(train_loader),1):.4f} '
              f'val_loss={val_loss:.4f} val_acc={accuracy:.4f}', flush=True)
    print(f'done minutes={(time.time()-started)/60:.1f} best_val_loss={best_loss:.4f}')


if __name__ == '__main__':
    main()

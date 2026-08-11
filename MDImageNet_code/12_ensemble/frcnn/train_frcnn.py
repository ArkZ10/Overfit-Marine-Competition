#!/usr/bin/env python3
"""Detector C: Faster R-CNN ResNet50-FPN v2 (torchvision), custom loop.

  python3 -m frcnn.train_frcnn                    # full 26-epoch run
  python3 -m frcnn.train_frcnn --resume
  python3 -m frcnn.train_frcnn --spike            # 2 epochs on 200 images

COCO-pretrained backbone+heads, box_predictor swapped to 35 classes
(34 NAMR33 + background). Per-epoch val AP50 with pycocotools against the
frozen preds/gt_val_namr33.json.
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ENS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENS_DIR))

from paths12 import GT_VAL_JSON, RUNS_DIR, SEED, TRAIN_LIST, VAL_LIST  # noqa: E402
from frcnn.dataset import YoloListDetection, collate  # noqa: E402

EPOCHS = 26
BATCH = 8
LR = 0.01           # torchvision reference: 0.02 @ global batch 16 -> 0.01 @ batch 8
WARMUP_EPOCHS = 1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 35    # 34 NAMR33 + background


def build_model(device):
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.transform.min_size = (640,)
    model.transform.max_size = 640
    return model.to(device)


@torch.no_grad()
def evaluate_ap50(model, loader, device, stem_to_id):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, io

    model.eval()
    dets = []
    for imgs, targets in loader:
        imgs = [i.to(device) for i in imgs]
        with torch.autocast("cuda"):
            outputs = model(imgs)
        for out, tgt in zip(outputs, targets):
            image_id = stem_to_id.get(tgt["stem"])
            if image_id is None:
                continue
            boxes = out["boxes"].float().cpu().numpy()
            scores = out["scores"].float().cpu().numpy()
            labels = out["labels"].cpu().numpy()
            for (x1, y1, x2, y2), s, lb in zip(boxes, scores, labels):
                dets.append({"image_id": image_id, "category_id": int(lb) - 1,
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             "score": float(s)})
    if not dets:
        return 0.0, 0.0, 0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(GT_VAL_JSON))
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[1]), float(ev.stats[0]), len(dets)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--spike", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True  # some detection ops lack deterministic kernels; seeds still fixed

    device = torch.device(args.device)
    run_name = "_frcnn_spike" if args.spike else "frcnn_r50v2"
    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = YoloListDetection(TRAIN_LIST, train=True)
    val_ds = YoloListDetection(VAL_LIST, train=False)
    if args.spike:
        train_ds.paths = train_ds.paths[:200]
        val_ds.paths = val_ds.paths[:50]
    epochs = args.epochs or (2 if args.spike else EPOCHS)

    gt = json.loads(GT_VAL_JSON.read_text())
    stem_to_id = {Path(im["file_name"]).stem: im["id"] for im in gt["images"]}

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=8,
                              collate_fn=collate, pin_memory=True, drop_last=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=8,
                            collate_fn=collate, pin_memory=True)

    model = build_model(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler()

    iters_per_epoch = len(train_loader)
    warmup_iters = WARMUP_EPOCHS * iters_per_epoch

    def lr_at(it):
        if it < warmup_iters:
            return LR * (it + 1) / warmup_iters
        prog = (it - warmup_iters) / max(1, epochs * iters_per_epoch - warmup_iters)
        return LR * 0.5 * (1 + math.cos(math.pi * prog))

    start_epoch, best_ap50 = 0, -1.0
    ckpt_last, ckpt_best = out_dir / "last.pth", out_dir / "best.pth"
    if args.resume and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_ap50 = ck["epoch"] + 1, ck["best_ap50"]
        print(f"resumed from epoch {ck['epoch']} (best_ap50={best_ap50:.4f})")

    csv_path = out_dir / "results.csv"
    if not csv_path.exists():
        csv_path.write_text("epoch,ap50,ap50_95,train_loss,lr,seconds\n")

    t_start = time.time()
    global_it = start_epoch * iters_per_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (imgs, targets) in enumerate(train_loader):
            for g in optimizer.param_groups:
                g["lr"] = lr_at(global_it)
            imgs = [im.to(device) for im in imgs]
            tgts = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
                    for t in targets]
            with torch.autocast("cuda"):
                loss_dict = model(imgs, tgts)
                loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            global_it += 1
            if i % 200 == 0:
                print(f"  ep{epoch} it{i}/{iters_per_epoch} loss={loss.item():.4f} lr={lr_at(global_it):.5f}",
                      flush=True)
            if not math.isfinite(loss.item()):
                raise SystemExit(f"non-finite loss at epoch {epoch} iter {i}: {loss_dict}")

        ap50, ap5095, n_dets = evaluate_ap50(model, val_loader, device, stem_to_id)
        secs = time.time() - t0
        avg_loss = running / max(1, iters_per_epoch)
        print(f"epoch {epoch}: AP50={ap50:.4f} AP50-95={ap5095:.4f} loss={avg_loss:.4f} ({secs:.0f}s)")
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{ap50:.5f},{ap5095:.5f},{avg_loss:.5f},{lr_at(global_it):.6f},{secs:.1f}\n")

        improved = ap50 > best_ap50
        best_ap50 = max(best_ap50, ap50)  # update BEFORE saving so last.pth carries the true best
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch, "best_ap50": best_ap50}
        torch.save(state, ckpt_last)
        if improved:
            torch.save(state, ckpt_best)

    wall = time.time() - t_start
    (out_dir / "wall_time_seconds.txt").write_text(f"{wall:.1f}\n")
    print(f"done in {wall / 3600:.3f} h; best AP50={best_ap50:.4f}; weights: {ckpt_best}")


if __name__ == "__main__":
    main()

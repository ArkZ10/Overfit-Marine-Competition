#!/usr/bin/env python3
"""Detector H: RF-DETR-Large, fine-tuned on the official marine train split.

  python3 spike_rfdetr.py                 # 2-epoch smoke test FIRST
  nohup python3 train_rfdetr.py > runs/rfdetr_l.log 2>&1 &

Licence: RF-DETR Nano..Large are Apache-2.0 (only XL/2XL need PML). The backbone is
DINOv2, which Meta relicensed to Apache-2.0 on 2023-08-31 -- it is NOT DINOv3, so the
project-wide DINOv3 ban does not apply. Verified 2026-08-15.

COCO reference: RF-DETR-L is AP50 75.1 / AP50-95 56.5, above D-FINE-L 54.0,
RT-DETR-l 53.0, RTMDet-L 51.3 -- the strongest legal starting point we have, and
architecturally unlike every current member (ViT vs HGNetv2 / CSPNeXt / ResNet).
"""
import argparse
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 42
RESOLUTION = 640          # must be a multiple of patch_size(16) * num_windows(2) = 32;
                          # 640 also matches every other member and the frozen imgsz


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(HERE / "data_spike"))
    ap.add_argument("--out", default=str(HERE / "runs" / "rfdetr_spike"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch * this")
    ap.add_argument("--resolution", type=int, default=RESOLUTION)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--early-stopping", action="store_true")
    args = ap.parse_args()

    from rfdetr import RFDETRLarge

    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = RFDETRLarge()
    model.train(
        dataset_dir=args.dataset,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        resolution=args.resolution,
        lr=args.lr,
        seed=SEED,
        num_workers=8,
        tensorboard=False,
        wandb=False,
        early_stopping=args.early_stopping,
    )
    mins = (time.time() - t0) / 60
    (Path(args.out) / "wall_time.txt").write_text(f"{mins:.1f} min\n")
    print(f"done in {mins:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()

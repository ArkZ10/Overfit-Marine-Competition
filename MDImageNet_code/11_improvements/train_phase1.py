#!/usr/bin/env python3
"""Phase 1 training driver: YOLOv11m control / RFS variant.

Fixed per spec: imgsz 640, epochs 150, seed 42, close_mosaic 20, COCO-pretrained
yolo11m.pt. Batch starts at 32 and backs off on CUDA OOM. No other hyperparameter
tuning, no TTA.

  python3 train_phase1.py --variant control
  python3 train_phase1.py --variant rfs --t 0.05
  python3 train_phase1.py --variant control --resume     # after a crash
"""
import argparse
import gc
import os
import time
from pathlib import Path

import torch

from paths import IMPROVE_DIR, RUNS_DIR, SEED

BATCH_LADDER = [32, 24, 16]
EPOCHS = 150
IMGSZ = 640
CLOSE_MOSAIC = 20
MODEL = "yolo11m.pt"  # Ultralytics COCO-pretrained YOLO11m (AGPL-3.0, open source)


def is_oom(err: BaseException) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["control", "rfs"], required=True)
    ap.add_argument("--t", type=float, default=None,
                    help="RFS t (required for --variant rfs); must match a generated data_rfs_t*.yaml")
    ap.add_argument("--batch", type=int, default=None, help="override; skips the OOM ladder")
    ap.add_argument("--patience", type=int, default=100,
                    help="stop after N epochs with no improvement (ultralytics default 100). "
                         "Does not change which checkpoint best.pt holds - that is always the "
                         "peak-fitness epoch - it only trims the plateau after the peak.")
    ap.add_argument("--resume", action="store_true", help="resume from last.pt of the same run name")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    if args.variant == "rfs":
        if args.t is None:
            raise SystemExit("--t is required for --variant rfs")
        tag = f"t{args.t:g}"
        data_yaml = IMPROVE_DIR / f"data_rfs_{tag}.yaml"
        run_name = f"y11m_rfs_{tag}"
    else:
        data_yaml = IMPROVE_DIR / "data_control.yaml"
        run_name = "y11m_control"

    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found - run make_rfs_list.py first")

    # keep auto-downloaded pretrained weights inside 11_improvements/ instead of CWD
    os.chdir(IMPROVE_DIR)

    from ultralytics import YOLO

    ladder = [args.batch] if args.batch else BATCH_LADDER
    last_err = None

    for batch in ladder:
        print(f"\n=== {run_name}: attempting batch={batch} ===")
        try:
            model = YOLO(MODEL)
            t0 = time.time()
            model.train(
                data=str(data_yaml),
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=batch,
                seed=SEED,
                deterministic=True,
                patience=args.patience,
                close_mosaic=CLOSE_MOSAIC,
                device=args.device,
                project=str(RUNS_DIR),
                name=run_name,
                exist_ok=True,
                resume=args.resume,
                plots=True,
            )
            wall = time.time() - t0
            out = RUNS_DIR / run_name
            (out / "wall_time_seconds.txt").write_text(f"{wall:.1f}\n")
            print(f"\n=== {run_name} finished in {wall / 3600:.3f} h at batch={batch} ===")
            print(f"weights: {out / 'weights' / 'best.pt'}")
            return
        except Exception as e:  # noqa: BLE001 - need the OOM check before deciding
            last_err = e
            if not is_oom(e):
                raise
            print(f"OOM at batch={batch}; backing off")
            try:
                del model
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    raise SystemExit(f"all batch sizes {ladder} OOMed; last error: {last_err}")


if __name__ == "__main__":
    main()

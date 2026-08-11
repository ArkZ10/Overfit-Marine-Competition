#!/usr/bin/env python3
"""Detector B: RT-DETR-l fine-tuned on the competition data (control list).

  python3 train_rtdetr.py                      # full run
  python3 train_rtdetr.py --resume             # after a crash
  python3 train_rtdetr.py --spike              # 2-epoch/200-image smoke test

COCO-pretrained rtdetr-l.pt. NMS-free transformer - the architectural
counterweight to YOLO in the ensemble.
"""
import argparse
import gc
import os
import time

import torch

from paths12 import DATA_CONTROL_YAML, ENS_DIR, IMPROVE_DIR, RUNS_DIR, SEED, VAL_LIST

EPOCHS = 100          # DETRs converge by ~72-100; 150 wastes budget
PATIENCE = 30
IMGSZ = 640
CLOSE_MOSAIC = 20
BATCH_LADDER = [16, 12, 8]  # decoder is VRAM-heavy; 32 does not fit at 640/24GB
MODEL = "rtdetr-l.pt"  # Ultralytics COCO-pretrained RT-DETR-l (AGPL-3.0)


def is_oom(err: BaseException) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def make_spike_yaml():
    """200-image train list + tiny val, reusing the control symlink tree."""
    import sys
    sys.path.insert(0, str(IMPROVE_DIR))
    from make_rfs_list import write_list, write_yaml
    from paths import variant_tree  # 11_improvements/paths.py

    train_imgs = sorted((variant_tree("control") / "images" / "train").glob("*.jpg"))[:200]
    val_lines = [ln for ln in VAL_LIST.read_text().splitlines() if ln][:50]
    spike_dir = ENS_DIR / "data"
    spike_dir.mkdir(exist_ok=True)
    write_list(spike_dir / "spike_train.txt", [str(p) for p in train_imgs])
    write_list(spike_dir / "spike_val.txt", val_lines)
    yaml_path = ENS_DIR / "data_spike.yaml"
    write_yaml(yaml_path, variant_tree("control"), spike_dir / "spike_train.txt",
               spike_dir / "spike_val.txt")
    return yaml_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--spike", action="store_true")
    args = ap.parse_args()

    os.chdir(ENS_DIR)  # keeps the auto-downloaded rtdetr-l.pt inside 12_ensemble/

    from ultralytics import RTDETR

    if args.spike:
        data, epochs, name, patience = make_spike_yaml(), 2, "_rtdetr_spike", 100
    else:
        data, epochs, name, patience = DATA_CONTROL_YAML, EPOCHS, "rtdetr_l", PATIENCE

    ladder = [args.batch] if args.batch else BATCH_LADDER
    last_err = None
    for batch in ladder:
        print(f"\n=== rtdetr_l: attempting batch={batch} ===")
        try:
            model = RTDETR(MODEL)
            t0 = time.time()
            model.train(
                data=str(data),
                epochs=epochs,
                imgsz=IMGSZ,
                batch=batch,
                seed=SEED,
                deterministic=True,
                patience=patience,
                close_mosaic=CLOSE_MOSAIC,
                device=args.device,
                project=str(RUNS_DIR),
                name=name,
                exist_ok=True,
                resume=args.resume,
                plots=not args.spike,
            )
            wall = time.time() - t0
            (RUNS_DIR / name / "wall_time_seconds.txt").write_text(f"{wall:.1f}\n")
            print(f"\n=== {name} finished in {wall / 3600:.3f} h at batch={batch} ===")
            print(f"weights: {RUNS_DIR / name / 'weights' / 'best.pt'}")
            return
        except Exception as e:  # noqa: BLE001
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

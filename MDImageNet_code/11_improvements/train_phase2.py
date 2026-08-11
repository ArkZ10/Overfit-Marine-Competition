#!/usr/bin/env python3
"""Phase 2: YOLO11m + frozen DINOv3 ViT-B/16 injected at P0 and P3.

Trains on the Phase 1 winner's data list (control, non-RFS) with the same
imgsz/epochs/seed so the numbers are comparable.

  python3 train_phase2.py
  python3 train_phase2.py --resume
"""
import argparse
import gc
import os
import time

import torch

from paths import IMPROVE_DIR, RUNS_DIR, SEED  # noqa: I001
import dino3_modules  # noqa: E402

CFG = IMPROVE_DIR / "yolo11m-dino3-p0p3.yaml"
DATA = IMPROVE_DIR / "data_control.yaml"  # Phase 1 winner
PRETRAINED = IMPROVE_DIR / "yolo11m.pt"
RUN_NAME = "y11m_dino_p0p3"

EPOCHS = 150
IMGSZ = 640
CLOSE_MOSAIC = 20
BATCH_LADDER = [32, 24, 16, 12, 8]


def remap_stock_index(i: int) -> int:
    """Stock yolo11 layer index -> our index.

    Layer 0 (P0 preprocessor) shifts stock 0-10 backbone / 11-23 head by +1;
    layer 6 (P3 fusion) shifts everything from stock 5 onward by a further +1.
    """
    return i + 1 if i <= 4 else i + 2


def load_pretrained(model, weights_path) -> tuple[int, int]:
    """Transfer COCO-pretrained YOLO11m weights across the shifted layer indices.

    Ultralytics' own .load() matches state_dict keys by name, which would transfer
    almost nothing here because every stock index moved. Returns (n_loaded, n_target).
    """
    from ultralytics import YOLO

    src = YOLO(str(weights_path)).model.float().state_dict()
    tgt = model.state_dict()

    remapped = {}
    for k, v in src.items():
        parts = k.split(".")
        if len(parts) < 2 or parts[0] != "model" or not parts[1].isdigit():
            continue
        new_key = ".".join(["model", str(remap_stock_index(int(parts[1])))] + parts[2:])
        if new_key in tgt and tgt[new_key].shape == v.shape:
            remapped[new_key] = v

    model.load_state_dict(remapped, strict=False)
    return len(remapped), len(tgt)


def is_oom(err: BaseException) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    os.chdir(IMPROVE_DIR)
    dino3_modules.register()

    from ultralytics import YOLO

    ladder = [args.batch] if args.batch else BATCH_LADDER
    last_err = None

    for batch in ladder:
        print(f"\n=== {RUN_NAME}: attempting batch={batch} ===")
        try:
            m = YOLO(str(CFG), task="detect")
            n_loaded, n_total = load_pretrained(m.model, PRETRAINED)
            print(f"  transferred {n_loaded}/{n_total} tensors from COCO-pretrained yolo11m.pt")
            if n_loaded < 300:
                raise SystemExit(
                    f"only {n_loaded} tensors transferred - the index remap is wrong; "
                    "aborting rather than training a randomly-initialised backbone"
                )

            t0 = time.time()
            m.train(
                data=str(DATA),
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=batch,
                seed=SEED,
                deterministic=True,
                patience=args.patience,
                close_mosaic=CLOSE_MOSAIC,
                freeze=dino3_modules.FREEZE_SPEC,  # without this the trainer un-freezes the ViT
                device=args.device,
                project=str(RUNS_DIR),
                name=RUN_NAME,
                exist_ok=True,
                resume=args.resume,
                plots=True,
            )
            wall = time.time() - t0
            (RUNS_DIR / RUN_NAME / "wall_time_seconds.txt").write_text(f"{wall:.1f}\n")
            print(f"\n=== {RUN_NAME} finished in {wall / 3600:.3f} h at batch={batch} ===")
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            if not is_oom(e):
                raise
            print(f"OOM at batch={batch}; backing off")
            try:
                del m
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    raise SystemExit(f"all batch sizes {ladder} OOMed; last error: {last_err}")


if __name__ == "__main__":
    main()

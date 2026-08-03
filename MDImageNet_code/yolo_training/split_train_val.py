#!/usr/bin/env python3
"""Split MDImageDataset/train into a train/val layout YOLO can consume directly.

Source images and NAMR33 labels are not copied; the split directory is
populated with symlinks back into MDImageDataset/train/{images,labels_NAMR33_train}.
"""
import argparse
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "MDImageDataset"
SRC_IMAGES = DATASET_ROOT / "train" / "images"
SRC_LABELS = DATASET_ROOT / "train" / "labels_NAMR33_train"


def build_split(out_dir: Path, val_frac: float, seed: int) -> None:
    stems = sorted(p.stem for p in SRC_IMAGES.glob("*.jpg"))
    if not stems:
        raise SystemExit(f"No images found under {SRC_IMAGES}")

    rng = random.Random(seed)
    stems_shuffled = stems[:]
    rng.shuffle(stems_shuffled)
    n_val = int(len(stems_shuffled) * val_frac)
    val_set = set(stems_shuffled[:n_val])

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_linked = {"train": 0, "val": 0}
    n_missing_label = 0
    for stem in stems:
        split = "val" if stem in val_set else "train"
        img_src = SRC_IMAGES / f"{stem}.jpg"
        lbl_src = SRC_LABELS / f"{stem}.txt"
        if not lbl_src.exists():
            n_missing_label += 1
            continue

        img_link = out_dir / "images" / split / img_src.name
        lbl_link = out_dir / "labels" / split / lbl_src.name
        if not img_link.exists():
            img_link.symlink_to(img_src)
        if not lbl_link.exists():
            lbl_link.symlink_to(lbl_src)
        n_linked[split] += 1

    print(f"train: {n_linked['train']} images")
    print(f"val:   {n_linked['val']} images")
    if n_missing_label:
        print(f"skipped {n_missing_label} images with no NAMR33 label file")
    print(f"split written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATASET_ROOT / "yolo_split")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_split(args.out_dir, args.val_frac, args.seed)

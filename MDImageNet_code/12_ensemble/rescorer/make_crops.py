#!/usr/bin/env python3
"""Stage-3 crop dataset: GT boxes -> 224px class crops + background/neighbor negatives.

  python3 -m rescorer.make_crops             # full train+val crop sets
  python3 -m rescorer.make_crops --limit 60  # spike: first N images per split

Positives: every GT box expanded 1.4x (clamped, min side 32px), class 0-33.
Class 34 (background/neighbor): (a) random crops with IoU < 0.1 vs all GT,
(b) GT boxes offset by 0.5-1.0x their size so the center object falls off-crop;
<= 2 negatives per image. Seeded (42).
"""
import argparse
import csv
import random
import sys
from pathlib import Path

from PIL import Image

ENS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENS_DIR))

from paths12 import CROPS_DIR, SEED, TRAIN_LIST, VAL_LIST  # noqa: E402
from frcnn.dataset import image_to_label_path  # noqa: E402

EXPAND = 1.4
CROP_SIZE = 224
MIN_SIDE = 32
NEG_PER_IMAGE = 2
BG_CLASS = 34


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_boxes(lbl_path, w, h):
    out = []
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            out.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                            (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def expand_clamp(box, w, h, factor=EXPAND):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max((x2 - x1) * factor, MIN_SIDE), max((y2 - y1) * factor, MIN_SIDE)
    nx1, ny1 = max(0, cx - bw / 2), max(0, cy - bh / 2)
    nx2, ny2 = min(w, cx + bw / 2), min(h, cy + bh / 2)
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return None
    return [nx1, ny1, nx2, ny2]


def crop_and_save(im, box, out_path):
    c = im.crop((box[0], box[1], box[2], box[3])).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c.convert("RGB").save(out_path, quality=90)


def negatives_for_image(rng, gts, w, h):
    """<=NEG_PER_IMAGE background/neighbor boxes for one image."""
    negs = []
    boxes_only = [b for _, b in gts]
    # (a) random crops, IoU < 0.1 vs every GT
    for _ in range(30):
        if len(negs) >= NEG_PER_IMAGE:
            break
        side = rng.uniform(MIN_SIDE, min(w, h) / 2)
        x1 = rng.uniform(0, w - side)
        y1 = rng.uniform(0, h - side)
        cand = [x1, y1, x1 + side, y1 + side]
        if all(iou(cand, g) < 0.1 for g in boxes_only):
            negs.append(cand)
    # (b) offset GT boxes - center object falls off-crop
    if gts and len(negs) < NEG_PER_IMAGE:
        for _, g in rng.sample(gts, min(len(gts), NEG_PER_IMAGE - len(negs))):
            bw, bh = g[2] - g[0], g[3] - g[1]
            dx = rng.uniform(0.5, 1.0) * bw * rng.choice([-1, 1])
            dy = rng.uniform(0.5, 1.0) * bh * rng.choice([-1, 1])
            cand = [g[0] + dx, g[1] + dy, g[2] + dx, g[3] + dy]
            cand = [max(0, cand[0]), max(0, cand[1]), min(w, cand[2]), min(h, cand[3])]
            if cand[2] - cand[0] >= MIN_SIDE and cand[3] - cand[1] >= MIN_SIDE:
                negs.append(cand)
    return negs[:NEG_PER_IMAGE]


def build_split(list_file, split, limit=None):
    rng = random.Random(SEED)
    paths = [Path(p) for p in list_file.read_text().splitlines() if p.strip()]
    if limit:
        paths = paths[:limit]
    manifest = []
    n_pos = n_neg = 0
    for img_path in paths:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            gts = read_boxes(image_to_label_path(img_path), w, h)
            for j, (cls, box) in enumerate(gts):
                eb = expand_clamp(box, w, h)
                if eb is None:
                    continue
                out = CROPS_DIR / split / str(cls) / f"{img_path.stem}_{j}.jpg"
                crop_and_save(im, eb, out)
                manifest.append((str(out.relative_to(CROPS_DIR)), cls))
                n_pos += 1
            for j, box in enumerate(negatives_for_image(rng, gts, w, h)):
                out = CROPS_DIR / split / str(BG_CLASS) / f"{img_path.stem}_neg{j}.jpg"
                crop_and_save(im, box, out)
                manifest.append((str(out.relative_to(CROPS_DIR)), BG_CLASS))
                n_neg += 1

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CROPS_DIR / f"{split}_manifest.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["path", "label"])
        wcsv.writerows(manifest)
    print(f"{split}: {n_pos} positives + {n_neg} background/neighbor = {len(manifest)} crops")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="first N images per split (spike)")
    a = ap.parse_args()
    build_split(TRAIN_LIST, "train", a.limit)
    build_split(VAL_LIST, "val", a.limit)

#!/usr/bin/env python3
"""Copy-Paste augmentation: composite SAM-cut rare-class instances into train images.

Builds a NEW training set = the official 13,577 images plus N synthetic images,
each an existing train image with 1-3 rare-class instances alpha-composited in.
Labels for pasted objects come from the paste geometry (exact by construction).

  python3 paste_augment.py --per-class 400
  python3 paste_augment.py --per-class 20 --smoke

Rules that keep the composites honest:
  * paste only onto images that do NOT already contain that class (avoids
    ambiguity with existing GT)
  * reject placements overlapping any existing GT box by IoU > 0.15, so we never
    occlude a real annotated object and invalidate its label
  * scale each instance relative to the target image so pasted objects keep a
    plausible size; jitter scale/flip/brightness so repeats are not identical
Writes: out/images/*.jpg, out/labels/*.txt, out/train_copypaste.txt
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

HERE = Path(__file__).resolve().parent
TRAIN_LIST = HERE.parent / "13_dfine" / "lists" / "official_train.txt"
INST = HERE / "instances"
OUT = HERE / "out"
SEED = 42

MAX_IOU_WITH_GT = 0.15
PASTE_PER_IMAGE = (1, 3)
REL_SIZE = (0.08, 0.35)     # pasted object's longer side as a fraction of image longer side
SCALE_JITTER = (0.75, 1.3)
BRIGHT_JITTER = (0.85, 1.15)


def read_labels(img_path: Path):
    lp = Path(str(img_path).replace("/images/", "/labels/", 1)).with_suffix(".txt")
    rows = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            q = line.split()
            if len(q) >= 5:
                rows.append((int(float(q[0])), *[float(v) for v in q[1:5]]))
    return rows


def iou_norm(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=400,
                    help="target number of pasted instances per class")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    rng = random.Random(SEED)
    manifest = json.loads((INST / "manifest.json").read_text())
    by_class = defaultdict(list)
    for e in manifest:
        by_class[e["class"]].append(e)
    classes = sorted(by_class)
    print(f"source instances: " + ", ".join(f"{c}:{len(by_class[c])}" for c in classes))

    train_paths = [Path(p) for p in TRAIN_LIST.read_text().split() if p.strip()]
    labels = {p: read_labels(p) for p in train_paths}
    has_class = defaultdict(set)
    for p, rows in labels.items():
        for r in rows:
            has_class[r[0]].add(p)

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    todo = {c: args.per_class for c in classes}
    made, pasted_count = 0, defaultdict(int)
    attempts = 0
    max_new = 40 if args.smoke else 10_000

    while any(v > 0 for v in todo.values()) and made < max_new and attempts < max_new * 20:
        attempts += 1
        cls = rng.choice([c for c in classes if todo[c] > 0])
        # never paste onto an image that already has this class
        candidates = [p for p in train_paths if p not in has_class[cls]]
        tgt = rng.choice(candidates)
        with Image.open(tgt) as im:
            im = im.convert("RGB")
            W, H = im.size
            base = im.copy()
        gt = list(labels[tgt])
        n_paste = rng.randint(*PASTE_PER_IMAGE)
        new_rows, placed = [], 0

        for _ in range(n_paste):
            e = rng.choice(by_class[cls])
            with Image.open(INST / e["file"]) as inst:
                inst = inst.convert("RGBA")
                if rng.random() < 0.5:
                    inst = inst.transpose(Image.FLIP_LEFT_RIGHT)
                rel = rng.uniform(*REL_SIZE) * rng.uniform(*SCALE_JITTER)
                target_long = max(W, H) * rel
                s = target_long / max(inst.size)
                nw, nh = max(8, int(inst.width * s)), max(8, int(inst.height * s))
                if nw >= W or nh >= H:
                    continue
                inst = inst.resize((nw, nh), Image.BILINEAR)
                b = ImageEnhance.Brightness(inst.convert("RGB")).enhance(
                    rng.uniform(*BRIGHT_JITTER))
                inst = Image.merge("RGBA", (*b.split(), inst.split()[3]))

            ok = False
            for _ in range(30):
                px, py = rng.randint(0, W - nw), rng.randint(0, H - nh)
                cand = ((px + nw / 2) / W, (py + nh / 2) / H, nw / W, nh / H)
                if all(iou_norm(cand, (g[1], g[2], g[3], g[4])) <= MAX_IOU_WITH_GT
                       for g in gt + new_rows_as_boxes(new_rows)):
                    ok = True
                    break
            if not ok:
                continue
            base.paste(inst, (px, py), inst)
            new_rows.append((cls, *cand))
            placed += 1

        if not placed:
            continue
        stem = f"cp_{cls}_{made:06d}"
        base.save(OUT / "images" / f"{stem}.jpg", quality=92)
        rows = gt + new_rows
        (OUT / "labels" / f"{stem}.txt").write_text(
            "\n".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in rows) + "\n")
        made += 1
        todo[cls] -= placed
        pasted_count[cls] += placed
        if made % 250 == 0:
            print(f"  {made} composites, pasted " +
                  ", ".join(f"{c}:{pasted_count[c]}" for c in classes), flush=True)

    new_imgs = sorted((OUT / "images").glob("*.jpg"))
    lines = [str(p) for p in train_paths] + [str(p) for p in new_imgs]
    (OUT / "train_copypaste.txt").write_text("\n".join(lines) + "\n")
    print(f"\ncreated {made} composite images ({len(lines)} total training images)")
    for c in classes:
        print(f"  class {c}: +{pasted_count[c]} pasted instances")
    print(f"wrote {OUT / 'train_copypaste.txt'}")


def new_rows_as_boxes(rows):
    return [(0, r[1], r[2], r[3], r[4]) for r in rows]


if __name__ == "__main__":
    main()

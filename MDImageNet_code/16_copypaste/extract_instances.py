#!/usr/bin/env python3
"""Cut out rare-class object instances from the official train split using SAM.

Each GT box becomes a box-prompt for SAM (facebook/sam-vit-huge, Apache-2.0,
ungated); the returned mask is used to save a tight RGBA cutout. Masks matter:
pasting rectangular crops would teach the detector to find rectangular patches
of background rather than the object.

  python3 extract_instances.py                    # all target classes
  python3 extract_instances.py --limit 5          # smoke test

Writes instances/<class_id>/<stem>_<n>.png (RGBA) + instances/manifest.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
TRAIN_LIST = HERE.parent / "13_dfine" / "lists" / "official_train.txt"
OUT = HERE / "instances"

# weak AND data-scarce -> the cases where added instances can actually help.
# 19 (other) and 26 (anthropogenic_fragment) are excluded deliberately: they are
# the largest headroom but have 1,611 / 5,733 boxes, so scarcity is not their problem.
TARGET_CLASSES = (24, 25, 29, 30, 31, 32)
MODEL_ID = "facebook/sam-vit-huge"
MIN_SIDE = 24          # skip specks; they carry no learnable appearance
MIN_MASK_FRAC = 0.05   # mask must cover >=5% of the prompt box to be trusted


def load_boxes(img_path: Path):
    lp = Path(str(img_path).replace("/images/", "/labels/", 1)).with_suffix(".txt")
    if not lp.exists():
        return []
    with Image.open(img_path) as im:
        w, h = im.size
    out = []
    for line in lp.read_text().splitlines():
        q = line.split()
        if len(q) < 5:
            continue
        c = int(float(q[0]))
        if c not in TARGET_CLASSES:
            continue
        cx, cy, bw, bh = (float(v) for v in q[1:5])
        x1, y1 = max(0, (cx - bw / 2) * w), max(0, (cy - bh / 2) * h)
        x2, y2 = min(w, (cx + bw / 2) * w), min(h, (cy + bh / 2) * h)
        if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
            continue
        out.append((c, [x1, y1, x2, y2]))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="max images per class (smoke test)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import SamModel, SamProcessor
    device = torch.device(args.device)
    model = SamModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = SamProcessor.from_pretrained(MODEL_ID)

    paths = [Path(p) for p in TRAIN_LIST.read_text().split() if p.strip()]
    per_class = {c: 0 for c in TARGET_CLASSES}
    manifest = []
    kept = skipped = 0

    for img_path in paths:
        boxes = load_boxes(img_path)
        if not boxes:
            continue
        if args.limit and all(per_class[c] >= args.limit for c, _ in boxes):
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            arr = np.array(im)
            for n, (cls, box) in enumerate(boxes):
                if args.limit and per_class[cls] >= args.limit:
                    continue
                inputs = proc(im, input_boxes=[[box]], return_tensors="pt").to(device)
                out = model(**inputs, multimask_output=False)
                masks = proc.image_processor.post_process_masks(
                    out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu())
                m = masks[0][0][0].numpy().astype(bool)

                x1, y1, x2, y2 = (int(round(v)) for v in box)
                sub = m[y1:y2, x1:x2]
                if sub.size == 0 or sub.mean() < MIN_MASK_FRAC:
                    skipped += 1
                    continue
                rgb = arr[y1:y2, x1:x2]
                rgba = np.dstack([rgb, (sub * 255).astype(np.uint8)])
                d = OUT / str(cls)
                d.mkdir(parents=True, exist_ok=True)
                fp = d / f"{img_path.stem}_{n}.png"
                Image.fromarray(rgba, mode="RGBA").save(fp)
                manifest.append({"class": cls, "file": str(fp.relative_to(OUT)),
                                 "src": img_path.name,
                                 "w": int(x2 - x1), "h": int(y2 - y1),
                                 "mask_frac": round(float(sub.mean()), 4)})
                per_class[cls] += 1
                kept += 1
        if kept and kept % 100 == 0:
            print(f"  {kept} instances cut ({skipped} skipped)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nkept {kept} instances, skipped {skipped} (mask < {MIN_MASK_FRAC} of box)")
    for c in TARGET_CLASSES:
        print(f"  class {c}: {per_class[c]}")
    print(f"wrote {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()

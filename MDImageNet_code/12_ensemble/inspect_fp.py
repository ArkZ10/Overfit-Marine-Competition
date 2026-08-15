#!/usr/bin/env python3
"""Contact sheet of a class's highest-confidence FALSE POSITIVES.

  python3 inspect_fp.py --cls 26 --n 24

Pulls detections of --cls that match no GT box of any class at IoU >= 0.5, sorts
by confidence, and crops each with context. If these land on visible unlabelled
objects of that class, the annotations are sparse and the model is being trained
to suppress what it is then scored on.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "10_diagnostics"))
from common import CLASS_NAMES  # noqa: E402

VAL_LIST = HERE.parent / "11_improvements" / "rfs" / "val_control.txt"
PAD = 1.6      # context around the box
TILE = 300


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default="preds/wbf_abef_rescored.val.json")
    ap.add_argument("--gt", default="preds/gt_val_namr33.json")
    ap.add_argument("--cls", type=int, default=26)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = json.loads((HERE / args.gt).read_text())
    dets = json.loads((HERE / args.dump).read_text())
    paths = {Path(p).name: Path(p) for p in VAL_LIST.read_text().split() if p.strip()}
    fname = {im["id"]: im["file_name"] for im in gt["images"]}

    gt_by_img = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_img[a["image_id"]].append(a)

    def iou(b, arr):
        if len(arr) == 0:
            return np.zeros(0)
        x1 = np.maximum(b[0], arr[:, 0]); y1 = np.maximum(b[1], arr[:, 1])
        x2 = np.minimum(b[0] + b[2], arr[:, 0] + arr[:, 2])
        y2 = np.minimum(b[1] + b[3], arr[:, 1] + arr[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        return inter / (b[2] * b[3] + arr[:, 2] * arr[:, 3] - inter + 1e-9)

    fps = []
    for d in dets:
        if d["category_id"] != args.cls or d["score"] < args.conf:
            continue
        gs = gt_by_img.get(d["image_id"], [])
        arr = np.array([a["bbox"] for a in gs], dtype=float).reshape(-1, 4)
        if len(arr) and iou(d["bbox"], arr).max() >= 0.5:
            continue
        fps.append(d)
    fps.sort(key=lambda d: -d["score"])
    fps = fps[:args.n]
    print(f"class {args.cls} ({CLASS_NAMES[args.cls]}): {len(fps)} FPs shown, "
          f"conf {fps[0]['score']:.3f} .. {fps[-1]['score']:.3f}")

    cols = args.cols
    rows = (len(fps) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TILE, rows * TILE), "white")
    dr = ImageDraw.Draw(sheet)
    for k, d in enumerate(fps):
        p = paths.get(fname[d["image_id"]])
        if p is None or not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        x, y, w, h = d["bbox"]
        cx, cy, s = x + w / 2, y + h / 2, max(w, h) * PAD / 2
        box = (max(0, cx - s), max(0, cy - s),
               min(im.width, cx + s), min(im.height, cy + s))
        crop = im.crop([int(v) for v in box]).resize((TILE, TILE), Image.LANCZOS)
        # the detection, in crop coords
        sx, sy = TILE / (box[2] - box[0]), TILE / (box[3] - box[1])
        cd = ImageDraw.Draw(crop)
        cd.rectangle([(x - box[0]) * sx, (y - box[1]) * sy,
                      (x + w - box[0]) * sx, (y + h - box[1]) * sy],
                     outline="#e03a2f", width=3)
        # any GT boxes that fall in view, for comparison
        for a in gt_by_img.get(d["image_id"], []):
            gx, gy, gw, gh = a["bbox"]
            if gx + gw < box[0] or gx > box[2] or gy + gh < box[1] or gy > box[3]:
                continue
            cd.rectangle([(gx - box[0]) * sx, (gy - box[1]) * sy,
                          (gx + gw - box[0]) * sx, (gy + gh - box[1]) * sy],
                         outline="#19c37d", width=3)
            cd.text(((gx - box[0]) * sx + 4, (gy - box[1]) * sy + 4),
                    CLASS_NAMES[a["category_id"]][:18], fill="#19c37d")
        cd.text((6, TILE - 16), f"conf {d['score']:.3f}", fill="#e03a2f")
        sheet.paste(crop, ((k % cols) * TILE, (k // cols) * TILE))
        dr.rectangle([(k % cols) * TILE, (k // cols) * TILE,
                      (k % cols + 1) * TILE - 1, (k // cols + 1) * TILE - 1],
                     outline="#cccccc")

    out = args.out or str(HERE / "scores" / f"fp_class{args.cls}.png")
    sheet.save(out)
    print(f"wrote {out}   (red = the false positive, green = existing GT in view)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render clean E/F consensus detections that are unmatched to official GT."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT = ROOT / "MDImageNet_code/12_ensemble/scores/clean_ef_audit.json"
GT = HERE / "data/val.json"
IMAGES = ROOT / "MDImageDataset2/train_dataset/images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-id", type=int, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    audit = json.loads(AUDIT.read_text())
    gt = json.loads(GT.read_text())
    images = {x["id"]: x for x in gt["images"]}
    anns = defaultdict(list)
    for a in gt["annotations"]:
        anns[a["image_id"]].append(a)
    rows, used_images = [], set()
    for x in audit["consensus_unmatched"]:
        if x["category_id"] != args.class_id or x["image_id"] in used_images:
            continue
        rows.append(x); used_images.add(x["image_id"])
        if len(rows) == args.n:
            break
    if not rows:
        raise SystemExit("no candidates stored for this class")
    tile = 320
    sheet = Image.new("RGB", (args.cols * tile, ((len(rows)+args.cols-1)//args.cols)*tile), "white")
    for k, x in enumerate(rows):
        info = images[x["image_id"]]
        im = Image.open(IMAGES / info["file_name"]).convert("RGB")
        bx, by, bw, bh = x["bbox"]
        cx, cy = bx + bw/2, by + bh/2
        radius = max(bw, bh) * 1.25
        cropbox = (max(0, cx-radius), max(0, cy-radius),
                   min(im.width, cx+radius), min(im.height, cy+radius))
        crop = im.crop(tuple(map(int, cropbox))).resize((tile, tile), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(crop)
        sx, sy = tile/(cropbox[2]-cropbox[0]), tile/(cropbox[3]-cropbox[1])
        def rect(box, color, width):
            x0,y0,w,h=box
            draw.rectangle(((x0-cropbox[0])*sx,(y0-cropbox[1])*sy,
                            (x0+w-cropbox[0])*sx,(y0+h-cropbox[1])*sy), outline=color,width=width)
        rect(x["bbox"], "#ff3030", 4)
        for a in anns[x["image_id"]]:
            rect(a["bbox"], "#20d080", 3)
        draw.rectangle((0, 0, tile, 22), fill="black")
        draw.text((4, 4), f"E {x['e_score']:.3f} F {x['f_score']:.3f} IoU {x['iou']:.2f}", fill="white")
        sheet.paste(crop, ((k % args.cols)*tile, (k // args.cols)*tile))
    out = args.out or (HERE / f"consensus_class{args.class_id}_review.png")
    sheet.save(out)
    print(f"wrote {out} ({len(rows)} unique images; red=consensus, green=official GT)")


if __name__ == "__main__":
    main()

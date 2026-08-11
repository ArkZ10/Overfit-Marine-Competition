#!/usr/bin/env python3
"""Fused detections -> competition submission.csv (ICC19 taxonomy).

  python3 make_submission.py --dump preds/wbf_test.json --manifest preds/<name>.<split>.meta.json \
      --images-dir <test images> --bbox-format <REQUIRED, see below> --out submission.csv

--bbox-format is REQUIRED and has NO default because example_submission.csv is
header-only and does not disambiguate the convention. Verify against organizer
docs before submitting:
  coco-abs         x,y = top-left corner, w,h in ABSOLUTE pixels
  yolo-norm-cxcywh x,y = box CENTER, w,h all NORMALIZED to [0,1]
  xyxy-abs         x,y = top-left, w,h = bottom-right (x2,y2) in absolute pixels
Also confirm whether label_id is 0- or 1-indexed ICC19 (--label-offset, default 0).
"""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from paths12 import CROSSWALK_CSV  # noqa: I001
from eval_yolo_mapped_icc19 import read_crosswalk  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="detections dump (NAMR33 ids, abs xywh)")
    ap.add_argument("--images-dir", required=True, type=Path,
                    help="the image folder the dump was produced from (same sorted-stem manifest)")
    ap.add_argument("--bbox-format", required=True,
                    choices=["coco-abs", "yolo-norm-cxcywh", "xyxy-abs"],
                    help="REQUIRED - confirm the organizer's convention first")
    ap.add_argument("--label-offset", type=int, default=0,
                    help="0 if label_id is 0-indexed ICC19, 1 if 1-indexed")
    ap.add_argument("--conf-min", type=float, default=0.001)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    crosswalk = read_crosswalk(CROSSWALK_CSV, "namr33")

    img_files = sorted(args.images_dir.glob("*.jpg"), key=lambda p: p.stem)
    if not img_files:
        raise SystemExit(f"no images in {args.images_dir}")
    id_to_file = {i: p for i, p in enumerate(img_files)}
    dims = {}
    for i, p in id_to_file.items():
        with Image.open(p) as im:
            dims[i] = im.size  # (w, h)

    dets = json.loads(open(args.dump).read())
    rows = []
    for d in dets:
        if d["score"] < args.conf_min:
            continue
        image_id = d["image_id"]
        if image_id not in id_to_file:
            raise SystemExit(f"image_id {image_id} not in the images-dir manifest - wrong folder?")
        w, h = dims[image_id]
        x, y, bw, bh = d["bbox"]
        if args.bbox_format == "coco-abs":
            ox, oy, ow, oh = x, y, bw, bh
        elif args.bbox_format == "xyxy-abs":
            ox, oy, ow, oh = x, y, x + bw, y + bh
        else:  # yolo-norm-cxcywh
            ox, oy = (x + bw / 2) / w, (y + bh / 2) / h
            ow, oh = bw / w, bh / h
        label_id = crosswalk[d["category_id"]] + args.label_offset
        rows.append((id_to_file[image_id].name, label_id,
                     round(ox, 6), round(oy, 6), round(ow, 6), round(oh, 6),
                     round(d["score"], 6)))

    rows.sort(key=lambda r: (r[0], -r[6]))
    with open(args.out, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["image_filename", "label_id", "x", "y", "w", "h", "confidence"])
        wcsv.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows over {len(img_files)} images "
          f"(format={args.bbox_format}, label_offset={args.label_offset})")


if __name__ == "__main__":
    main()

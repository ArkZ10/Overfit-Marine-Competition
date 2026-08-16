#!/usr/bin/env python3
"""Draw predicted boxes with class names onto images.

  # 12 random test images from the submission, at conf >= 0.3
  python3 visualize.py --n 12

  # one specific image, everything above 0.15
  python3 visualize.py --image 00188e7dd3274f83.jpg --conf 0.15

  # a contact sheet instead of one file per image
  python3 visualize.py --n 24 --grid

  # val predictions with ground truth overlaid (dashed = GT) to eyeball errors
  python3 visualize.py --split val --n 8 --show-gt

  # only images containing a given class, e.g. 26 anthropogenic_fragment
  python3 visualize.py --n 12 --only-class 26

Reads submission.csv by default; --dump reads a COCO json dump instead.
Writes to viz/ (gitignored).
"""
import argparse
import colorsys
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "10_diagnostics"))
from common import CLASS_NAMES  # noqa: E402

TEST_DIR = Path("/root/Overfit-Marine-Competition/MDImageDataset/test/images")
VAL_DIR = Path("/root/Overfit-Marine-Competition/MDImageDataset/yolo_split/images/val")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
NC = 34


def class_colors():
    """34 visually distinct colours: spread hue, alternate value so neighbours differ."""
    out = {}
    for c in range(NC):
        h = (c * 0.6180339887) % 1.0          # golden-ratio hue spacing
        s = 0.95 if c % 2 else 0.75
        v = 1.00 if c % 3 else 0.80
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out[c] = (int(r * 255), int(g * 255), int(b * 255))
    return out


COLORS = class_colors()


def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def rows_from_submission(path):
    by_img = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by_img[r["image_filename"]].append(
                (int(r["label_id"]),
                 [float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])],
                 float(r["confidence"])))
    return by_img


def rows_from_dump(dump, manifest):
    names = {i["id"]: i["file_name"] for i in json.loads(Path(manifest).read_text())["images"]}
    by_img = defaultdict(list)
    for d in json.loads(Path(dump).read_text()):
        fn = names.get(d["image_id"])
        if fn:
            by_img[fn].append((d["category_id"], d["bbox"], d["score"]))
    return by_img


def gt_rows(manifest):
    doc = json.loads(Path(manifest).read_text())
    names = {i["id"]: i["file_name"] for i in doc["images"]}
    by_img = defaultdict(list)
    for a in doc.get("annotations", []):
        by_img[names[a["image_id"]]].append((a["category_id"], a["bbox"]))
    return by_img


def draw_one(img_path, dets, gts=None, conf=0.3, max_side=1400):
    im = Image.open(img_path).convert("RGB")
    scale = min(1.0, max_side / max(im.size))
    if scale < 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    dr = ImageDraw.Draw(im, "RGBA")
    fsz = max(13, int(im.width / 62))
    font, mono = load_font(FONT, fsz), load_font(FONT_MONO, int(fsz * 0.85))
    lw = max(2, int(im.width / 420))

    # ground truth first, underneath: dashed-looking thin white boxes
    for c, (x, y, w, h) in (gts or []):
        x, y, w, h = x * scale, y * scale, w * scale, h * scale
        dr.rectangle([x, y, x + w, y + h], outline=(255, 255, 255, 230), width=lw)
        dr.rectangle([x + lw, y + lw, x + w - lw, y + h - lw], outline=(0, 0, 0, 160), width=1)

    kept = sorted([d for d in dets if d[2] >= conf], key=lambda d: d[2])
    for c, (x, y, w, h), s in kept:
        x, y, w, h = x * scale, y * scale, w * scale, h * scale
        col = COLORS[c]
        dr.rectangle([x, y, x + w, y + h], outline=col + (255,), width=lw)
        label = f"{CLASS_NAMES[c]} {s:.2f}"
        tw = dr.textlength(label, font=font)
        th = fsz + 6
        ty = y - th if y - th >= 0 else y            # flip inside if it would clip off-image
        dr.rectangle([x, ty, x + tw + 10, ty + th], fill=col + (235,))
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        dr.text((x + 5, ty + 3), label, font=font,
                fill=(0, 0, 0) if lum > 140 else (255, 255, 255))

    cap = f"{Path(img_path).name}   {len(kept)} boxes >= {conf}"
    if gts is not None:
        cap += f"   |  white = ground truth ({len(gts)})"
    cw = dr.textlength(cap, font=mono)
    dr.rectangle([0, im.height - fsz - 10, cw + 14, im.height], fill=(0, 0, 0, 190))
    dr.text((7, im.height - fsz - 7), cap, font=mono, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--dump", default=None, help="use a COCO json dump instead of the csv")
    ap.add_argument("--manifest", default=None, help="images json for --dump (auto by --split)")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--image", default=None, help="one filename")
    ap.add_argument("--n", type=int, default=12, help="how many random images")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--only-class", type=int, default=None,
                    help="only images with a prediction of this class above --conf")
    ap.add_argument("--show-gt", action="store_true", help="overlay ground truth (val only)")
    ap.add_argument("--grid", action="store_true", help="one contact sheet instead of N files")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="viz")
    args = ap.parse_args()

    img_dir = TEST_DIR if args.split == "test" else VAL_DIR
    manifest = args.manifest or str(
        HERE / "preds" / ("gt_test_manifest.json" if args.split == "test"
                          else "gt_val_namr33.json"))

    if args.dump:
        by_img = rows_from_dump(args.dump, manifest)
        src = args.dump
    else:
        by_img = rows_from_submission(args.submission)
        src = args.submission
    gts = gt_rows(manifest) if args.show_gt else None

    if args.image:
        picks = [args.image]
    else:
        pool = sorted(by_img)
        if args.only_class is not None:
            pool = [f for f in pool
                    if any(c == args.only_class and s >= args.conf for c, _, s in by_img[f])]
            if not pool:
                raise SystemExit(f"no image has class {args.only_class} above conf {args.conf}")
        random.Random(args.seed).shuffle(pool)
        picks = pool[:args.n]

    out = HERE / args.out
    out.mkdir(exist_ok=True)
    print(f"source: {src}   images: {len(picks)}   conf >= {args.conf}")

    ims = []
    for fn in picks:
        p = img_dir / fn
        if not p.exists():
            print(f"  missing {p}")
            continue
        im = draw_one(p, by_img.get(fn, []), gts.get(fn) if gts else None, args.conf,
                      max_side=700 if args.grid else 1400)
        ims.append((fn, im))
        if not args.grid:
            dst = out / f"{Path(fn).stem}_pred.jpg"
            im.save(dst, quality=92)
            print(f"  {dst.name}  ({sum(1 for _, _, s in by_img.get(fn, []) if s >= args.conf)} boxes)")

    if args.grid and ims:
        cols = min(args.cols, len(ims))
        rows = (len(ims) + cols - 1) // cols
        # uniform square cells, each image scaled to fit and centred -- portrait and
        # landscape shots otherwise leave ragged white gaps
        cell = max(max(i.size) for _, i in ims)
        sheet = Image.new("RGB", (cols * cell, rows * cell), (24, 26, 28))
        for k, (_, im) in enumerate(ims):
            s = min(cell / im.width, cell / im.height)
            im2 = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                            Image.LANCZOS)
            ox = (k % cols) * cell + (cell - im2.width) // 2
            oy = (k // cols) * cell + (cell - im2.height) // 2
            sheet.paste(im2, (ox, oy))
        dst = out / f"grid_{args.split}_conf{args.conf}.jpg"
        sheet.save(dst, quality=90)
        print(f"  wrote {dst}  ({sheet.width}x{sheet.height})")

    print(f"\nclass colours are stable across runs (golden-ratio hue by class id)")


if __name__ == "__main__":
    main()

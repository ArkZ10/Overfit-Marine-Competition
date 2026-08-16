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

  # several classes at once, one grid, stratified so each class actually appears
  python3 visualize.py --n 12 --only-class 22 29 --grid

  # TRAINING DATA ground truth (no predictions involved) for a set of classes
  python3 visualize.py --source gt --split train --only-class 26 22 29 --grid \
      --out viz_train_gt

Reads submission.csv by default; --dump reads a COCO json dump instead;
--source gt draws ground-truth boxes only (works for train/val, since only they
have labels). Writes to viz/ (gitignored) unless --out overrides it.
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
TRAIN_DIR = Path("/root/Overfit-Marine-Competition/MDImageDataset/yolo_split/images/train")
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


def draw_one(img_path, dets, gts=None, conf=0.3, max_side=1400, show_conf=True):
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
        label = f"{CLASS_NAMES[c]} {s:.2f}" if show_conf else CLASS_NAMES[c]
        tw = dr.textlength(label, font=font)
        th = fsz + 6
        ty = y - th if y - th >= 0 else y            # flip inside if it would clip off-image
        dr.rectangle([x, ty, x + tw + 10, ty + th], fill=col + (235,))
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        dr.text((x + 5, ty + 3), label, font=font,
                fill=(0, 0, 0) if lum > 140 else (255, 255, 255))

    if show_conf:
        cap = f"{Path(img_path).name}   {len(kept)} boxes >= {conf}"
    else:
        cap = f"{Path(img_path).name}   {len(kept)} ground-truth boxes"
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
    ap.add_argument("--source", choices=["submission", "dump", "gt"], default=None,
                    help="box source; default is 'dump' if --dump given else 'submission'. "
                         "'gt' draws ground-truth boxes only -- needs --split train or val, "
                         "since test has no labels")
    ap.add_argument("--manifest", default=None, help="images json (auto by --split)")
    ap.add_argument("--split", choices=["test", "val", "train"], default="test")
    ap.add_argument("--image", default=None, help="one filename")
    ap.add_argument("--n", type=int, default=12, help="how many random images")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--only-class", type=int, nargs="+", default=None,
                    help="only images containing one of these class ids (space-separated). "
                         "With --grid, sampling is STRATIFIED across them so every requested "
                         "class actually shows up instead of the grid being dominated by "
                         "whichever class has the most images.")
    ap.add_argument("--show-gt", action="store_true", help="overlay ground truth (val only)")
    ap.add_argument("--grid", action="store_true", help="one contact sheet instead of N files")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="viz")
    args = ap.parse_args()

    dirs = {"test": TEST_DIR, "val": VAL_DIR, "train": TRAIN_DIR}
    img_dir = dirs[args.split]
    default_manifest = {
        "test": "gt_test_manifest.json",
        "val": "gt_val_namr33.json",
        "train": str((HERE.parent / "14_deim" / "data" / "train_official.json")),
    }[args.split]
    manifest = args.manifest or (default_manifest if "/" in default_manifest
                                 else str(HERE / "preds" / default_manifest))

    source = args.source or ("dump" if args.dump else "submission")
    if source == "gt" and args.split == "test":
        raise SystemExit("--source gt needs --split train or val; test has no labels")

    show_conf = True
    if source == "dump":
        by_img = rows_from_dump(args.dump, manifest)
        src = args.dump
    elif source == "gt":
        raw = gt_rows(manifest)                      # {file: [(cls, bbox), ...]}
        by_img = {f: [(c, b, 1.0) for c, b in v] for f, v in raw.items()}
        src = f"{manifest}  (ground truth, {args.split})"
        show_conf = False
    else:
        by_img = rows_from_submission(args.submission)
        src = args.submission
    gts = gt_rows(manifest) if (args.show_gt and source != "gt") else None

    if args.image:
        picks = [args.image]
    elif args.only_class and len(args.only_class) > 1:
        # stratified: sample per requested class so a grid always shows all of them,
        # rather than random sampling letting the most common class crowd out the rest
        rng = random.Random(args.seed)
        per_class = -(-args.n // len(args.only_class))   # ceil
        picks, seen = [], set()
        for cls in args.only_class:
            pool = [f for f in sorted(by_img)
                    if f not in seen
                    and any(c == cls and s >= args.conf for c, _, s in by_img[f])]
            if not pool:
                print(f"  (no image has class {cls} '{CLASS_NAMES[cls]}' above conf {args.conf})")
                continue
            rng.shuffle(pool)
            for f in pool[:per_class]:
                picks.append(f)
                seen.add(f)
        rng.shuffle(picks)
        picks = picks[:args.n] if len(picks) > args.n else picks
    else:
        pool = sorted(by_img)
        if args.only_class is not None:
            classes = set(args.only_class)
            pool = [f for f in pool
                    if any(c in classes and s >= args.conf for c, _, s in by_img[f])]
            if not pool:
                raise SystemExit(f"no image has class(es) {args.only_class} above conf {args.conf}")
        random.Random(args.seed).shuffle(pool)
        picks = pool[:args.n]

    out = HERE / args.out
    out.mkdir(exist_ok=True)
    tag = ("_".join(str(c) for c in args.only_class)) if args.only_class else None
    print(f"source: {src}   images: {len(picks)}   conf >= {args.conf}"
          + (f"   classes: {tag}" if tag else ""))

    ims = []
    for fn in picks:
        p = img_dir / fn
        if not p.exists():
            print(f"  missing {p}")
            continue
        im = draw_one(p, by_img.get(fn, []), gts.get(fn) if gts else None, args.conf,
                      max_side=700 if args.grid else 1400, show_conf=show_conf)
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
        namebit = f"_classes_{tag}" if tag else ""
        dst = out / f"grid_{args.split}_{source}_conf{args.conf}{namebit}.jpg"
        sheet.save(dst, quality=90)
        print(f"  wrote {dst}  ({sheet.width}x{sheet.height})")

    print(f"\nclass colours are stable across runs (golden-ratio hue by class id)")


if __name__ == "__main__":
    main()

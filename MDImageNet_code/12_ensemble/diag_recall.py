#!/usr/bin/env python3
"""Two diagnostics that decide where the remaining headroom is.

  python3 diag_recall.py --dump preds/wbf_abef_rescored.val.json

(A) SIZE: GT box area in ORIGINAL image coords -> COCO small/medium/large, plus
    the effective size each box has after the image is letterboxed to 640, which
    is what the network actually sees.

(B) PROPOSAL vs RANKING: for every GT box, find the best-IoU detection at
    conf >= 0.001 (the floor mAP integrates to, NOT the 0.25 the matrix used) and
    split the failures into
      * never proposed        -> no detection of any class reaches IoU 0.5
      * proposed, wrong class -> localized, misnamed
      * proposed, low conf    -> correct class present but ranked below `--conf`
    Only the first is a resolution/architecture problem.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parent / "10_diagnostics"))
from common import CLASS_NAMES  # noqa: E402

IMGSZ = 640


def ious_to(g, dets):
    """IoU of one GT xywh against an (N,4) xywh array."""
    if len(dets) == 0:
        return np.zeros(0)
    gx1, gy1, gx2, gy2 = g[0], g[1], g[0] + g[2], g[1] + g[3]
    dx1, dy1 = dets[:, 0], dets[:, 1]
    dx2, dy2 = dets[:, 0] + dets[:, 2], dets[:, 1] + dets[:, 3]
    iw = np.clip(np.minimum(gx2, dx2) - np.maximum(gx1, dx1), 0, None)
    ih = np.clip(np.minimum(gy2, dy2) - np.maximum(gy1, dy1), 0, None)
    inter = iw * ih
    return inter / (g[2] * g[3] + dets[:, 2] * dets[:, 3] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="preds/wbf_abef_rescored.val.json")
    ap.add_argument("--gt", default="preds/gt_val_namr33.json")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="the threshold the confusion matrix was drawn at")
    ap.add_argument("--floor", type=float, default=0.001, help="the floor mAP integrates to")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="scores/recall_diag.json")
    args = ap.parse_args()

    gt = json.loads((HERE / args.gt).read_text())
    dets = json.loads((HERE / args.dump).read_text())
    dims = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}

    # ---------- (A) size ----------
    print("=" * 78)
    print("(A)  GT BOX SIZE  —  original coords vs what the 640 network sees")
    print("=" * 78)
    areas, eff_sides, by_cls = [], [], defaultdict(list)
    for a in gt["annotations"]:
        w, h = a["bbox"][2], a["bbox"][3]
        W, H = dims[a["image_id"]]
        s = IMGSZ / max(W, H)                       # letterbox scale factor
        areas.append(w * h)
        eff_sides.append(np.sqrt(w * h) * s)
        by_cls[a["category_id"]].append(w * h)
    areas = np.array(areas)
    eff = np.array(eff_sides)
    n = len(areas)
    buckets = [("small  (<32^2)", areas < 32 ** 2),
               ("medium (32^2-96^2)", (areas >= 32 ** 2) & (areas < 96 ** 2)),
               ("large  (>=96^2)", areas >= 96 ** 2)]
    print(f"  {n} GT boxes;  image longest side median "
          f"{np.median([max(dims[a['image_id']]) for a in gt['annotations']]):.0f} px")
    for lab, mask in buckets:
        print(f"  {lab:<20} {mask.sum():>5} boxes ({100 * mask.sum() / n:>5.1f}%)   "
              f"median side at 640 input: {np.median(eff[mask]) if mask.sum() else 0:>6.1f} px")
    print(f"\n  effective side at 640:  <8px {100 * (eff < 8).mean():.1f}%   "
          f"<16px {100 * (eff < 16).mean():.1f}%   <32px {100 * (eff < 32).mean():.1f}%")
    print(f"  median scale factor orig->640: {IMGSZ / np.median([max(dims[a['image_id']]) for a in gt['annotations']]):.3f}")
    print("\n  smallest classes by median box area (original coords):")
    med = sorted(((np.median(v), c, len(v)) for c, v in by_cls.items()))
    for m, c, k in med[:8]:
        print(f"    {c:>2} {CLASS_NAMES[c]:<32} median {np.sqrt(m):>6.1f} px side, {k} boxes")

    # ---------- (B) proposal vs ranking ----------
    print()
    print("=" * 78)
    print(f"(B)  THE 'MISSED' BOXES  —  re-examined down to conf {args.floor}")
    print("=" * 78)
    det_by_img = defaultdict(list)
    for d in dets:
        if d["score"] >= args.floor:
            det_by_img[d["image_id"]].append(d)
    gt_by_img = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_img[a["image_id"]].append(a)

    cat = defaultdict(int)
    cat_by_cls = defaultdict(lambda: defaultdict(int))
    conf_of_recovered = []
    for img, gs in gt_by_img.items():
        ds = det_by_img.get(img, [])
        arr = np.array([d["bbox"] for d in ds], dtype=float).reshape(-1, 4)
        cls = np.array([d["category_id"] for d in ds], dtype=int)
        scr = np.array([d["score"] for d in ds], dtype=float)
        for a in gs:
            c = a["category_id"]
            iou = ious_to(a["bbox"], arr)
            hit = iou >= args.iou
            if not hit.any():
                k = "never proposed"
            else:
                same = hit & (cls == c)
                if not same.any():
                    k = "proposed, wrong class only"
                else:
                    best = scr[same].max()
                    if best >= args.conf:
                        k = f"found at conf>={args.conf}"
                    else:
                        k = f"proposed, conf {args.floor}-{args.conf}"
                        conf_of_recovered.append(best)
            cat[k] += 1
            cat_by_cls[c][k] += 1

    tot = sum(cat.values())
    order = [f"found at conf>={args.conf}", f"proposed, conf {args.floor}-{args.conf}",
             "proposed, wrong class only", "never proposed"]
    for k in order:
        print(f"  {k:<34} {cat[k]:>5}  ({100 * cat[k] / tot:>5.1f}% of all GT)")
    missed = tot - cat[f"found at conf>={args.conf}"]
    print(f"\n  of the {missed} not found at conf {args.conf}:")
    for k in order[1:]:
        print(f"    {k:<32} {cat[k]:>5}  ({100 * cat[k] / missed:>5.1f}%)")
    if conf_of_recovered:
        q = np.percentile(conf_of_recovered, [25, 50, 75])
        print(f"\n  the under-confident ones sit at conf "
              f"p25 {q[0]:.3f} / median {q[1]:.3f} / p75 {q[2]:.3f}")

    print("\n  worst classes by 'never proposed' share:")
    rows = []
    for c, dd in cat_by_cls.items():
        t = sum(dd.values())
        rows.append((dd["never proposed"] / t, dd["never proposed"], t, c))
    for frac, np_, t, c in sorted(rows, reverse=True)[:10]:
        print(f"    {c:>2} {CLASS_NAMES[c]:<32} {np_:>4}/{t:<4} never proposed ({100 * frac:>5.1f}%)")

    out = {"size": {lab: int(m.sum()) for lab, m in buckets},
           "eff_side_median_by_bucket": {lab: float(np.median(eff[m])) if m.sum() else 0
                                         for lab, m in buckets},
           "proposal": dict(cat),
           "per_class": {str(c): dict(v) for c, v in cat_by_cls.items()}}
    (HERE / args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {HERE / args.out}")


if __name__ == "__main__":
    main()

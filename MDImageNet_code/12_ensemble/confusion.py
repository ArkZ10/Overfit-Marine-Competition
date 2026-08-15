#!/usr/bin/env python3
"""Confusion matrix for any detection dump, against the frozen NAMR33 val GT.

  python3 confusion.py --dump preds/wbf_abef_rescored.val.json --name best_pipeline
  python3 confusion.py --dump preds/deim_dfine_l.val.json --conf 0.25

Matching follows the standard detection-confusion convention: within an image,
detections above --conf are taken in descending score order and greedily matched
to the highest-IoU unclaimed GT box at IoU >= --iou, REGARDLESS of class -- that
is what makes the off-diagonal cells mean "found the object, named it wrong".
Unmatched GT -> row 34 (missed); unmatched detection -> column 34 (hallucinated).

Writes scores/<name>.confusion.json (matrix + per-class precision/recall + the
ranked confusion pairs) and prints the worst pairs.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "10_diagnostics"))
from common import CLASS_NAMES  # noqa: E402

NC = 34
BG = NC  # index of the background row/column


def iou_matrix(a, b):
    """a: (N,4) xywh, b: (M,4) xywh -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    return inter / (area_a + area_b - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--gt", default="preds/gt_val_namr33.json")
    ap.add_argument("--name", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    name = args.name or Path(args.dump).name.split(".")[0]
    gt = json.loads((HERE / args.gt).read_text() if not Path(args.gt).is_absolute()
                    else Path(args.gt).read_text())
    dets = json.loads((HERE / args.dump).read_text() if not Path(args.dump).is_absolute()
                      else Path(args.dump).read_text())

    gt_by_img = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_img[a["image_id"]].append((a["category_id"], a["bbox"]))
    det_by_img = defaultdict(list)
    for d in dets:
        if d["score"] >= args.conf:
            det_by_img[d["image_id"]].append((d["category_id"], d["bbox"], d["score"]))

    m = np.zeros((NC + 1, NC + 1), dtype=np.int64)  # [pred, gt]

    for img in set(gt_by_img) | set(det_by_img):
        g = gt_by_img.get(img, [])
        p = sorted(det_by_img.get(img, []), key=lambda x: -x[2])
        gb = np.array([x[1] for x in g], dtype=float).reshape(-1, 4)
        pb = np.array([x[1] for x in p], dtype=float).reshape(-1, 4)
        ious = iou_matrix(pb, gb)
        claimed = set()
        for i in range(len(p)):
            j, best = -1, args.iou
            for k in range(len(g)):
                if k in claimed:
                    continue
                if ious[i, k] >= best:
                    best, j = ious[i, k], k
            if j >= 0:
                claimed.add(j)
                m[p[i][0], g[j][0]] += 1
            else:
                m[p[i][0], BG] += 1
        for k in range(len(g)):
            if k not in claimed:
                m[BG, g[k][0]] += 1

    # per-class precision / recall from the matrix
    per_class = {}
    for c in range(NC):
        tp = int(m[c, c])
        pred_tot = int(m[c, :].sum())
        gt_tot = int(m[:, c].sum())
        per_class[c] = {
            "name": CLASS_NAMES[c],
            "gt": gt_tot,
            "tp": tp,
            "predicted": pred_tot,
            "precision": tp / pred_tot if pred_tot else 0.0,
            "recall": tp / gt_tot if gt_tot else 0.0,
            "missed": int(m[BG, c]),
            "false_pos": int(m[c, BG]),
        }

    pairs = []
    for pc in range(NC + 1):
        for gc in range(NC + 1):
            if pc == gc or m[pc, gc] == 0:
                continue
            gt_tot = int(m[:, gc].sum())
            pairs.append({
                "pred": pc, "gt": gc, "n": int(m[pc, gc]),
                "pred_name": "MISSED (background)" if pc == BG else CLASS_NAMES[pc],
                "gt_name": "FALSE POSITIVE (background)" if gc == BG else CLASS_NAMES[gc],
                "frac_of_gt_class": (m[pc, gc] / gt_tot) if gt_tot else 0.0,
            })
    pairs.sort(key=lambda d: -d["n"])

    out = {
        "name": name, "dump": args.dump, "conf": args.conf, "iou": args.iou,
        "class_names": {c: CLASS_NAMES[c] for c in range(NC)},
        "matrix": m.tolist(), "per_class": per_class, "pairs": pairs,
    }
    dst = HERE / "scores" / f"{name}.confusion.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))

    print(f"{name}  conf>={args.conf}  IoU>={args.iou}")
    print(f"  GT boxes {int(m[:, :NC].sum())}   detections {int(m[:NC, :].sum())}   "
          f"correct {int(np.trace(m[:NC, :NC]))}")
    print(f"  missed {int(m[BG, :NC].sum())}   false positives {int(m[:NC, BG].sum())}")
    print(f"\n  top {args.top} confusions (excluding the diagonal):")
    print(f"  {'n':>5}  {'predicted as':<32} {'actually':<32} {'%of GT class':>12}")
    for d in pairs[:args.top]:
        print(f"  {d['n']:>5}  {d['pred_name']:<32} {d['gt_name']:<32} "
              f"{100 * d['frac_of_gt_class']:>11.1f}%")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()

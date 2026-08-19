#!/usr/bin/env python3
"""Size-conditioned AP50-like ranking and proposal recall for a COCO dump.

Bins use GT/predicted box area divided by image area, matching the teammate
shared audit: <0.1%, 0.1--1%, and >=1%. A detection that matches an out-of-bin
GT is ignored for that bin; unmatched detections are assigned by their own
relative area. AP is computed per class and macro-averaged over classes that
have GT in the bin. Proposal recall uses every prediction in the dump.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BINS = (("tiny_lt_0.1pct", 0.0, 0.001),
        ("small_0.1_to_1pct", 0.001, 0.01),
        ("large_ge_1pct", 0.01, float("inf")))


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def ap101(labels, n_gt):
    if n_gt == 0:
        return float("nan")
    if not labels:
        return 0.0
    tp = np.cumsum(np.asarray(labels, dtype=float))
    fp = np.cumsum(1.0 - np.asarray(labels, dtype=float))
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    return float(np.mean([prec[rec >= r].max() if np.any(rec >= r) else 0.0
                          for r in np.linspace(0, 1, 101)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text())
    dets = json.loads(Path(args.dump).read_text())
    dims = {int(x["id"]): (x["width"], x["height"]) for x in gt["images"]}
    gby = defaultdict(list)
    for idx, a in enumerate(gt["annotations"]):
        iid = int(a["image_id"]); w, h = dims[iid]
        gby[(iid, int(a["category_id"]))].append({
            "idx": idx, "bbox": a["bbox"], "rel": a["bbox"][2] * a["bbox"][3] / (w * h)
        })
    dby = defaultdict(list)
    for d in dets:
        dby[(int(d["image_id"]), int(d["category_id"]))].append(d)
    for rows in dby.values():
        rows.sort(key=lambda x: -x["score"])

    result = {"dump": args.dump, "gt": args.gt, "iou": 0.5, "bins": {}}
    for name, lo, hi in BINS:
        n_gt_cls = defaultdict(int)
        for rows in gby.values():
            for g in rows:
                if lo <= g["rel"] < hi:
                    # category is supplied by the dictionary key below instead
                    pass
        ranked = defaultdict(list)
        matched_in_bin = set()
        total_gt = 0
        for (iid, cls), grows in gby.items():
            n_here = sum(lo <= g["rel"] < hi for g in grows)
            n_gt_cls[cls] += n_here
            total_gt += n_here
            used = set()
            for d in dby.get((iid, cls), []):
                candidates = [(iou(d["bbox"], g["bbox"]), j, g)
                              for j, g in enumerate(grows) if j not in used]
                best = max(candidates, default=(0.0, -1, None), key=lambda x: x[0])
                if best[0] >= 0.5:
                    used.add(best[1])
                    if lo <= best[2]["rel"] < hi:
                        ranked[cls].append((d["score"], 1))
                        matched_in_bin.add(best[2]["idx"])
                    # A correct match to another size bin is ignored.
                else:
                    w, h = dims[iid]
                    rel = d["bbox"][2] * d["bbox"][3] / (w * h)
                    if lo <= rel < hi:
                        ranked[cls].append((d["score"], 0))

        per_class = {}
        for cls, n in sorted(n_gt_cls.items()):
            if n == 0:
                continue
            labels = [y for _, y in sorted(ranked[cls], reverse=True)]
            per_class[str(cls)] = {"n_gt": n, "ap50": ap101(labels, n)}
        macro = float(np.mean([x["ap50"] for x in per_class.values()])) if per_class else float("nan")
        result["bins"][name] = {
            "range": [lo, None if np.isinf(hi) else hi],
            "n_gt": total_gt,
            "matched": len(matched_in_bin),
            "proposal_recall50": len(matched_in_bin) / total_gt if total_gt else float("nan"),
            "macro_ap50": macro,
            "per_class": per_class,
        }
        print(f"{name:22s} GT={total_gt:4d} recall50={result['bins'][name]['proposal_recall50']:.4f} "
              f"macro_AP50={macro:.4f}")

    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

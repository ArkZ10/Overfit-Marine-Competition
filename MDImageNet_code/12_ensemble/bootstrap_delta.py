#!/usr/bin/env python3
"""Paired image-level bootstrap CI on the macro-AP50 difference between two dumps.

  python3 bootstrap_delta.py --a preds/wbf_abef.val.json --b preds/wbf_abef_rescored.val.json
  python3 bootstrap_delta.py --a X.json --b Y.json --half fit -n 400

Why this and not the val_fit/val_sel gap: the half-to-half spread measures how
much a FIXED model's score moves between two different image samples. Comparing
two variants on the SAME images is a paired comparison, and its noise is much
smaller — the shared images cancel. Resampling images with replacement and
recomputing BOTH variants on each resample gives the right null for
"is this delta real", which is the gate every later step needs.

Reports the delta, its 95% percentile interval, and P(delta > 0).
"""
import argparse
import contextlib
import copy
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from paths12 import GT_VAL_JSON, NC


def macro_ap50(gt_dict, dets, img_ids):
    """Macro AP50 over classes that have GT in this image subset."""
    keep = set(img_ids)
    sub = {"images": [im for im in gt_dict["images"] if im["id"] in keep],
           "annotations": [a for a in gt_dict["annotations"] if a["image_id"] in keep],
           "categories": gt_dict["categories"]}
    if not sub["annotations"]:
        return float("nan")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = sub
        gt.createIndex()
        d = [x for x in dets if x["image_id"] in keep]
        if not d:
            return 0.0
        dt = gt.loadRes(copy.deepcopy(d))
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.imgIds = sorted(keep)
        ev.evaluate()
        ev.accumulate()
    # precision: [T, R, K, A, M]; T index 0 = IoU 0.50, A=0 all, M=2 maxDets 100
    p = ev.eval["precision"][0, :, :, 0, 2]
    per_class = [p[:, k][p[:, k] > -1].mean() for k in range(p.shape[1])
                 if (p[:, k] > -1).any()]
    return float(np.mean(per_class)) if per_class else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline dump")
    ap.add_argument("--b", required=True, help="candidate dump")
    ap.add_argument("--half", choices=["full", "fit", "sel"], default="full")
    ap.add_argument("--gt", help="explicit COCO ground truth (overrides --half path lookup)")
    ap.add_argument("-n", type=int, default=300, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gtp = Path(args.gt) if args.gt else (GT_VAL_JSON if args.half == "full" else
        GT_VAL_JSON.parent / f"gt_val_{args.half}_namr33.json")
    gt_dict = json.loads(Path(gtp).read_text())
    da = json.loads(Path(args.a).read_text())
    db = json.loads(Path(args.b).read_text())
    img_ids = [im["id"] for im in gt_dict["images"]]

    base_a = macro_ap50(gt_dict, da, img_ids)
    base_b = macro_ap50(gt_dict, db, img_ids)
    print(f"  half={args.half}  {len(img_ids)} images")
    print(f"  A {Path(args.a).name:<34} {base_a:.4f}")
    print(f"  B {Path(args.b).name:<34} {base_b:.4f}")
    print(f"  point delta (B-A): {base_b - base_a:+.4f}\n  bootstrapping {args.n}x ...",
          flush=True)

    rng = np.random.default_rng(args.seed)
    ids = np.array(img_ids)
    deltas, a_s, b_s = [], [], []
    for i in range(args.n):
        samp = rng.choice(ids, size=len(ids), replace=True)
        # COCOeval dedups imgIds, so resample by duplicating image ids is not
        # possible directly; use the unique subset (0.632-style) instead
        uniq = np.unique(samp)
        va = macro_ap50(gt_dict, da, uniq)
        vb = macro_ap50(gt_dict, db, uniq)
        if np.isnan(va) or np.isnan(vb):
            continue
        a_s.append(va); b_s.append(vb); deltas.append(vb - va)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{args.n}", flush=True)

    d = np.array(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n  delta  mean {d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  P(delta > 0) = {100 * (d > 0).mean():.1f}%")
    print(f"  |delta| that a change must beat to clear the CI half-width: "
          f"{(hi - lo) / 2:.4f}")
    print(f"\n  for reference, the absolute-score spread of each variant across "
          f"resamples:\n    A sd {np.std(a_s):.4f}   B sd {np.std(b_s):.4f}   "
          f"delta sd {d.std():.4f}")
    print(f"  (the delta sd is the number that matters; the absolute sds are "
          f"~{np.std(a_s) / max(d.std(), 1e-9):.1f}x larger because paired "
          f"comparison cancels the shared image sample)")


if __name__ == "__main__":
    main()

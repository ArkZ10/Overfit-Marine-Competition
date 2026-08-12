#!/usr/bin/env python3
"""Grid-search WBF settings on the val split.

Fuses and scores every combo in-process. Selection metric is the competition
metric: 34-class NAMR33 AP@0.50 via the COCO API.

  python3 sweep_wbf.py --topk 3
"""
import argparse
import contextlib
import io
import itertools
import json
import time
from pathlib import Path

from paths12 import GT_VAL_JSON, PREDS_DIR, SCORES_DIR  # noqa: I001
from wbf_fuse import fuse, load_dims  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

DUMPS = [
    PREDS_DIR / "y11m_control.val.json",
    PREDS_DIR / "rtdetr_l.val.json",
    PREDS_DIR / "frcnn_r50v2.val.json",
    PREDS_DIR / "dfine_l.val.json",
]
AP50 = {"y11m_control": 0.6049, "rtdetr_l": 0.6684, "frcnn_r50v2": 0.5130, "dfine_l": 0.5755}

NORMALIZE = ["temperature", "minmax", "none"]
IOU_THRS = [0.50, 0.55, 0.60, 0.65]
SKIP_THRS = [0.0, 0.001]
WEIGHTS = {
    "equal": [1.0, 1.0, 1.0, 1.0],
    "ap50prop": [AP50["y11m_control"], AP50["rtdetr_l"], AP50["frcnn_r50v2"], AP50["dfine_l"]],
    "B_heavy": [1.0, 2.0, 1.0, 1.0],
    "detr_heavy": [1.0, 2.0, 1.0, 1.5],   # the two DETRs carry the localisation quality
}


def quick_ap50(gt: COCO, dets):
    if not dets:
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes([dict(d) for d in dets])
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    dims = load_dims(GT_VAL_JSON)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(GT_VAL_JSON))

    combos = list(itertools.product(NORMALIZE, IOU_THRS, WEIGHTS.items(), SKIP_THRS))
    print(f"sweeping {len(combos)} combos\n")
    results = []
    t0 = time.time()
    for i, (norm, iou, (wname, w), skip) in enumerate(combos, 1):
        fused = fuse([str(p) for p in DUMPS], dims, iou, skip, w, norm)
        ap50 = quick_ap50(gt, fused)
        results.append({"normalize": norm, "iou_thr": iou, "weights": wname,
                        "skip_box_thr": skip, "ap50": ap50, "n_boxes": len(fused)})
        print(f"[{i:>3}/{len(combos)}] norm={norm:<11} iou={iou:.2f} w={wname:<9} "
              f"skip={skip:<5} -> AP50={ap50:.4f}  ({len(fused)} boxes)", flush=True)

    results.sort(key=lambda r: -r["ap50"])
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    (SCORES_DIR / "wbf4_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"\nswept in {(time.time() - t0) / 60:.1f} min")

    print(f"\n=== top {args.topk} ===")
    for r in results[:args.topk]:
        print(f"  AP50={r['ap50']:.4f}  norm={r['normalize']:<11} iou={r['iou_thr']:.2f} "
              f"w={r['weights']:<9} skip={r['skip_box_thr']}")

    # materialise the winner as a real dump for full scoring downstream
    best = results[0]
    fused = fuse([str(p) for p in DUMPS], dims, best["iou_thr"], best["skip_box_thr"],
                 WEIGHTS[best["weights"]], best["normalize"])
    out = PREDS_DIR / "wbf4_best.val.json"
    out.write_text(json.dumps(fused))
    (PREDS_DIR / "wbf4_best.val.meta.json").write_text(json.dumps(best, indent=2))
    print(f"\nwrote {out} with the winning config")


if __name__ == "__main__":
    main()

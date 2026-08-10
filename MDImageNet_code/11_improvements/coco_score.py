#!/usr/bin/env python3
"""Score any checkpoint on the val split with pycocotools, overall + per class.

Reuses the COCO conversion pipeline from 10_diagnostics/task3_coco_eval.py
(build_gt_coco / run_predictions / yolo_to_coco_bbox) rather than reimplementing
it - real image dimensions, normalized cxcywh -> absolute xywh.

  python3 coco_score.py --weights <path/to/best.pt> --name y11m_control
"""
import argparse
import contextlib
import io
import json
import random

import numpy as np
import torch

from paths import NC, SCORES_DIR, SEED  # noqa: I001 - puts 10_diagnostics on sys.path
from task3_coco_eval import build_gt_coco, run_predictions  # noqa: E402

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IOU50_IDX = 0    # precision axis T: IoU 0.50:0.05:0.95
AREA_ALL_IDX = 0  # precision axis A: all / small / medium / large
MAXDET_100_IDX = 2  # precision axis M: maxDets 1 / 10 / 100


def per_class_ap(ev):
    """-> {class_id: {'ap50':float, 'ap50_95':float}} from COCOeval precision array.

    precision has shape [T, R, K, A, M]; -1 marks recall levels with no data.
    """
    precision = ev.eval["precision"]
    cat_ids = list(ev.params.catIds)
    out = {}
    for k, cat_id in enumerate(cat_ids):
        p50 = precision[IOU50_IDX, :, k, AREA_ALL_IDX, MAXDET_100_IDX]
        p50 = p50[p50 > -1]
        p_all = precision[:, :, k, AREA_ALL_IDX, MAXDET_100_IDX]
        p_all = p_all[p_all > -1]
        out[int(cat_id)] = {
            "ap50": float(p50.mean()) if p50.size else float("nan"),
            "ap50_95": float(p_all.mean()) if p_all.size else float("nan"),
        }
    return out


def score(weights, name):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    gt_path = SCORES_DIR / f"_gt_{name}.json"
    pred_path = SCORES_DIR / f"_pred_{name}.json"

    coco_gt, stem_to_id, img_files = build_gt_coco()
    gt_path.write_text(json.dumps(coco_gt))

    preds = run_predictions(stem_to_id, img_files, weights=weights)
    pred_path.write_text(json.dumps(preds))

    with contextlib.redirect_stdout(io.StringIO()):
        gt_obj = COCO(str(gt_path))
        dt_obj = gt_obj.loadRes(str(pred_path))

    ev = COCOeval(gt_obj, dt_obj, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.summarize()
    summary = buf.getvalue()

    result = {
        "name": name,
        "weights": str(weights),
        "n_predictions": len(preds),
        "ap50": float(ev.stats[1]),
        "ap50_95": float(ev.stats[0]),
        "coco_stats": [float(x) for x in ev.stats],
        "coco_summary": summary,
        "per_class": {str(c): per_class_ap(ev).get(c, {"ap50": float("nan"), "ap50_95": float("nan")})
                      for c in range(NC)},
    }

    out_path = SCORES_DIR / f"{name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    gt_path.unlink()
    pred_path.unlink()

    print(summary)
    print(f"{name}: pycocotools AP@0.50 = {result['ap50']:.4f}   AP@0.50:0.95 = {result['ap50_95']:.4f}")
    print(f"wrote {out_path}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True, help="label for the output json, e.g. y11m_control")
    a = ap.parse_args()
    score(a.weights, a.name)

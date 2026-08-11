#!/usr/bin/env python3
"""Score a prediction dump with pycocotools on BOTH taxonomies.

  python3 score_preds.py --dump preds/y11m_control.val.json --name y11m_control

Writes scores/<name>.json:
  - namr33: 34-class COCOeval vs preds/gt_val_namr33.json (+ per-class AP50/AP50-95)
  - icc19: 20-class after crosswalk remap of BOTH GT and detections
           (summarize_coco_eval from 06_evaluation, +1 category convention)
"""
import argparse
import contextlib
import copy
import io
import json

import numpy as np

from paths12 import (  # noqa: I001
    CROSSWALK_CSV, GT_VAL_JSON, NC, NC_ICC19, SCORES_DIR,
)
from coco_score import per_class_ap  # noqa: E402  (11_improvements)
from eval_yolo_mapped_icc19 import read_crosswalk, summarize_coco_eval  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402


def load_gt():
    if not GT_VAL_JSON.exists():
        raise SystemExit(f"{GT_VAL_JSON} missing - run dump_preds.py --split val once first")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(GT_VAL_JSON))
    return gt


def score_namr33(gt: COCO, dets: list[dict]):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(copy.deepcopy(dets))
    ev = COCOeval(gt, dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.summarize()
    pc = per_class_ap(ev)
    return {
        "ap50": float(ev.stats[1]),
        "ap50_95": float(ev.stats[0]),
        "coco_summary": buf.getvalue(),
        "per_class": {str(c): pc.get(c, {"ap50": float("nan"), "ap50_95": float("nan")})
                      for c in range(NC)},
    }


def score_icc19(gt: COCO, dets: list[dict]):
    """Remap GT + detections NAMR33 -> ICC19 (many-to-one), score with the
    06_evaluation path (category ids are icc19_class_id + 1 on both sides)."""
    crosswalk = read_crosswalk(CROSSWALK_CSV, "namr33")

    gt_icc = copy.deepcopy(gt.dataset)
    names = {}
    import csv as _csv
    with open(CROSSWALK_CSV) as f:
        for row in _csv.DictReader(f):
            if row["taxonomy"] == "icc19":
                names[int(row["icc19_class_id"])] = row["icc19_class_name"]
    gt_icc["categories"] = [{"id": c + 1, "name": names.get(c, str(c))} for c in range(NC_ICC19)]
    for ann in gt_icc["annotations"]:
        ann["category_id"] = crosswalk[ann["category_id"]] + 1

    with contextlib.redirect_stdout(io.StringIO()):
        gt_obj = COCO()
        gt_obj.dataset = gt_icc
        gt_obj.createIndex()

    dets_icc = [
        {**d, "category_id": crosswalk[d["category_id"]] + 1}
        for d in dets
    ]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary, per_class = summarize_coco_eval(gt_obj, dets_icc)
    return {
        "map50": summary["mAP50"],
        "map50_95": summary["mAP50-95"],
        "summary": summary,
        "per_class": {str(c): per_class.get(c, {"AP50": float("nan"), "AP50-95": float("nan")})
                      for c in range(NC_ICC19)},
        "coco_summary": buf.getvalue(),
    }


def score(dump_path, name):
    dets = json.loads(open(dump_path).read())
    gt = load_gt()

    n33 = score_namr33(gt, dets)
    icc = score_icc19(gt, dets)

    result = {
        "name": name,
        "dump": str(dump_path),
        "n_detections": len(dets),
        "namr33": n33,
        "icc19": icc,
    }
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out = SCORES_DIR / f"{name}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"{name}: NAMR33 AP@0.50 = {n33['ap50']:.4f}  (AP@0.50:0.95 = {n33['ap50_95']:.4f})")
    print(f"{name}: ICC19  mAP50   = {icc['map50']:.4f}  (mAP50-95    = {icc['map50_95']:.4f})")
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    score(a.dump, a.name)

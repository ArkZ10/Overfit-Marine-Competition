#!/usr/bin/env python3
"""Select a compact clean-EF WBF recipe on val_fit and materialize full val."""

import contextlib
import io
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENS = HERE.parent / "12_ensemble"
sys.path.insert(0, str(ENS))

from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from wbf_fuse import fuse, load_dims  # noqa: E402

DATA = HERE / "data"
PREDS = ENS / "preds"
SCORES = ENS / "scores"


def ap50(gt, dets):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[1])


def main():
    fit_gt_path = DATA / "val_fit.json"
    fit_dumps = [PREDS / "clean_e.clean_val_fit.json",
                 PREDS / "clean_f.clean_val_fit.json"]
    full_dumps = [PREDS / "clean_e.clean_val.json",
                  PREDS / "clean_f.clean_val.json"]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(fit_gt_path))
    dims = load_dims(fit_gt_path)

    rows = []
    for norm, iou, weights in itertools.product(
            ("temperature", "none"), (0.55, 0.60, 0.65),
            ([1.0, 1.0], [1.1, 1.0], [1.2, 1.0], [1.4, 1.0])):
        dets = fuse(fit_dumps, dims, iou, 0.001, weights, norm, "avg")
        score = ap50(gt, dets)
        row = {"normalize": norm, "iou_thr": iou, "weights": weights,
               "skip_box_thr": 0.001, "conf_type": "avg", "fit_ap50": score,
               "n_fit_boxes": len(dets)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    rows.sort(key=lambda r: -r["fit_ap50"])
    best = rows[0]
    full = fuse(full_dumps, load_dims(DATA / "val.json"), best["iou_thr"],
                best["skip_box_thr"], best["weights"], best["normalize"],
                best["conf_type"])
    out = PREDS / "clean_ef_frozen.clean_val.json"
    out.write_text(json.dumps(full))
    (PREDS / "clean_ef_frozen.clean_val.meta.json").write_text(
        json.dumps(best, indent=2))
    (SCORES / "clean_ef_fit_sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"BEST {json.dumps(best)}")
    print(f"wrote {out} ({len(full)} boxes)")


if __name__ == "__main__":
    main()

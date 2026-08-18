#!/usr/bin/env python3
"""Compare global score calibrators for clean E/F on val_fit only.

Fits temperature, Platt-on-logit, and isotonic mappings to greedy IoU correctness labels,
then runs the already-frozen EF WBF geometry/weights.  val_sel is intentionally untouched.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import random
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENS = ROOT / "MDImageNet_code" / "12_ensemble"
sys.path.insert(0, str(ENS))
from calibrate import apply_temperature, ece, fit_temperature, match_detections  # noqa: E402
from wbf_fuse import fuse, load_dims  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402


DATA = HERE / "data"
PREDS = ENS / "preds"
SCORES = ENS / "scores"
GT_PATH = DATA / "val_fit.json"
EPS = 1e-6


def logit(scores):
    s = np.clip(np.asarray(scores), EPS, 1-EPS)
    return np.log(s/(1-s))


def metrics(prob, y):
    p = np.clip(prob, EPS, 1-EPS)
    nll = float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))
    brier = float(np.mean((p-y)**2))
    e, _ = ece(p, y)
    return {"nll": nll, "brier": brier, "ece": e}


def ap50(gt, dets):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1])


def transform_dump(dets, kind, fitted):
    scores = np.asarray([d["score"] for d in dets])
    if kind == "temperature":
        prob = apply_temperature(scores, fitted["temperature"])
    elif kind == "platt":
        prob = fitted["estimator"].predict_proba(logit(scores).reshape(-1,1))[:,1]
    elif kind == "isotonic":
        prob = fitted["estimator"].predict(scores)
    else:
        prob = scores
    return [{**d, "score": float(p)} for d, p in zip(dets, prob)]


def fit_one(scores, y, kind):
    if kind == "temperature":
        return {"temperature": fit_temperature(scores, y)}
    if kind == "platt":
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logit(scores).reshape(-1,1), y.astype(int))
        return {"estimator": model}
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1-EPS)
        model.fit(scores, y)
        return {"estimator": model}
    return {}


def oof_transform(dets, gt_anns, image_folds, kind):
    """Fit on four image folds and transform the held-out fold, repeated five times."""
    result = []
    for heldout in range(5):
        train_ids = {i for i, fold in image_folds.items() if fold != heldout}
        held_ids = {i for i, fold in image_folds.items() if fold == heldout}
        train_dets = [d for d in dets if d["image_id"] in train_ids]
        train_gt = [a for a in gt_anns if a["image_id"] in train_ids]
        scores, y = match_detections(train_dets, train_gt)
        fitted = fit_one(scores, y, kind)
        result.extend(transform_dump([d for d in dets if d["image_id"] in held_ids],
                                     kind, fitted))
    return result


def main():
    gt_data = json.loads(GT_PATH.read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(GT_PATH))
    model_paths = {
        "clean_e": PREDS / "clean_e.clean_val_fit.json",
        "clean_f": PREDS / "clean_f.clean_val_fit.json",
    }
    dumps = {name: json.loads(path.read_text()) for name, path in model_paths.items()}
    labels, fits, calibration = {}, {}, {}
    for name, dets in dumps.items():
        scores, y = match_detections(dets, gt_data["annotations"])
        labels[name] = (scores, y)
        T = fit_temperature(scores, y)
        platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        platt.fit(logit(scores).reshape(-1,1), y.astype(int))
        iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1-EPS)
        iso.fit(scores, y)
        fits[name] = {
            "temperature": {"temperature": T},
            "platt": {"estimator": platt},
            "isotonic": {"estimator": iso},
            "none": {},
        }
        calibration[name] = {
            "n_detections": len(scores), "n_positive": int(y.sum()),
            "none": metrics(scores, y),
            "temperature": metrics(apply_temperature(scores,T), y),
            "platt": metrics(platt.predict_proba(logit(scores).reshape(-1,1))[:,1], y),
            "isotonic": metrics(iso.predict(scores), y),
            "parameters": {
                "temperature": T,
                "platt": {"coef": float(platt.coef_[0,0]),
                          "intercept": float(platt.intercept_[0])},
                "isotonic": {"x_thresholds": iso.X_thresholds_.tolist(),
                             "y_thresholds": iso.y_thresholds_.tolist()},
            },
        }

    fusion = []
    with tempfile.TemporaryDirectory(prefix="clean_ef_calib_") as td:
        td = Path(td)
        for kind in ("none", "temperature", "platt", "isotonic"):
            paths = []
            for name in ("clean_e", "clean_f"):
                p = td / f"{name}_{kind}.json"
                p.write_text(json.dumps(transform_dump(dumps[name], kind, fits[name][kind])))
                paths.append(p)
            dets = fuse(paths, load_dims(GT_PATH), 0.65, 0.001, [1.1,1.0], "none", "avg")
            fusion.append({"method": kind, "fit_ap50": ap50(gt,dets), "n_fused": len(dets)})
            print(f"{kind:11} AP50={fusion[-1]['fit_ap50']:.6f} boxes={len(dets)}")

    # Honest within-val_fit comparison: no image is transformed by a calibrator fitted on it.
    image_ids = sorted(im["id"] for im in gt_data["images"])
    random.Random(42).shuffle(image_ids)
    image_folds = {image_id: k % 5 for k, image_id in enumerate(image_ids)}
    oof_fusion = []
    with tempfile.TemporaryDirectory(prefix="clean_ef_oof_calib_") as td:
        td = Path(td)
        for kind in ("temperature", "platt", "isotonic"):
            paths = []
            for name in ("clean_e", "clean_f"):
                transformed = oof_transform(dumps[name], gt_data["annotations"], image_folds, kind)
                p = td / f"{name}_{kind}.json"
                p.write_text(json.dumps(transformed)); paths.append(p)
            dets = fuse(paths, load_dims(GT_PATH), 0.65, 0.001, [1.1,1.0], "none", "avg")
            oof_fusion.append({"method":kind,"fit_ap50":ap50(gt,dets),"n_fused":len(dets)})
            print(f"OOF {kind:7} AP50={oof_fusion[-1]['fit_ap50']:.6f} boxes={len(dets)}")

    result = {
        "fit_gt": str(GT_PATH),
        "warning": "In-sample val_fit comparison; isotonic flexibility can overfit. val_sel untouched.",
        "frozen_wbf": {"iou_thr":0.65,"skip_box_thr":0.001,"weights":[1.1,1.0],"conf_type":"avg"},
        "calibration": calibration, "fusion": fusion,
        "oof_fusion": oof_fusion, "oof_folds": 5, "oof_image_seed": 42,
    }
    out = SCORES / "clean_ef_calibrator_comparison.json"
    out.write_text(json.dumps(result,indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

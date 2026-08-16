#!/usr/bin/env python3
"""Fuse N prediction dumps with Weighted Boxes Fusion.

  python3 wbf_fuse.py --dumps preds/a.val.json preds/b.val.json preds/c.val.json \
      --gt preds/gt_val_namr33.json --out preds/wbf_abc.val.json \
      --iou-thr 0.55 --skip-box-thr 0.001 --weights 1 1 1 \
      --normalize-scores temperature

Per-model score normalization runs BEFORE fusion (WBF uses confidence as the
multiplier in its coordinate average, so uncalibrated cross-architecture scores
spatially bias the fused boxes):
  none        - raw scores
  temperature - sigmoid(logit(s)/T) with T from scores/<model>.calib.json
                (<model> = dump filename stem before the first '.')
  minmax      - per model, per image: (s - min) / (max - min)

Image dims come from --gt (val) or --manifest (an images/{id,width,height} json
for test mode). Output is the same dump format, fusable/scoreable downstream.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

from paths12 import GT_VAL_JSON, SCORES_DIR  # noqa: I001

EPS = 1e-6


def load_dims(gt_path):
    data = json.loads(open(gt_path).read())
    return {im["id"]: (im["width"], im["height"]) for im in data["images"]}


def normalize_scores(dets, mode, model_name):
    if mode == "none":
        return dets
    if mode == "temperature":
        calib = SCORES_DIR / f"{model_name}.calib.json"
        if not calib.exists():
            raise SystemExit(f"{calib} missing - run calibrate.py --name {model_name} first")
        T = json.loads(calib.read_text())["temperature"]
        out = []
        for d in dets:
            s = min(max(d["score"], EPS), 1 - EPS)
            logit = np.log(s / (1 - s))
            out.append({**d, "score": float(1 / (1 + np.exp(-logit / T)))})
        return out
    if mode == "minmax":
        by_img = defaultdict(list)
        for d in dets:
            by_img[d["image_id"]].append(d)
        out = []
        for img_dets in by_img.values():
            ss = [d["score"] for d in img_dets]
            lo, hi = min(ss), max(ss)
            rng = hi - lo
            for d in img_dets:
                s = (d["score"] - lo) / rng if rng > 0 else 1.0
                out.append({**d, "score": float(max(s, EPS))})
        return out
    raise ValueError(mode)


def fuse(dump_paths, dims, iou_thr, skip_box_thr, weights, norm_mode,
         conf_type="avg"):
    models = []
    for p in dump_paths:
        name = Path(p).name.split(".")[0]
        dets = json.loads(open(p).read())
        dets = normalize_scores(dets, norm_mode, name)
        by_img = defaultdict(list)
        for d in dets:
            by_img[d["image_id"]].append(d)
        models.append(by_img)

    fused = []
    for image_id, (w, h) in dims.items():
        boxes_list, scores_list, labels_list = [], [], []
        for by_img in models:
            dets = by_img.get(image_id, [])
            boxes, scores, labels = [], [], []
            for d in dets:
                x, y, bw, bh = d["bbox"]
                boxes.append([
                    min(max(x / w, 0.0), 1.0), min(max(y / h, 0.0), 1.0),
                    min(max((x + bw) / w, 0.0), 1.0), min(max((y + bh) / h, 0.0), 1.0),
                ])
                scores.append(d["score"])
                labels.append(d["category_id"])
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        if not any(scores_list):
            continue
        fb, fs, fl = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr,
            conf_type=conf_type,
        )
        for (x1, y1, x2, y2), s, lb in zip(fb, fs, fl):
            fused.append({
                "image_id": image_id,
                "category_id": int(lb),
                "bbox": [round(float(x1 * w), 3), round(float(y1 * h), 3),
                         round(float((x2 - x1) * w), 3), round(float((y2 - y1) * h), 3)],
                "score": round(float(s), 6),
            })
    return fused


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--gt", default=str(GT_VAL_JSON), help="json with images[{id,width,height}]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou-thr", type=float, default=0.55)
    ap.add_argument("--skip-box-thr", type=float, default=0.001)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--normalize-scores", choices=["none", "temperature", "minmax"], default="none")
    ap.add_argument("--conf-type",
                    choices=["avg", "max", "box_and_model_avg",
                             "absent_model_aware_avg"],
                    default="avg",
                    help="how WBF sets a fused box score. 'avg' (the default, and "
                         "what every result so far used) rescales by "
                         "min(n_models, n_boxes)/weights.sum(), so a box only one "
                         "member found is multiplied by 1/weights.sum(). 'max' "
                         "drops that penalty (divides by weights.max() instead).")
    args = ap.parse_args()

    if args.weights is not None and len(args.weights) != len(args.dumps):
        raise SystemExit("--weights count must match --dumps count")

    dims = load_dims(args.gt)
    fused = fuse(args.dumps, dims, args.iou_thr, args.skip_box_thr, args.weights,
                 args.normalize_scores, args.conf_type)
    Path(args.out).write_text(json.dumps(fused))
    meta = {
        "dumps": [str(p) for p in args.dumps],
        "iou_thr": args.iou_thr,
        "skip_box_thr": args.skip_box_thr,
        "weights": args.weights,
        "normalize_scores": args.normalize_scores,
        "conf_type": args.conf_type,
        "n_fused": len(fused),
    }
    Path(str(args.out).replace(".json", ".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out}  ({len(fused)} fused boxes from {len(args.dumps)} models)")


if __name__ == "__main__":
    main()

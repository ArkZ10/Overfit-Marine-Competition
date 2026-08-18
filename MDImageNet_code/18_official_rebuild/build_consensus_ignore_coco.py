#!/usr/bin/env python3
"""Add conservative all-teacher unmatched consensus as COCO ignore regions.

Official annotations are copied byte-for-byte at the object level. A mined box is
added with ``iscrowd=1`` so MMDetection routes it to ``gt_instances_ignore``:
classification/background anchors overlapping the region are ignored, and the box
does not supervise classification or regression as a pseudo-positive.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def iou_matrix(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a2, b2 = a[:, :2] + a[:, 2:], b[:, :2] + b[:, 2:]
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a2[:, None, :], b2[None, :, :])
    wh = np.clip(hi - lo, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    return inter / (area_a + area_b - inter + 1e-9)


def calibrated_score(score, temperature):
    score = min(max(float(score), 1e-6), 1 - 1e-6)
    logit = np.log(score / (1 - score))
    return float(1 / (1 + np.exp(-logit / temperature)))


def by_image(rows, threshold, temperature):
    result = defaultdict(list)
    for row in rows:
        score = calibrated_score(row["score"], temperature)
        if score >= threshold:
            result[row["image_id"]].append({**row, "score": score})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--dumps", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        help="one fitted temperature per dump; defaults to raw scores")
    parser.add_argument("--agreement-iou", type=float, default=0.5)
    parser.add_argument("--gt-iou", type=float, default=0.3,
                        help="reject consensus overlapping any official annotation")
    args = parser.parse_args()
    if len(args.dumps) < 2:
        parser.error("at least two teacher dumps are required")
    if args.temperatures is not None and len(args.temperatures) != len(args.dumps):
        parser.error("--temperatures count must match --dumps")

    coco = json.loads(args.gt.read_text())
    images = {im["id"]: im for im in coco["images"]}
    gt_by = defaultdict(list)
    for ann in coco["annotations"]:
        gt_by[ann["image_id"]].append(ann)

    teachers = []
    temperatures = args.temperatures or [1.0] * len(args.dumps)
    for path, temperature in zip(args.dumps, temperatures):
        rows = json.loads(path.read_text())
        extra = {r["image_id"] for r in rows} - set(images)
        if extra:
            raise SystemExit(f"{path}: {len(extra)} image IDs are outside --gt")
        teachers.append(by_image(rows, args.min_score, temperature))

    candidates = []
    for image_id in sorted(images):
        gt_boxes = [a["bbox"] for a in gt_by[image_id]]
        for base in teachers[0].get(image_id, []):
            if gt_boxes and iou_matrix([base["bbox"]], gt_boxes).max() >= args.gt_iou:
                continue
            matched = []
            for teacher in teachers[1:]:
                same = [r for r in teacher.get(image_id, [])
                        if r["category_id"] == base["category_id"]]
                if not same:
                    break
                ious = iou_matrix([base["bbox"]], [r["bbox"] for r in same])[0]
                index = int(np.argmax(ious))
                if ious[index] < args.agreement_iou:
                    break
                matched.append((same[index], float(ious[index])))
            else:
                candidates.append({
                    **base,
                    "min_score": min([float(base["score"])] +
                                     [float(row["score"]) for row, _ in matched]),
                    "mean_iou": float(np.mean([iou for _, iou in matched])),
                })

    # Deduplicate independent base detections of the same object.
    kept = []
    for row in sorted(candidates, key=lambda r: (-r["min_score"], -r["mean_iou"])):
        peers = [x for x in kept if x["image_id"] == row["image_id"] and
                 x["category_id"] == row["category_id"]]
        if peers and iou_matrix([row["bbox"]], [x["bbox"] for x in peers]).max() >= args.agreement_iou:
            continue
        kept.append(row)

    annotations = list(coco["annotations"])
    next_id = max((int(a["id"]) for a in annotations), default=0) + 1
    for row in kept:
        x, y, w, h = map(float, row["bbox"])
        annotations.append({
            "id": next_id,
            "image_id": row["image_id"],
            "category_id": int(row["category_id"]),
            "bbox": [x, y, w, h],
            "area": w * h,
            # MMDetection's CocoDataset converts iscrowd to ignore_flag=1.
            # Do not also write `ignore`: its parser drops such annotations.
            "iscrowd": 1,
            "min_teacher_score": row["min_score"],
            "mean_teacher_iou": row["mean_iou"],
        })
        next_id += 1

    output = {**coco, "annotations": annotations}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output))
    counts = Counter(int(x["category_id"]) for x in kept)
    meta = {
        "source_gt": str(args.gt),
        "teacher_dumps": [str(p) for p in args.dumps],
        "parameters": {"min_score": args.min_score,
                       "temperatures": temperatures,
                       "agreement_iou": args.agreement_iou, "gt_iou": args.gt_iou},
        "official_annotations": len(coco["annotations"]),
        "ignore_annotations": len(kept),
        "images_with_ignore": len({x["image_id"] for x in kept}),
        "ignore_by_class": dict(sorted(counts.items())),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

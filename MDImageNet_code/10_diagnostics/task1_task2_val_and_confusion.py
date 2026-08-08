#!/usr/bin/env python3
"""TASK 1 (per-class AP table) + TASK 2 (confusion matrix), from one validator run.

Numbers only. Read-only on weights/data.
"""
import csv
import random
import numpy as np
import torch

from common import (
    DATA_YAML, WEIGHTS, DIAG_DIR, NC, CLASS_NAMES, count_boxes_and_images,
)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from ultralytics.models.yolo.detect import DetectionValidator


def run_validator():
    args = dict(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=16,
        device="0",
        plots=True,  # required: DetectionValidator only calls confusion_matrix.process_batch when plots=True
        save_json=False,
        verbose=False,
        seed=0,
        workers=0,
        project=str(DIAG_DIR),
        name="_val_run",
        exist_ok=True,
    )
    validator = DetectionValidator(args=args)
    validator(model=str(WEIGHTS))
    return validator


def task1(validator, train_boxes, train_images, val_boxes, val_images):
    box = validator.metrics.box
    ap_class_index = list(box.ap_class_index)
    ap50_by_class = {int(c): float(box.ap50[i]) for i, c in enumerate(ap_class_index)}
    ap5095_by_class = {int(c): float(box.ap[i]) for i, c in enumerate(ap_class_index)}
    p_by_class = {int(c): float(box.p[i]) for i, c in enumerate(ap_class_index)}
    r_by_class = {int(c): float(box.r[i]) for i, c in enumerate(ap_class_index)}

    rows = []
    for c in range(NC):
        rows.append({
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "ap50": round(ap50_by_class.get(c, 0.0), 6),
            "ap50_95": round(ap5095_by_class.get(c, 0.0), 6),
            "precision": round(p_by_class.get(c, 0.0), 6),
            "recall": round(r_by_class.get(c, 0.0), 6),
            "train_boxes": train_boxes.get(c, 0),
            "train_images": train_images.get(c, 0),
            "val_boxes": val_boxes.get(c, 0),
            "val_images": val_images.get(c, 0),
        })
    rows.sort(key=lambda r: r["ap50"])

    out_csv = DIAG_DIR / "per_class_ap.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    header = f"{'class_id':>8} {'class_name':<32} {'ap50':>8} {'ap50_95':>8} {'prec':>8} {'rec':>8} {'trn_box':>8} {'trn_img':>8} {'val_box':>8} {'val_img':>8}"
    print(header)
    for r in rows:
        print(f"{r['class_id']:>8} {r['class_name']:<32} {r['ap50']:>8.4f} {r['ap50_95']:>8.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['train_boxes']:>8} {r['train_images']:>8} "
              f"{r['val_boxes']:>8} {r['val_images']:>8}")
    return rows


def task2(validator):
    # ultralytics matrix is matrix[pred, gt], shape (nc+1, nc+1); last idx = background
    m = validator.confusion_matrix.matrix.astype(np.float64)
    gt_pred = m.T  # now gt_pred[gt, pred]

    labels = [CLASS_NAMES[c] for c in range(NC)] + ["background"]

    row_sums = gt_pred.sum(axis=1, keepdims=True)
    norm = np.divide(gt_pred, row_sums, out=np.zeros_like(gt_pred), where=row_sums != 0)

    out_csv = DIAG_DIR / "confusion_matrix.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt_class\\pred_class"] + labels)
        for i, lbl in enumerate(labels):
            w.writerow([lbl] + [round(float(x), 6) for x in norm[i]])

    # top 30 off-diagonal cells (raw counts), sorted desc by count
    entries = []
    n = NC + 1
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            count = gt_pred[i, j]
            if count <= 0:
                continue
            frac = norm[i, j]
            entries.append((labels[i], labels[j], int(count), float(frac)))
    entries.sort(key=lambda e: e[2], reverse=True)
    top30 = entries[:30]

    out_csv2 = DIAG_DIR / "confusions_top30.csv"
    with open(out_csv2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt_class", "pred_class", "count", "fraction_of_gt_class"])
        for gt, pred, count, frac in top30:
            w.writerow([gt, pred, count, round(frac, 6)])

    print("\ntop 5 confusion pairs (gt -> pred, count, frac_of_gt):")
    for gt, pred, count, frac in top30[:5]:
        print(f"  {gt} -> {pred}: {count} ({frac:.3f})")

    return top30


if __name__ == "__main__":
    import shutil

    validator = run_validator()
    train_boxes, train_images = count_boxes_and_images("train")
    val_boxes, val_images = count_boxes_and_images("val")
    ap_rows = task1(validator, train_boxes, train_images, val_boxes, val_images)
    top30 = task2(validator)

    # clean up ultralytics' plot-run scratch dir (labels.jpg, PR curves, etc.) -
    # we only wanted the confusion matrix data extracted above, not the plot images.
    for scratch in (DIAG_DIR / "_val_run", DIAG_DIR / "runs"):
        if scratch.exists():
            shutil.rmtree(scratch)

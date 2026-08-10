#!/usr/bin/env python3
"""TASK 3 - COCO API recalibration.

Exports val ground truth + best.pt predictions to COCO JSON (reading each
image's real dimensions, not assuming 640x640), scores with pycocotools
COCOeval, and writes the standard summary plus a one-line comparison against
ultralytics' own mAP50.
"""
import contextlib
import io
import json
import random

import numpy as np
import torch
from PIL import Image

from common import CLASS_NAMES, DIAG_DIR, NC, WEIGHTS, resolve_image_files, resolve_label_files

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def ultralytics_map50_from_per_class_csv():
    """mean(ap50) over all 34 classes == ultralytics' own reported box.map50, since
    per_class_ap.csv (written by task1_task2_val_and_confusion.py) has one row per
    class with 0.0 filled in for any class absent from ap_class_index."""
    import csv
    path = DIAG_DIR / "per_class_ap.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run task1_task2_val_and_confusion.py first")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return sum(float(r["ap50"]) for r in rows) / len(rows)


def yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h):
    """Normalized YOLO cx,cy,w,h -> absolute COCO x_min,y_min,w,h."""
    abs_w = w * img_w
    abs_h = h * img_h
    x_min = cx * img_w - abs_w / 2
    y_min = cy * img_h - abs_h / 2
    return [x_min, y_min, abs_w, abs_h]


def build_gt_coco():
    images = []
    annotations = []
    categories = [{"id": c, "name": CLASS_NAMES[c]} for c in range(NC)]

    img_files = resolve_image_files("val")
    stem_to_id = {p.stem: i for i, p in enumerate(img_files)}

    ann_id = 0
    for img_path in img_files:
        real_img_path = img_path.resolve()
        with Image.open(real_img_path) as im:
            img_w, img_h = im.size
        image_id = stem_to_id[img_path.stem]
        images.append({"id": image_id, "file_name": img_path.name, "width": img_w, "height": img_h})

        lbl_path = img_path.parent.parent.parent / "labels" / "val" / f"{img_path.stem}.txt"
        real_lbl_path = lbl_path.resolve()
        if real_lbl_path.exists():
            with open(real_lbl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls = int(parts[0])
                    cx, cy, w, h = (float(x) for x in parts[1:5])
                    bbox = yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h)
                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls,
                        "bbox": [round(v, 3) for v in bbox],
                        "area": round(bbox[2] * bbox[3], 3),
                        "iscrowd": 0,
                    })
                    ann_id += 1

    coco_gt = {"images": images, "annotations": annotations, "categories": categories}
    return coco_gt, stem_to_id, img_files


def run_predictions(stem_to_id, img_files, weights=WEIGHTS):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    predictions = []
    val_dir = img_files[0].parent
    results = model.predict(
        source=str(val_dir),
        imgsz=640,
        conf=0.001,
        iou=0.7,
        device="0",
        verbose=False,
        save=False,
        seed=0,
        stream=True,
    )
    for r in results:
        stem = __import__("pathlib").Path(r.path).stem
        if stem not in stem_to_id:
            continue
        image_id = stem_to_id[stem]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            predictions.append({
                "image_id": image_id,
                "category_id": int(k),
                "bbox": [round(float(x1), 3), round(float(y1), 3), round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
                "score": round(float(c), 6),
            })
    return predictions


def main():
    coco_gt, stem_to_id, img_files = build_gt_coco()
    gt_path = DIAG_DIR / "_coco_gt.json"
    with open(gt_path, "w") as f:
        json.dump(coco_gt, f)

    predictions = run_predictions(stem_to_id, img_files)
    pred_path = DIAG_DIR / "_coco_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt_obj = COCO(str(gt_path))
        coco_dt_obj = coco_gt_obj.loadRes(str(pred_path))

    ev = COCOeval(coco_gt_obj, coco_dt_obj, iouType="bbox")
    ev.evaluate()
    ev.accumulate()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.summarize()
    summary_text = buf.getvalue()
    print(summary_text)

    pycoco_ap50 = float(ev.stats[1])  # AP @ IoU=0.50
    ultralytics_map50 = ultralytics_map50_from_per_class_csv()
    delta = pycoco_ap50 - ultralytics_map50
    comparison_line = (
        f"ultralytics mAP50={ultralytics_map50:.4f} vs pycocotools AP@0.50={pycoco_ap50:.4f} "
        f"(delta={delta:+.4f})"
    )
    print(comparison_line)

    out_path = DIAG_DIR / "coco_eval.txt"
    with open(out_path, "w") as f:
        f.write(summary_text)
        f.write(comparison_line + "\n")

    gt_path.unlink()
    pred_path.unlink()

    return pycoco_ap50, delta


if __name__ == "__main__":
    main()

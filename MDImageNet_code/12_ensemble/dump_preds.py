#!/usr/bin/env python3
"""Dump any detector's predictions in the shared COCO-detections format.

  python3 dump_preds.py --model-type yolo   --weights W --name y11m_control --split val
  python3 dump_preds.py --model-type rtdetr --weights W --name rtdetr_l     --split val
  python3 dump_preds.py --model-type frcnn  --weights W --name frcnn_r50v2  --split val
  python3 dump_preds.py --model-type yolo --weights W --name X --images-dir /path/to/imgs

Output: preds/<name>.<split>.json  = [{image_id, category_id (0-33), bbox [x,y,w,h] abs px, score}]
        preds/<name>.<split>.meta.json = provenance + image_id manifest source.

The val-split image_id convention is build_gt_coco()'s enumerate of sorted stems
(10_diagnostics/task3_coco_eval.py) - identical for every model, which is what
makes the dumps fusable. --images-dir mode writes its own sorted-stem manifest.
"""
import argparse
import datetime
import json
import random
from pathlib import Path

import numpy as np
import torch

from paths12 import (  # noqa: I001 - installs sys.path hooks first
    CONF_THR, GT_VAL_JSON, IMGSZ, MAX_DET, PREDS_DIR, SEED,
)
from task3_coco_eval import build_gt_coco  # noqa: E402  (10_diagnostics)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NMS_IOU = 0.7  # matches 11_improvements/coco_score.py; RT-DETR is NMS-free and ignores it


def val_manifest():
    """(stem_to_id, ordered image paths) for the val split; writes GT json once."""
    coco_gt, stem_to_id, img_files = build_gt_coco()
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    if not GT_VAL_JSON.exists():
        GT_VAL_JSON.write_text(json.dumps(coco_gt))
        print(f"wrote {GT_VAL_JSON}")
    return stem_to_id, img_files, "build_gt_coco(val)"


def dir_manifest(images_dir: Path):
    img_files = sorted(images_dir.glob("*.jpg"), key=lambda p: p.stem)
    if not img_files:
        raise SystemExit(f"no .jpg images in {images_dir}")
    stem_to_id = {p.stem: i for i, p in enumerate(img_files)}
    return stem_to_id, img_files, f"sorted-stem enumerate of {images_dir}"


def predict_ultralytics(model_type: str, weights: str, img_files, stem_to_id):
    from ultralytics import RTDETR, YOLO

    model = RTDETR(str(weights)) if model_type == "rtdetr" else YOLO(str(weights))
    dets = []
    # Predict over the images' parent DIRECTORY, not a Python list: ultralytics
    # treats a bare list as one giant batch (40GB conv OOM on 1,661 images) and
    # rewrites r.path to 'image0'. Directory sources stream at batch 1 with real
    # paths - this is the same code path 11_improvements/coco_score.py uses.
    src_dir = img_files[0].parent
    results = model.predict(
        source=str(src_dir),
        imgsz=IMGSZ,
        conf=CONF_THR,
        iou=NMS_IOU,
        max_det=MAX_DET,
        device="0",
        verbose=False,
        save=False,
        stream=True,
    )
    for r in results:
        stem = Path(r.path).stem
        if stem not in stem_to_id:  # stray file in the folder, not in the manifest
            continue
        image_id = stem_to_id[stem]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
            dets.append({
                "image_id": image_id,
                "category_id": int(c),
                "bbox": [round(float(x1), 3), round(float(y1), 3),
                         round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
                "score": round(float(s), 6),
            })
    return dets


def predict_frcnn(weights: str, img_files, stem_to_id):
    from frcnn.predict_frcnn import run_inference

    return run_inference(weights, img_files, stem_to_id)


def predict_dfine(weights: str, img_files, stem_to_id):
    import sys
    sys.path.insert(0, str(PREDS_DIR.parent.parent / "13_dfine"))
    from predict_dfine import run_inference

    return run_inference(weights, img_files, stem_to_id)


def predict_rtmdet(weights: str, img_files, stem_to_id, tta: bool = False):
    import sys
    sys.path.insert(0, str(PREDS_DIR.parent.parent / "15_rtmdet"))
    from predict_rtmdet import run_inference

    return run_inference(weights, img_files, stem_to_id, tta=tta)


def predict_deim(weights: str, img_files, stem_to_id):
    import sys
    sys.path.insert(0, str(PREDS_DIR.parent.parent / "14_deim"))
    from predict_deim import run_inference

    return run_inference(weights, img_files, stem_to_id)


def predict_rfdetr(weights: str, img_files, stem_to_id):
    import sys
    sys.path.insert(0, str(PREDS_DIR.parent.parent / "17_rfdetr"))
    from predict_rfdetr import run_inference

    return run_inference(weights, img_files, stem_to_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-type", choices=["yolo", "rtdetr", "frcnn", "dfine", "deim", "rtmdet", "rfdetr"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val", help="label for the dump; 'val' uses the GT manifest")
    ap.add_argument("--tta", action="store_true",
                    help="rtmdet only: mmdet's built-in 3-scale x 2-flip TTA")
    ap.add_argument("--images-dir", type=Path, default=None,
                    help="predict a folder instead of the val split (test/pseudo mode)")
    args = ap.parse_args()

    if args.images_dir is not None:
        split = args.split if args.split != "val" else args.images_dir.name
        stem_to_id, img_files, manifest_src = dir_manifest(args.images_dir)
    else:
        split = "val"
        stem_to_id, img_files, manifest_src = val_manifest()

    if args.model_type == "frcnn":
        dets = predict_frcnn(args.weights, img_files, stem_to_id)
    elif args.model_type == "dfine":
        dets = predict_dfine(args.weights, img_files, stem_to_id)
    elif args.model_type == "deim":
        dets = predict_deim(args.weights, img_files, stem_to_id)
    elif args.model_type == "rtmdet":
        dets = predict_rtmdet(args.weights, img_files, stem_to_id, tta=args.tta)
    elif args.model_type == "rfdetr":
        dets = predict_rfdetr(args.weights, img_files, stem_to_id)
    else:
        dets = predict_ultralytics(args.model_type, args.weights, img_files, stem_to_id)

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    out = PREDS_DIR / f"{args.name}.{split}.json"
    out.write_text(json.dumps(dets))
    meta = {
        "name": args.name,
        "split": split,
        "model_type": args.model_type,
        "weights": str(Path(args.weights).resolve()),
        "conf": CONF_THR,
        "nms_iou": NMS_IOU,
        "max_det": MAX_DET,
        "imgsz": IMGSZ,
        "n_images": len(img_files),
        "n_detections": len(dets),
        "image_id_manifest": manifest_src,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (PREDS_DIR / f"{args.name}.{split}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}  ({len(dets)} detections over {len(img_files)} images)")


if __name__ == "__main__":
    main()

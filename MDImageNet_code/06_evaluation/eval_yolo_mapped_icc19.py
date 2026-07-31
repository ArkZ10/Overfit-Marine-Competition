#!/usr/bin/env python3
"""Evaluate a YOLO detector after mapping source classes into ICC19 classes.

Purpose:
    Apply a taxonomy crosswalk to ground truth and predictions, then export
    ICC19-aligned AP, confusion, and fixed-threshold metrics.

Usage:
    python 06_evaluation/eval_yolo_mapped_icc19.py --model <weights.pt> --data <dataset.yaml> --labels-dir <labels-dir> --crosswalk <crosswalk.csv> --taxonomy namr26 --icc-names-yaml <icc19.yaml>
    python 06_evaluation/eval_yolo_mapped_icc19.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO, settings


IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class GTBox:
    image_id: int
    class_id: int
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    image_id: int
    class_id: int
    score: float
    xyxy: tuple[float, float, float, float]


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def names_to_dict(names: dict | list) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise TypeError(f"Unsupported names type: {type(names).__name__}")


def resolve_split_dirs(data_yaml: Path, split: str) -> tuple[Path, Path]:
    data = read_yaml(data_yaml)
    if split not in data:
        raise KeyError(f"Split {split!r} not found in {data_yaml}")
    dataset_root = Path(data.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    image_dir = Path(data[split])
    if not image_dir.is_absolute():
        image_dir = dataset_root / image_dir
    label_dir = image_dir.parent / "labels"
    return image_dir.resolve(), label_dir.resolve()


def image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def yolo_to_xyxy(values: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, w, h = values
    box_w = w * width
    box_h = h * height
    x1 = cx * width - box_w / 2.0
    y1 = cy * height - box_h / 2.0
    return (x1, y1, x1 + box_w, y1 + box_h)


def xyxy_to_xywh(box: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def read_crosswalk(path: Path, taxonomy: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["taxonomy"] != taxonomy:
                continue
            mapping[int(row["source_class_id"])] = int(row["icc19_class_id"])
    if not mapping:
        raise ValueError(f"No crosswalk rows found for taxonomy {taxonomy!r} in {path}")
    return mapping


def load_ground_truth(
    image_dir: Path,
    label_dir: Path,
    names: dict[int, str],
    crosswalk: dict[int, int],
) -> tuple[COCO, dict[str, int], dict[int, list[GTBox]], list[dict], list[dict], dict[int, int]]:
    images: list[dict] = []
    annotations: list[dict] = []
    image_name_to_id: dict[str, int] = {}
    gt_by_image: dict[int, list[GTBox]] = defaultdict(list)
    instances_by_class: Counter[int] = Counter()
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths(image_dir), 1):
        with Image.open(image_path) as im:
            width, height = im.size
        image_name_to_id[image_path.name] = image_id
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}: {line!r}")
            source_class = int(float(parts[0]))
            if source_class not in crosswalk:
                raise KeyError(f"No ICC19 crosswalk for source class {source_class} at {label_path}:{line_number}")
            class_id = crosswalk[source_class]
            xyxy = yolo_to_xyxy([float(v) for v in parts[1:5]], width, height)
            xywh = xyxy_to_xywh(xyxy)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": xywh,
                    "area": xywh[2] * xywh[3],
                    "iscrowd": 0,
                }
            )
            gt_by_image[image_id].append(GTBox(image_id=image_id, class_id=class_id, xyxy=xyxy))
            instances_by_class[class_id] += 1
            annotation_id += 1

    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": class_id + 1, "name": names[class_id]} for class_id in sorted(names)],
        "info": {"description": "YOLO labels mapped to ICC19"},
        "licenses": [],
    }
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco, image_name_to_id, gt_by_image, images, annotations, dict(instances_by_class)


def predict_mapped(
    model: YOLO,
    image_dir: Path,
    image_name_to_id: dict[str, int],
    crosswalk: dict[int, int],
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
) -> tuple[list[dict], dict[int, list[Detection]]]:
    detections_json: list[dict] = []
    detections_by_image: dict[int, list[Detection]] = defaultdict(list)
    predict_kwargs = {
        "source": str(image_dir),
        "imgsz": imgsz,
        "batch": batch,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "stream": True,
        "verbose": False,
    }
    if device.strip():
        predict_kwargs["device"] = device.strip()

    for result in model.predict(**predict_kwargs):
        image_name = Path(result.path).name
        image_id = image_name_to_id[image_name]
        if result.boxes is None:
            continue
        xyxy_values = result.boxes.xyxy.cpu().numpy()
        class_values = result.boxes.cls.cpu().numpy()
        conf_values = result.boxes.conf.cpu().numpy()
        for xyxy_arr, source_class_raw, score_raw in zip(xyxy_values, class_values, conf_values):
            source_class = int(source_class_raw)
            if source_class not in crosswalk:
                raise KeyError(f"No ICC19 crosswalk for predicted source class {source_class}")
            class_id = crosswalk[source_class]
            score = float(score_raw)
            xyxy = tuple(float(v) for v in xyxy_arr.tolist())
            detections_json.append(
                {
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": xyxy_to_xywh(xyxy),
                    "score": score,
                }
            )
            detections_by_image[image_id].append(
                Detection(image_id=image_id, class_id=class_id, score=score, xyxy=xyxy)
            )
    return detections_json, detections_by_image


def summarize_coco_eval(coco_gt: COCO, detections_json: list[dict]) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    if detections_json:
        coco_dt = coco_gt.loadRes(detections_json)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {"images": coco_gt.dataset["images"], "categories": coco_gt.dataset["categories"], "annotations": []}
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.params.catIds = sorted(coco_gt.getCatIds())
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    summary = {
        "mAP50-95": float(stats[0]),
        "mAP50": float(stats[1]),
        "mAP75": float(stats[2]),
        "AP_S": float(stats[3]),
        "AP_M": float(stats[4]),
        "AP_L": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
    }

    precision = evaluator.eval["precision"]
    iou_thrs = evaluator.params.iouThrs
    ap50_index = int(np.argmin(np.abs(iou_thrs - 0.5)))
    max_det_index = len(evaluator.params.maxDets) - 1
    per_class: dict[int, dict[str, float]] = {}
    for class_index, category_id in enumerate(evaluator.params.catIds):
        class_precision = precision[:, :, class_index, 0, max_det_index]
        valid = class_precision[class_precision > -1]
        class_ap = float(np.mean(valid)) if valid.size else 0.0
        class_ap50_precision = precision[ap50_index, :, class_index, 0, max_det_index]
        valid50 = class_ap50_precision[class_ap50_precision > -1]
        class_ap50 = float(np.mean(valid50)) if valid50.size else 0.0
        per_class[category_id - 1] = {"AP50": class_ap50, "AP50-95": class_ap}
    return summary, per_class


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else inter / denom


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def fixed_threshold_metrics(
    gt_by_image: dict[int, list[GTBox]],
    detections_by_image: dict[int, list[Detection]],
    nc: int,
    conf: float,
    iou: float,
) -> tuple[dict[str, float], list[dict], np.ndarray]:
    tp: Counter[int] = Counter()
    fp: Counter[int] = Counter()
    fn: Counter[int] = Counter()
    matrix = np.zeros((nc + 1, nc + 1), dtype=int)
    background = nc

    for image_id, gt_boxes in gt_by_image.items():
        detections = [det for det in detections_by_image.get(image_id, []) if det.score >= conf]
        detections = sorted(detections, key=lambda det: det.score, reverse=True)
        used_gt: set[int] = set()
        for det in detections:
            best_idx = -1
            best_iou = 0.0
            for idx, gt in enumerate(gt_boxes):
                if idx in used_gt:
                    continue
                value = box_iou(det.xyxy, gt.xyxy)
                if value > best_iou:
                    best_iou = value
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou:
                gt = gt_boxes[best_idx]
                used_gt.add(best_idx)
                matrix[gt.class_id, det.class_id] += 1
                if gt.class_id == det.class_id:
                    tp[det.class_id] += 1
                else:
                    fp[det.class_id] += 1
                    fn[gt.class_id] += 1
            else:
                matrix[background, det.class_id] += 1
                fp[det.class_id] += 1
        for idx, gt in enumerate(gt_boxes):
            if idx not in used_gt:
                matrix[gt.class_id, background] += 1
                fn[gt.class_id] += 1

    rows: list[dict] = []
    total_tp = total_fp = total_fn = 0
    for class_id in range(nc):
        class_tp = tp[class_id]
        class_fp = fp[class_id]
        class_fn = fn[class_id]
        total_tp += class_tp
        total_fp += class_fp
        total_fn += class_fn
        precision = class_tp / (class_tp + class_fp) if class_tp + class_fp else 0.0
        recall = class_tp / (class_tp + class_fn) if class_tp + class_fn else 0.0
        rows.append(
            {
                "class_id": class_id,
                "tp": class_tp,
                "fp": class_fp,
                "fn": class_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1(precision, recall),
            }
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    summary = {
        "conf_threshold": conf,
        "iou_threshold": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }
    return summary, rows, matrix


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(path: Path, matrix: np.ndarray, names: dict[int, str]) -> None:
    labels = [names[i] for i in sorted(names)] + ["background"]
    rows = []
    for true_idx, true_name in enumerate(labels):
        row = {"true_class": true_name}
        for pred_idx, pred_name in enumerate(labels):
            row[pred_name] = int(matrix[true_idx, pred_idx])
        rows.append(row)
    write_csv(path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to YOLO .pt weights.")
    parser.add_argument("--data", type=Path, required=True, help="Path to source-taxonomy YOLO dataset.yaml.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help=(
            "Optional source-taxonomy YOLO labels directory. "
            "If omitted, uses the standard labels directory beside the images directory."
        ),
    )
    parser.add_argument("--crosswalk", type=Path, required=True, help="CSV with taxonomy,source_class_id,icc19_class_id.")
    parser.add_argument("--taxonomy", default="namr26", help="Taxonomy name to select from the crosswalk CSV.")
    parser.add_argument(
        "--icc-names-yaml",
        type=Path,
        required=True,
        help="ICC19 dataset yaml used only for class names.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for mapped evaluation outputs.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001, help="Prediction confidence threshold for AP curves.")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="Prediction NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--fixed-conf", type=float, default=0.25)
    parser.add_argument("--fixed-iou", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_yaml = args.data.resolve()
    crosswalk_path = args.crosswalk.resolve()
    icc_names_yaml = args.icc_names_yaml.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")
    if not crosswalk_path.exists():
        raise FileNotFoundError(f"Crosswalk CSV not found: {crosswalk_path}")
    if not icc_names_yaml.exists():
        raise FileNotFoundError(f"ICC names yaml not found: {icc_names_yaml}")

    default_output = model_path.parent.parent / f"{args.split}_icc19_mapped_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = (args.output_dir or default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    settings.update({"wandb": False})
    image_dir, inferred_label_dir = resolve_split_dirs(data_yaml, args.split)
    label_dir = args.labels_dir.resolve() if args.labels_dir else inferred_label_dir
    if not label_dir.is_dir():
        raise NotADirectoryError(f"Labels directory not found: {label_dir}")
    icc_names = names_to_dict(read_yaml(icc_names_yaml)["names"])
    crosswalk = read_crosswalk(crosswalk_path, args.taxonomy)

    coco_gt, image_name_to_id, gt_by_image, images, annotations, instances_by_class = load_ground_truth(
        image_dir=image_dir,
        label_dir=label_dir,
        names=icc_names,
        crosswalk=crosswalk,
    )
    with (output_dir / "mapped_icc19_gt_coco.json").open("w", encoding="utf-8") as f:
        json.dump(coco_gt.dataset, f, ensure_ascii=False)

    model = YOLO(str(model_path))
    detections_json, detections_by_image = predict_mapped(
        model=model,
        image_dir=image_dir,
        image_name_to_id=image_name_to_id,
        crosswalk=crosswalk,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.nms_iou,
        max_det=args.max_det,
    )
    with (output_dir / "mapped_icc19_predictions_coco.json").open("w", encoding="utf-8") as f:
        json.dump(detections_json, f, ensure_ascii=False)

    coco_summary, per_class = summarize_coco_eval(coco_gt, detections_json)
    fixed_summary, fixed_rows, matrix = fixed_threshold_metrics(
        gt_by_image=gt_by_image,
        detections_by_image=detections_by_image,
        nc=len(icc_names),
        conf=args.fixed_conf,
        iou=args.fixed_iou,
    )

    fixed_by_class = {int(row["class_id"]): row for row in fixed_rows}
    rows: list[dict] = [
        {
            "class_id": "all",
            "class_name": "all",
            "images": len(images),
            "instances": len(annotations),
            "AP50": coco_summary["mAP50"],
            "AP50-95": coco_summary["mAP50-95"],
            "precision@fixed": fixed_summary["precision"],
            "recall@fixed": fixed_summary["recall"],
            "f1@fixed": fixed_summary["f1"],
        }
    ]
    for class_id in sorted(icc_names):
        fixed = fixed_by_class[class_id]
        rows.append(
            {
                "class_id": class_id,
                "class_name": icc_names[class_id],
                "images": "",
                "instances": instances_by_class.get(class_id, 0),
                "AP50": per_class[class_id]["AP50"],
                "AP50-95": per_class[class_id]["AP50-95"],
                "precision@fixed": fixed["precision"],
                "recall@fixed": fixed["recall"],
                "f1@fixed": fixed["f1"],
            }
        )

    write_csv(output_dir / "mapped_icc19_metrics.csv", rows)
    write_csv(output_dir / "mapped_icc19_fixed_conf_iou.csv", fixed_rows)
    write_confusion_matrix(output_dir / "mapped_icc19_confusion_matrix.csv", matrix, icc_names)

    summary = {
        "model": str(model_path),
        "data_yaml": str(data_yaml),
        "split": args.split,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "crosswalk": str(crosswalk_path),
        "taxonomy": args.taxonomy,
        "icc_names_yaml": str(icc_names_yaml),
        "params": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "conf": args.conf,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "fixed_conf": args.fixed_conf,
            "fixed_iou": args.fixed_iou,
        },
        "totals": {
            **coco_summary,
            "precision@fixed": fixed_summary["precision"],
            "recall@fixed": fixed_summary["recall"],
            "f1@fixed": fixed_summary["f1"],
        },
        "output_dir": str(output_dir),
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Mapped ICC19 evaluation exported:")
    print(f"- output_dir: {output_dir}")
    print(f"- images: {len(images)}")
    print(f"- instances: {len(annotations)}")
    print(f"- mAP50: {coco_summary['mAP50']:.6f}")
    print(f"- mAP50-95: {coco_summary['mAP50-95']:.6f}")
    print(f"- metrics_csv: {output_dir / 'mapped_icc19_metrics.csv'}")


if __name__ == "__main__":
    main()

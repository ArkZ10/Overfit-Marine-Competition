#!/usr/bin/env python3
"""First-pass baseline: fine-tune pretrained YOLOv11 on the NAMR33 (34-class) split.

Pretrained weights (yolo11n.pt) are Ultralytics' open-source YOLO11 release
(AGPL-3.0), downloaded automatically on first run.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data_NAMR33_split.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(HERE / "runs"))
    parser.add_argument("--name", default="namr33_yolo11n_baseline")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

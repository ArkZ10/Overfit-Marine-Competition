"""YOLO-txt-list -> torchvision detection dataset.

Built from a txt list of image paths (11_improvements/rfs/*.txt). Label path is
derived by the same '/images/' -> '/labels/' substitution ultralytics uses.
Labels are shifted +1 (torchvision reserves 0 for background). Boxes with any
side < 1 px after conversion are dropped.
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def yolo_txt_to_xyxy(label_path: Path, w: int, h: int):
    """-> (boxes xyxy abs, labels 1-based). Missing/empty file -> empty tensors."""
    boxes, labels = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:  # degenerate after clamping
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(c + 1)
    if boxes:
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
    return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)


def image_to_label_path(img_path: Path) -> Path:
    return Path(str(img_path).replace("/images/", "/labels/", 1)).with_suffix(".txt")


class YoloListDetection(Dataset):
    def __init__(self, list_file: Path, train: bool = False):
        self.paths = [Path(p) for p in Path(list_file).read_text().splitlines() if p.strip()]
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            img = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
            img = img.view(h, w, 3).permute(2, 0, 1).float() / 255.0

        boxes, labels = yolo_txt_to_xyxy(image_to_label_path(img_path), w, h)

        if self.train and torch.rand(1).item() < 0.5:  # horizontal flip
            img = img.flip(-1)
            if len(boxes):
                boxes = boxes.clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "stem": img_path.stem,
        }
        return img, target


def collate(batch):
    return tuple(zip(*batch))

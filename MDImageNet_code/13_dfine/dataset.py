"""YOLO-txt-list -> D-FINE (transformers) detection dataset.

D-FINE's label format is {'class_labels': int64[N], 'boxes': float32[N,4]} with
boxes as NORMALIZED cxcywh - identical to a YOLO txt line, so no box conversion
is needed. Resizing to a fixed 640x640 also leaves normalized boxes untouched.

Preprocessing matches the model's own processor: resize 640x640, rescale to
[0,1], and NO ImageNet mean/std normalization (do_normalize=False for D-FINE).
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

IMGSZ = 640


def image_to_label_path(img_path: Path) -> Path:
    return Path(str(img_path).replace("/images/", "/labels/", 1)).with_suffix(".txt")


def read_yolo(label_path: Path):
    cls, boxes = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            cx, cy, w, h = (float(v) for v in p[1:5])
            if w <= 0 or h <= 0:
                continue
            cls.append(c)
            boxes.append([cx, cy, w, h])
    if boxes:
        return torch.tensor(cls, dtype=torch.int64), torch.tensor(boxes, dtype=torch.float32)
    return torch.zeros((0,), dtype=torch.int64), torch.zeros((0, 4), dtype=torch.float32)


class DFineDataset(Dataset):
    def __init__(self, list_file, train: bool = False, imgsz: int = IMGSZ):
        self.paths = [Path(p) for p in Path(list_file).read_text().splitlines() if p.strip()]
        self.train = train
        self.imgsz = imgsz

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB").resize((self.imgsz, self.imgsz), Image.BILINEAR)
            t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
            t = t.view(self.imgsz, self.imgsz, 3).permute(2, 0, 1).float() / 255.0

        cls, boxes = read_yolo(image_to_label_path(p))

        if self.train and torch.rand(1).item() < 0.5:  # hflip: cx -> 1 - cx
            t = t.flip(-1)
            if len(boxes):
                boxes = boxes.clone()
                boxes[:, 0] = 1.0 - boxes[:, 0]

        return t, {"class_labels": cls, "boxes": boxes}, p.stem


def collate(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    stems = [b[2] for b in batch]
    return pixel_values, labels, stems

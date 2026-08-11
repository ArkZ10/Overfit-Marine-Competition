"""Faster R-CNN checkpoint -> shared dump-format detections (called by dump_preds.py)."""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

ENS_DIR = Path(__file__).resolve().parents[1]
if str(ENS_DIR) not in sys.path:
    sys.path.insert(0, str(ENS_DIR))

from paths12 import CONF_THR, MAX_DET  # noqa: E402
from frcnn.train_frcnn import build_model  # noqa: E402


class ImageOnly(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            img = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
            img = img.view(h, w, 3).permute(2, 0, 1).float() / 255.0
        return img, p.stem


def collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def run_inference(weights, img_files, stem_to_id, device="cuda:0", batch=8):
    device = torch.device(device)
    model = build_model(device)
    ck = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    model.roi_heads.score_thresh = CONF_THR
    model.roi_heads.detections_per_img = MAX_DET

    loader = DataLoader(ImageOnly([Path(p) for p in img_files]), batch_size=batch,
                        num_workers=8, collate_fn=collate, pin_memory=True)
    dets = []
    for imgs, stems in loader:
        imgs = [i.to(device) for i in imgs]
        with torch.autocast("cuda"):
            outputs = model(imgs)
        for out, stem in zip(outputs, stems):
            if stem not in stem_to_id:
                continue
            image_id = stem_to_id[stem]
            boxes = out["boxes"].float().cpu().numpy()
            scores = out["scores"].float().cpu().numpy()
            labels = out["labels"].cpu().numpy()
            for (x1, y1, x2, y2), s, lb in zip(boxes, scores, labels):
                dets.append({
                    "image_id": image_id,
                    "category_id": int(lb) - 1,  # back to 0-33
                    "bbox": [round(float(x1), 3), round(float(y1), 3),
                             round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
                    "score": round(float(s), 6),
                })
    return dets

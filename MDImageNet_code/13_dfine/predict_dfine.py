"""D-FINE checkpoint -> shared 12_ensemble dump format (called by dump_preds.py)."""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_dfine import MODEL_ID, build_model  # noqa: E402

IMGSZ = 640
CONF = 0.001
MAX_DET = 300


class ImageOnly(Dataset):
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            r = im.resize((IMGSZ, IMGSZ), Image.BILINEAR)
            t = torch.frombuffer(bytearray(r.tobytes()), dtype=torch.uint8)
            t = t.view(IMGSZ, IMGSZ, 3).permute(2, 0, 1).float() / 255.0
        return t, p.stem, h, w


def collate(b):
    return torch.stack([x[0] for x in b]), [x[1] for x in b], [x[2] for x in b], [x[3] for x in b]


@torch.no_grad()
def run_inference(weights, img_files, stem_to_id, device="cuda:0", batch=16):
    from transformers import AutoImageProcessor

    device = torch.device(device)
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = build_model(device)
    ck = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    loader = DataLoader(ImageOnly(img_files), batch_size=batch, num_workers=8,
                        collate_fn=collate, pin_memory=True)
    dets = []
    for px, stems, hs, ws in loader:
        px = px.to(device, non_blocking=True)
        with torch.autocast("cuda"):
            out = model(pixel_values=px)
        tgt = torch.tensor(list(zip(hs, ws)), device=device)
        res = proc.post_process_object_detection(out, threshold=CONF, target_sizes=tgt)
        for r, s in zip(res, stems):
            if s not in stem_to_id:
                continue
            iid = stem_to_id[s]
            keep = r["scores"].argsort(descending=True)[:MAX_DET]
            boxes = r["boxes"][keep].float().cpu().numpy()
            scores = r["scores"][keep].float().cpu().numpy()
            labels = r["labels"][keep].cpu().numpy()
            for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
                dets.append({
                    "image_id": iid,
                    "category_id": int(lb),
                    "bbox": [round(float(x1), 3), round(float(y1), 3),
                             round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
                    "score": round(float(sc), 6),
                })
    return dets

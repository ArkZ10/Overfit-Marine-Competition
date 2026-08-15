"""DEIM checkpoint -> shared 12_ensemble dump format (called by dump_preds.py).

Uses DEIM's own YAMLConfig/model/postprocessor so inference matches training
exactly. Outputs absolute-pixel xywh boxes with NAMR33 category ids 0-33.
"""
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CONFIG = HERE / "configs" / "deim_dfine" / "deim_hgnetv2_l_marine.yml"
IMGSZ = 640
CONF = 0.001
MAX_DET = 300


class ImageOnly(Dataset):
    def __init__(self, paths, imgsz=IMGSZ):
        self.paths = [Path(p) for p in paths]
        self.imgsz = imgsz

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = self.imgsz
            r = im.resize((s, s), Image.BILINEAR)
            t = torch.frombuffer(bytearray(r.tobytes()), dtype=torch.uint8)
            t = t.view(s, s, 3).permute(2, 0, 1).float() / 255.0
        return t, p.stem, h, w


def collate(b):
    return (torch.stack([x[0] for x in b]), [x[1] for x in b],
            [x[2] for x in b], [x[3] for x in b])


@torch.no_grad()
def run_inference(weights, img_files, stem_to_id, device="cuda:0", batch=16,
                  config=None, imgsz=IMGSZ):
    from engine.core import YAMLConfig

    device = torch.device(device)
    # HGNetv2.__init__ tries to fetch ImageNet backbone weights and, on failure,
    # its error handler calls torch.distributed.get_rank() - which raises outside
    # a process group and masks the real error. Irrelevant here: our fine-tuned
    # checkpoint overwrites the whole backbone, so skip that fetch entirely.
    # DEIM bakes positional embeddings from eval_spatial_size at build time, so a
    # multi-scale TTA pass must rebuild the model at that scale, else the encoder
    # throws "size of tensor a (289) must match tensor b (400)".
    overrides = {"HGNetv2": {"pretrained": False}}
    if imgsz != IMGSZ:
        overrides["eval_spatial_size"] = [imgsz, imgsz]
    cfg = YAMLConfig(str(config or CONFIG), **overrides)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    state = ck.get("ema", {}).get("module", ck.get("model"))  # EMA weights when present
    if imgsz != IMGSZ:
        # decoder.anchors / decoder.valid_mask are BUFFERS sized from eval_spatial_size,
        # not learned weights - the freshly built model already has correct ones for this
        # scale, so drop the checkpoint's 640-sized copies rather than force a mismatch.
        state = {k: v for k, v in state.items()
                 if k not in ("decoder.anchors", "decoder.valid_mask")}
        cfg.model.load_state_dict(state, strict=False)
    else:
        cfg.model.load_state_dict(state)
    model = cfg.model.deploy().to(device).eval()
    post = cfg.postprocessor.deploy().to(device).eval()

    loader = DataLoader(ImageOnly(img_files, imgsz), batch_size=batch, num_workers=8,
                        collate_fn=collate, pin_memory=True)
    dets = []
    for px, stems, hs, ws in loader:
        px = px.to(device, non_blocking=True)
        # postprocessor rescales normalised boxes onto the ORIGINAL image size
        orig = torch.tensor([[w, h] for w, h in zip(ws, hs)], device=device)
        with torch.autocast("cuda"):
            out = model(px)
        labels, boxes, scores = post(out, orig)
        for lb, bx, sc, s in zip(labels, boxes, scores, stems):
            if s not in stem_to_id:
                continue
            iid = stem_to_id[s]
            keep = sc.argsort(descending=True)[:MAX_DET]
            lb, bx, sc = lb[keep], bx[keep], sc[keep]
            m = sc >= CONF
            for l_, b_, s_ in zip(lb[m].cpu().numpy(), bx[m].float().cpu().numpy(),
                                  sc[m].float().cpu().numpy()):
                x1, y1, x2, y2 = b_
                dets.append({
                    "image_id": iid,
                    "category_id": int(l_),
                    "bbox": [round(float(x1), 3), round(float(y1), 3),
                             round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
                    "score": round(float(s_), 6),
                })
    return dets

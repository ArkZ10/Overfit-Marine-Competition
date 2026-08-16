#!/usr/bin/env python3
"""Inference adapter for detector H (RF-DETR-L) -> the shared dump format.

Called by 12_ensemble/dump_preds.py --model-type rfdetr. Boxes come back as
absolute-pixel xyxy in ORIGINAL image coords, so the xyxy->xywh conversion is ours.

CLASS IDS ARE ALREADY 0..33 -- NO OFFSET. Training uses an identity cat_id -> label
map (build_dataset.py asserts this) and `predict()` on a trained checkpoint returns
class_id in 0..33 directly.

Do not "fix" this by subtracting 1. An earlier version did, based on a 2-epoch spike
checkpoint that reported 1..34; on that model GT agreement was 0/54 vs 2/54, i.e. pure
noise, and the -1 was an over-read of it. Re-measured on the TRAINED checkpoint over
298 IoU-matched val boxes: raw class_id agrees with GT **72.8%**, class_id-1 agrees
**2.0%**. The range assertion below is what caught the mistake -- keep it.

BACKGROUND SLOT: at the dump threshold (conf 0.001) the head also emits class_id == 34,
an extra no-object slot that never appears above ~0.4. Those are not predictions of any
real class and are dropped. The assertion now only fires for ids outside 0..34, i.e. a
genuine convention change.
"""
from pathlib import Path

CONF_THR = 0.001
MAX_DET = 300
RESOLUTION = 640
NC = 34
CLASS_ID_OFFSET = 0   # verified: predict() is already 0-indexed on trained ckpts


def run_inference(weights, img_files, stem_to_id, conf=CONF_THR, max_det=MAX_DET):
    from PIL import Image
    from rfdetr import RFDETRLarge

    model = RFDETRLarge(pretrain_weights=str(weights), resolution=RESOLUTION)
    model.optimize_for_inference()

    dets = []
    n_bg = [0]
    for k, p in enumerate(img_files):
        stem = Path(p).stem
        if stem not in stem_to_id:
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            d = model.predict(im, threshold=conf)
        xyxy = d.xyxy
        if len(d.class_id) and not (0 <= int(d.class_id.min())
                                    and int(d.class_id.max()) <= NC):
            raise SystemExit(
                f"rfdetr class_id outside the expected 0..{NC} range "
                f"({int(d.class_id.min())}..{int(d.class_id.max())}) - the convention "
                f"changed, re-verify against GT before trusting this dump")
        order = (-d.confidence).argsort()[:max_det]
        for i in order:
            cls = int(d.class_id[i]) - CLASS_ID_OFFSET
            if cls >= NC:            # the no-object slot, not a real class
                n_bg[0] += 1
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            dets.append({
                "image_id": stem_to_id[stem],
                "category_id": cls,
                "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                "score": round(float(d.confidence[i]), 6),
            })
        if (k + 1) % 250 == 0:
            print(f"    {k+1}/{len(img_files)} images, {len(dets)} dets", flush=True)
    print(f"  dropped {n_bg[0]} no-object-slot (class {NC}) predictions")
    return dets

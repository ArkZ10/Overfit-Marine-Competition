#!/usr/bin/env python3
"""Test-time augmentation: 3 scales x {none, hflip} per member -> intra-member WBF.

  python3 tta_dump.py --member E            # one member
  python3 tta_dump.py --member all          # all four

Flips are done by materialising a mirrored copy of the val images ONCE, so no
flip handling has to be duplicated inside four different predictors. Every
predictor already maps boxes back to original-image pixel coords, so:
  * scale needs no correction on the output
  * hflip needs x' = W - (x + w)

Output: preds/<name>_tta.val.json  (same shared dump format, fusable as normal)
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image

from paths12 import CONF_THR, GT_VAL_JSON, MAX_DET, PREDS_DIR, SEED  # noqa: I001
from wbf_fuse import fuse, load_dims

SCALES = (544, 640, 768)          # ~0.85, 1.0, 1.2 x 640, all multiples of 32
INTRA_IOU = 0.65                  # swept on member A: 0.65 peaks (0.6410 vs 0.6386@0.55, 0.6387@0.80)
INTRA_SKIP = 0.001
VAL_DIR = Path("/root/Overfit-Marine-Competition/MDImageDataset/yolo_split/images/val")
FLIP_DIR = Path("/tmp/claude-0/-root/fc668498-8e4a-4786-b529-4ec1fa0c2799/scratchpad/val_hflip")

MEMBERS = {
    "A": ("y11m_control", "yolo",
          "../11_improvements/runs/y11m_control/weights/best.pt"),
    "B": ("rtdetr_l", "rtdetr", "runs/rtdetr_l/weights/best.pt"),
    "E": ("deim_dfine_l", "deim",
          "../14_deim/runs/deim_dfine_l/best_stg1.pth"),
    "F": ("rtmdet_l", "rtmdet",
          "../15_rtmdet/runs/rtmdet_l/best_coco_bbox_mAP_50_epoch_40.pth"),
}


def make_flipped(paths):
    FLIP_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in paths:
        out = FLIP_DIR / p.name
        if out.exists():
            continue
        with Image.open(p) as im:
            im.convert("RGB").transpose(Image.FLIP_LEFT_RIGHT).save(out, quality=95)
        n += 1
    print(f"  mirrored images ready in {FLIP_DIR} ({n} newly written)")


def predict(kind, weights, img_files, stem_to_id, imgsz):
    if kind in ("yolo", "rtdetr"):
        from ultralytics import RTDETR, YOLO
        model = (RTDETR if kind == "rtdetr" else YOLO)(str(weights))
        dets = []
        results = model.predict(source=str(img_files[0].parent), imgsz=imgsz,
                                conf=CONF_THR, iou=0.7, max_det=MAX_DET, device="0",
                                verbose=False, save=False, stream=True)
        for r in results:
            stem = Path(r.path).stem
            if stem not in stem_to_id or r.boxes is None or len(r.boxes) == 0:
                continue
            iid = stem_to_id[stem]
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
                dets.append({"image_id": iid, "category_id": int(c),
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             "score": float(s)})
        return dets
    if kind == "deim":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "14_deim"))
        from predict_deim import run_inference
        return run_inference(weights, img_files, stem_to_id, imgsz=imgsz)
    if kind == "rtmdet":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "15_rtmdet"))
        from predict_rtmdet import run_inference
        return run_inference(weights, img_files, stem_to_id, imgsz=imgsz)
    raise ValueError(kind)


def unflip(dets, dims):
    """Mirror boxes produced on a hflipped image back to original coords."""
    out = []
    for d in dets:
        w_img, _ = dims[d["image_id"]]
        x, y, bw, bh = d["bbox"]
        out.append({**d, "bbox": [w_img - (x + bw), y, bw, bh]})
    return out


def run_member(key, dims, stem_to_id, val_paths, flip_paths, keep_passes=False, intra_iou=INTRA_IOU):
    name, kind, weights = MEMBERS[key]
    weights = str((Path(__file__).resolve().parent / weights).resolve())
    passes = []
    for imgsz in SCALES:
        for flipped in (False, True):
            t0 = time.time()
            files = flip_paths if flipped else val_paths
            dets = predict(kind, weights, files, stem_to_id, imgsz)
            if flipped:
                dets = unflip(dets, dims)
            tag = f"{imgsz}{'_flip' if flipped else ''}"
            p = PREDS_DIR / f"_tta_{name}_{tag}.val.json"
            p.write_text(json.dumps(dets))
            passes.append(str(p))
            print(f"    {name} @ {tag}: {len(dets)} dets ({time.time()-t0:.0f}s)", flush=True)

    fused = fuse(passes, dims, intra_iou, INTRA_SKIP, None, "none")
    out = PREDS_DIR / f"{name}_tta.val.json"
    out.write_text(json.dumps(fused))
    (PREDS_DIR / f"{name}_tta.val.meta.json").write_text(json.dumps(
        {"member": key, "name": name, "scales": SCALES, "flips": [False, True],
         "intra_iou": intra_iou, "intra_skip": INTRA_SKIP, "n_fused": len(fused)}, indent=2))
    print(f"  -> {out.name}: {len(fused)} boxes after intra-member WBF")
    if not keep_passes:
        for p in passes:
            Path(p).unlink()
    return out, passes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--member", required=True, help="A|B|E|F|all")
    ap.add_argument("--keep-passes", action="store_true")
    ap.add_argument("--intra-iou", type=float, default=INTRA_IOU)
    a = ap.parse_args()

    gt = json.loads(GT_VAL_JSON.read_text())
    stem_to_id = {Path(i["file_name"]).stem: i["id"] for i in gt["images"]}
    dims = load_dims(GT_VAL_JSON)
    val_paths = sorted(VAL_DIR.glob("*.jpg"), key=lambda p: p.stem)
    make_flipped(val_paths)
    flip_paths = sorted(FLIP_DIR.glob("*.jpg"), key=lambda p: p.stem)
    assert len(flip_paths) == len(val_paths), "mirrored set incomplete"

    keys = list(MEMBERS) if a.member == "all" else [a.member]
    for k in keys:
        print(f"\n=== member {k} ({MEMBERS[k][0]}) ===", flush=True)
        run_member(k, dims, stem_to_id, val_paths, flip_paths, a.keep_passes, a.intra_iou)

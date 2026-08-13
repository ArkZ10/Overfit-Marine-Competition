#!/usr/bin/env python3
"""Build COCO-format jsons for DEIM from our YOLO labels + the official split lists.

DEIM's CocoDetection with remap_mscoco_category=False uses the raw `category_id`
as the class index, so categories are written as ids 0-33 (matching NAMR33 and
the organisers' own train_label.json).

The val json reuses the image ids from 12_ensemble/preds/gt_val_namr33.json so
DEIM predictions line up with every other dump in the ensemble.

  python3 build_coco_jsons.py
"""
import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE / "10_diagnostics"))
from common import CLASS_NAMES  # noqa: E402

LISTS = CODE / "13_dfine" / "lists"
GT_VAL = CODE / "12_ensemble" / "preds" / "gt_val_namr33.json"
OUT = HERE / "data"
NC = 34


def label_path(img: Path) -> Path:
    return Path(str(img).replace("/images/", "/labels/", 1)).with_suffix(".txt")


def build(list_file: Path, out_json: Path, id_map=None):
    paths = [Path(p) for p in list_file.read_text().splitlines() if p.strip()]
    images, annotations = [], []
    ann_id = 0
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            w, h = im.size
        img_id = id_map[p.stem] if id_map else i
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lp = label_path(p)
        if not lp.exists():
            continue
        for line in lp.read_text().splitlines():
            q = line.split()
            if len(q) < 5:
                continue
            c = int(float(q[0]))
            cx, cy, bw, bh = (float(v) for v in q[1:5])
            x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
            aw, ah = bw * w, bh * h
            x, y = max(0.0, x), max(0.0, y)
            aw, ah = min(aw, w - x), min(ah, h - y)
            if aw < 1 or ah < 1:
                continue
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": c,
                "bbox": [round(x, 2), round(y, 2), round(aw, 2), round(ah, 2)],
                "area": round(aw * ah, 2), "iscrowd": 0,
            })
            ann_id += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": c, "name": CLASS_NAMES[c], "supercategory": CLASS_NAMES[c]}
                       for c in range(NC)],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco))
    print(f"{out_json.name}: {len(images)} images, {len(annotations)} annotations")
    return coco


if __name__ == "__main__":
    gt = json.loads(GT_VAL.read_text())
    val_ids = {Path(i["file_name"]).stem: i["id"] for i in gt["images"]}

    build(LISTS / "official_train.txt", OUT / "train_official.json")
    v = build(LISTS / "official_val.txt", OUT / "val.json", id_map=val_ids)

    # cross-check the val json against the frozen ensemble GT
    assert len(v["images"]) == len(gt["images"]), "val image count mismatch vs gt_val_namr33"
    assert {i["id"] for i in v["images"]} == {i["id"] for i in gt["images"]}, "val id mismatch"
    print(f"val json image ids match gt_val_namr33.json ({len(v['images'])} images) OK")
    print(f"val annotations: ours {len(v['annotations'])} vs gt {len(gt['annotations'])}")

#!/usr/bin/env python3
"""Build a COCO-shaped images manifest for the test split.

  python3 make_test_manifest.py

wbf_fuse.py needs {id, width, height} to normalise boxes before fusion, and
make_submission.py needs {id, file_name} to write rows. The test split has no GT,
so this writes an images-only json.

CRITICAL: image_id must match dump_preds.dir_manifest exactly -- `sorted(glob("*.jpg"),
key=stem)` then enumerate. If these two disagree, fusion silently mixes boxes from
different photographs.
"""
import json
from pathlib import Path

from PIL import Image

from paths12 import NC, PREDS_DIR

TEST_DIR = Path("/root/Overfit-Marine-Competition/MDImageDataset/test/images")
OUT = PREDS_DIR / "gt_test_manifest.json"


def main():
    files = sorted(TEST_DIR.glob("*.jpg"), key=lambda p: p.stem)
    if not files:
        raise SystemExit(f"no .jpg in {TEST_DIR}")
    images = []
    for i, p in enumerate(files):
        with Image.open(p) as im:
            w, h = im.size
        images.append({"id": i, "file_name": p.name, "width": w, "height": h})
    doc = {
        "images": images,
        "annotations": [],
        "categories": [{"id": c, "name": str(c)} for c in range(NC)],
    }
    OUT.write_text(json.dumps(doc))
    print(f"wrote {OUT}  ({len(images)} images)")
    print(f"  id 0 = {images[0]['file_name']} ({images[0]['width']}x{images[0]['height']})")
    print(f"  id {len(images)-1} = {images[-1]['file_name']}")


if __name__ == "__main__":
    main()

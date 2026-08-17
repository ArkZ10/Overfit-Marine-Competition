#!/usr/bin/env python3
"""Build RF-DETR's directory layout from the leakage-free official split."""

import argparse
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLEAN = ROOT / "MDImageNet_code" / "18_official_rebuild" / "data"
IMAGES = ROOT / "MDImageDataset2" / "train_dataset" / "images"
SRC = {"train": CLEAN / "train.json", "valid": CLEAN / "val.json"}


def check_identity(train_json):
    from rfdetr.datasets.coco import annotated_category_ids, filter_parent_categories

    data = json.loads(train_json.read_text())
    kept = filter_parent_categories(data["categories"], annotated_category_ids(data))
    mapping = {int(category["id"]): label for label, category in enumerate(kept)}
    bad = {category: label for category, label in mapping.items() if category != label}
    if len(kept) != 34 or bad:
        raise SystemExit(f"category map is not identity: kept={len(kept)} shifts={bad}")
    print("category map verified: 34 classes, cat_id == label index")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE / "data_clean"))
    parser.add_argument("--spike", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out + ("_spike" if args.spike else ""))
    if out.exists():
        shutil.rmtree(out)
    check_identity(SRC["train"])

    for split, annotation_path in SRC.items():
        data = json.loads(annotation_path.read_text())
        count = args.spike if split == "train" else max(1, args.spike // 4)
        if args.spike:
            keep = {im["id"] for im in data["images"][:count]}
            data["images"] = [im for im in data["images"] if im["id"] in keep]
            data["annotations"] = [a for a in data["annotations"] if a["image_id"] in keep]
        destination = out / split
        destination.mkdir(parents=True)
        for image in data["images"]:
            source = IMAGES / image["file_name"]
            if not source.is_file():
                raise SystemExit(f"missing image: {source}")
            (destination / image["file_name"]).symlink_to(source)
        (destination / "_annotations.coco.json").write_text(json.dumps(data))
        print(f"{split}: {len(data['images'])} images, {len(data['annotations'])} boxes")

    (out / "test").symlink_to(out / "valid")
    print(f"wrote {out} (test -> valid)")


if __name__ == "__main__":
    main()


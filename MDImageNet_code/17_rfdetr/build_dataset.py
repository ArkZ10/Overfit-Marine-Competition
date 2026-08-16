#!/usr/bin/env python3
"""Build the Roboflow-COCO directory layout RF-DETR expects, by symlinking.

  python3 build_dataset.py            # full
  python3 build_dataset.py --spike 200   # tiny copy for the smoke test

RF-DETR's build_roboflow wants <root>/{train,valid,test}/_annotations.coco.json
with the images beside the json. We already have exactly the right COCO files
from the DEIM work, so this only re-homes them and symlinks the images (no copy
-- the images are ~14 GB).

VERIFIED before writing: all 34 categories are annotated in train, so
rfdetr's filter_parent_categories keeps all 34 and its cat_id -> label map is the
IDENTITY. If a class ever drops to zero train boxes, rfdetr silently re-indexes
every later label and the model trains on shifted classes.
"""
import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEIM = HERE.parent / "14_deim" / "data"
IMG = Path("/root/Overfit-Marine-Competition/MDImageDataset/yolo_split/images")
SRC = {"train": (DEIM / "train_official.json", IMG / "train"),
       "valid": (DEIM / "val.json", IMG / "val")}


def check_identity(train_json):
    from rfdetr.datasets.coco import annotated_category_ids, filter_parent_categories
    d = json.loads(Path(train_json).read_text())
    kept = filter_parent_categories(d["categories"], annotated_category_ids(d))
    m = {int(c["id"]): l for l, c in enumerate(kept)}
    bad = {k: v for k, v in m.items() if k != v}
    if len(kept) != 34 or bad:
        raise SystemExit(f"category map is NOT identity: kept={len(kept)} shifts={bad}")
    print(f"  category map verified: 34 classes, cat_id == label index")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--spike", type=int, default=0, help="keep only N train / N//4 valid images")
    args = ap.parse_args()

    out = Path(args.out + ("_spike" if args.spike else ""))
    if out.exists():
        shutil.rmtree(out)
    check_identity(SRC["train"][0])

    for split, (js, img_dir) in SRC.items():
        d = json.loads(js.read_text())
        n = args.spike if split == "train" else max(1, args.spike // 4)
        if args.spike:
            keep_ids = {im["id"] for im in d["images"][:n]}
            d["images"] = [im for im in d["images"] if im["id"] in keep_ids]
            d["annotations"] = [a for a in d["annotations"] if a["image_id"] in keep_ids]
        dst = out / split
        dst.mkdir(parents=True)
        miss = 0
        for im in d["images"]:
            src = img_dir / im["file_name"]
            if not src.exists():
                miss += 1
                continue
            (dst / im["file_name"]).symlink_to(src)
        if miss:
            raise SystemExit(f"{miss} images missing from {img_dir}")
        (dst / "_annotations.coco.json").write_text(json.dumps(d))
        print(f"  {split:6} {len(d['images']):>6} images, {len(d['annotations']):>6} boxes -> {dst}")

    # rfdetr looks for a test split; point it at valid so eval never crashes
    (out / "test").symlink_to(out / "valid")
    print(f"\nwrote {out}   (test -> valid)")


if __name__ == "__main__":
    main()

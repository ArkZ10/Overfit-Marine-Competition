#!/usr/bin/env python3
"""Build a leakage-free split from the official competition archive.

The official JSON uses string image ids and ``filename``; COCO consumers in
this repository expect integer ids and ``file_name``.  Images include both
.jpg and .jpeg files.  This script normalizes those details and creates:

  data/train.json, data/val.json, data/val_fit.json, data/val_sel.json
  data/{train,val,val_fit,val_sel}.txt
  data/split.json

The split is deterministic iterative multilabel stratification on per-image
class presence (Sechidis et al., 2011): 90% train, 10% validation, followed by
an equal fit/selection division of validation.  Existing marine checkpoints
must not be evaluated on this split because they may have trained on its val
images; E/F/H must be restarted from their public pretrained weights.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "MDImageDataset2" / "train_dataset"
SOURCE_JSON = SOURCE / "train_label.json"
IMAGE_DIR = SOURCE / "images"
OUT = HERE / "data"
SEED = 42
NC = 34


def iterative_split(labels_by_id, ratios, seed):
    """Assign ids to folds using iterative multilabel stratification.

    ``labels_by_id`` contains sets of labels (class presence, not box count).
    The implementation follows the rarest-label-first allocation principle,
    maintaining both per-label and total-image quotas for every fold.
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to one: {ratios}")
    rng = random.Random(seed)
    folds = tuple(ratios)
    ids = set(labels_by_id)
    total_images = len(ids)
    raw_targets = {f: total_images * ratios[f] for f in folds}
    target_images = {f: math.floor(raw_targets[f]) for f in folds}
    remainder = total_images - sum(target_images.values())
    fold_order = {f: i for i, f in enumerate(folds)}
    for f in sorted(folds, key=lambda x: (raw_targets[x] - target_images[x],
                                          -fold_order[x]), reverse=True)[:remainder]:
        target_images[f] += 1
    label_totals = Counter(c for labs in labels_by_id.values() for c in labs)
    want_images = dict(target_images)
    want_labels = {
        f: {c: label_totals[c] * ratios[f] for c in range(NC)} for f in folds
    }
    assigned = {}
    remaining = set(ids)

    while remaining:
        pending = Counter(c for i in remaining for c in labels_by_id[i])
        if not pending:
            break
        rare = min(pending, key=lambda c: (pending[c], c))
        pool = sorted(i for i in remaining if rare in labels_by_id[i])
        rng.shuffle(pool)
        for image_id in pool:
            if image_id not in remaining:
                continue
            available = [f for f in folds if want_images[f] > 0]
            if not available:
                raise RuntimeError("all fold capacities exhausted before assignment completed")
            # Most under quota for the rare class, then across all labels on
            # this image, then for total images. Random is a seeded final tie.
            fold = max(
                available,
                key=lambda f: (
                    want_labels[f][rare],
                    sum(want_labels[f][c] for c in labels_by_id[image_id]),
                    want_images[f],
                    rng.random(),
                ),
            )
            assigned[image_id] = fold
            remaining.remove(image_id)
            want_images[fold] -= 1
            for c in labels_by_id[image_id]:
                want_labels[fold][c] -= 1

    # Images without annotations are balanced only by the image quota.
    tail = sorted(remaining)
    rng.shuffle(tail)
    for image_id in tail:
        available = [f for f in folds if want_images[f] > 0]
        if not available:
            raise RuntimeError("all fold capacities exhausted before tail assignment completed")
        fold = max(available, key=lambda f: (want_images[f], rng.random()))
        assigned[image_id] = fold
        want_images[fold] -= 1

    assert len(assigned) == total_images
    got = Counter(assigned.values())
    assert all(got[f] == target_images[f] for f in folds), (got, target_images)
    return assigned


def subset(coco, keep_ids):
    keep = set(keep_ids)
    return {
        "images": [im for im in coco["images"] if im["id"] in keep],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
        "categories": coco["categories"],
    }


def audit_split(name, coco):
    images = coco["images"]
    annotations = coco["annotations"]
    box_counts = Counter(a["category_id"] for a in annotations)
    image_counts = Counter()
    present = defaultdict(set)
    for a in annotations:
        present[a["image_id"]].add(a["category_id"])
    for labs in present.values():
        image_counts.update(labs)
    missing_classes = sorted(set(range(NC)) - set(box_counts))
    print(
        f"{name:8} {len(images):5} images  {len(annotations):5} boxes  "
        f"classes={len(box_counts):2}"
    )
    if missing_classes:
        print(f"         missing classes: {missing_classes}")
    return box_counts, image_counts


def main():
    if not SOURCE_JSON.exists() or not IMAGE_DIR.is_dir():
        raise SystemExit(
            f"official data is not extracted at {SOURCE}; extract "
            "MDImageDataset2/train_dataset.zip first"
        )

    raw = json.loads(SOURCE_JSON.read_text())
    if set(raw) != {"images", "categories", "annotations"}:
        raise SystemExit(f"unexpected top-level keys: {raw.keys()}")
    if sorted(c["id"] for c in raw["categories"]) != list(range(NC)):
        raise SystemExit("categories must be the identity ids 0..33")
    if not all(isinstance(im["id"], str) for im in raw["images"]):
        raise SystemExit("expected official image ids to be strings")

    # Deterministic integer mapping: lexical order of the official string ids.
    ordered = sorted(raw["images"], key=lambda im: im["id"])
    id_map = {im["id"]: i for i, im in enumerate(ordered)}
    images = []
    ext_counts = Counter()
    missing = []
    for im in ordered:
        filename = im.get("filename")
        if not filename:
            raise SystemExit(f"image {im['id']} lacks the official 'filename' field")
        ext_counts[Path(filename).suffix.lower()] += 1
        if not (IMAGE_DIR / filename).is_file():
            missing.append(filename)
        images.append(
            {
                "id": id_map[im["id"]],
                "file_name": filename,
                "width": int(im["width"]),
                "height": int(im["height"]),
                "official_id": im["id"],
            }
        )
    if missing:
        raise SystemExit(f"{len(missing)} official images missing; first: {missing[:5]}")
    if ext_counts != Counter({".jpg": 15044, ".jpeg": 83}):
        raise SystemExit(f"unexpected extension counts: {dict(ext_counts)}")

    annotations = []
    labels_by_id = {im["id"]: set() for im in images}
    for ann in raw["annotations"]:
        old_id = ann["image_id"]
        if old_id not in id_map:
            raise SystemExit(f"annotation references unknown image_id {old_id}")
        c = int(ann["category_id"])
        if not 0 <= c < NC:
            raise SystemExit(f"category outside 0..33: {c}")
        new_id = id_map[old_id]
        a = dict(ann)
        a["id"] = int(ann["id"])
        a["image_id"] = new_id
        a["category_id"] = c
        a["bbox"] = [float(v) for v in ann["bbox"]]
        a["area"] = float(ann.get("area", a["bbox"][2] * a["bbox"][3]))
        a["iscrowd"] = int(ann.get("iscrowd", 0))
        annotations.append(a)
        labels_by_id[new_id].add(c)

    categories = [
        {
            "id": int(c["id"]),
            "name": c["name"],
            "supercategory": c.get("supercategory", c["name"]),
        }
        for c in sorted(raw["categories"], key=lambda c: c["id"])
    ]
    coco = {"images": images, "annotations": annotations, "categories": categories}

    outer = iterative_split(labels_by_id, {"train": 0.9, "val": 0.1}, SEED)
    train_ids = sorted(i for i, fold in outer.items() if fold == "train")
    val_ids = sorted(i for i, fold in outer.items() if fold == "val")
    val_labels = {i: labels_by_id[i] for i in val_ids}
    inner = iterative_split(val_labels, {"fit": 0.5, "sel": 0.5}, SEED + 1)
    fit_ids = sorted(i for i, fold in inner.items() if fold == "fit")
    sel_ids = sorted(i for i, fold in inner.items() if fold == "sel")

    groups = {
        "train": train_ids,
        "val": val_ids,
        "val_fit": fit_ids,
        "val_sel": sel_ids,
    }
    if set(train_ids) & set(val_ids) or set(fit_ids) & set(sel_ids):
        raise SystemExit("split overlap detected")
    if set(train_ids) | set(val_ids) != set(labels_by_id):
        raise SystemExit("outer split does not cover every image exactly once")
    if set(fit_ids) | set(sel_ids) != set(val_ids):
        raise SystemExit("inner split does not cover validation exactly once")

    OUT.mkdir(parents=True, exist_ok=True)
    audits = {}
    image_by_id = {im["id"]: im for im in images}
    for name, ids in groups.items():
        part = subset(coco, ids)
        (OUT / f"{name}.json").write_text(json.dumps(part))
        (OUT / f"{name}.txt").write_text(
            "".join(str(IMAGE_DIR / image_by_id[i]["file_name"]) + "\n" for i in ids)
        )
        boxes, class_images = audit_split(name, part)
        audits[name] = {
            "images": len(ids),
            "boxes": len(part["annotations"]),
            "boxes_per_class": {str(c): boxes[c] for c in range(NC)},
            "images_per_class": {str(c): class_images[c] for c in range(NC)},
        }

    manifest = {
        "seed": SEED,
        "method": "iterative multilabel stratification on per-image class presence",
        "source": str(SOURCE_JSON),
        "image_dir": str(IMAGE_DIR),
        "extensions": dict(sorted(ext_counts.items())),
        "old_string_id_to_new_int_id": id_map,
        "ids": groups,
        "audit": audits,
    }
    (OUT / "split.json").write_text(json.dumps(manifest, indent=1))
    print(f"extensions: {dict(ext_counts)}")
    print(f"wrote leakage-free official split to {OUT}")


if __name__ == "__main__":
    main()

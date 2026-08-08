"""Shared path constants and label-counting helpers for the diagnostics scripts.

Read-only: nothing here writes into MDImageDataset/ or 09_training/runs/.
"""
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "MDImageNet_code" / "09_training"
DATA_YAML = TRAIN_DIR / "data_NAMR33_split.yaml"
WEIGHTS = TRAIN_DIR / "runs" / "namr33_yolo11n_baseline" / "weights" / "best.pt"

SPLIT_ROOT = (REPO_ROOT / "MDImageDataset" / "yolo_split").resolve()
IMAGES = {
    "train": (SPLIT_ROOT / "images" / "train").resolve(),
    "val": (SPLIT_ROOT / "images" / "val").resolve(),
}
LABELS = {
    "train": (SPLIT_ROOT / "labels" / "train").resolve(),
    "val": (SPLIT_ROOT / "labels" / "val").resolve(),
}

DIAG_DIR = Path(__file__).resolve().parent
NC = 34

CLASS_NAMES = {
    0: "plastic_bottle", 1: "plastic_bottle_cap", 2: "non_pet_food_beverage_container",
    3: "non_food_plastic_bottle", 4: "takeaway_beverage_cup", 5: "straw",
    6: "disposable_tableware", 7: "plastic_bag", 8: "food_wrapper", 9: "metal_can",
    10: "cigarette_butt", 11: "lighter", 12: "drink_carton", 13: "glass_bottle",
    14: "fishing_net_rope", 15: "fishing_buoy_float", 16: "fishing_gear",
    17: "syringe_needle", 18: "toothbrush", 19: "other", 20: "plastic_lid",
    21: "cup", 22: "foam_buoy_float", 23: "cigarette_pack", 24: "textile",
    25: "net_like_item", 26: "anthropogenic_fragment", 27: "disposable_food_container",
    28: "soft_float", 29: "foam_container", 30: "non_pet_food_container",
    31: "non_food_plastic_container", 32: "aluminum_packaging", 33: "fish_trap_and_bait",
}


def resolve_label_files(split: str):
    """Resolve every label .txt symlink in a split to its real path (sorted by stem)."""
    d = LABELS[split]
    return sorted(d.glob("*.txt"), key=lambda p: p.stem)


def count_boxes_and_images(split: str):
    """Return (boxes_per_class, images_per_class) dicts keyed by class_id, for a split.

    boxes_per_class[c]  = total number of annotated boxes of class c
    images_per_class[c] = number of distinct images containing >=1 box of class c
    """
    boxes = defaultdict(int)
    images = defaultdict(int)
    for lbl_path in resolve_label_files(split):
        real_path = lbl_path.resolve()
        seen_classes_this_image = set()
        with open(real_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls = int(line.split()[0])
                boxes[cls] += 1
                seen_classes_this_image.add(cls)
        for cls in seen_classes_this_image:
            images[cls] += 1
    return boxes, images


def resolve_image_files(split: str):
    d = IMAGES[split]
    return sorted(d.glob("*.jpg"), key=lambda p: p.stem)

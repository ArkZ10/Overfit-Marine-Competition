"""Shared path constants for Phase 1 / Phase 2 improvement work.

Read-only on MDImageDataset/ and 09_training/runs/. Everything this package
writes lands under 11_improvements/.
"""
import sys
from pathlib import Path

IMPROVE_DIR = Path(__file__).resolve().parent
CODE_ROOT = IMPROVE_DIR.parent
REPO_ROOT = CODE_ROOT.parent

DIAG_DIR = CODE_ROOT / "10_diagnostics"
TRAIN09_DIR = CODE_ROOT / "09_training"

# reuse 10_diagnostics helpers (CLASS_NAMES, COCO conversion) rather than re-deriving them
if str(DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(DIAG_DIR))

SPLIT_ROOT = REPO_ROOT / "MDImageDataset" / "yolo_split"
SRC_IMAGES = {s: SPLIT_ROOT / "images" / s for s in ("train", "val")}
SRC_LABELS = {s: SPLIT_ROOT / "labels" / s for s in ("train", "val")}

BASELINE_N_WEIGHTS = TRAIN09_DIR / "runs" / "namr33_yolo11n_baseline" / "weights" / "best.pt"

DATA_DIR = IMPROVE_DIR / "data"          # per-variant symlink trees (keeps .cache out of MDImageDataset/)
RFS_DIR = IMPROVE_DIR / "rfs"            # generated repeat lists + factor tables
RUNS_DIR = IMPROVE_DIR / "runs"
SCORES_DIR = IMPROVE_DIR / "scores"      # per-model pycocotools score dumps

NC = 34
SEED = 42

# classes called out as known failures in the diagnostics pass
FOCUS_CLASSES = [30, 31, 32, 26, 19]


def variant_tree(variant: str) -> Path:
    """Root of the images/labels symlink tree for a training variant."""
    return DATA_DIR / variant


def ensure_variant_tree(variant: str) -> Path:
    """Create data/<variant>/{images,labels}/{train,val} as directory symlinks.

    Ultralytics derives label paths by replacing '/images/' with '/labels/' and
    writes its label .cache next to the label dir, so each variant needs its own
    tree to avoid (a) writing into read-only MDImageDataset/ and (b) two variants
    fighting over one cache file.
    """
    root = variant_tree(variant)
    for kind, src_map in (("images", SRC_IMAGES), ("labels", SRC_LABELS)):
        (root / kind).mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            link = root / kind / split
            target = src_map[split]
            if link.is_symlink() or link.exists():
                continue
            link.symlink_to(target, target_is_directory=True)
    return root

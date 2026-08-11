"""Shared path constants for the 12_ensemble pipeline.

Read-only on MDImageDataset/ and every pre-existing runs/ directory. All outputs
land under 12_ensemble/. sys.path hooks expose the reused helpers from
10_diagnostics, 11_improvements, and 06_evaluation.
"""
import sys
from pathlib import Path

ENS_DIR = Path(__file__).resolve().parent
CODE_ROOT = ENS_DIR.parent
REPO_ROOT = CODE_ROOT.parent

DIAG_DIR = CODE_ROOT / "10_diagnostics"
IMPROVE_DIR = CODE_ROOT / "11_improvements"
EVAL_DIR = CODE_ROOT / "06_evaluation"

for p in (DIAG_DIR, IMPROVE_DIR, EVAL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PREDS_DIR = ENS_DIR / "preds"
SCORES_DIR = ENS_DIR / "scores"
RUNS_DIR = ENS_DIR / "runs"
DATA_DIR = ENS_DIR / "data"
CROPS_DIR = ENS_DIR / "crops"

GT_VAL_JSON = PREDS_DIR / "gt_val_namr33.json"

# reused inputs
DETECTOR_A_WEIGHTS = IMPROVE_DIR / "runs" / "y11m_control" / "weights" / "best.pt"
DATA_CONTROL_YAML = IMPROVE_DIR / "data_control.yaml"
TRAIN_LIST = IMPROVE_DIR / "rfs" / "train_control.txt"
VAL_LIST = IMPROVE_DIR / "rfs" / "val_control.txt"
CROSSWALK_CSV = CODE_ROOT / "crosswalk_to_icc19.csv"
ICC19_YAML = REPO_ROOT / "MDImageDataset" / "data_TaiwanIcc19.yaml"

NC = 34
NC_ICC19 = 20
SEED = 42

# pinned inference protocol shared by all detectors
CONF_THR = 0.001
MAX_DET = 300
IMGSZ = 640

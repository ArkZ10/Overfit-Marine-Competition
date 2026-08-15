#!/usr/bin/env python3
"""Live status for a DEIM run.

  python3 watch_deim.py --run deim_dfine_l_cp        # the Copy-Paste model
  python3 watch_deim.py --run deim_dfine_l           # the original
  python3 watch_deim.py --run deim_dfine_l_cp -f     # refresh every 60s

DEIM appends one JSON line per epoch to <run>/log.txt; test_coco_eval_bbox holds
the 12 COCO stats, index 1 = AP@0.50 (the competition metric), index 0 = AP@0.50:0.95.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOTAL_EPOCHS = 32

BARS = [
    ("plain D-FINE-L (detector D)", 0.5755),
    ("RT-DETR-l (detector B)", 0.6684),
    ("RTMDet-L (detector F)", 0.6518),
    ("DEIM without Copy-Paste (E)  <- the one to beat", 0.7069),
    ("A+B+E+F fused", 0.7479),
    ("fused + rescorer  <- pipeline best", 0.7684),
]


def rows(run):
    log = HERE / "runs" / run / "log.txt"
    if not log.exists():
        return []
    out = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = d.get("test_coco_eval_bbox") or []
        out.append({"epoch": d.get("epoch"),
                    "ap50": s[1] if len(s) > 1 else float("nan"),
                    "ap": s[0] if s else float("nan"),
                    "loss": d.get("train_loss", float("nan"))})
    return out


def alive(run):
    cfg = "deim_hgnetv2_l_marine_cp.yml" if run.endswith("_cp") else "deim_hgnetv2_l_marine.yml"
    out = subprocess.run(["pgrep", "-af", "train.py"], capture_output=True, text=True).stdout
    return any(cfg in ln for ln in out.splitlines())


def current_iter(run):
    stdout = HERE / "runs" / ("deim_cp.log" if run.endswith("_cp") else f"{run}.log")
    if not stdout.exists():
        return ""
    tail = stdout.read_text(errors="ignore")[-6000:].replace("\r", "\n").splitlines()
    for ln in reversed(tail):
        if "Epoch: [" in ln:
            return ln.strip()[:105]
    return ""


def show(run, n):
    r = rows(run)
    running = alive(run)
    print("=" * 74)
    print(f"DEIM run '{run}'   running={running}   epochs done: {len(r)}/{TOTAL_EPOCHS}")
    if not r:
        print("  no completed epoch yet")
        cur = current_iter(run)
        if cur:
            print(f"  in progress: {cur}")
        print("=" * 74)
        return
    print(f"  {'ep':>4} {'AP50':>8} {'AP50-95':>9} {'train_loss':>11}")
    for d in r[-n:]:
        print(f"  {d['epoch']:>4} {d['ap50']:>8.4f} {d['ap']:>9.4f} {d['loss']:>11.4f}")
    best = max(r, key=lambda d: d["ap50"])
    print(f"  best: AP50={best['ap50']:.4f} @ epoch {best['epoch']}")
    cur = current_iter(run)
    if running and cur:
        print(f"  in progress: {cur}")
    print("-" * 74)
    for label, v in BARS:
        mark = "PASSED" if best["ap50"] > v else f"{best['ap50'] - v:+.4f}"
        print(f"  vs {label:<44} {v:.4f}  {mark}")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="deim_dfine_l_cp",
                    help="run dir under runs/ (default: the Copy-Paste model)")
    ap.add_argument("-n", type=int, default=12, help="epochs of history to show")
    ap.add_argument("-f", action="store_true", help="follow: refresh every 60s")
    a = ap.parse_args()
    if not a.f:
        show(a.run, a.n)
    else:
        while True:
            print("\033[2J\033[H", end="")
            show(a.run, a.n)
            if not alive(a.run) and rows(a.run):
                print("training finished")
                break
            time.sleep(60)

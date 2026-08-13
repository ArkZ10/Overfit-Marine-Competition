#!/usr/bin/env python3
"""Live status for the DEIM-D-FINE-L run.

  python3 watch_deim.py            # snapshot
  python3 watch_deim.py -n 20      # more history
  python3 watch_deim.py -f         # refresh every 60s until training ends

DEIM appends one JSON line per epoch to <output_dir>/log.txt; test_coco_eval_bbox
holds the 12 COCO stats, index 1 = AP@0.50 (the competition metric).
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent / "runs" / "deim_dfine_l"
LOG = RUN / "log.txt"
STDOUT = Path(__file__).resolve().parent / "runs" / "deim_dfine_l.log"
TOTAL_EPOCHS = 32

BARS = [
    ("D-FINE-L solo (detector D)", 0.5755),
    ("best single (RT-DETR-l)", 0.6684),
    ("4-model fused", 0.7187),
    ("fused + rescorer  <- overall best", 0.7506),
]


def alive():
    return bool(subprocess.run(["pgrep", "-f", "deim_hgnetv2_l_marine"],
                               capture_output=True, text=True).stdout.strip())


def rows():
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        stats = d.get("test_coco_eval_bbox") or []
        out.append({
            "epoch": d.get("epoch"),
            "ap50": stats[1] if len(stats) > 1 else float("nan"),
            "ap": stats[0] if stats else float("nan"),
            "loss": d.get("train_loss", float("nan")),
        })
    return out


def current_iter():
    """Last in-epoch progress line from stdout, so a long epoch doesn't look stalled."""
    if not STDOUT.exists():
        return ""
    tail = STDOUT.read_text(errors="ignore")[-4000:].replace("\r", "\n").splitlines()
    for ln in reversed(tail):
        if "Epoch:" in ln or "eta:" in ln:
            return ln.strip()[:110]
    return ""


def show(n):
    running = alive()
    r = rows()
    print("=" * 72)
    print(f"DEIM-D-FINE-L   running={running}   epochs done: {len(r)}/{TOTAL_EPOCHS}")
    if not r:
        print("  no completed epoch yet")
        cur = current_iter()
        if cur:
            print(f"  in progress: {cur}")
        print("=" * 72)
        return
    print(f"  {'ep':>4} {'AP50':>8} {'AP50-95':>9} {'train_loss':>11}")
    for d in r[-n:]:
        print(f"  {d['epoch']:>4} {d['ap50']:>8.4f} {d['ap']:>9.4f} {d['loss']:>11.4f}")
    best = max(r, key=lambda d: d["ap50"])
    print(f"  best: AP50={best['ap50']:.4f} @ epoch {best['epoch']}")
    cur = current_iter()
    if running and cur:
        print(f"  in progress: {cur}")
    print("-" * 72)
    for label, v in BARS:
        mark = "PASSED" if best["ap50"] > v else f"{best['ap50'] - v:+.4f}"
        print(f"  vs {label:<36} {v:.4f}   {mark}")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=10, help="epochs of history to show")
    ap.add_argument("-f", action="store_true", help="follow: refresh every 60s")
    a = ap.parse_args()
    if not a.f:
        show(a.n)
    else:
        while True:
            print("\033[2J\033[H", end="")
            show(a.n)
            if not alive() and rows():
                print("training finished")
                break
            time.sleep(60)

#!/usr/bin/env python3
"""Live status for the RTMDet-L run.

  python3 watch_rtmdet.py          # snapshot
  python3 watch_rtmdet.py -n 20    # more history
  python3 watch_rtmdet.py -f       # refresh every 60s until training ends

mmengine writes per-step records to <work_dir>/<timestamp>/vis_data/scalars.json
(one JSON object per line); validation lines carry coco/bbox_mAP_50.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent / "runs" / "rtmdet_l"
MAX_EPOCHS = 40
BARS = [
    ("D-FINE-L (detector D)", 0.5755),
    ("RT-DETR-l (detector B)", 0.6684),
    ("DEIM-D-FINE-L (E, best single)", 0.7069),
    ("A+B+D+E fused", 0.7402),
    ("fused + rescorer  <- overall best", 0.7647),
]


def scalars():
    files = sorted(RUN.glob("*/vis_data/scalars.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return []
    out = []
    for line in files[-1].read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def alive():
    return bool(subprocess.run(["pgrep", "-f", "rtmdet_l_marine"],
                               capture_output=True, text=True).stdout.strip())


def show(n):
    recs = scalars()
    vals = [r for r in recs if "coco/bbox_mAP_50" in r]
    trains = [r for r in recs if "loss" in r]
    print("=" * 74)
    print(f"RTMDet-L   running={alive()}   validations: {len(vals)}/{MAX_EPOCHS}")
    if trains:
        t = trains[-1]
        print(f"  latest train step {t.get('step','?')}: loss={t.get('loss',float('nan')):.4f} "
              f"lr={t.get('lr',float('nan')):.2e}")
    if not vals:
        print("  no validation yet (first epoch still running)")
        print("=" * 74)
        return
    print(f"  {'ep':>4} {'mAP50':>8} {'mAP50-95':>9}")
    for i, r in enumerate(vals[-n:], start=max(1, len(vals) - n + 1)):
        print(f"  {i:>4} {r['coco/bbox_mAP_50']:>8.4f} {r.get('coco/bbox_mAP', float('nan')):>9.4f}")
    best = max(vals, key=lambda r: r["coco/bbox_mAP_50"])
    print(f"  best: mAP50={best['coco/bbox_mAP_50']:.4f} @ validation #{vals.index(best)+1}")
    print("-" * 74)
    for label, v in BARS:
        mark = "PASSED" if best["coco/bbox_mAP_50"] > v else f"{best['coco/bbox_mAP_50'] - v:+.4f}"
        print(f"  vs {label:<38} {v:.4f}   {mark}")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("-f", action="store_true", help="follow: refresh every 60s")
    a = ap.parse_args()
    if not a.f:
        show(a.n)
    else:
        while True:
            print("\033[2J\033[H", end="")
            show(a.n)
            if not alive() and scalars():
                print("training finished")
                break
            time.sleep(60)

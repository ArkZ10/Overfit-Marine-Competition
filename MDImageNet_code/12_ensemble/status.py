#!/usr/bin/env python3
"""One-glance status for the Phase 1 trainings.

  python3 status.py           # both runs, last 8 epochs each
  python3 status.py -n 20     # more history
"""
import argparse
import csv
import subprocess
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def rows(csv_path):
    if not csv_path.exists():
        return []
    return [{k.strip(): v for k, v in d.items()} for d in csv.DictReader(open(csv_path))]


def alive(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return bool(out.stdout.strip())


def show_rtdetr(n):
    print("=" * 62)
    print(f"RT-DETR-l (GPU 0)   running={alive('train_rtdetr')}")
    r = rows(RUNS / "rtdetr_l" / "results.csv")
    if not r:
        print("  no results.csv yet (still warming up / scanning dataset)")
        return
    print(f"  epochs done: {len(r)}/100")
    print(f"  {'ep':>4} {'mAP50':>8} {'mAP50-95':>9} {'elapsed':>9}")
    for d in r[-n:]:
        print(f"  {d['epoch']:>4} {float(d['metrics/mAP50(B)']):>8.4f} "
              f"{float(d['metrics/mAP50-95(B)']):>9.4f} {float(d['time']) / 3600:>8.2f}h")
    best = max(r, key=lambda d: float(d["metrics/mAP50(B)"]))
    print(f"  best: mAP50={float(best['metrics/mAP50(B)']):.4f} @ epoch {best['epoch']}")


def show_frcnn(n):
    print("=" * 62)
    print(f"Faster R-CNN (GPU 1)  running={alive('train_frcnn')}")
    r = rows(RUNS / "frcnn_r50v2" / "results.csv")
    if not r:
        print("  no results.csv yet (epoch 0 in progress)")
        return
    print(f"  epochs done: {len(r)}/26")
    print(f"  {'ep':>4} {'AP50':>8} {'AP50-95':>9} {'loss':>8} {'secs':>7}")
    for d in r[-n:]:
        print(f"  {d['epoch']:>4} {float(d['ap50']):>8.4f} {float(d['ap50_95']):>9.4f} "
              f"{float(d['train_loss']):>8.4f} {float(d['seconds']):>7.0f}")
    best = max(r, key=lambda d: float(d["ap50"]))
    print(f"  best: AP50={float(best['ap50']):.4f} @ epoch {best['epoch']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=8, help="epochs of history to show")
    a = ap.parse_args()
    show_rtdetr(a.n)
    show_frcnn(a.n)
    print("=" * 62)
    print("reference: detector A (YOLOv11m) val AP50 = 0.6049 (34-class) / 0.6744 (ICC19)")

#!/usr/bin/env python3
"""Per-member score dispersion inside the under-confident fused clusters.

  python3 diag_cluster.py

A cluster's fused score is WBF's weighted MEAN, so a low fused score is equally
consistent with every member hedging (uniform) or with one member being confident
and the rest near zero (dispersed). conf_type='max' takes max/weights.max(), so
the two cases behave completely differently. This measures which one we have,
using TEMPERATURE-CALIBRATED member scores, i.e. what fusion actually sees.

Also breaks the under-confident population down by class, to test whether it is
concentrated in the sparse-annotation classes (26/19/22/28) — if so, a rescorer
trained on the same labels inherits the same bias.
"""
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parent / "10_diagnostics"))
from common import CLASS_NAMES  # noqa: E402

MEMBERS = [("A", "y11m_control"), ("B", "rtdetr_l"), ("E", "deim_dfine_l"), ("F", "rtmdet_l")]
W = np.array([0.6049, 0.6684, 1.4, 0.6518])
FUSE_IOU, MATCH_IOU, CONF = 0.65, 0.5, 0.25
SPARSE = {26, 19, 22, 28}


def calibrated(name):
    T = json.loads((HERE / "scores" / f"{name}.calib.json").read_text())["temperature"]
    out = defaultdict(list)
    for d in json.loads((HERE / "preds" / f"{name}.val.json").read_text()):
        s = min(max(d["score"], 1e-6), 1 - 1e-6)
        s = 1 / (1 + np.exp(-np.log(s / (1 - s)) / T))
        out[(d["image_id"], d["category_id"])].append((d["bbox"], float(s)))
    return T, out


def iou(b, arr):
    if not len(arr):
        return np.zeros(0)
    x1 = np.maximum(b[0], arr[:, 0]); y1 = np.maximum(b[1], arr[:, 1])
    x2 = np.minimum(b[0] + b[2], arr[:, 0] + arr[:, 2])
    y2 = np.minimum(b[1] + b[3], arr[:, 1] + arr[:, 3])
    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return it / (b[2] * b[3] + arr[:, 2] * arr[:, 3] - it + 1e-9)


def main():
    gt = json.loads((HERE / "preds" / "gt_val_namr33.json").read_text())
    md = {}
    for k, name in MEMBERS:
        T, d = calibrated(name)
        md[k] = d
        print(f"  member {k} ({name}) T={T:.3f}")

    for dump in ("wbf_abef.val.json", "wbf_abef_rescored.val.json"):
        fused = json.loads((HERE / "preds" / dump).read_text())
        fd = defaultdict(list)
        for x in fused:
            fd[(x["image_id"], x["category_id"])].append(x)

        bands = defaultdict(lambda: {"hi": 0, "lo": 0})
        rows, no_cluster, no_match = [], 0, 0
        for a in gt["annotations"]:
            key = (a["image_id"], a["category_id"])
            cand = fd.get(key, [])
            arr = np.array([c["bbox"] for c in cand], dtype=float).reshape(-1, 4)
            ii = iou(a["bbox"], arr)
            if not len(ii) or ii.max() < MATCH_IOU:
                no_match += 1
                continue
            best = cand[int(ii.argmax())]
            ms = {}
            for k, _ in MEMBERS:
                mb = md[k].get(key, [])
                if not mb:
                    continue
                marr = np.array([b for b, _ in mb], dtype=float)
                mi = iou(best["bbox"], marr)
                sel = mi >= FUSE_IOU
                if sel.any():
                    ms[k] = max(s for (b, s), t in zip(mb, sel) if t)
            n = len(ms)
            if n == 0:
                no_cluster += 1
            band = bands[n]
            band["hi" if best["score"] >= CONF else "lo"] += 1
            if best["score"] < CONF and n >= 1:
                rows.append({"n": n, "cls": a["category_id"], "fused": best["score"],
                             "scores": ms})

        print(f"\n{'=' * 74}\n{dump}\n{'=' * 74}")
        tot_lo = sum(b["lo"] for b in bands.values())
        print(f"  GT matched to a same-class fused box at IoU>={MATCH_IOU}: "
              f"{sum(b['hi'] + b['lo'] for b in bands.values())} of {len(gt['annotations'])}"
              f"   (unmatched: {no_match})")
        print(f"  {'members':>8} {'conf>=.25':>10} {'conf<.25':>9}")
        for n in range(5):
            print(f"  {n:>8} {bands[n]['hi']:>10} {bands[n]['lo']:>9}")
        print(f"  under-confident total: {tot_lo}")

        if dump.startswith("wbf_abef."):
            for target in (4, 3):
                sub = [r for r in rows if r["n"] == target]
                if not sub:
                    continue
                mx = np.array([max(r["scores"].values()) for r in sub])
                mn = np.array([np.mean(list(r["scores"].values())) for r in sub])
                sd = np.array([np.std(list(r["scores"].values())) for r in sub])
                sim = mx / W.max()          # what conf_type='max' would produce
                print(f"\n  --- {len(sub)} under-confident {target}-member clusters ---")
                print(f"    within-cluster std   : median {np.median(sd):.4f}  p90 {np.percentile(sd, 90):.4f}")
                print(f"    max/mean ratio       : median {np.median(mx / np.maximum(mn, 1e-9)):.2f}"
                      f"  p90 {np.percentile(mx / np.maximum(mn, 1e-9), 90):.2f}")
                print(f"    member calib. scores : max median {np.median(mx):.4f}   mean median {np.median(mn):.4f}")
                arg = Counter(max(r['scores'], key=r['scores'].get) for r in sub)
                print(f"    most-confident member: " +
                      ", ".join(f"{k} {100 * v / len(sub):.0f}%" for k, v in arg.most_common()))
                print(f"    simulated conf_type='max' (max/{W.max()}): median {np.median(sim):.4f}"
                      f"   would cross {CONF}: {int((sim >= CONF).sum())}/{len(sub)}"
                      f" ({100 * (sim >= CONF).mean():.0f}%)")

            print(f"\n  --- class breakdown of the {len([r for r in rows if r['n'] >= 3])} "
                  f"under-confident 3-and-4-member boxes ---")
            cc = Counter(r["cls"] for r in rows if r["n"] >= 3)
            tot = sum(cc.values())
            sp = sum(v for c, v in cc.items() if c in SPARSE)
            for c, v in cc.most_common(10):
                tag = "  <-- sparse-annotation class" if c in SPARSE else ""
                print(f"    {c:>2} {CLASS_NAMES[c]:<32} {v:>4}  ({100 * v / tot:>4.1f}%){tag}")
            print(f"    sparse-annotation classes (26/19/22/28): {sp}/{tot} = {100 * sp / tot:.1f}%")


if __name__ == "__main__":
    main()

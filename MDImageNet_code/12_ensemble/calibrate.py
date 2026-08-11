#!/usr/bin/env python3
"""Fit a per-model temperature on the val split so confidence scores are
comparable across architectures before WBF.

Detections are matched to GT (greedy, per image+class, IoU >= 0.5, descending
score) -> binary correctness labels. T minimizes NLL of sigmoid(logit(s)/T).
T > 1 flattens overconfident models (Faster R-CNN softmax), T < 1 sharpens
under-confident ones (RT-DETR's flat Hungarian-matched scores).

  python3 calibrate.py --dump preds/y11m_control.val.json --name y11m_control

Writes scores/<name>.calib.json: {temperature, ece_before, ece_after, n_matched, bins}.
Test-time fusion reuses the frozen T fitted here.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from paths12 import GT_VAL_JSON, SCORES_DIR  # noqa: I001

EPS = 1e-6
IOU_MATCH = 0.5


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_detections(dets, gt_anns):
    """-> (scores array, correct array) via greedy per-image/per-class matching."""
    gt_by_key = defaultdict(list)
    for ann in gt_anns:
        gt_by_key[(ann["image_id"], ann["category_id"])].append(ann["bbox"])

    scores, correct = [], []
    dets_sorted = sorted(dets, key=lambda d: -d["score"])
    used = defaultdict(set)
    for d in dets_sorted:
        key = (d["image_id"], d["category_id"])
        gts = gt_by_key.get(key, [])
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in used[key]:
                continue
            i = iou_xywh(d["bbox"], g)
            if i > best_iou:
                best_iou, best_j = i, j
        hit = best_iou >= IOU_MATCH
        if hit:
            used[key].add(best_j)
        scores.append(d["score"])
        correct.append(1.0 if hit else 0.0)
    return np.array(scores), np.array(correct)


def apply_temperature(scores, T):
    logits = np.log(np.clip(scores, EPS, 1 - EPS) / np.clip(1 - scores, EPS, 1 - EPS))
    return 1.0 / (1.0 + np.exp(-logits / T))


def nll(scores, correct, T):
    p = np.clip(apply_temperature(scores, T), EPS, 1 - EPS)
    return float(-np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p)))


def fit_temperature(scores, correct):
    """Golden-section search on log T in [1/20, 20] - NLL(T) is unimodal here."""
    lo, hi = np.log(1 / 20), np.log(20.0)
    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(scores, correct, np.exp(c)), nll(scores, correct, np.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(scores, correct, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(scores, correct, np.exp(d))
    return float(np.exp((a + b) / 2))


def ece(scores, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(scores)
    e = 0.0
    detail = []
    for i in range(n_bins):
        m = (scores >= bins[i]) & (scores < bins[i + 1] if i < n_bins - 1 else scores <= bins[i + 1])
        if m.sum() == 0:
            continue
        conf, acc = float(scores[m].mean()), float(correct[m].mean())
        e += (m.sum() / total) * abs(conf - acc)
        detail.append({"bin": [round(float(bins[i]), 3), round(float(bins[i + 1]), 3)],
                       "n": int(m.sum()), "mean_conf": round(conf, 4), "frac_correct": round(acc, 4)})
    return float(e), detail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--gt", default=str(GT_VAL_JSON))
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    dets = json.loads(open(args.dump).read())
    gt = json.loads(open(args.gt).read())

    scores, correct = match_detections(dets, gt["annotations"])
    T = fit_temperature(scores, correct)
    ece_before, bins_before = ece(scores, correct)
    ece_after, bins_after = ece(apply_temperature(scores, T), correct)

    result = {
        "name": args.name,
        "dump": args.dump,
        "temperature": round(T, 4),
        "n_detections": len(scores),
        "n_matched_correct": int(correct.sum()),
        "nll_before": round(nll(scores, correct, 1.0), 5),
        "nll_after": round(nll(scores, correct, T), 5),
        "ece_before": round(ece_before, 5),
        "ece_after": round(ece_after, 5),
        "bins_before": bins_before,
        "bins_after": bins_after,
    }
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out = SCORES_DIR / f"{args.name}.calib.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"{args.name}: T={T:.3f}  NLL {result['nll_before']:.4f} -> {result['nll_after']:.4f}  "
          f"ECE {ece_before:.4f} -> {ece_after:.4f}  ({len(scores)} dets, {int(correct.sum())} correct)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

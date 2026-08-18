#!/usr/bin/env python3
"""Audit clean E/F errors and complementarity on the official validation split.

This is deliberately CPU-only.  It produces machine-readable JSON/CSV plus a concise
Markdown report.  Predictions are assigned one of five mutually exclusive outcomes in
descending confidence order: TP, duplicate, class confusion, localization, unmatched.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENS = ROOT / "MDImageNet_code" / "12_ensemble"
PREDS = ENS / "preds"
SCORES = ENS / "scores"
GT_PATH = HERE / "data" / "val.json"
OUT_JSON = SCORES / "clean_ef_audit.json"
OUT_CSV = SCORES / "clean_ef_per_class_audit.csv"
OUT_MD = HERE / "CLEAN_EF_AUDIT.md"
IOU_TP = 0.5
IOU_LOC = 0.1
CONSENSUS_CONF = 0.05


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax2 = a[:, :2] + a[:, 2:]
    bx2 = b[:, :2] + b[:, 2:]
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(ax2[:, None, :], bx2[None, :, :])
    wh = np.clip(hi - lo, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] * a[:, 3])[:, None]
    ba = (b[:, 2] * b[:, 3])[None, :]
    return inter / (aa + ba - inter + 1e-9)


def grouped(rows):
    out = defaultdict(list)
    for row in rows:
        out[row["image_id"]].append(row)
    return out


def classify_predictions(gt_by_img, det_by_img):
    outcomes = Counter()
    by_class = defaultdict(Counter)
    gt_hits = {}
    records = []
    for image_id in sorted(set(gt_by_img) | set(det_by_img)):
        gs = gt_by_img.get(image_id, [])
        ps = sorted(det_by_img.get(image_id, []), key=lambda x: -x["score"])
        gb = np.asarray([g["bbox"] for g in gs], dtype=float).reshape(-1, 4)
        pb = np.asarray([p["bbox"] for p in ps], dtype=float).reshape(-1, 4)
        ious = iou_matrix(pb, gb)
        claimed = set()
        for i, p in enumerate(ps):
            pc = p["category_id"]
            same = [j for j, g in enumerate(gs) if g["category_id"] == pc]
            same_good = [j for j in same if ious[i, j] >= IOU_TP]
            free = [j for j in same_good if j not in claimed]
            if free:
                j = max(free, key=lambda x: ious[i, x])
                claimed.add(j)
                outcome = "tp"
                gt_hits[(image_id, gs[j]["id"])] = p["score"]
            elif same_good:
                outcome = "duplicate"
            elif len(gs) and float(ious[i].max()) >= IOU_TP:
                outcome = "class_confusion"
            elif same and float(ious[i, same].max()) >= IOU_LOC:
                outcome = "localization"
            else:
                outcome = "unmatched"
            outcomes[outcome] += 1
            by_class[pc][outcome] += 1
            records.append({**p, "outcome": outcome})
    return outcomes, by_class, gt_hits, records


def consensus_unmatched(e_records, f_records):
    e_by = grouped([x for x in e_records if x["outcome"] == "unmatched" and
                    x["score"] >= CONSENSUS_CONF])
    f_by = grouped([x for x in f_records if x["outcome"] == "unmatched" and
                    x["score"] >= CONSENSUS_CONF])
    pairs = []
    for image_id in sorted(set(e_by) & set(f_by)):
        es, fs = e_by[image_id], f_by[image_id]
        eb = np.asarray([x["bbox"] for x in es], dtype=float)
        fb = np.asarray([x["bbox"] for x in fs], dtype=float)
        ious = iou_matrix(eb, fb)
        candidates = []
        for i, e in enumerate(es):
            for j, f in enumerate(fs):
                if e["category_id"] == f["category_id"] and ious[i, j] >= IOU_TP:
                    candidates.append((min(e["score"], f["score"]), float(ious[i, j]), i, j))
        used_e, used_f = set(), set()
        for support, iou, i, j in sorted(candidates, reverse=True):
            if i in used_e or j in used_f:
                continue
            used_e.add(i); used_f.add(j)
            e, f = es[i], fs[j]
            pairs.append({
                "image_id": image_id, "category_id": e["category_id"],
                "e_score": e["score"], "f_score": f["score"], "iou": iou,
                "support": support, "bbox": e["bbox"],
            })
    return sorted(pairs, key=lambda x: -x["support"])


def main():
    gt = json.loads(GT_PATH.read_text())
    names = {c["id"]: c["name"] for c in gt["categories"]}
    gt_by = grouped(gt["annotations"])
    gt_class = {a["id"]: a["category_id"] for a in gt["annotations"]}
    models = {
        "E": json.loads((PREDS / "clean_e.clean_val.json").read_text()),
        "F": json.loads((PREDS / "clean_f.clean_val.json").read_text()),
    }
    results, hits, records = {}, {}, {}
    for model, dets in models.items():
        outcome, per_class, model_hits, model_records = classify_predictions(gt_by, grouped(dets))
        results[model] = {"outcomes": dict(outcome),
                          "per_class": {str(c): dict(per_class[c]) for c in names}}
        hits[model] = model_hits
        records[model] = model_records

    all_gt_keys = {(a["image_id"], a["id"]) for a in gt["annotations"]}
    ekeys, fkeys = set(hits["E"]), set(hits["F"])
    coverage = {
        "gt_boxes": len(all_gt_keys), "E": len(ekeys), "F": len(fkeys),
        "both": len(ekeys & fkeys), "E_only": len(ekeys - fkeys),
        "F_only": len(fkeys - ekeys), "neither": len(all_gt_keys - (ekeys | fkeys)),
        "union": len(ekeys | fkeys),
    }
    consensus = consensus_unmatched(records["E"], records["F"])
    consensus_by_class = Counter(x["category_id"] for x in consensus)

    per_class_rows = []
    gt_counts = Counter(a["category_id"] for a in gt["annotations"])
    for c in sorted(names):
        ckeys = {k for k in all_gt_keys if gt_class[k[1]] == c}
        row = {"class_id": c, "class_name": names[c], "gt": gt_counts[c]}
        for model in ("E", "F"):
            row[f"{model}_tp"] = len(ckeys & set(hits[model]))
            for outcome in ("duplicate", "class_confusion", "localization", "unmatched"):
                row[f"{model}_{outcome}"] = results[model]["per_class"][str(c)].get(outcome, 0)
        row.update({
            "both_tp": len(ckeys & ekeys & fkeys),
            "E_only_tp": len((ckeys & ekeys) - fkeys),
            "F_only_tp": len((ckeys & fkeys) - ekeys),
            "neither_tp": len(ckeys - (ekeys | fkeys)),
            "consensus_unmatched": consensus_by_class[c],
        })
        per_class_rows.append(row)

    payload = {
        "ground_truth": str(GT_PATH), "iou_tp": IOU_TP, "iou_localization_floor": IOU_LOC,
        "min_dump_confidence": 0.001, "consensus_min_each_score": CONSENSUS_CONF,
        "models": results, "gt_coverage": coverage,
        "consensus_unmatched_count": len(consensus),
        "consensus_unmatched_by_class": {str(c): consensus_by_class[c] for c in names},
        "consensus_unmatched": consensus,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_class_rows[0].keys())
        writer.writeheader(); writer.writerows(per_class_rows)

    def pct(n, d): return 100 * n / d if d else 0
    lines = [
        "# Clean E/F audit", "",
        "Official clean validation: 1,513 images, 3,233 boxes. Predictions are evaluated down "
        "to the stored 0.001 floor. These counts diagnose proposal coverage and error type; "
        "they are not a replacement for COCO AP.", "", "## Ground-truth coverage at IoU≥0.5", "",
        "| detector coverage | boxes | percent |", "|---|---:|---:|",
        f"| E | {coverage['E']} | {pct(coverage['E'], coverage['gt_boxes']):.1f}% |",
        f"| F | {coverage['F']} | {pct(coverage['F'], coverage['gt_boxes']):.1f}% |",
        f"| Both | {coverage['both']} | {pct(coverage['both'], coverage['gt_boxes']):.1f}% |",
        f"| E only | {coverage['E_only']} | {pct(coverage['E_only'], coverage['gt_boxes']):.1f}% |",
        f"| F only | {coverage['F_only']} | {pct(coverage['F_only'], coverage['gt_boxes']):.1f}% |",
        f"| Union | {coverage['union']} | {pct(coverage['union'], coverage['gt_boxes']):.1f}% |",
        f"| Neither | {coverage['neither']} | {pct(coverage['neither'], coverage['gt_boxes']):.1f}% |",
        "", "## Prediction outcomes at the 0.001 dump floor", "",
        "| model | TP | duplicate | class confusion | localization | unmatched |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("E", "F"):
        o = results[model]["outcomes"]
        lines.append(f"| {model} | {o.get('tp',0)} | {o.get('duplicate',0)} | "
                     f"{o.get('class_confusion',0)} | {o.get('localization',0)} | "
                     f"{o.get('unmatched',0)} |")
    top = sorted(per_class_rows, key=lambda r: -r["consensus_unmatched"])[:10]
    lines += ["", "## E/F consensus detections unmatched to official GT", "",
              f"At score≥{CONSENSUS_CONF} for both models and same-class IoU≥0.5: "
              f"**{len(consensus)} pairs**. These are review candidates, not proven positives.", "",
              "| class | count |", "|---|---:|"]
    lines += [f"| {r['class_id']} {r['class_name']} | {r['consensus_unmatched']} |" for r in top]
    lines += ["", "## Interpretation guardrails", "",
              "- `unmatched` means unmatched to the official annotation, not necessarily background.",
              "- Consensus boxes must be manually sampled before defining ignore thresholds.",
              "- E/F-only consensus is provisional; clean S will provide the stronger three-model test.",
              "- The confusion outputs at 0.001, 0.05, and 0.25 live beside this report in `12_ensemble/scores/`.", ""]
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    print(json.dumps(coverage, indent=2))
    print(f"consensus unmatched: {len(consensus)}")


if __name__ == "__main__":
    main()

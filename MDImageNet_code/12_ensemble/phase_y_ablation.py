#!/usr/bin/env python3
"""Bounded clean-val_fit member ablation for DEIMv2-X candidate Y."""
import itertools
import json
from pathlib import Path

from score_preds import load_gt, score_namr33
from wbf_fuse import fuse, load_dims

HERE = Path(__file__).resolve().parent
P = HERE / "preds"
S = HERE / "scores"
GT = HERE.parent / "18_official_rebuild" / "data" / "val_fit.json"

MEMBERS = {
    "E": ("clean_e.clean_val_fit.json", 1.1),
    "F": ("clean_f.clean_val_fit.json", 0.9),
    "S": ("clean_s.clean_val_fit.json", 1.0),
    "X": ("deim_dfine_x_clean.clean_val_fit.json", 1.1),
    "Y": ("deimv2_dinov3_x_clean.clean_val_fit.json", 1.2),
}


def ap50(dets, gt):
    return score_namr33(gt, dets)["ap50"]


def main():
    dims = load_dims(GT)
    gt = load_gt(explicit_gt=GT)
    rows = []
    keys = list(MEMBERS)
    subsets = [c for n in range(2, len(keys) + 1)
               for c in itertools.combinations(keys, n) if "Y" in c]
    for sub in subsets:
        paths = [str(P / MEMBERS[k][0]) for k in sub]
        weights = [MEMBERS[k][1] for k in sub]
        dets = fuse(paths, dims, 0.65, 0.001, weights, "temperature", "avg")
        score = ap50(dets, gt)
        rows.append({"members": "".join(sub), "iou": 0.65, "conf_type": "avg",
                     "weights": weights, "ap50": score})
        print(f"{''.join(sub):<6} {score:.6f}", flush=True)
    rows.sort(key=lambda x: -x["ap50"])

    # Only the best three member sets enter the bounded hyperparameter grid.
    top_sets = [tuple(r["members"]) for r in rows[:3]]
    grid = []
    for sub in top_sets:
        paths = [str(P / MEMBERS[k][0]) for k in sub]
        for iou in (0.55, 0.60, 0.65, 0.70, 0.75):
            for conf_type in ("avg", "max"):
                for yw in (0.8, 1.1, 1.4):
                    weights = [yw if k == "Y" else MEMBERS[k][1] for k in sub]
                    dets = fuse(paths, dims, iou, 0.001, weights,
                                "temperature", conf_type)
                    score = ap50(dets, gt)
                    grid.append({"members": "".join(sub), "iou": iou,
                                 "conf_type": conf_type, "weights": weights,
                                 "ap50": score})
                    print(f"grid {''.join(sub):<6} iou={iou:.2f} {conf_type:<3} "
                          f"yw={yw:.1f} {score:.6f}", flush=True)
    grid.sort(key=lambda x: -x["ap50"])
    out = {"fixed_ablation": rows, "grid": grid, "winner": grid[0]}
    (S / "phase_y_wbf_search.json").write_text(json.dumps(out, indent=2))
    print("WINNER", json.dumps(grid[0]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Member-set ablation: fuse every subset and score it, tuned on val_fit.

  python3 ablate_members.py                      # all subsets of size >= 2
  python3 ablate_members.py --min-size 3 --only A B E F H

Why this exists as a standing step, not a one-off: a weak member stops earning its
slot the moment a stronger, more DIVERSE one arrives. C (Faster R-CNN) earned its
place until E (DEIM) landed, then dropping it GAINED +0.002; D (D-FINE) earned its
place until F (RTMDet) landed. Appending a new detector without re-ablating has
twice left a member in that was costing points.

Fusion settings are the frozen ones (temperature normalisation, iou 0.65,
skip 0.001, conf_type avg -- all validated on both halves). Weights are each
member's solo val_fit AP50, with E boosted to 1.4 as in the frozen config.

Reports val_fit AND val_sel, plus the macro restricted to the 26 classes with
>20 val boxes -- the val->test gap showed the all-34 number is inflated ~0.023 by
three classes holding 16 boxes between them.
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from paths12 import GT_VAL_JSON, PREDS_DIR, SCORES_DIR
from wbf_fuse import fuse, load_dims

HERE = Path(__file__).resolve().parent

MEMBERS = {
    "A": ("y11m_control", 0.6165),
    "B": ("rtdetr_l", 0.6764),
    "E": ("deim_dfine_l", 1.4),      # E-heavy, as in the frozen config
    "F": ("rtmdet_l", 0.6608),
    "H": ("rfdetr_l", 0.6662),
}
IOU, SKIP, NORM, CONF_TYPE = 0.65, 0.001, "temperature", "avg"


def score(dump, name, half):
    r = subprocess.run(
        [sys.executable, "score_preds.py", "--dump", dump, "--name", name, "--half", half],
        capture_output=True, text=True, cwd=str(HERE))
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"scoring failed: {dump}")
    return json.loads((SCORES_DIR / f"{name}.json").read_text())


def macro_well_measured(res, split_json, min_boxes=20):
    """Macro AP50 over classes with > min_boxes val boxes -- tracks test far better."""
    pc = res["namr33"]["per_class"]
    per = json.loads(Path(split_json).read_text())["per_class"]
    keep = [c for c in range(34) if per[str(c)]["fit"] + per[str(c)]["sel"] > min_boxes]
    vals = [pc[str(c)]["ap50"] for c in keep if not np.isnan(pc[str(c)]["ap50"])]
    return float(np.mean(vals)), len(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=list(MEMBERS))
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--out", default="scores/member_ablation.json")
    args = ap.parse_args()

    keys = [k for k in MEMBERS if k in args.only]
    dims = load_dims(GT_VAL_JSON)
    split_json = HERE / "data" / "val_split.json"

    subsets = [c for n in range(args.min_size, len(keys) + 1)
               for c in itertools.combinations(keys, n)]
    print(f"{len(subsets)} subsets of {''.join(keys)}   "
          f"(temperature, iou {IOU}, skip {SKIP}, conf_type {CONF_TYPE})\n")
    print(f"  {'members':<8} {'fit':>7} {'sel':>7} {'fit>20box':>10} {'sel>20box':>10}")

    rows = []
    for sub in subsets:
        tag = "".join(sub)
        dumps = [str(PREDS_DIR / f"{MEMBERS[k][0]}.val.json") for k in sub]
        weights = [MEMBERS[k][1] for k in sub]
        out = PREDS_DIR / f"_abl_{tag}.val.json"
        out.write_text(json.dumps(
            fuse(dumps, dims, IOU, SKIP, weights, NORM, CONF_TYPE)))
        r = {"members": tag}
        for half in ("fit", "sel"):
            res = score(str(out), f"_abl_{tag}_{half}", half)
            r[half] = res["namr33"]["ap50"]
            r[f"{half}_wm"], nkeep = macro_well_measured(res, split_json)
        out.unlink()                      # fused dumps are large; keep only scores
        rows.append(r)
        print(f"  {tag:<8} {r['fit']:>7.4f} {r['sel']:>7.4f} "
              f"{r['fit_wm']:>10.4f} {r['sel_wm']:>10.4f}", flush=True)

    rows.sort(key=lambda d: -d["fit"])
    (HERE / args.out).write_text(json.dumps({"iou": IOU, "skip": SKIP,
                                             "conf_type": CONF_TYPE,
                                             "weights": {k: MEMBERS[k][1] for k in keys},
                                             "rows": rows}, indent=1))
    print(f"\n  best on val_fit: {rows[0]['members']} = {rows[0]['fit']:.4f} "
          f"(sel {rows[0]['sel']:.4f})")
    print(f"  frozen ABEF for reference: fit 0.7565  sel 0.7395")
    print(f"  gate: beat ABEF by >0.008 on fit AND hold on sel")
    print(f"\nwrote {HERE / args.out}")


if __name__ == "__main__":
    main()

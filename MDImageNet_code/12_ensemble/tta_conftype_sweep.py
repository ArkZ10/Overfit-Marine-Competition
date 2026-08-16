#!/usr/bin/env python3
"""Re-fuse a member's kept TTA passes under each WBF conf_type, and score solo.

  python3 tta_conftype_sweep.py --member B E

Requires `tta_dump.py --member X --keep-passes` to have left the six per-pass
dumps in preds/_tta_<name>_<tag>.val.json.

Why this is not the same test as the cross-member conf_type sweep: intra-member
TTA fuses 6 passes with unit weights, so weights.sum()=6 and a box found in ONE
pass is cut to x0.167 -- roughly twice the cross-member penalty (x0.301 at
weights.sum()=3.3251). DETRs emit 300 unsuppressed queries, so cross-pass
matching is least stable exactly where the penalty is harshest; that is the
stated cause of E losing -0.277 to TTA. Also the asymmetry runs the other way
here: a STABLE false positive appears in all 6 passes and is untouched by the
penalty, while an unstable TRUE box appears once.

Scores on val_fit and val_sel; solo baselines are B 0.6684, E 0.7069 on full val.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from paths12 import GT_VAL_JSON, PREDS_DIR
from tta_dump import INTRA_IOU, INTRA_SKIP, MEMBERS, SCALES
from wbf_fuse import fuse, load_dims

CONF_TYPES = ("avg", "max", "box_and_model_avg", "absent_model_aware_avg")


def passes_for(name):
    out = []
    for imgsz in SCALES:
        for flipped in (False, True):
            tag = f"{imgsz}{'_flip' if flipped else ''}"
            p = PREDS_DIR / f"_tta_{name}_{tag}.val.json"
            if not p.exists():
                raise SystemExit(f"{p} missing - run tta_dump.py --member ? --keep-passes")
            out.append(str(p))
    return out


def score(dump, name, half):
    r = subprocess.run(
        [sys.executable, "score_preds.py", "--dump", dump, "--name", name, "--half", half],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent))
    for line in r.stdout.splitlines():
        if "NAMR33" in line:
            return float(line.split("=")[1].split("(")[0].strip())
    print(r.stdout, r.stderr)
    raise SystemExit(f"scoring failed for {dump}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", nargs="+", required=True)
    ap.add_argument("--intra-iou", type=float, default=INTRA_IOU)
    args = ap.parse_args()

    dims = load_dims(GT_VAL_JSON)
    results = {}
    for key in args.member:
        name = MEMBERS[key][0]
        ps = passes_for(name)
        print(f"\n=== member {key} ({name}) — {len(ps)} passes, intra_iou {args.intra_iou} ===")
        # the un-augmented member, for reference
        base_fit = score(str(PREDS_DIR / f"{name}.val.json"), f"_base_{name}_fit", "fit")
        base_sel = score(str(PREDS_DIR / f"{name}.val.json"), f"_base_{name}_sel", "sel")
        print(f"  {'no TTA':<26} fit {base_fit:.4f}   sel {base_sel:.4f}")
        results[key] = {"no_tta": {"fit": base_fit, "sel": base_sel}}
        for ct in CONF_TYPES:
            fused = fuse(ps, dims, args.intra_iou, INTRA_SKIP, None, "none", ct)
            out = PREDS_DIR / f"{name}_tta_{ct}.val.json"
            out.write_text(json.dumps(fused))
            f = score(str(out), f"tta_{name}_{ct}_fit", "fit")
            s = score(str(out), f"tta_{name}_{ct}_sel", "sel")
            df, ds = f - base_fit, s - base_sel
            flag = "  <-- beats no-TTA on BOTH" if df > 0 and ds > 0 else ""
            print(f"  TTA conf_type={ct:<22} fit {f:.4f} ({df:+.4f})   "
                  f"sel {s:.4f} ({ds:+.4f}){flag}")
            results[key][ct] = {"fit": f, "sel": s, "d_fit": df, "d_sel": ds}

    # merge into any previous run's results rather than clobbering them, so
    # sweeping members one at a time still leaves one complete record
    out = Path(__file__).resolve().parent / "scores" / "tta_conftype_sweep.json"
    prev = json.loads(out.read_text())["results"] if out.exists() else {}
    prev.update(results)
    out.write_text(json.dumps({"intra_iou": args.intra_iou, "results": prev}, indent=1))
    print(f"\nwrote {out}")
    print("\nGate: must beat no-TTA by >0.008 on fit AND hold the sign on sel.")


if __name__ == "__main__":
    main()

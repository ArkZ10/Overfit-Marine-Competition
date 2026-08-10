#!/usr/bin/env python3
"""Assemble phase1_report.md + phase1_per_class.csv from the scores/ dumps.

Numbers only - no interpretation.

  python3 make_phase1_report.py --rfs-t 0.05
"""
import argparse
import csv
import json
import math

from paths import FOCUS_CLASSES, IMPROVE_DIR, NC, RUNS_DIR, SCORES_DIR  # noqa: I001
from common import CLASS_NAMES  # noqa: E402


def load(name):
    p = SCORES_DIR / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def wall_time(run_name):
    """(seconds, source) - explicit marker first, else last cumulative time in results.csv."""
    run_dir = RUNS_DIR / run_name
    marker = run_dir / "wall_time_seconds.txt"
    if marker.exists():
        return float(marker.read_text().strip()), "wall_time_seconds.txt"
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(open(csv_path)))
        if rows:
            key = next((k for k in rows[-1] if k.strip() == "time"), None)
            if key:
                return float(rows[-1][key]), "results.csv (cumulative)"
    return None, "not found"


def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rfs-t", type=float, required=True)
    a = ap.parse_args()

    tag = f"t{a.rfs_t:g}"
    models = [
        ("baseline_y11n", "baseline YOLOv11n", None),
        ("y11m_control", "YOLOv11m control", "y11m_control"),
        (f"y11m_rfs_{tag}", f"YOLOv11m RFS ({tag})", f"y11m_rfs_{tag}"),
    ]

    loaded = []
    for score_name, label, run_name in models:
        d = load(score_name)
        if d is None:
            print(f"WARNING: scores/{score_name}.json missing - column will read n/a")
        loaded.append((score_name, label, run_name, d))

    lines = ["# Phase 1 - RFS resampling on YOLOv11m", ""]
    lines.append("All AP numbers are pycocotools COCOeval (bbox) on the val split (1,661 images).")
    lines.append("")

    lines.append("## AP@0.50 overall")
    lines.append("")
    lines.append("| model | pycocotools AP@0.50 | pycocotools AP@0.50:0.95 |")
    lines.append("|---|---|---|")
    for _, label, _, d in loaded:
        lines.append(f"| {label} | {fmt(d['ap50']) if d else 'n/a'} | {fmt(d['ap50_95']) if d else 'n/a'} |")
    lines.append("")

    lines.append("## Per-class AP@0.50 - focus classes")
    lines.append("")
    header = "| class_id | class_name | " + " | ".join(l for _, l, _, _ in loaded) + " |"
    lines.append(header)
    lines.append("|---" * (2 + len(loaded)) + "|")
    for c in FOCUS_CLASSES:
        cells = [fmt(d["per_class"][str(c)]["ap50"]) if d else "n/a" for _, _, _, d in loaded]
        lines.append(f"| {c} | {CLASS_NAMES[c]} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Training wall-time")
    lines.append("")
    lines.append("| run | seconds | hours | source |")
    lines.append("|---|---|---|---|")
    for _, label, run_name, _ in loaded:
        if run_name is None:
            lines.append(f"| {label} | n/a (pre-existing) | n/a | n/a |")
            continue
        secs, src = wall_time(run_name)
        if secs is None:
            lines.append(f"| {label} | n/a | n/a | {src} |")
        else:
            lines.append(f"| {label} | {secs:.0f} | {secs / 3600:.3f} | {src} |")
    lines.append("")
    lines.append("Full per-class table: `phase1_per_class.csv`")
    lines.append("")

    report = "\n".join(lines)
    (IMPROVE_DIR / "phase1_report.md").write_text(report)

    with open(IMPROVE_DIR / "phase1_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["class_id", "class_name"]
        for score_name, _, _, _ in loaded:
            cols += [f"{score_name}_ap50", f"{score_name}_ap50_95"]
        w.writerow(cols)
        for c in range(NC):
            row = [c, CLASS_NAMES[c]]
            for _, _, _, d in loaded:
                if d is None:
                    row += ["", ""]
                else:
                    pc = d["per_class"][str(c)]
                    row += [round(pc["ap50"], 6), round(pc["ap50_95"], 6)]
            w.writerow(row)

    print(report)
    print(f"wrote {IMPROVE_DIR / 'phase1_report.md'}")
    print(f"wrote {IMPROVE_DIR / 'phase1_per_class.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assemble phase2_report.md. Numbers only - no interpretation.

  python3 make_phase2_report.py
"""
import csv
import json
import math

from paths import FOCUS_CLASSES, IMPROVE_DIR, NC, RUNS_DIR, SCORES_DIR  # noqa: I001
from common import CLASS_NAMES  # noqa: E402

WINNER = ("y11m_control", "YOLOv11m control (Phase 1 winner)")
DINO = ("y11m_dino_p0p3", "YOLOv11m + DINOv3 P0+P3")


def load(name, suffix=""):
    p = SCORES_DIR / f"{name}{suffix}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def wall_time(run_name):
    marker = RUNS_DIR / run_name / "wall_time_seconds.txt"
    if marker.exists():
        return float(marker.read_text().strip())
    csv_path = RUNS_DIR / run_name / "results.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(open(csv_path)))
        if rows:
            key = next((k for k in rows[-1] if k.strip() == "time"), None)
            if key:
                return float(rows[-1][key])
    return None


def main():
    models = [WINNER, DINO]
    scores = {n: load(n) for n, _ in models}
    costs = {n: load(n, "_cost") for n, _ in models}
    for n, _ in models:
        if scores[n] is None:
            print(f"WARNING: scores/{n}.json missing")
        if costs[n] is None:
            print(f"WARNING: scores/{n}_cost.json missing")

    L = ["# Phase 2 - DINOv3 P0+P3 injection", ""]
    L.append("pycocotools COCOeval (bbox) on the val split (1,661 images). "
             "Both models trained on the Phase 1 winning list (control, non-RFS), "
             "imgsz 640, seed 42.")
    L.append("")

    L.append("## AP@0.50")
    L.append("")
    L.append("| model | AP@0.50 | AP@0.50:0.95 |")
    L.append("|---|---|---|")
    for n, label in models:
        d = scores[n]
        L.append(f"| {label} | {fmt(d['ap50']) if d else 'n/a'} | {fmt(d['ap50_95']) if d else 'n/a'} |")
    a, b = scores[WINNER[0]], scores[DINO[0]]
    if a and b:
        L.append(f"| **delta (DINO - control)** | **{b['ap50'] - a['ap50']:+.4f}** | "
                 f"**{b['ap50_95'] - a['ap50_95']:+.4f}** |")
    L.append("")

    L.append("## Per-class AP@0.50 - focus classes")
    L.append("")
    L.append("| class_id | class_name | " + " | ".join(l for _, l in models) + " | delta |")
    L.append("|---|---|---|---|---|")
    for c in FOCUS_CLASSES:
        cells = [fmt(scores[n]["per_class"][str(c)]["ap50"]) if scores[n] else "n/a" for n, _ in models]
        delta = (f"{scores[DINO[0]]['per_class'][str(c)]['ap50'] - scores[WINNER[0]]['per_class'][str(c)]['ap50']:+.4f}"
                 if a and b else "n/a")
        L.append(f"| {c} | {CLASS_NAMES[c]} | " + " | ".join(cells) + f" | {delta} |")
    L.append("")

    L.append("## Size / FLOPs / latency")
    L.append("")
    L.append("| metric | " + " | ".join(l for _, l in models) + " |")
    L.append("|---|---|---|")
    rows = [
        ("total parameters", "params_total"),
        ("trainable parameters", "params_trainable"),
        ("frozen parameters", "params_frozen"),
        ("checkpoint size (MB)", "checkpoint_mb"),
        ("GFLOPs @ 640", "gflops_640"),
        ("latency ms/img (bs=1)", "latency_ms_per_img_bs1"),
    ]
    for label, key in rows:
        cells = []
        for n, _ in models:
            c = costs[n]
            v = c.get(key) if c else None
            cells.append(f"{v:,}" if isinstance(v, int) else fmt(v, 2))
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Training wall-time")
    L.append("")
    L.append("| run | seconds | hours |")
    L.append("|---|---|---|")
    for n, label in models:
        s = wall_time(n)
        L.append(f"| {label} | {s:.0f} | {s / 3600:.3f} |" if s else f"| {label} | n/a | n/a |")
    L.append("")
    L.append("Full per-class table: `phase2_per_class.csv`")
    L.append("")

    report = "\n".join(L)
    (IMPROVE_DIR / "phase2_report.md").write_text(report)

    with open(IMPROVE_DIR / "phase2_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["class_id", "class_name"]
        for n, _ in models:
            cols += [f"{n}_ap50", f"{n}_ap50_95"]
        cols.append("delta_ap50")
        w.writerow(cols)
        for c in range(NC):
            row = [c, CLASS_NAMES[c]]
            vals = []
            for n, _ in models:
                d = scores[n]
                if d is None:
                    row += ["", ""]
                    vals.append(None)
                else:
                    pc = d["per_class"][str(c)]
                    row += [round(pc["ap50"], 6), round(pc["ap50_95"], 6)]
                    vals.append(pc["ap50"])
            row.append(round(vals[1] - vals[0], 6) if all(v is not None for v in vals) else "")
            w.writerow(row)

    print(report)
    print(f"wrote {IMPROVE_DIR / 'phase2_report.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render scores/<name>.confusion.json as a PNG.

  python3 plot_confusion.py --name best_pipeline
  python3 plot_confusion.py --name best_pipeline --normalize   # column-normalized

Two panels: the 35x35 matrix (34 classes + background) and the error budget.
Log colour scale, because the diagonal is ~100x the off-diagonal cells.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = Path(__file__).resolve().parent
NC, BG = 34, 34

INK, DIM, RULE = "#11242c", "#5d727b", "#d2dcdf"
HIT, MISS, FP, MIX = "#1a7f7b", "#ab4636", "#b07d31", "#5555a0"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="best_pipeline")
    ap.add_argument("--out", default=None)
    ap.add_argument("--normalize", action="store_true",
                    help="show each cell as a share of its GT column instead of raw counts")
    args = ap.parse_args()

    d = json.loads((HERE / "scores" / f"{args.name}.confusion.json").read_text())
    m = np.array(d["matrix"], dtype=float)
    names = {int(k): v for k, v in d["class_names"].items()}
    pc = {int(k): v for k, v in d["per_class"].items()}
    thin = {c for c in pc if pc[c]["gt"] <= 10}

    hit = float(np.trace(m[:NC, :NC]))
    miss = float(m[BG, :NC].sum())
    fp = float(m[:NC, BG].sum())
    mix = float(m[:NC, :NC].sum() - hit)
    err = miss + fp + mix

    row_lab = [f"{c}  {names[c]}" for c in range(NC)] + ["nothing detected"]
    col_lab = [f"{names[c]}  {c}" for c in range(NC)] + ["background"]

    disp = m.copy()
    if args.normalize:
        colsum = m.sum(axis=0, keepdims=True)
        disp = np.divide(m, np.where(colsum == 0, 1, colsum)) * 100

    # top band is reserved for the rotated column labels, which are up to ~3.2in long
    fig = plt.figure(figsize=(19.5, 20.5), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[15, 1.7], hspace=0.06,
                          left=0.185, right=0.94, top=0.795, bottom=0.035)
    ax = fig.add_subplot(gs[0])

    cmap = LinearSegmentedColormap.from_list(
        "sea", ["#ffffff", "#d8e8e6", "#8cc4bf", "#3f9b95", "#136b66", "#0a3f45"])
    masked = np.ma.masked_where(disp <= 0, disp)
    vmax = disp.max()
    norm = None if args.normalize else LogNorm(vmin=1, vmax=vmax)
    im = ax.imshow(masked, cmap=cmap, norm=norm,
                   vmin=None if norm else 0, vmax=None if norm else vmax,
                   aspect="equal", interpolation="nearest")

    thr = (np.log(vmax) * 0.45) if not args.normalize else vmax * 0.45
    for i in range(NC + 1):
        for j in range(NC + 1):
            v = disp[i, j]
            if v <= 0:
                continue
            val = np.log(max(v, 1)) if not args.normalize else v
            txt = f"{v:.0f}" if not args.normalize else (f"{v:.0f}" if v >= 1 else "")
            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.4,
                        color="white" if val > thr else INK,
                        fontfamily="monospace")

    for i in range(NC):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor=HIT, lw=1.1, zorder=4))
    ax.axhline(NC - .5, color=INK, lw=1.6)
    ax.axvline(NC - .5, color=INK, lw=1.6)

    ax.set_yticks(range(NC + 1))
    ax.set_yticklabels(row_lab, fontsize=7.6, fontfamily="monospace")
    for k, t in enumerate(ax.get_yticklabels()):
        t.set_color(MISS if k in thin else (INK if k < NC else MISS))

    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xticks(range(NC + 1))
    ax.set_xticklabels(col_lab, rotation=90, fontsize=7.6, fontfamily="monospace")
    for k, t in enumerate(ax.get_xticklabels()):
        t.set_color(MISS if k in thin else (INK if k < NC else FP))

    ax.tick_params(length=0)
    ax.set_xlabel("ACTUALLY  WAS   →", fontsize=10, labelpad=11,
                  fontfamily="monospace", color=INK)
    ax.set_ylabel("←   PIPELINE  SAID", fontsize=10, labelpad=11,
                  fontfamily="monospace", color=INK)
    for s in ax.spines.values():
        s.set_edgecolor(RULE)
    ax.set_xticks(np.arange(-.5, NC + 1), minor=True)
    ax.set_yticks(np.arange(-.5, NC + 1), minor=True)
    ax.grid(which="minor", color=RULE, lw=.45)

    cb = fig.colorbar(im, ax=ax, fraction=0.017, pad=0.012)
    cb.set_label("% of that true class" if args.normalize else "boxes (log scale)",
                 fontsize=8.5, color=DIM)
    cb.ax.tick_params(labelsize=7.5, colors=DIM)
    cb.outline.set_edgecolor(RULE)

    bx = fig.add_subplot(gs[1])
    segs = [(hit, HIT, "found and named"), (miss, MISS, "missed entirely"),
            (fp, FP, "false positive"), (mix, MIX, "misnamed")]
    tot = sum(s[0] for s in segs)
    left = 0.0
    for v, c, lab in segs:
        bx.barh(0, v, left=left, color=c, height=.62)
        bx.text(left + v / 2, 0, f"{int(v):,}\n{lab}\n{100 * v / tot:.1f}%",
                ha="center", va="center", color="white", fontsize=8.5,
                fontfamily="monospace", linespacing=1.5)
        left += v
    bx.set_xlim(0, tot)
    bx.set_ylim(-.62, .62)
    bx.axis("off")
    bx.text(0, -.55,
            f"{int(err):,} errors total   ·   only {int(mix):,} ({100 * mix / err:.1f}%) are "
            f"a found object with the wrong name — a recall problem, not a taxonomy problem",
            fontsize=9.5, color=DIM, fontfamily="monospace", va="top")

    fig.text(0.045, 0.978, "Where the ensemble goes wrong",
             fontsize=21, fontweight="semibold", color=INK, ha="left", va="top")
    fig.text(0.045, 0.9605,
             "A + B + E + F   →   WBF   →   crop rescorer        "
             f"NAMR33 val, 1,661 images   ·   conf ≥ {d['conf']}   ·   "
             f"IoU ≥ {d['iou']}, class-agnostic matching",
             fontsize=10.5, color=DIM, ha="left", va="top", fontfamily="monospace")
    fig.text(0.045, 0.9465,
             "rows = what the pipeline said   ·   columns = what was actually there   ·   "
             "red labels = ≤10 val boxes, unmeasurable",
             fontsize=10.5, color=DIM, ha="left", va="top", fontfamily="monospace")

    out = args.out or str(HERE / "scores" /
                          f"{args.name}.confusion{'_norm' if args.normalize else ''}.png")
    fig.savefig(out, dpi=145, facecolor="white")
    print(f"wrote {out}")
    print(f"  correct {int(hit)} | missed {int(miss)} | false pos {int(fp)} | misnamed {int(mix)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Split the val set once, deterministically, into val_fit and val_sel.

  python3 make_val_split.py

Why: val has absorbed ~250 selection decisions, so 0.7684 is optimistically
biased by an unknown amount. From here, tune on val_fit and confirm on val_sel;
a gain that does not survive the second half does not count.

Splitting on images (annotations are per-image) with ITERATIVE STRATIFICATION
(Sechidis et al. 2011): images are assigned rarest-class-first to whichever half
is furthest below its quota for that class. Plain random splitting on a 34-class
long tail routinely lands a whole rare class on one side.

Writes preds/gt_val_{fit,sel}_namr33.json (COCO GT for each half) and
data/val_split.json (the image-id lists + the per-class audit).
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from paths12 import GT_VAL_JSON, NC, PREDS_DIR, ENS_DIR

SEED = 42
HALVES = ("fit", "sel")
UNMEASURABLE = 10       # classes with <= this many boxes cannot be split meaningfully


def iterative_stratify(img_classes, rng):
    """img_classes: {image_id: Counter(class -> n boxes)} -> {image_id: 'fit'|'sel'}."""
    # desired share of each class per half, and of images overall
    want = {h: defaultdict(float) for h in HALVES}
    total = Counter()
    for cc in img_classes.values():
        total.update(cc)
    for c, n in total.items():
        for h in HALVES:
            want[h][c] = n / 2.0
    want_imgs = {h: len(img_classes) / 2.0 for h in HALVES}

    assign = {}
    remaining = set(img_classes)
    # rarest class first; images with no annotations are handled at the end
    while remaining:
        pending = {c: sum(img_classes[i][c] for i in remaining if c in img_classes[i])
                   for c in total}
        pending = {c: v for c, v in pending.items() if v > 0}
        if not pending:
            break
        c = min(pending, key=lambda k: (pending[k], k))
        pool = sorted(i for i in remaining if img_classes[i].get(c))
        rng.shuffle(pool)
        for img in pool:
            # the half most under-quota for this class; ties -> most under-quota overall
            best = max(HALVES, key=lambda h: (want[h][c], want_imgs[h], rng.random()))
            assign[img] = best
            remaining.discard(img)
            for k, n in img_classes[img].items():
                want[best][k] -= n
            want_imgs[best] -= 1

    for img in sorted(remaining):     # images with zero annotations
        best = max(HALVES, key=lambda h: (want_imgs[h], rng.random()))
        assign[img] = best
        want_imgs[best] -= 1
    return assign


def main():
    rng = random.Random(SEED)
    gt = json.loads(GT_VAL_JSON.read_text())
    img_classes = {im["id"]: Counter() for im in gt["images"]}
    for a in gt["annotations"]:
        img_classes[a["image_id"]][a["category_id"]] += 1

    assign = iterative_stratify(img_classes, rng)
    ids = {h: sorted(i for i, v in assign.items() if v == h) for h in HALVES}

    # ---- write a COCO GT per half ----
    for h in HALVES:
        keep = set(ids[h])
        sub = {
            "images": [im for im in gt["images"] if im["id"] in keep],
            "annotations": [a for a in gt["annotations"] if a["image_id"] in keep],
            "categories": gt["categories"],
        }
        out = PREDS_DIR / f"gt_val_{h}_namr33.json"
        out.write_text(json.dumps(sub))
        print(f"wrote {out}  ({len(sub['images'])} images, {len(sub['annotations'])} boxes)")

    # ---- audit ----
    counts = {h: Counter() for h in HALVES}
    for a in gt["annotations"]:
        counts[assign[a["image_id"]]][a["category_id"]] += 1
    print(f"\n  {'cls':>3} {'name':<32} {'fit':>6} {'sel':>6}   note")
    import sys
    sys.path.insert(0, str(ENS_DIR.parent / "10_diagnostics"))
    from common import CLASS_NAMES
    unmeasurable = []
    for c in range(NC):
        f, s = counts["fit"][c], counts["sel"][c]
        note = ""
        if f + s <= UNMEASURABLE:
            note = "UNMEASURABLE in both halves"
            unmeasurable.append(c)
        elif min(f, s) < 5:
            note = "thin on one side"
        print(f"  {c:>3} {CLASS_NAMES[c]:<32} {f:>6} {s:>6}   {note}")
    print(f"\n  images: fit {len(ids['fit'])}  sel {len(ids['sel'])}")
    print(f"  boxes : fit {sum(counts['fit'].values())}  sel {sum(counts['sel'].values())}")
    print(f"  classes unmeasurable in both halves (<= {UNMEASURABLE} boxes total): {unmeasurable}")

    (ENS_DIR / "data").mkdir(exist_ok=True)
    out = ENS_DIR / "data" / "val_split.json"
    out.write_text(json.dumps({
        "seed": SEED, "method": "iterative stratification on class presence",
        "ids": ids,
        "per_class": {str(c): {"fit": counts["fit"][c], "sel": counts["sel"][c]}
                      for c in range(NC)},
        "unmeasurable": unmeasurable,
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

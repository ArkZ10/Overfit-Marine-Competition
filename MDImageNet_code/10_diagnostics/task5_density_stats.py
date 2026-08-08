#!/usr/bin/env python3
"""TASK 5 - annotation density stats, over the full dataset (train+val combined).

boxes_per_image histogram buckets are conditional on images containing >=1 box
of that class (denominator matches the "images" column in density_stats.csv).
"""
import csv
import math
from collections import defaultdict

import numpy as np

from common import CLASS_NAMES, DIAG_DIR, NC, resolve_label_files


def iter_all_label_files():
    for split in ("train", "val"):
        for p in resolve_label_files(split):
            yield p.resolve()


def main():
    rel_sizes = defaultdict(list)          # class_id -> [sqrt(w*h), ...]
    boxes_per_class = defaultdict(int)
    images_per_class = defaultdict(int)
    hist = defaultdict(lambda: defaultdict(int))  # class_id -> {1:.., 2:.., 3:.., 4:.., '5+':..}

    for lbl_path in iter_all_label_files():
        per_image_count = defaultdict(int)
        with open(lbl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                rel_sizes[cls].append(math.sqrt(w * h))
                boxes_per_class[cls] += 1
                per_image_count[cls] += 1
        for cls, cnt in per_image_count.items():
            images_per_class[cls] += 1
            bucket = cnt if cnt <= 4 else "5+"
            hist[cls][bucket] += 1

    # density_stats.csv
    stats_rows = []
    for c in range(NC):
        sizes = rel_sizes.get(c, [])
        boxes = boxes_per_class.get(c, 0)
        images = images_per_class.get(c, 0)
        bpi = boxes / images if images else 0.0
        if sizes:
            p10, p50, p90 = np.percentile(sizes, [10, 50, 90])
        else:
            p10 = p50 = p90 = 0.0
        stats_rows.append({
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "boxes": boxes,
            "images": images,
            "boxes_per_image": round(bpi, 6),
            "p50_rel_size": round(float(p50), 6),
            "p10_rel_size": round(float(p10), 6),
            "p90_rel_size": round(float(p90), 6),
        })

    with open(DIAG_DIR / "density_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)

    # density_hist.csv
    bucket_labels = [1, 2, 3, 4, "5+"]
    with open(DIAG_DIR / "density_hist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name"] + [f"images_with_{b}_boxes" for b in bucket_labels])
        for c in range(NC):
            row = [c, CLASS_NAMES[c]] + [hist.get(c, {}).get(b, 0) for b in bucket_labels]
            w.writerow(row)

    print(f"{'class_id':>8} {'class_name':<32} {'boxes':>7} {'images':>7} {'bpi':>6} {'p10':>7} {'p50':>7} {'p90':>7}")
    for r in stats_rows:
        print(f"{r['class_id']:>8} {r['class_name']:<32} {r['boxes']:>7} {r['images']:>7} "
              f"{r['boxes_per_image']:>6.2f} {r['p10_rel_size']:>7.4f} {r['p50_rel_size']:>7.4f} {r['p90_rel_size']:>7.4f}")

    return stats_rows


if __name__ == "__main__":
    main()

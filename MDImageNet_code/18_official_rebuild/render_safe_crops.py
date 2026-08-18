#!/usr/bin/env python3
"""Render positive and teacher-vetoed negative rows from a safe crop manifest.

Ambiguous rows are deliberately not rendered and never enter classifier loss.
"""

import argparse
import csv
from pathlib import Path

from PIL import Image

EXPAND = 1.4
SIZE = 224


def expanded(box, width, height):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    w, h = max(w * EXPAND, 32), max(h * EXPAND, 32)
    x1, y1 = max(0, cx - w / 2), max(0, cy - h / 2)
    x2, y2 = min(width, cx + w / 2), min(height, cy + h / 2)
    return (x1, y1, x2, y2) if x2 - x1 >= 8 and y2 - y1 >= 8 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--images-root', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--out-manifest', type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.manifest.open()))
    kept = [r for r in rows if r['state'] in ('positive', 'negative')]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = []
    current_name = None
    image = None
    try:
        for index, row in enumerate(kept):
            if row['file_name'] != current_name:
                if image is not None:
                    image.close()
                image = Image.open(args.images_root / row['file_name']).convert('RGB')
                current_name = row['file_name']
            box = expanded([float(row[k]) for k in ('x', 'y', 'w', 'h')], *image.size)
            if box is None:
                continue
            label = int(row['target'])
            rel = Path(str(label)) / f"{int(row['image_id'])}_{row['state']}_{index}.jpg"
            dest = args.out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                crop = image.crop(box).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
                crop.save(dest, quality=90)
            output.append({'path': str(rel), 'label': label, 'state': row['state'],
                           'image_id': row['image_id']})
    finally:
        if image is not None:
            image.close()

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader(); writer.writerows(output)
    positives = sum(r['state'] == 'positive' for r in output)
    negatives = sum(r['state'] == 'negative' for r in output)
    print({'rendered': len(output), 'positive': positives, 'negative': negatives,
           'ambiguous_excluded': len(rows) - len(kept)})


if __name__ == '__main__':
    main()

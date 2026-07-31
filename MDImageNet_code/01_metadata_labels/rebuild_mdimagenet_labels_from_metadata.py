#!/usr/bin/env python3
"""Rebuild the three MDImageNet v1.0 YOLO label schemes from metadata.

Metadata is authoritative. By default this writes a staging tree and comparison
report. --apply atomically swaps the six official label directories while
retaining the previous directories in a timestamped backup.

Usage:
    python 01_metadata_labels/rebuild_mdimagenet_labels_from_metadata.py --dataset-root <dataset-root>
    python 01_metadata_labels/rebuild_mdimagenet_labels_from_metadata.py --dataset-root <dataset-root> --apply
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SPLITS = ("train", "test")
IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TAXONOMIES = {
    "icc19": ("icc_tw_class_id", 20, "labels_TaiwanIcc19_{split}"),
    "namr26": ("namr26_class_id", 27, "labels_NAMR26_{split}"),
    "namr33": ("namr33_class_id", 34, "labels_NAMR33_{split}"),
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def metadata_path(root: Path, split: str) -> Path:
    pattern = f"metadata_annotation_{split}*.csv"
    matches = sorted(
        {
            path.resolve()
            for directory in (root, root / "dataset_info")
            for path in directory.glob(pattern)
            if path.is_file()
        }
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {pattern} at the dataset root or in dataset_info, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]

def label_dir(root: Path, taxonomy: str, split: str) -> Path:
    return root / split / TAXONOMIES[taxonomy][2].format(split=split)

def parse_bbox_name(value: str) -> tuple[str, int, int]:
    parts = value.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid bounding_box_name: {value!r}")
    stem, total_raw, sequence_raw = parts
    total, sequence = int(total_raw), int(sequence_raw)
    if not stem or total <= 0 or sequence <= 0 or sequence > total:
        raise ValueError(f"Invalid bounding_box_name counters: {value!r}")
    return stem, total, sequence

def bounded_float(value: str, field: str, path: Path, row_no: int) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{path}:{row_no}: {field}={value!r} is outside finite 0..1")
    return number

def load_metadata(root: Path, split: str) -> tuple[dict[str, list[dict]], dict]:
    path = metadata_path(root, split)
    grouped: dict[str, list[dict]] = defaultdict(list)
    bbox_names: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "bounding_box_name", "bbox1", "bbox2", "bbox3", "bbox4",
            *(values[0] for values in TAXONOMIES.values()),
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row_no, row in enumerate(reader, 2):
            bbox_name = row["bounding_box_name"].strip()
            if bbox_name in bbox_names:
                raise ValueError(f"{path}:{row_no}: duplicate bounding_box_name={bbox_name}")
            bbox_names.add(bbox_name)
            stem, total, sequence = parse_bbox_name(bbox_name)
            bbox = tuple(
                bounded_float(row[f"bbox{i}"], f"bbox{i}", path, row_no)
                for i in range(1, 5)
            )
            if bbox[2] <= 0.0 or bbox[3] <= 0.0:
                raise ValueError(f"{path}:{row_no}: bbox width/height must be positive")
            classes = {}
            for taxonomy, (column, nc, _) in TAXONOMIES.items():
                class_id = int(row[column])
                if class_id < 0 or class_id >= nc:
                    raise ValueError(
                        f"{path}:{row_no}: {taxonomy} class {class_id} outside 0..{nc - 1}"
                    )
                classes[taxonomy] = class_id
            grouped[stem].append(
                {"name": bbox_name, "total": total, "sequence": sequence,
                 "bbox": bbox, "classes": classes}
            )
    counter_issues = []
    for stem, rows in grouped.items():
        rows.sort(key=lambda item: item["sequence"])
        totals = {item["total"] for item in rows}
        sequences = [item["sequence"] for item in rows]
        if len(sequences) != len(set(sequences)):
            raise ValueError(f"{path}: {stem} contains duplicate bbox sequence numbers")
        declared = max(totals) if totals else 0
        if len(totals) != 1 or sequences != list(range(1, declared + 1)):
            counter_issues.append(
                {"image_stem": stem, "declared_totals": sorted(totals), "sequences": sequences}
            )
    return dict(grouped), {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(bbox_names),
        "bbox_counter_issue_count": len(counter_issues),
        "bbox_counter_issue_examples": counter_issues[:20],
    }

def render_label(rows: list[dict], taxonomy: str) -> str:
    lines = []
    for row in rows:
        cx, cy, width, height = row["bbox"]
        lines.append(
            f"{row['classes'][taxonomy]} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
        )
    # Preserve the release's no-trailing-newline convention so metadata-identical
    # files remain byte-identical and comparison reports contain only real edits.
    return "\n".join(lines)

def parsed_label_rows(text: str) -> list[tuple[int, float, float, float, float]]:
    parsed = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Existing label row is not YOLO bbox format: {line!r}")
        parsed.append((int(parts[0]), *(round(float(value), 6) for value in parts[1:])))
    return parsed

def rebuild(root: Path, staging: Path) -> dict:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "authority": (
            "metadata_annotation_{train,test}*.csv at the dataset root "
            "(dataset_info fallback supported)"
        ),
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "metadata": {}, "splits": {}, "differences": [],
    }
    for split in SPLITS:
        grouped, source = load_metadata(root, split)
        report["metadata"][split] = source
        images_dir = root / split / "images"
        if not images_dir.is_dir():
            raise NotADirectoryError(f"Images directory not found: {images_dir}")
        image_stems = {
            path.stem
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        }
        if image_stems != set(grouped):
            raise ValueError(
                f"{split}: metadata/image mismatch; "
                f"missing_metadata={sorted(image_stems - set(grouped))[:5]}, "
                f"missing_images={sorted(set(grouped) - image_stems)[:5]}"
            )
        split_result = {
            "images": len(grouped),
            "annotations": sum(len(rows) for rows in grouped.values()),
            "taxonomies": {},
        }
        for taxonomy in TAXONOMIES:
            rebuilt_dir = label_dir(staging, taxonomy, split)
            current_dir = label_dir(root, taxonomy, split)
            rebuilt_dir.mkdir(parents=True)
            expected_names = set()
            counts = {
                "unchanged": 0, "reordered_only": 0, "content_differs": 0,
                "missing": 0, "extra": 0,
            }
            for stem in sorted(grouped):
                name = f"{stem}.txt"
                expected_names.add(name)
                output = rebuilt_dir / name
                output.write_text(render_label(grouped[stem], taxonomy), encoding="utf-8")
                current = current_dir / name
                rebuilt_text = output.read_text(encoding="utf-8")
                if not current.exists():
                    status = "missing"
                else:
                    current_rows = parsed_label_rows(current.read_text(encoding="utf-8-sig"))
                    rebuilt_rows = parsed_label_rows(rebuilt_text)
                    if current_rows == rebuilt_rows:
                        status = "unchanged"
                    elif Counter(current_rows) == Counter(rebuilt_rows):
                        status = "reordered_only"
                    else:
                        status = "content_differs"
                counts[status] += 1
                if status != "unchanged":
                    item = {"split": split, "taxonomy": taxonomy, "file": name, "status": status}
                    if current.exists():
                        item["current_sha256"] = sha256_file(current)
                        item["current_rows"] = len(current_rows)
                        item["rebuilt_rows"] = len(rebuilt_rows)
                    item["rebuilt_sha256"] = sha256_file(output)
                    report["differences"].append(item)
            actual_names = {path.name for path in current_dir.glob("*.txt")}
            for name in sorted(actual_names - expected_names):
                counts["extra"] += 1
                report["differences"].append(
                    {"split": split, "taxonomy": taxonomy, "file": name, "status": "extra"}
                )
            split_result["taxonomies"][taxonomy] = counts
        report["splits"][split] = split_result
    return report

def apply_rebuilt(root: Path, staging: Path, report: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / "dataset_info" / "label_backups" / stamp
    completed: list[tuple[Path, Path]] = []
    try:
        for split in SPLITS:
            for taxonomy in TAXONOMIES:
                current = label_dir(root, taxonomy, split)
                rebuilt = label_dir(staging, taxonomy, split)
                saved = backup / split / current.name
                saved.parent.mkdir(parents=True, exist_ok=True)
                os.replace(current, saved)
                try:
                    os.replace(rebuilt, current)
                except Exception:
                    os.replace(saved, current)
                    raise
                completed.append((current, saved))
    except Exception:
        for current, saved in reversed(completed):
            failed_rebuilt = staging / "rollback" / current.parent.name / current.name
            failed_rebuilt.parent.mkdir(parents=True, exist_ok=True)
            if current.exists():
                os.replace(current, failed_rebuilt)
            if saved.exists():
                os.replace(saved, current)
        raise
    report["applied_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report["backup_root"] = str(backup)
    return backup

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "MDImageNet dataset root containing annotation metadata CSV files, "
            "train, and test. Metadata may alternatively be stored in dataset_info."
        ),
    )
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    staging = (
        args.staging_root.expanduser().resolve()
        if args.staging_root
        else dataset_root / "dataset_info/labels_rebuilt_from_metadata"
    )
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else dataset_root / "dataset_info/label_rebuild_report.json"
    )
    report = rebuild(dataset_root, staging)
    if args.apply:
        backup = apply_rebuilt(dataset_root, staging, report)
        print(f"Applied metadata-derived labels; previous labels retained at: {backup}")
    else:
        print("Dry run: official labels unchanged. Review the report, then pass --apply.")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Report: {report_path}")
    print(f"Differences: {len(report['differences'])}")

if __name__ == "__main__":
    main()

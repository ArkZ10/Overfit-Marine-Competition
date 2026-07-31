#!/usr/bin/env python3
"""Count YOLO label IDs and write a Markdown summary beside the labels.

Usage:
    python 08_statistics/countYoloLabels.py --labels-dir <labels-dir>
    python 08_statistics/countYoloLabels.py --labels-dir <labels-root> --recursive
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import sys


RECURSIVE = False
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
REPORT_FILE_NAME = "readme.md"
PROGRESS_UPDATE_INTERVAL = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count YOLO class ids in label txt files.")
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=RECURSIVE)
    parser.add_argument("--report-file-name", default=REPORT_FILE_NAME)
    return parser.parse_args()


@dataclass(frozen=True)
class FileStats:
    path: Path
    class_counts: Counter[int]
    invalid_lines: tuple[tuple[int, str], ...]

    @property
    def has_boxes(self) -> bool:
        return bool(self.class_counts)

    @property
    def class_ids_in_file(self) -> set[int]:
        return set(self.class_counts)


def iter_label_files(labels_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(
            path for path in labels_dir.rglob("*.txt")
            if path.is_file()
        )

    direct_files = sorted(
        path for path in labels_dir.glob("*.txt")
        if path.is_file()
    )
    if direct_files:
        return direct_files

    # Retain support for roots containing train/labels, val/labels, or test/labels.
    return sorted(labels_dir.glob("*/labels/*.txt"))


def read_label_file(label_path: Path) -> FileStats:
    class_counts: Counter[int] = Counter()
    invalid_lines: list[tuple[int, str]] = []

    with label_path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                invalid_lines.append((line_number, line))
                continue

            try:
                class_counts[int(parts[0])] += 1
            except ValueError:
                invalid_lines.append((line_number, line))

    return FileStats(label_path, class_counts, tuple(invalid_lines))


def print_progress(done: int, total: int, width: int = 40, force: bool = False) -> None:
    if total <= 0:
        return
    if not force and done != total and done % PROGRESS_UPDATE_INTERVAL != 0:
        return

    ratio = done / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = ratio * 100
    sys.stdout.write(f"\rProgress: [{bar}] {done}/{total} ({percent:5.1f}%)")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def build_report(
    labels_dir: Path,
    label_file_count: int,
    files_with_boxes: int,
    class_counts: Counter[int],
    class_image_counts: Counter[int],
    invalid_line_count: int,
    invalid_files: list[Path],
) -> str:
    total_annotations = sum(class_counts.values())
    class_ids = sorted(class_counts)

    lines = [
        "# YOLO bbox annotation summary",
        "",
        f"- 標註資料夾: `{labels_dir}`",
        f"- 標註 txt 檔案數量: {label_file_count}",
        f"- 含有標註的圖片/檔案數量: {files_with_boxes}",
        f"- 標註數量 bbox 總數: {total_annotations}",
        f"- 類別數量: {len(class_ids)}",
        f"- 類別 ID: {', '.join(str(class_id) for class_id in class_ids) if class_ids else 'None'}",
        "",
        "## 各類別統計",
        "",
        "| 類別 ID | 標註數量 | 圖片數量 |",
        "|---:|---:|---:|",
    ]

    if class_ids:
        for class_id in class_ids:
            lines.append(f"| {class_id} | {class_counts[class_id]} | {class_image_counts[class_id]} |")
    else:
        lines.append("| None | 0 | 0 |")

    lines.extend(
        [
            "",
            "## 無效標註列",
            "",
            f"- 已略過的無效列數: {invalid_line_count}",
        ]
    )

    if invalid_files:
        lines.append("- 含有無效列的檔案:")
        for invalid_file in invalid_files:
            lines.append(f"  - `{invalid_file}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    # Path.resolve() interprets relative paths from the caller's current
    # working directory, rather than from this script's installation folder.
    labels_dir = args.labels_dir.expanduser().resolve()
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"Labels directory not found: {labels_dir}")

    label_files = iter_label_files(labels_dir, args.recursive)
    class_counts: Counter[int] = Counter()
    class_image_counts: Counter[int] = Counter()
    files_with_boxes = 0
    invalid_line_count = 0
    invalid_files: list[Path] = []

    print(f"Labels directory: {labels_dir}")
    print(f"Label txt files: {len(label_files)}")
    print(f"Workers: {MAX_WORKERS}")
    print_progress(0, len(label_files), force=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(read_label_file, label_path) for label_path in label_files]
        for done, future in enumerate(as_completed(futures), start=1):
            stats = future.result()
            if stats.has_boxes:
                files_with_boxes += 1
            if stats.invalid_lines:
                invalid_line_count += len(stats.invalid_lines)
                invalid_files.append(stats.path)

            class_counts.update(stats.class_counts)
            class_image_counts.update(stats.class_ids_in_file)
            print_progress(done, len(label_files))

    invalid_files.sort()

    total_annotations = sum(class_counts.values())
    unique_class_count = len(class_counts)
    report_path = labels_dir / args.report_file_name
    report_path.write_text(
        build_report(
            labels_dir,
            len(label_files),
            files_with_boxes,
            class_counts,
            class_image_counts,
            invalid_line_count,
            invalid_files,
        ),
        encoding="utf-8",
    )

    print(f"Files with annotations: {files_with_boxes}")
    print(f"Total annotation IDs / bboxes: {total_annotations}")
    print(f"Unique ID classes: {unique_class_count}")
    print(f"Report written: {report_path}")

    if class_counts:
        print("\nPer-class counts:")
        for class_id in sorted(class_counts):
            print(
                f"  ID {class_id}: "
                f"{class_counts[class_id]} annotations, "
                f"{class_image_counts[class_id]} images"
            )

    if invalid_line_count:
        print(f"\nInvalid lines skipped: {invalid_line_count}")
        print("Files containing invalid lines:")
        for invalid_file in invalid_files:
            print(f"  {invalid_file}")


if __name__ == "__main__":
    main()

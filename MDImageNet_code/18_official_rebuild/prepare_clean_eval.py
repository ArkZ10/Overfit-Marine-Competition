#!/usr/bin/env python3
"""Validate and split clean full-validation prediction dumps by official IDs."""

import argparse
import json
from pathlib import Path


def ids(path: Path) -> set[int]:
    return {int(im["id"]) for im in json.loads(path.read_text())["images"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = ap.parse_args()

    split_ids = {s: ids(args.data / f"val_{s}.json") for s in ("fit", "sel")}
    full_ids = ids(args.data / "val.json")
    assert split_ids["fit"].isdisjoint(split_ids["sel"])
    assert split_ids["fit"] | split_ids["sel"] == full_ids

    for dump in args.dumps:
        dets = json.loads(dump.read_text())
        seen = {int(d["image_id"]) for d in dets}
        extra = seen - full_ids
        if extra:
            raise SystemExit(f"{dump}: {len(extra)} prediction image IDs outside clean val")
        base = dump.name.removesuffix(".clean_val.json")
        for split, keep in split_ids.items():
            out = dump.with_name(f"{base}.clean_val_{split}.json")
            rows = [d for d in dets if int(d["image_id"]) in keep]
            out.write_text(json.dumps(rows))
            print(f"wrote {out} ({len(rows)} detections, {len(keep)} images)")


if __name__ == "__main__":
    main()

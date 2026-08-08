#!/usr/bin/env python3
"""TASK 4 - near-duplicate leakage check between train and val.

phash every image (train + val), cluster at Hamming distance <= 5, report
train/val leakage. Read-only; does not touch the split.
"""
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import DIAG_DIR, REPO_ROOT, resolve_image_files

random.seed(0)
np.random.seed(0)

HASH_SIZE = 8  # imagehash.phash default -> 64-bit hash
DIST_THRESHOLD = 5
CHUNK = 500


def _phash_one(path_str):
    import imagehash
    from PIL import Image
    try:
        with Image.open(path_str) as im:
            h = imagehash.phash(im, hash_size=HASH_SIZE)
        return int(str(h), 16)
    except Exception:
        return None


def compute_hashes(paths):
    real_paths = [str(p.resolve()) for p in paths]
    hashes = [None] * len(real_paths)
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as ex:
        for i, h in enumerate(ex.map(_phash_one, real_paths, chunksize=64)):
            hashes[i] = h
    return hashes


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_edges(hashes, n):
    """All (i, j, dist) with i < j and dist <= DIST_THRESHOLD, vectorized in row chunks."""
    edges = []
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        chunk = hashes[start:end]  # (c,)
        xor = chunk[:, None] ^ hashes[None, :]  # (c, n) uint64
        dist = np.bitwise_count(xor)  # (c, n) uint8
        for row in range(end - start):
            gi = start + row
            js = np.nonzero(dist[row, gi + 1:] <= DIST_THRESHOLD)[0] + gi + 1
            for gj in js:
                edges.append((gi, int(gj), int(dist[row][gj])))
    return edges


def main():
    train_paths = resolve_image_files("train")
    val_paths = resolve_image_files("val")
    n_train = len(train_paths)
    n_val = len(val_paths)

    all_paths = train_paths + val_paths
    split = np.array([0] * n_train + [1] * n_val)  # 0=train, 1=val
    # NOTE: report split-relative (symlink) paths, not .resolve()'d ones - both the
    # train and val symlink trees point back into the same physical
    # MDImageDataset/train/images/ pool, so resolving would erase which split an
    # image was assigned to. Only actual file I/O (hashing) needs the resolved path.
    real_paths = [p.resolve() for p in all_paths]
    display_paths = all_paths

    print(f"hashing {len(all_paths)} images ({n_train} train + {n_val} val)...")
    hash_list = compute_hashes(all_paths)
    missing = sum(1 for h in hash_list if h is None)
    if missing:
        print(f"warning: {missing} images failed to hash, excluding them")
    valid_idx = [i for i, h in enumerate(hash_list) if h is not None]
    hashes = np.array([hash_list[i] for i in valid_idx], dtype=np.uint64)
    n = len(hashes)
    # map from position-in-hashes -> original all_paths index
    idx_map = valid_idx

    print(f"computing pairwise Hamming distances over {n} images (threshold <= {DIST_THRESHOLD})...")
    edges = find_edges(hashes, n)
    print(f"found {len(edges)} raw edges")

    dsu = DSU(n)
    cross_edges = []  # (i, j, dist) in hashes-index space, one train one val
    for i, j, d in edges:
        dsu.union(i, j)
        oi, oj = idx_map[i], idx_map[j]
        if split[oi] != split[oj]:
            cross_edges.append((i, j, d))

    # cluster spanning check
    from collections import defaultdict
    clusters = defaultdict(set)  # root -> set of splits present
    cluster_members = defaultdict(list)
    for pos in range(n):
        root = dsu.find(pos)
        orig = idx_map[pos]
        clusters[root].add(int(split[orig]))
        cluster_members[root].append(orig)

    spanning_clusters = [root for root, splits in clusters.items() if splits == {0, 1}]
    n_spanning = len(spanning_clusters)

    # val images with a DIRECT train neighbor within threshold
    val_with_train_dup = set()
    example_pairs = []  # (train_path, val_path, dist)
    for i, j, d in cross_edges:
        oi, oj = idx_map[i], idx_map[j]
        train_pos, val_pos = (oi, oj) if split[oi] == 0 else (oj, oi)
        val_with_train_dup.add(val_pos)
        example_pairs.append((
            os.path.relpath(display_paths[train_pos], REPO_ROOT),
            os.path.relpath(display_paths[val_pos], REPO_ROOT),
            d,
        ))

    example_pairs.sort(key=lambda t: (t[2], t[0], t[1]))
    n_leaked = len(val_with_train_dup)
    pct_leaked = 100.0 * n_leaked / n_val if n_val else 0.0

    lines = []
    lines.append(f"images hashed: {n} ({n_train} train + {n_val} val), hash=phash{HASH_SIZE}x{HASH_SIZE}, distance threshold <= {DIST_THRESHOLD}")
    lines.append(f"total near-duplicate edges (any pair, dist<=5): {len(edges)}")
    lines.append(f"cross-split (train<->val) near-duplicate edges: {len(cross_edges)}")
    lines.append(f"clusters spanning both train and val: {n_spanning}")
    lines.append(f"val images with a train near-duplicate: {n_leaked}")
    lines.append(f"val images with a train near-duplicate as % of {n_val} val images: {pct_leaked:.4f}%")
    lines.append("")
    lines.append("10 example (train_path, val_path, hamming_distance) pairs:")
    for tp, vp, d in example_pairs[:10]:
        lines.append(f"  {tp} | {vp} | {d}")

    report = "\n".join(lines)
    print(report)

    with open(DIAG_DIR / "leakage_report.txt", "w") as f:
        f.write(report + "\n")

    return n_leaked, pct_leaked, n_spanning


if __name__ == "__main__":
    main()

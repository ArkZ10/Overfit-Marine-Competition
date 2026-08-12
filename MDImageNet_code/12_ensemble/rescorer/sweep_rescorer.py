#!/usr/bin/env python3
"""Sweep rescorer variants against the Phase-2 fused baseline.

Classifier probabilities for every fused box are computed ONCE and cached, then
each (alpha, bg_suppress, reassign) variant is applied and scored from the cache.

  python3 -m rescorer.sweep_rescorer
"""
import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ENS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENS_DIR))

from paths12 import GT_VAL_JSON, PREDS_DIR, RUNS_DIR, SCORES_DIR  # noqa: E402
from rescorer.make_crops import expand_clamp  # noqa: E402
from rescorer.train_rescorer import IMAGENET_MEAN, IMAGENET_STD, build_model  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

CROP_SIZE = 224
BG_CLASS = 34
BASELINE = None  # set from --baseline; the fused score this must beat


@torch.no_grad()
def compute_probs(dump, weights, images_root, device="cuda:0", batch=256):
    tag = Path(dump).name.split(".")[0]          # cache is per-dump, not global
    cache = SCORES_DIR / f"rescorer_probs_{tag}.npy"
    idx_cache = SCORES_DIR / f"rescorer_probs_{tag}_idx.json"
    if cache.exists() and idx_cache.exists():
        print(f"reusing cached probabilities from {cache}")
        return np.load(cache), json.loads(idx_cache.read_text())

    model = build_model(torch.device(device))
    ck = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded rescorer (val_acc={ck.get('acc')})")

    gt = json.loads(GT_VAL_JSON.read_text())
    info = {im["id"]: im for im in gt["images"]}
    dets = json.loads(Path(dump).read_text())
    by_img = defaultdict(list)
    for i, d in enumerate(dets):
        by_img[d["image_id"]].append(i)

    probs = np.zeros((len(dets), BG_CLASS + 1), dtype=np.float32)
    scored = []
    buf_t, buf_i = [], []

    def flush():
        nonlocal buf_t, buf_i
        if not buf_t:
            return
        x = torch.stack(buf_t).to(device)
        with torch.autocast("cuda"):
            p = torch.softmax(model(x).float(), 1).cpu().numpy()
        probs[buf_i] = p
        scored.extend(buf_i)
        buf_t, buf_i = [], []

    root = Path(images_root)
    for n, (image_id, det_ids) in enumerate(by_img.items(), 1):
        with Image.open(root / info[image_id]["file_name"]) as im:
            im = im.convert("RGB")
            w, h = im.size
            for di in det_ids:
                x, y, bw, bh = dets[di]["bbox"]
                eb = expand_clamp([x, y, x + bw, y + bh], w, h)
                if eb is None:
                    continue
                c = im.crop(tuple(eb)).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
                t = torch.frombuffer(bytearray(c.tobytes()), dtype=torch.uint8)
                t = t.view(CROP_SIZE, CROP_SIZE, 3).permute(2, 0, 1).float() / 255.0
                buf_t.append((t - IMAGENET_MEAN) / IMAGENET_STD)
                buf_i.append(di)
                if len(buf_t) >= batch:
                    flush()
        if n % 300 == 0:
            print(f"  {n}/{len(by_img)} images", flush=True)
    flush()

    np.save(cache, probs)
    idx_cache.write_text(json.dumps(sorted(scored)))
    print(f"cached probabilities for {len(scored)}/{len(dets)} boxes")
    return probs, sorted(scored)


def apply_variant(dets, probs, scored_set, alpha, bg_suppress, reassign, thr=0.6):
    out, n_re = [], 0
    for i, d in enumerate(dets):
        if i not in scored_set:
            out.append(d)
            continue
        p = probs[i]
        c = d["category_id"]
        if reassign:
            top = int(p[:BG_CLASS].argmax())
            if p[top] > thr and top != c:
                c, n_re = top, n_re + 1
        s = d["score"] * (1 - alpha + alpha * float(p[c]))
        if bg_suppress:
            s *= 1 - float(p[BG_CLASS])
        out.append({**d, "category_id": c, "score": float(s)})
    return out, n_re


def ap50(gt, dets):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes([dict(d) for d in dets])
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default=str(PREDS_DIR / "wbf4_best.val.json"))
    ap.add_argument("--baseline", type=float, required=True, help="fused AP50 the rescorer must beat")
    ap.add_argument("--weights", default=str(RUNS_DIR / "rescorer" / "best.pth"))
    ap.add_argument("--images-root",
                    default="/root/Overfit-Marine-Competition/MDImageDataset/yolo_split/images/val")
    args = ap.parse_args()

    global BASELINE
    BASELINE = args.baseline
    probs, scored = compute_probs(args.dump, args.weights, args.images_root)
    scored_set = set(scored)
    dets = json.loads(Path(args.dump).read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(GT_VAL_JSON))

    variants = []
    for alpha in (0.25, 0.5, 1.0):
        for bg in (False, True):
            for re_ in (False, True):
                variants.append((alpha, bg, re_))

    print(f"\nbaseline (Phase-2 fused): {BASELINE:.4f}\n")
    print(f"{'alpha':>6} {'bg_sup':>7} {'reassign':>9} {'AP50':>8} {'delta':>8} {'n_reassigned':>13}")
    results = []
    for alpha, bg, re_ in variants:
        out, n_re = apply_variant(dets, probs, scored_set, alpha, bg, re_)
        a = ap50(gt, out)
        results.append({"alpha": alpha, "bg_suppress": bg, "reassign": re_,
                        "ap50": a, "delta": a - BASELINE, "n_reassigned": n_re})
        print(f"{alpha:>6} {str(bg):>7} {str(re_):>9} {a:>8.4f} {a - BASELINE:>+8.4f} {n_re:>13}")

    results.sort(key=lambda r: -r["ap50"])
    (SCORES_DIR / "rescorer_sweep.json").write_text(json.dumps(results, indent=2))
    best = results[0]
    print(f"\nbest: alpha={best['alpha']} bg_suppress={best['bg_suppress']} "
          f"reassign={best['reassign']} -> {best['ap50']:.4f} ({best['delta']:+.4f})")
    print("GATE: " + ("PASS - ship the rescorer" if best["delta"] > 0
                      else "FAIL - ship stage 3 disabled, use the Phase-2 fused dump"))

    if best["delta"] > 0:
        out, _ = apply_variant(dets, probs, scored_set, best["alpha"],
                               best["bg_suppress"], best["reassign"])
        p = PREDS_DIR / (Path(args.dump).name.split(".")[0] + "_rescored.val.json")
        p.write_text(json.dumps(out))
        (PREDS_DIR / (Path(args.dump).name.split(".")[0] + "_rescored.val.meta.json")).write_text(json.dumps(best, indent=2))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

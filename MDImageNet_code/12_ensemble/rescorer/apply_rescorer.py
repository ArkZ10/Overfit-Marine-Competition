#!/usr/bin/env python3
"""Apply the crop rescorer to a fused dump: s' = s * (1 - alpha + alpha * p[class]).

  python3 -m rescorer.apply_rescorer --dump preds/wbf_abc.val.json \
      --weights runs/rescorer/best.pth --images-root <val images dir> \
      --alpha 0.5 [--bg-suppress] [--reassign] --out preds/wbf_abc_rs.val.json

Interpolated form keeps ranking stable when the classifier is unsure.
--bg-suppress additionally multiplies by (1 - p[background]).
--reassign switches category_id to argmax when p > 0.6 (default off).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

ENS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENS_DIR))

from paths12 import GT_VAL_JSON  # noqa: E402
from rescorer.make_crops import expand_clamp  # noqa: E402
from rescorer.train_rescorer import IMAGENET_MEAN, IMAGENET_STD, build_model  # noqa: E402

CROP_SIZE = 224
BG_CLASS = 34


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt", default=str(GT_VAL_JSON), help="for image filenames/dims")
    ap.add_argument("--images-root", required=True, help="directory holding the images")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--bg-suppress", action="store_true")
    ap.add_argument("--reassign", action="store_true")
    ap.add_argument("--reassign-thr", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = build_model(device)
    ck = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    gt = json.loads(open(args.gt).read())
    img_info = {im["id"]: im for im in gt["images"]}
    dets = json.loads(open(args.dump).read())
    by_img = defaultdict(list)
    for i, d in enumerate(dets):
        by_img[d["image_id"]].append(i)

    probs = [None] * len(dets)
    batch_tensors, batch_idx = [], []

    def flush():
        nonlocal batch_tensors, batch_idx
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        with torch.autocast("cuda"):
            p = torch.softmax(model(x).float(), dim=1).cpu()
        for j, bi in enumerate(batch_idx):
            probs[bi] = p[j]
        batch_tensors, batch_idx = [], []

    root = Path(args.images_root)
    for image_id, det_ids in by_img.items():
        info = img_info[image_id]
        path = root / info["file_name"]
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            for di in det_ids:
                x, y, bw, bh = dets[di]["bbox"]
                eb = expand_clamp([x, y, x + bw, y + bh], w, h)
                if eb is None:
                    continue
                crop = im.crop((eb[0], eb[1], eb[2], eb[3])).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
                t = torch.frombuffer(bytearray(crop.tobytes()), dtype=torch.uint8)
                t = t.view(CROP_SIZE, CROP_SIZE, 3).permute(2, 0, 1).float() / 255.0
                t = (t - IMAGENET_MEAN) / IMAGENET_STD
                batch_tensors.append(t)
                batch_idx.append(di)
                if len(batch_tensors) >= args.batch:
                    flush()
    flush()

    out_dets = []
    n_reassigned = 0
    for d, p in zip(dets, probs):
        if p is None:
            out_dets.append(d)
            continue
        c = d["category_id"]
        if args.reassign:
            top = int(p[:BG_CLASS].argmax())
            if float(p[top]) > args.reassign_thr and top != c:
                c = top
                n_reassigned += 1
        s = d["score"] * (1 - args.alpha + args.alpha * float(p[c]))
        if args.bg_suppress:
            s *= 1 - float(p[BG_CLASS])
        out_dets.append({**d, "category_id": c, "score": round(float(s), 6)})

    Path(args.out).write_text(json.dumps(out_dets))
    meta = {"dump": args.dump, "weights": args.weights, "alpha": args.alpha,
            "bg_suppress": args.bg_suppress, "reassign": args.reassign,
            "n_reassigned": n_reassigned, "n_dets": len(out_dets)}
    Path(str(args.out).replace(".json", ".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out}  (alpha={args.alpha}, bg_suppress={args.bg_suppress}, "
          f"reassigned={n_reassigned})")


if __name__ == "__main__":
    main()

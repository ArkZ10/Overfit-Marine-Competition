#!/usr/bin/env python3
"""Detector D: D-FINE-L (ICLR 2025, Apache-2.0) fine-tuned on the OFFICIAL train set.

  python3 train_dfine.py                 # full run
  python3 train_dfine.py --resume        # after a crash
  python3 train_dfine.py --spike         # 2 epochs on 200 images

Trains on lists/official_train.txt = the competition's own train_dataset.zip
membership, minus anything in our val split. Validates on lists/official_val.txt
(= the existing val split), so scores stay directly comparable with detectors
A/B/C and the dumps stay fusable.
"""
import argparse
import contextlib
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
GT_VAL = HERE.parent / "12_ensemble" / "preds" / "gt_val_namr33.json"
LISTS = HERE / "lists"
RUNS = HERE / "runs"

from dataset import DFineDataset, collate  # noqa: E402

MODEL_ID = "ustc-community/dfine-large-coco"  # D-FINE-L, COCO-pretrained, Apache-2.0
NUM_LABELS = 34
EPOCHS = 60
PATIENCE = 20
BATCH_LADDER = [16, 12, 8]
LR = 1e-4
LR_BACKBONE = 1e-5
WD = 1e-4
WARMUP_EPOCHS = 1
SEED = 42


def build_model(device):
    from transformers import DFineForObjectDetection
    m = DFineForObjectDetection.from_pretrained(
        MODEL_ID, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )
    return m.to(device)


@torch.no_grad()
def evaluate(model, loader, device, stem_to_id, sizes):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from transformers import AutoImageProcessor

    proc = evaluate._proc
    model.eval()
    dets = []
    for px, _, stems in loader:
        px = px.to(device, non_blocking=True)
        with torch.autocast("cuda"):
            out = model(pixel_values=px)
        target_sizes = torch.tensor([sizes[s] for s in stems], device=device)  # (h, w)
        res = proc.post_process_object_detection(out, threshold=0.001, target_sizes=target_sizes)
        for r, s in zip(res, stems):
            if s not in stem_to_id:
                continue
            iid = stem_to_id[s]
            boxes = r["boxes"].float().cpu().numpy()
            scores = r["scores"].float().cpu().numpy()
            labels = r["labels"].cpu().numpy()
            for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
                dets.append({"image_id": iid, "category_id": int(lb),
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             "score": float(sc)})
    if not dets:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(GT_VAL))
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1]), float(ev.stats[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--spike", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    from transformers import AutoImageProcessor
    evaluate._proc = AutoImageProcessor.from_pretrained(MODEL_ID)

    device = torch.device(args.device)
    name = "_dfine_spike" if args.spike else "dfine_l"
    out_dir = RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DFineDataset(LISTS / "official_train.txt", train=True)
    val_ds = DFineDataset(LISTS / "official_val.txt", train=False)
    if args.spike:
        train_ds.paths = train_ds.paths[:200]
        val_ds.paths = val_ds.paths[:50]
    epochs = args.epochs or (2 if args.spike else EPOCHS)

    gt = json.loads(GT_VAL.read_text())
    stem_to_id = {Path(i["file_name"]).stem: i["id"] for i in gt["images"]}
    sizes = {Path(i["file_name"]).stem: (i["height"], i["width"]) for i in gt["images"]}

    ladder = [args.batch] if args.batch else BATCH_LADDER
    last_err = None
    for batch in ladder:
        print(f"\n=== {name}: attempting batch={batch} ===", flush=True)
        try:
            train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=8,
                                      collate_fn=collate, pin_memory=True, drop_last=True,
                                      generator=torch.Generator().manual_seed(SEED))
            val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=8,
                                    collate_fn=collate, pin_memory=True)
            model = build_model(device)
            backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
            rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
            opt = torch.optim.AdamW([{"params": rest, "lr": LR},
                                     {"params": backbone, "lr": LR_BACKBONE}], weight_decay=WD)
            scaler = torch.amp.GradScaler()

            ipe = len(train_loader)
            warm = WARMUP_EPOCHS * ipe

            def scale_at(it):
                if it < warm:
                    return (it + 1) / warm
                prog = (it - warm) / max(1, epochs * ipe - warm)
                return 0.5 * (1 + math.cos(math.pi * prog))

            start_ep, best = 0, -1.0
            ck_last, ck_best = out_dir / "last.pth", out_dir / "best.pth"
            if args.resume and ck_last.exists():
                ck = torch.load(ck_last, map_location=device, weights_only=False)
                model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
                scaler.load_state_dict(ck["scaler"]); start_ep, best = ck["epoch"] + 1, ck["best_ap50"]
                print(f"resumed from epoch {ck['epoch']} (best_ap50={best:.4f})")

            csv_path = out_dir / "results.csv"
            if not csv_path.exists():
                csv_path.write_text("epoch,ap50,ap50_95,train_loss,lr,seconds\n")

            t_start = time.time(); it = start_ep * ipe; bad = 0
            for ep in range(start_ep, epochs):
                model.train(); t0 = time.time(); run = 0.0
                for i, (px, labels, _) in enumerate(train_loader):
                    sc = scale_at(it)
                    opt.param_groups[0]["lr"] = LR * sc
                    opt.param_groups[1]["lr"] = LR_BACKBONE * sc
                    px = px.to(device, non_blocking=True)
                    labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
                    with torch.autocast("cuda"):
                        loss = model(pixel_values=px, labels=labels).loss
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(opt); scaler.update()
                    run += loss.item(); it += 1
                    if not math.isfinite(loss.item()):
                        raise SystemExit(f"non-finite loss at epoch {ep} iter {i}")
                    if i % 200 == 0:
                        print(f"  ep{ep} it{i}/{ipe} loss={loss.item():.4f} lr={LR*sc:.2e}", flush=True)

                ap50, ap5095 = evaluate(model, val_loader, device, stem_to_id, sizes)
                secs = time.time() - t0
                avg = run / max(1, ipe)
                print(f"epoch {ep}: AP50={ap50:.4f} AP50-95={ap5095:.4f} loss={avg:.4f} ({secs:.0f}s)",
                      flush=True)
                with open(csv_path, "a") as f:
                    f.write(f"{ep},{ap50:.5f},{ap5095:.5f},{avg:.5f},{LR*scale_at(it):.6f},{secs:.1f}\n")

                improved = ap50 > best
                best = max(best, ap50)
                state = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                         "scaler": scaler.state_dict(), "epoch": ep, "best_ap50": best}
                torch.save(state, ck_last)
                if improved:
                    torch.save(state, ck_best); bad = 0
                else:
                    bad += 1
                    if bad >= PATIENCE and not args.spike:
                        print(f"early stop: {PATIENCE} epochs without improvement")
                        break

            wall = time.time() - t_start
            (out_dir / "wall_time_seconds.txt").write_text(f"{wall:.1f}\n")
            print(f"\ndone in {wall/3600:.3f} h at batch={batch}; best AP50={best:.4f}; {ck_best}")
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            if not (isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()):
                raise
            print(f"OOM at batch={batch}; backing off")
            for v in ("model", "opt", "scaler"):
                if v in dir(): pass
            import gc; gc.collect(); torch.cuda.empty_cache()

    raise SystemExit(f"all batch sizes {ladder} OOMed; last: {last_err}")


if __name__ == "__main__":
    main()

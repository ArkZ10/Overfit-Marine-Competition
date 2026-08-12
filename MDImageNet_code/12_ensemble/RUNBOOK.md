# 12_ensemble runbook — 3-detector WBF pipeline

All commands run from:

```bash
cd /root/Overfit-Marine-Competition/MDImageNet_code/12_ensemble
```

Detectors: **A** YOLOv11m (trained, `../11_improvements/runs/y11m_control/weights/best.pt`, val AP50 0.6049) ·
**B** RT-DETR-l (ultralytics) · **C** Faster R-CNN R50-FPN v2 (torchvision).
All mAP numbers are pycocotools. Val GT is frozen at `preds/gt_val_namr33.json`.

## Phase 0 — scaffold + smokes (DONE)

| smoke | result |
|---|---|
| dump+score A reproduces 0.6049 | ✅ 0.6049 exactly; ICC19-mapped = **0.6744** |
| ICC19 path vs organizer script | ✅ 0.6744 vs 0.6715 — Δ0.003 from predict batching (they batch 64 rect, we stream batch 1), same crosswalk+scorer |
| WBF of A alone ≈ identity | ✅ 0.6023 (Δ0.0026 — WBF self-merges same-class overlaps at iou 0.55; affects every variant equally) |
| calibration sanity | ✅ T=1.109, NLL 0.1426→0.1414, ECE 0.0238→0.0226; monotone → AP unchanged |
| RT-DETR spike | ✅ losses falling, batch 16 fits, dump round-trips (300 dets/img at flat low scores — why calibration exists) |
| FRCNN adapter + spike + resume | ✅ 5/5 label conversions exact; loss 4.09→0.49; resume works |
| rescorer spike (2k crops) | ✅ val_acc 0.816 vs 0.601 majority baseline in 3 epochs |
| submission round-trip | ✅ 3,519 rows on val; refuses to run without `--bbox-format` |

## Phase 1 — train B and C (user, overnight, both GPUs)

```bash
# GPU 0 — RT-DETR-l (~9–13 h)
nohup python3 train_rtdetr.py > runs/rtdetr_l.log 2>&1 &

# GPU 1 — Faster R-CNN (~5–6 h)
nohup python3 -m frcnn.train_frcnn --device cuda:1 > runs/frcnn_r50v2.log 2>&1 &
```

Monitor: `tail -f runs/rtdetr_l.log` / `tail -f runs/frcnn_r50v2.log`.
Crash recovery (resumes, don't restart): append `--resume` to the same command.

## Phase 2 — dump, calibrate, fuse, tune (after Phase 1)

```bash
# dumps (A's already exists from Phase 0)
python3 dump_preds.py --model-type rtdetr --weights runs/rtdetr_l/weights/best.pt --name rtdetr_l
python3 dump_preds.py --model-type frcnn  --weights runs/frcnn_r50v2/best.pth     --name frcnn_r50v2

# individual scores
python3 score_preds.py --dump preds/rtdetr_l.val.json    --name rtdetr_l
python3 score_preds.py --dump preds/frcnn_r50v2.val.json --name frcnn_r50v2

# per-model temperatures (A's already exists)
python3 calibrate.py --dump preds/rtdetr_l.val.json    --name rtdetr_l
python3 calibrate.py --dump preds/frcnn_r50v2.val.json --name frcnn_r50v2

# one fused variant looks like:
python3 wbf_fuse.py --dumps preds/y11m_control.val.json preds/rtdetr_l.val.json preds/frcnn_r50v2.val.json \
  --out preds/wbf_abc.val.json --iou-thr 0.55 --skip-box-thr 0.001 --normalize-scores temperature
python3 score_preds.py --dump preds/wbf_abc.val.json --name wbf_abc
```

The full grid (72 combos: normalize {temperature,minmax,none} × iou_thr {0.50,0.55,0.60,0.65} ×
weights {[1,1,1], AP50-proportional, [2,1,1]} × skip_box_thr {0,0.001}) is Phase-2 work —
ask the assistant to sweep it; selection on 34-class val AP50, ICC19 as consistency check.
**Go/no-go: fused must beat 0.6049.**

## Phase 2 — DONE (see `phase2_report.md`)

| model | NAMR33 AP50 | ICC19 mAP50 |
|---|---|---|
| A YOLOv11m | 0.6049 | 0.6744 |
| B RT-DETR-l | 0.6684 | 0.7419 |
| C Faster R-CNN | 0.5130 | 0.6213 |
| **A+B+C WBF** | **0.7017** | **0.7659** |

**Decision metric is NAMR33 (34-class), per the published rules:** "averages the AP values
across the 33+1 classes ... using the COCO API". ICC19 is reported only as a secondary
diagnostic — `06_evaluation/eval_yolo_mapped_icc19.py` is a crosswalk analysis tool
(it defaults to `namr26`), NOT the competition scorer.

**CURRENT BEST — 4-model + rescorer = 0.7506.** Frozen config, reuse verbatim for test
(dump order must be A B C D):
`--normalize-scores temperature --iou-thr 0.65 --skip-box-thr 0.001 --weights 0.6049 0.6684 0.5130 0.5755`
then `apply_rescorer --alpha 1.0 --bg-suppress`.

| model | val AP@0.50 |
|---|---|
| A YOLOv11m 0.6049 · B RT-DETR-l 0.6684 · C Faster R-CNN 0.5130 · **D D-FINE-L 0.5755** | |
| A+B+C fused 0.7017 → +rescorer 0.7408 | |
| **A+B+C+D fused 0.7187 → +rescorer 0.7506** | |

Temperatures are already fitted and stored in `scores/*.calib.json` — **never refit on test.**

## Phase 3 — crop rescorer (gated)

```bash
python3 -m rescorer.make_crops                 # full crop sets (~30 min)
python3 -m rescorer.train_rescorer             # ~1–2 h
python3 -m rescorer.apply_rescorer --dump preds/wbf_best.val.json \
  --weights runs/rescorer/best.pth \
  --images-root /root/Overfit-Marine-Competition/MDImageDataset/yolo_split/images/val \
  --alpha 0.5 --out preds/wbf_best_rs05.val.json
python3 score_preds.py --dump preds/wbf_best_rs05.val.json --name wbf_best_rs05
```

**DONE — gate PASSED.** Shipping config: `--alpha 1.0 --bg-suppress` (never `--reassign`,
which costs −0.03). Final: **0.7408** (+0.0391 over fused). Sweep: `scores/rescorer_sweep.json`.

Sweep α ∈ {0.25, 0.5, 1.0}, optionally `--bg-suppress` / `--reassign`.
**Gate: ships only if 34-class NAMR33 AP@0.50 improves.** Note this makes the rescorer
*more* promising than first scoped: fixing `anthropogenic_fragment` vs `other` confusion
scores real points on the 34-class metric, whereas ICC19 merges them away for free.

## Phase 4 — on test release

1. **Confirm with organizer docs first:** bbox convention (pixel top-left xywh vs
   normalized center xywh) and whether `label_id` is 0- or 1-indexed. Taxonomy defaults
   to NAMR33 (0-33), matching the published 33+1-class metric.
2. ```bash
   python3 dump_preds.py --model-type yolo   --weights ../11_improvements/runs/y11m_control/weights/best.pt --name y11m_control --split test --images-dir <TESTDIR>
   python3 dump_preds.py --model-type rtdetr --weights runs/rtdetr_l/weights/best.pt --name rtdetr_l --split test --images-dir <TESTDIR>
   python3 dump_preds.py --model-type frcnn  --weights runs/frcnn_r50v2/best.pth --name frcnn_r50v2 --split test --images-dir <TESTDIR>
   python3 wbf_fuse.py --dumps preds/y11m_control.test.json preds/rtdetr_l.test.json preds/frcnn_r50v2.test.json \
     --normalize-scores temperature --iou-thr 0.65 --skip-box-thr 0.001 --weights 0.6049 0.6684 0.5130 \
     --gt <a json with images[{id,width,height}] for TESTDIR — build like gt but annotation-free> \
     --out preds/wbf_abc.test.json  <FROZEN Phase-2 params + temperatures>
   # rescorer PASSED its gate - apply it before writing the submission:
   python3 -m rescorer.apply_rescorer --dump preds/wbf_abc.test.json \
     --weights runs/rescorer/best.pth --images-root <TESTDIR> \
     --alpha 1.0 --bg-suppress --out preds/wbf_abc_rescored.test.json
   python3 make_submission.py --dump preds/wbf_abc.test.json --images-dir <TESTDIR> \
     --bbox-format <confirmed> --label-offset <confirmed> --out submission.csv
   # (--taxonomy namr33 is the default; do NOT pass --taxonomy icc19 for a real submission)
   ```

## Submission discipline

- **Maximum 3 uploads per day** (rules.md line 7).
- The **LAST upload of each day** is that day's score; ties break by **earlier** submission time.
- Never upload an experiment after the day's best submission.
- Qualification gate: **mAP@0.5 >= 0.6 in the Preliminary Round** (rules.md line 17). Current
  val standing: 0.7408 (fused + rescorer). Submit good runs **early in the day**, then stop.
- Log every upload here: time, dump file, expected val score.

| date | time | dump | expected (NAMR33 val AP50) | actual |
|---|---|---|---|---|

## Notes

- Inference protocol pinned everywhere: conf 0.001, max_det 300, imgsz 640, stream batch 1.
  Never dump one model with a different protocol — fusion fairness depends on it.
- `dump_preds.py` predicts the images' parent **directory** (a bare Python list is one
  giant batch in ultralytics → 40 GB OOM, and `r.path` degrades to 'image0').
- Calibration is fitted on val once per model and **frozen** for test — never refit on test.
- ensemble-boxes 1.0.9 (MIT), torchvision COCO weights (BSD-3), timm ConvNeXt (MIT),
  ultralytics YOLO/RT-DETR weights (AGPL-3.0) — sources documented for the rules packet.
- FRCNN val AP50 during training is computed against the full frozen GT — comparable
  to every other number in this pipeline.

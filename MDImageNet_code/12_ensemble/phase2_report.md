# Phase 2 — calibration + WBF fusion results

All numbers are pycocotools COCOeval on the val split (1,661 images), scored against the
frozen `preds/gt_val_namr33.json`.

**NAMR33 (34-class) is the competition metric** — the published criteria state the score
"averages the AP values across the 33+1 classes of marine debris objects ... using the
COCO API", and `data_NAMR33.yaml` is titled "NAMR 33+1" with `nc: 34`. ICC19 columns below
are a secondary diagnostic only; `06_evaluation/eval_yolo_mapped_icc19.py` is a crosswalk
analysis utility (its default taxonomy is `namr26`), not the competition scorer.

## Headline

| model | NAMR33 AP@0.50 | NAMR33 AP@0.50:0.95 | ICC19 mAP50 | ICC19 mAP50-95 |
|---|---|---|---|---|
| A — YOLOv11m (Phase 1 winner) | 0.6049 | 0.5224 | 0.6744 | 0.5731 |
| B — RT-DETR-l | **0.6684** | 0.5686 | **0.7419** | 0.6244 |
| C — Faster R-CNN R50-FPN v2 | 0.5130 | 0.4031 | 0.6213 | 0.4820 |
| **A+B+C — WBF fused** | **0.7017** | **0.6003** | **0.7659** | **0.6451** |

- Fused vs best single model (B): **+0.0333** NAMR33 / **+0.0240** ICC19
- Fused vs previous baseline (A): **+0.0968** NAMR33 / **+0.0915** ICC19

**Go/no-go: PASS.** On the competition metric the fused model scores **0.7017**, clearing
both the 0.6049 baseline and the mAP ≥ 0.6 qualification gate. The uplift over the best
single model (+0.033) is at the top of the +0.01–0.03 range typical for 3-model WBF.

## Winning WBF configuration

```
normalize_scores = temperature
iou_thr          = 0.65
weights          = ap50-proportional  [0.6049, 0.6684, 0.5130]
skip_box_thr     = 0.001
```

Materialised at `preds/wbf_best.val.json` (159,695 fused boxes); full sweep in
`scores/wbf_sweep.json`.

## Calibration was the single largest factor

Best combo achievable under each normalization mode:

| normalization | best NAMR33 AP50 |
|---|---|
| **temperature** | **0.7017** |
| minmax | 0.6371 |
| none (raw scores) | 0.6343 |

**All 10 top-scoring combos use temperature normalization; zero use minmax or none.**
Without calibration the 3-model ensemble scores 0.6343 — *worse than RT-DETR alone*
(0.6684). Uncalibrated fusion actively destroys accuracy here.

Fitted temperatures explain why:

| model | T | ECE before → after | reading |
|---|---|---|---|
| A — YOLOv11m | 1.109 | 0.0238 → 0.0226 | mildly overconfident |
| B — RT-DETR-l | **0.519** | 0.0248 → **0.0043** | badly under-confident, flat |
| C — Faster R-CNN | 1.196 | 0.0232 → 0.0269 | overconfident softmax |

A 2.3× spread between B and C. RT-DETR emits ~300 low-flat-scored boxes per image
(498,300 detections total) against Faster R-CNN's 67,484 saturated ones; without
rescaling, C's confident-but-coarser boxes dominate WBF's coordinate average and drag
fused boxes off the better RT-DETR localisations. T=0.519 sharpens B by 5.7× on ECE.

(C's ECE rises slightly while its NLL falls — temperature minimises NLL, not ECE. Its
fused contribution is what matters, and the grid confirms the setting.)

## Subset ablation — does every model earn its slot?

Same winning settings, restricted to pairs:

| subset | NAMR33 AP50 |
|---|---|
| A+B | 0.6875 |
| B+C | 0.6738 |
| A+C | 0.6559 |
| **A+B+C** | **0.7017** |

All three contribute: dropping C (the weakest alone at 0.5130) still costs −0.0142.
Architectural diversity is doing real work — the CNN one-stage, transformer, and
two-stage families make different mistakes.

## Cost (NetScore raw inputs)

| model | params | ckpt MB | GFLOPs @640 | ms/img (bs=1, 3090) |
|---|---|---|---|---|
| A — YOLOv11m | 20,079,222 | 38.7 | 68.3 | 6.42 |
| B — RT-DETR-l | 32,875,946 | 63.3 | 108.1 | 15.54 |
| C — Faster R-CNN | 43,425,278 | 330.8 | 452.4 | 24.60 |
| **ensemble sum** | **96,380,446** | **432.8** | **628.8** | **46.55** |

Note the ICC19 numbers run ~0.06 higher than NAMR33 throughout, because the crosswalk
merges classes the model confuses (all of `anthropogenic_fragment`, `textile`,
`net_like_item`, `cigarette_pack`, `foam_container`, `other` collapse into ICC19 19).
Those confusions DO cost points on the real 34-class metric.

Caveat on C's 330.8 MB: that checkpoint still carries SGD momentum + AMP scaler state.
Weights alone are 43.4M params ≈ **174 MB** fp32. Strip the optimizer before quoting an
ensemble size — the deployable sum is ≈ 276 MB, not 433 MB.

The ensemble costs **9.2× the GFLOPs and 7.3× the latency of detector A** for +0.097 AP50.
Worth knowing if the lightweight-award NetScore ever competes with the accuracy track.

## Training record

| run | epochs | best epoch | wall-time | notes |
|---|---|---|---|---|
| RT-DETR-l | 73/100 | 43 | 8.86 h | early-stopped, patience 30 |
| Faster R-CNN | 26/26 | 19 | 5.80 h | full schedule |

Both ran concurrently on separate GPUs; ~8.9 h wall total. RT-DETR peaked at epoch 43 —
a 60-epoch schedule would have reached the same checkpoint in ~5 h.

## Artifacts

- Dumps: `preds/{y11m_control,rtdetr_l,frcnn_r50v2,wbf_best}.val.json`
- Scores: `scores/{y11m_control,rtdetr_l,frcnn_r50v2,wbf_best}.json`
- Calibration: `scores/*.calib.json`
- Sweep (96 combos): `scores/wbf_sweep.json`
- Costs: `scores/costs.json`

---

# Phase 3 — crop rescorer: GATE PASSED

Classifier: timm `convnext_tiny.fb_in22k_ft_in1k`, 35 classes (34 NAMR33 + background/neighbor),
60,639 train crops / 6,786 val crops, 20 epochs, 28.8 min.
Val accuracy on GT crops **0.8970** vs 0.4895 majority baseline.

## Variant sweep (competition metric, baseline = fused 0.7017)

| alpha | bg_suppress | reassign | AP@0.50 | delta |
|---|---|---|---|---|
| 1.0 | **True** | False | **0.7408** | **+0.0391** |
| 1.0 | False | False | 0.7405 | +0.0388 |
| 0.5 | True | False | 0.7310 | +0.0293 |
| 0.5 | False | False | 0.7297 | +0.0280 |
| 0.25 | True | False | 0.7202 | +0.0185 |
| 0.25 | False | False | 0.7182 | +0.0165 |
| 1.0 | True | **True** | 0.6699 | **−0.0318** |
| 0.5 | True | True | 0.6775 | −0.0242 |

**Shipping config: `--alpha 1.0 --bg-suppress`** (no `--reassign`).

Three clear patterns:

1. **Higher alpha is monotonically better.** At alpha=1.0 the score becomes `s x p[class]` —
   the classifier's probability fully re-weights the detector's confidence. A 224px
   classifier that sees only the cropped object is a better class-confidence estimator
   than the fused detection score.
2. **`--bg-suppress` adds a small, consistent +0.002–0.003.** The background/neighbor
   class is doing real false-positive suppression.
3. **`--reassign` is destructive: −0.022 to −0.032.** It relabels 70,167 of ~160k boxes;
   each wrong relabel creates a false positive in the new class *and* destroys a true
   positive in the old one. The classifier's argmax on *detection* crops is not reliable
   enough for hard relabeling, even gated at p>0.6. Left off by default.

## Per-class effect

27 of 34 classes improved, 6 regressed.

| class | fused | rescored | delta |
|---|---|---|---|
| cigarette_pack | 0.750 | 0.896 | +0.146 |
| non_pet_food_container | 0.452 | 0.581 | +0.128 |
| drink_carton | 0.773 | 0.895 | +0.122 |
| disposable_food_container | 0.702 | 0.792 | +0.091 |
| food_wrapper | 0.636 | 0.723 | +0.087 |
| … | | | |
| foam_buoy_float | 0.597 | 0.580 | −0.017 |
| non_food_plastic_container | 0.604 | 0.525 | −0.079 |

## Final pipeline standing (competition metric, NAMR33 AP@0.50)

| stage | AP@0.50 |
|---|---|
| original YOLOv11n baseline | 0.4767 |
| YOLOv11m (Phase 1 winner) | 0.6049 |
| A+B+C WBF fused (Phase 2) | 0.7017 |
| **+ crop rescorer (Phase 3)** | **0.7408** |

**+0.264 over the starting baseline; qualification gate is 0.60.**

Caveat: the 0.8970 classifier accuracy is measured on ground-truth crops, a cleaner
distribution than the detection crops it sees in deployment. The +0.0391 mAP gain is
measured end-to-end on real fused detections and is the number that counts.

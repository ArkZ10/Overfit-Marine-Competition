# Current execution plan — audit first, then clean EFS, ranking, and missing labels

Last updated: 2026-08-18 UTC.

This is the operational source of truth: what has been established, what is running, what
comes next, and which dependencies block each step. `CLAUDE.md` remains the detailed
experiment ledger and competition-rules reference.

Historical EFH/ABEF plans are superseded. A, B, and H are dropped. Our available local
detectors are E, F, and the completed clean S. Teammate models are not currently available,
so they do not enter the critical path.

## 1. Objective and decision rule

Build a legal, official-data-only pipeline whose gains survive the clean split and transfer to
the leaderboard. We will diagnose the failure mechanism before spending GPU time on another
large detector.

The working sequence is:

1. Audit clean E/F immediately while S trains.
2. Finish and measure clean S; establish whether EFS adds genuinely complementary errors.
3. Repair calibration/ranking and unsafe crop-negative construction.
4. Test pseudo-positive **ignore** mining through a controlled F retrain.
5. Only then choose a new heavy detector targeted at the remaining error mechanism.

## 2. Current diagnosis

The principal failure is low precision and poor candidate ranking under incomplete/selective
annotations—not inadequate recall, localization, or conventional long-tail imbalance.

Evidence accumulated so far:

- Maximum recall reaches about 0.95, while precision is only about 0.37 at half recall.
- AP50 to AP75 falls only about 6%, so localization is already comparatively strong.
- `corr(log10(n_train), AP50) = -0.324`: abundant classes are worse, not better. Another
  class-balancing method does not follow from the evidence.
- At confidence 0.25, 87.6% of apparent misses still have a correct proposal by confidence
  0.001. The model often sees the object but ranks it too low.
- Classes 19 and 26 are universally weak, around 0.29–0.37 AP50.
- Visual audits found roughly 67–72% of apparent false positives for classes 22, 26, and 28
  were visible objects without matching annotations.
- Geometric signals are separable but redundant with confidence; geometric reranking did not
  supply independent signal.
- Global temperature calibration materially improved fusion in the old experiments, showing
  that member score scales are incompatible. It does not by itself solve selective-label noise.

Likely mechanism:

1. Visible but unannotated objects are treated as background during detector training.
2. The old crop rescorer samples annotation-empty regions as hard background, even when they
   contain real unlabeled debris.
3. Calibration can also be corrupted when unmatched predictions are automatically treated as
   negatives.
4. AP rewards within-class ordering, but the current post-processing stack does not directly
   optimize ranking under ambiguous labels.

The test annotations may follow the same selective policy. Therefore unmatched visible objects
must not automatically become hard positives either. The safe first intervention is to ignore
suspected unlabeled foreground as negative evidence, then learn the benchmark's ranking policy.

## 3. Completed work

### Dataset and validity

- [x] Read and record the competition rules; submission `x,y,w,h` are integers.
- [x] Remove A and B: they used non-official training data, introduced AGPL exposure, and were
  weak ensemble members.
- [x] Drop H after three weak/non-additive tests. Do not retry H.
- [x] Convert all 15,127 official `MDImageDataset2` images, including 83 `.jpeg` files, string
  image IDs, and the `filename` field.
- [x] Replace the contaminated validation setup with a fresh seed-42 official split:
  13,614 train / 1,513 validation / 757 `val_fit` / 756 `val_sel`.
- [x] Treat every number from the old contaminated validation split—including the old 0.7582
  headline—as historical evidence only.

### Clean baselines

- [x] Retrain clean E. Full clean-val AP50: **0.6941**; AP50:95: **0.5971**.
- [x] Retrain clean F. Full clean-val AP50: **0.6638**; AP50:95: **0.5581**. Best epoch: 39.
- [x] Confirm the late RTMDet behavior: heavy augmentation suppresses early metrics, followed
  by a large gain when the pipeline switches after epoch 30. Never judge an RTMDet-family run
  before epoch 32.
- [x] Produce exact-ID clean E/F prediction dumps and subset ground truth.
- [x] Fit temperatures on `val_fit`: E `T=0.533`, F `T=0.461`.
- [x] Freeze clean EF WBF on `val_fit`: temperature normalization, IoU 0.65, skip 0.001,
  weights 1.1/1.0, `conf_type=avg`.
- [x] Open `val_sel` once: EF **0.7257**, E **0.7116**, delta **+0.0141**.
- [x] Submit `clean_ef_submission.csv`: test score **0.6981512**, about 0.0198 below the old
  ~0.718 result. Clean EF is valid evidence but not a leaderboard upgrade; do not spend another
  upload tuning EF.

### Architecture evidence

- [x] Establish S versus F as a backbone-controlled experiment: same RTMDet head, neck, loss,
  schedule, and training list; the principal change is Swin-L/ImageNet-22k versus the lighter
  backbone.
- [x] On the old split S beat F by **+0.0445** despite a randomly initialized neck/head and
  batch 4 versus F's 16. This makes stronger pretrained representations a proven direction,
  though clean S must still confirm it.
- [x] Old-split EFS reached **0.7421**, exceeding old ABEF **0.7395** with fewer, legal members.
  This is motivation, not clean proof.
- [x] Finish clean S. Best checkpoint is epoch 34: full clean-val AP50 **0.6858** and
  AP50:95 **0.5799** (fit AP50 **0.6768**, sel AP50 **0.6977**). Epoch 40 regressed to
  AP50 0.649, so use epoch 34.
- [x] Fit S temperature on `val_fit`: **T=0.462**.
- [x] Freeze clean EFS WBF on `val_fit`: temperature normalization, IoU **0.70**, skip
  0.001, weights **1.1/0.9/1.0** for E/F/S, `conf_type=avg`.
- [x] Open `val_sel` once: EFS **0.7497** versus EF **0.7257**, delta **+0.0240**.
  Full clean-val EFS is **0.7399 AP50 / 0.6377 AP50:95**.
- [x] Compare Cross-Model NMS and a WBF-plus-NMS hybrid using `val_fit` only. They failed:
  best Cross-Model NMS **0.6995**, hybrid **0.7127**, plain calibrated WBF **0.7312**.
  Do not put Cross-Model NMS after EFS WBF; it removes useful complementary boxes.

## 4. Currently underway

- [x] Paired image-level bootstrap of frozen EF versus frozen EFS on clean `val_sel`:
  mean delta **+0.0234**, 95% CI **[+0.0161, +0.0323]**, P(delta>0) **100%** (400 resamples).
- [x] Run epoch-34 S test inference and generate the frozen EFS submission:
  `12_ensemble/clean_efs_submission.csv` (145,589 rows, all 2,092 images; schema and bounds
  validated). This file is ready for manual upload; leaderboard result is pending.
- Controlled F missing-label retrain is running on GPU 0. Full-train E/F/S teacher dumps are
  complete. Calibrated all-three same-class consensus (score >=0.30, agreement IoU >=0.50,
  GT IoU <0.30) produced **806 ignore regions across 543 train images**, versus 28,956 official
  annotations. The clean validation set is unchanged. The run passed epoch 1 iteration 100;
  initial ETA was about 5.5 hours.
- Safe crop-feature rescorer is running on GPU 1. Its supervised train set has **28,956
  official positives + 27,171 E/F/S-vetoed negatives**; **77,578 ambiguous consensus crops
  are excluded from loss**. `val_fit` has 1,644 positives + 1,513 safe negatives, with 4,467
  ambiguous crops excluded. It uses ConvNeXt-Tiny IN-22k, natural sampling (no class-balanced
  prior distortion), and selects by validation loss. Training finished; epoch 3 was selected.
  Frozen EFS `val_fit` improved from **0.7312 to 0.7427 (+0.0115)** with alpha 0.5,
  background suppression on, and class reassignment off. The one-time frozen `val_sel`
  confirmation **passed: 0.7497 -> 0.7640 (+0.0143 AP50)**; AP50:95 improved 0.6499 ->
  0.6599. Paired bootstrap is running. No further rescorer tuning is authorized on `val_sel`.
  Frozen EFS test rescoring is running on GPU 1 over 268,488 candidates; once complete, create
  and schema-check `clean_efs_safe_rescored_submission.csv` for manual upload.

## 5. Work that can proceed now — not blocked by S

The local audit of E and F is the current priority and can run while S trains:

- [x] Rebuild clean E, F, and EF confusion/error summaries at confidence 0.001, 0.05, and 0.25.
- [x] Produce threshold-free error decomposition: correct class/correct box, class confusion,
  localization miss, duplicate, and unmatched foreground/background candidate.
- [ ] Generate per-class precision-recall curves and AP deltas for E, F, and EF.
- [x] Measure E/F proposal complementarity: unique true positives, shared coverage, and
  annotation-unmatched agreement.
- [x] Mine high-confidence E/F consensus boxes unmatched to GT and create contact sheets for
  manual review.
- [ ] Estimate class- and context-dependent annotation propensity from reviewed samples.
- [x] Audit crop-rescorer data construction and identify every path that labels random or
  annotation-empty crops as background.
- [x] Implement and smoke-test a safe crop-manifest builder supporting positive, teacher-vetoed
  clean-negative, and ambiguous
  / ignore states.
- [x] Compare global temperature, Platt scaling, and isotonic regression with five-fold
  image-level out-of-fold evaluation. Temperature remains frozen; flexible isotonic overfit.
- [ ] Prepare per-class calibration shrinkage where support permits.
  and per-class shrinkage where support permits.
- [ ] Prepare the candidate-table schema for the later EFS ranker.
- [x] Test a conservative E/F metadata-only reranker out of fold. It failed: unchanged EF
  0.69956 AP50; every nonzero reranker strength reduced AP. Do not use this version.

None of these tasks requires clean S. Any E/F-only conclusion must be marked provisional until
S is included where ensemble consensus matters.

## 6. Work formerly blocked by S

The following cannot be completed until clean S finishes and its predictions are dumped:

- [x] Clean solo-S AP and aggregate error profile.
- [x] Fit S calibration on `val_fit`.
- [ ] Clean subset ablation: E, F, S, EF, ES, FS, and EFS.
- [ ] Three-model consensus/discordance analysis.
- [x] Final EFS WBF weights and fusion recipe.
- [ ] Final EFS candidate table and EFS ranker.
- [x] EFS `val_sel` confirmation and paired bootstrap; test inference remains.
- [ ] E/F/S consensus pseudo-positive-ignore mining at full strength.

Teammate model weights, manifests, and prediction dumps are unavailable. That blocks a
cross-team audit and cross-team fusion only; it does **not** block the local E/F audit, S
training, or the clean EFS pipeline.

## 7. Dependency map

```text
clean E/F dumps
    ├──> E/F error audit ──> safe crop-data design ──> provisional calibration/ranker work
    └──> two-model unmatched consensus ──> manual annotation audit

clean S training (running)
    └──> S dump + calibration
            ├──> E/F/S subset ablation ──> freeze EFS ──> val_sel once ──> submit/no-submit
            ├──> final EFS candidate table ──> safe rescorer/ranker
            └──> 3-model consensus mining ──> controlled F ignore retrain

controlled F ignore retrain
    └──> redump + recalibrate + re-ablate ──> decide whether the mechanism is proven
                                                    └──> choose targeted heavy detector

teammate artifacts (unavailable)
    └──> cross-team audit/fusion only; no dependency for the local critical path
```

## 8. Phase A completion — immediately after S

1. Select the valid post-switch S checkpoint using `val_fit` under the existing rules.
2. Dump S predictions with exact official image IDs for full val, `val_fit`, `val_sel`, and test.
3. Measure solo S on full validation and both halves.
4. Fit S temperature/calibration using `val_fit` only.
5. Run all seven E/F/S subset evaluations with calibration applied inside WBF.
6. Tune only on `val_fit`; freeze the best legal recipe.
7. Open `val_sel` once and use paired bootstrap to quantify the delta and uncertainty.
8. Create a test submission only if the gain clears the upload gate below.

## 9. Phase B — safe calibration, rescorer, and ranker

1. Rebuild crop training data without arbitrary random-background labels.
2. Mark detector-supported annotation-unmatched regions as ambiguous/ignore.
3. Use hard negatives only where the evidence strongly supports background; do not infer
   background solely from absence of an annotation.
4. Remove class-balanced sampling from probability-producing models or explicitly correct the
   induced prior shift.
5. Separate object/background confidence from conditional class confidence where practical.
6. Compare temperature, Platt, and isotonic calibration on `val_fit`; use model-by-class fits
   only with shrinkage and enough support.
7. Build one row per fused candidate with raw member scores, support mask/count, pairwise IoUs,
   score statistics, calibrated/fused score, box geometry, neighborhood density, same-class
   count, and crop/image context.
8. Train a low-capacity class-conditional or pairwise ranker. Same-class IoU>=0.5 matches are
   positives; ambiguous unmatched candidates are ignored, downweighted, or handled with a
   positive-unlabeled objective—not automatically made negative.
9. Freeze on `val_fit`; confirm once on `val_sel` with paired bootstrap.

## 10. Phase C — pseudo-positive ignore mining and controlled F retraining

1. Generate out-of-fold predictions or use a properly separated teacher over official train.
2. Mine high-consensus E/F/S boxes that do not overlap annotations.
3. Convert those regions first to classification/background **ignore areas**, not hard ground
   truth boxes.
4. Keep box regression supervised only by official annotations.
5. Retrain F with the same checkpoint, seed, schedule, clean boundary, and all other settings
   held fixed. The ignore policy is the controlled variable.
6. Redump, recalibrate, rerun the subset ablation, and test the paired delta.
7. Only if ignore mining wins, consider consistency learning or carefully weighted soft labels
   on ambiguous regions.

## 11. Phase D — targeted new model, not blind model addition

A “heavy complete detector” means adding/replacing a detector with an end-to-end pretrained
backbone, neck, and head—not merely swapping in another backbone with random task heads. It is
an additional model experiment, but only after the audit identifies what complementary signal
is missing.

- Do not duplicate the teammate's Co-DETR/Co-DINO experiment (~0.71 test) without evidence of a
  materially different configuration or error profile.
- Prefer a detector whose training objective or pretraining addresses the measured weakness:
  ranking quality, sparse/incomplete annotation robustness, or representations complementary
  to E/F/S.
- Test higher **training** resolution only if the audit shows scale-dependent misses. Do not
  default to SAHI/high-resolution inference; localization and maximum recall are not the main
  failure.
- The new detector must earn inclusion through solo quality plus unique corrected errors, not
  simply increase member count.

## 12. Evaluation and submission gates

- Tune only on clean `val_fit`; `val_sel` is a one-time confirmation set.
- Report all-34 AP plus a macro over classes with adequate support. Do not optimize around
  classes 23, 31, or 32, whose support is too thin.
- Require approximately **+0.008 paired local improvement** as a minimum research signal.
- For another leaderboard upload, require more than a marginal gain: clean EF's +0.0141
  `val_sel` improvement failed to transfer, producing a 0.698 test score. Favor a larger,
  bootstrap-supported gain or a mechanism independently validated by the audit.
- Validate submission schema: integer `x,y,w,h`, correct image IDs, no illegal data/models, and
  exact row format.
- Record every frozen recipe and leaderboard outcome in `CLAUDE.md` immediately.

## 13. Do not re-propose without new contradictory evidence

- H, or indiscriminately adding ensemble members.
- A/B or non-official training images.
- More class balancing, repeat-factor sampling, or Copy-Paste as the primary fix.
- Better box regression as the primary fix.
- SAHI or high-resolution inference as the default fix.
- Hard crop-class reassignment.
- Blind hard pseudo-labeling of annotation-unmatched detections.
- Treating every unmatched prediction as a calibration/rescorer negative.
- Tuning on the contaminated split or repeatedly opening clean `val_sel`.
- Reading an RTMDet-family run before epoch 32 as a final result.

## 14. Immediate execution order

1. Keep clean S running on GPU 0.
2. Use the available capacity for clean E/F auditing and safe crop-data/ranker infrastructure.
3. When S finishes, dump and calibrate it before any new training experiment.
4. Run the complete E/F/S subset ablation and freeze EFS.
5. Decide whether EFS clears the submission gate.
6. Run the safe rescorer/calibration/ranker experiment.
7. Run the controlled F pseudo-positive-ignore retrain.
8. Use the resulting evidence to select—or reject—a new heavy complete detector.

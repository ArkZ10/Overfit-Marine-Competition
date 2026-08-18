# Clean E/F audit

Official clean validation: 1,513 images, 3,233 boxes. Predictions are evaluated down to the stored 0.001 floor. These counts diagnose proposal coverage and error type; they are not a replacement for COCO AP.

## Ground-truth coverage at IoU≥0.5

| detector coverage | boxes | percent |
|---|---:|---:|
| E | 3087 | 95.5% |
| F | 3052 | 94.4% |
| Both | 3013 | 93.2% |
| E only | 74 | 2.3% |
| F only | 39 | 1.2% |
| Union | 3126 | 96.7% |
| Neither | 107 | 3.3% |

## Prediction outcomes at the 0.001 dump floor

| model | TP | duplicate | class confusion | localization | unmatched |
|---|---:|---:|---:|---:|---:|
| E | 3087 | 12614 | 74825 | 21756 | 341618 |
| F | 3052 | 5214 | 44624 | 28224 | 372786 |

## E/F consensus detections unmatched to official GT

At score≥0.05 for both models and same-class IoU≥0.5: **11895 pairs**. These are review candidates, not proven positives.

| class | count |
|---|---:|
| 26 anthropogenic_fragment | 4987 |
| 19 other | 1107 |
| 22 foam_buoy_float | 823 |
| 14 fishing_net_rope | 715 |
| 28 soft_float | 676 |
| 0 plastic_bottle | 613 |
| 1 plastic_bottle_cap | 352 |
| 7 plastic_bag | 302 |
| 15 fishing_buoy_float | 220 |
| 5 straw | 202 |

## Visual review of the highest-supported candidates

Thirty unique-image candidates were rendered for each of classes 26, 19, 22, and 28.

- Class 26 (`anthropogenic_fragment`): the high-confidence sample is dominated by visible
  unannotated foam/plastic fragments. Some class-boundary ambiguity with foam remains.
- Class 22 (`foam_buoy_float`): the sample is dominated by visible foam pieces without a
  matching official box.
- Class 28 (`soft_float`): many candidates are plausible visible floats/fragments, with some
  ambiguity against neighboring material classes.
- Class 19 (`other`): mixed. It includes real unannotated debris, but also driftwood, rocks,
  and ambiguous background. Agreement between E and F is therefore insufficient for hard GT.

This supports using high-consensus unmatched regions as **ignore/ambiguous supervision** first.
It does not support turning the entire pool into hard pseudo-labels.

## Crop-rescorer construction audit

The old `rescorer/make_crops.py` creates class-34 background in two unsafe ways:

1. random crops whose IoU is below 0.1 against every *annotated* GT box;
2. GT-neighbor crops shifted by 0.5–1.0 box widths/heights.

With selective annotations, neither condition proves that a crop contains no object. The
trainer then uses inverse-frequency `WeightedRandomSampler`, so the resulting softmax also
does not represent the deployment class prior. The old rescorer remains historical evidence;
new crop data must carry positive, clean-negative, and ambiguous/ignore states.

## Calibration comparison

Global temperature, Platt-on-logit, and isotonic mappings were fitted on `val_fit`. The
in-sample isotonic result looked promising (EF AP50 0.7078 versus temperature 0.6996), but
five-fold image-level out-of-fold evaluation removed the gain:

| mapping | out-of-fold EF AP50 |
|---|---:|
| temperature | **0.69894** |
| isotonic | 0.69832 |
| Platt | 0.69758 |

Therefore global temperature remains the frozen mapping. `val_sel` was not opened. The full
comparison is in `12_ensemble/scores/clean_ef_calibrator_comparison.json`.

## Safe crop-manifest builder

`build_safe_crop_manifest.py` now emits three explicit states without copying image data:
official positives, all-teacher consensus unmatched regions as `target=-1` ambiguous/ignore,
and random negatives vetoed by both GT and every teacher proposal above a low score floor.
The `val_fit` smoke test produced 1,644 positives, 6,458 ambiguous regions, and 1,514 clean
negative candidates with no sampling shortfall. The smoke manifest is an infrastructure test,
not training data. The final train manifest requires train-split teacher dumps and, preferably,
clean S consensus.

## Conservative E/F reranker result

`rank_clean_ef.py` trained a small gradient-boosted model using fused confidence, calibrated
E/F scores, model/box agreement, geometry, and class. Official true positives were positives;
duplicates, wrong-class overlaps, and localization failures were negatives. The 49,621
annotation-unmatched candidates were excluded rather than falsely labelled background.

| reranker strength | five-fold out-of-fold AP50 |
|---:|---:|
| 0 (unchanged EF) | **0.69956** |
| 0.25 | 0.69726 |
| 0.50 | 0.69391 |
| 1.00 | 0.69000 |
| 2.00 | 0.68368 |

The safe decision is **no reranking**. This objective adds no useful ranking signal beyond EF
confidence. `val_sel` was not opened. It does not disprove a crop-feature or E/F/S ranker, but
it rules out this E/F metadata-only version.

## Interpretation guardrails

- `unmatched` means unmatched to the official annotation, not necessarily background.
- Consensus boxes must be manually sampled before defining ignore thresholds.
- E/F-only consensus is provisional; clean S will provide the stronger three-model test.
- The confusion outputs at 0.001, 0.05, and 0.25 live beside this report in `12_ensemble/scores/`.

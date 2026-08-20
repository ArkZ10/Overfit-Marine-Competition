# Combined Team Audit and Execution Direction

**Competition:** Marine Debris Image Recognition Competition  
**Metric:** pycocotools macro mAP@0.50 over all 34 NAMR classes  
**State date:** 2026-08-19 UTC  
**Sources:** our clean official-data experiments in [`CLAUDE.md`](CLAUDE.md) and
[`CURRENT_PLAN.md`](CURRENT_PLAN.md), plus the teammate's Research Dashboard v7
([overview](https://tilarayhan.lol/), [models](https://tilarayhan.lol/models),
[ensembles](https://tilarayhan.lol/ensembles), [analysis](https://tilarayhan.lol/analysis),
[submissions](https://tilarayhan.lol/submissions)).

## 1. Executive conclusion

The combined evidence identifies **two different residual-error populations**:

1. **Objects that at least one detector sees:** our audit shows that candidate recall is high,
   but confidence ordering and precision are poor. Selective/missing annotations make this
   worse by teaching visible objects as background and by corrupting naive calibration or
   crop-negative construction.
2. **Objects that every detector misses:** the teammate's 12-model shared audit shows that
   these are overwhelmingly tiny. Objects below 1% of image area are only 29.6% of validation
   GT, but account for 78.9% of universal misses.

Therefore the next system should not be "one more blind ensemble member." The justified shape
is:

```text
higher-resolution strong detector
    -> calibrated complementary fusion
    -> safe crop-feature rescoring
    -> frozen validation gate
    -> hidden submission
```

Higher resolution attacks **missing candidates**. Calibration and safe crop rescoring attack
**bad ranking among existing candidates**. Neither intervention replaces the other.

## 2. Comparability boundary

The two teams' local validation figures must not be placed in one numerical leaderboard unless
the exact image IDs and annotations are confirmed identical.

| Evidence source | Validation boundary | Safe use |
|---|---|---|
| Our clean rebuild | 1,513 official images; seed-42 split; 757 `val_fit` / 756 `val_sel` | Controlled E/F/S/X and rescorer decisions |
| Teammate Full Shared Audit V1 | 3,022 images; 6,449 GT objects; 12 models | Cross-model, class, size, and density diagnosis |
| AIdea preliminary | Organizer hidden test | Competition-facing transfer evidence |

For example, our DEIM-X `0.71627` and the teammate's Co-DINO `0.71652` are numerically close,
but must not be called a tie until evaluated on the same local image IDs. Hidden submission
scores are directly comparable because they use the same organizer test.

## 3. Our branch: clean official-data evidence

### 3.1 Current clean detectors

| Detector | Architecture | Clean AP50 | Decision |
|---|---|---:|---|
| **X** | DEIM-D-FINE-X / HGNetv2-X | **0.71627** | New strongest solo model; fusion evaluation pending |
| **E** | DEIM-D-FINE-L / HGNetv2 | **0.6941** | Retain |
| **S** | RTMDet-L / Swin-L IN-22k | **0.6858** | Retain; strong representation evidence |
| **F** | RTMDet-L / CSPNeXt | **0.6638** | Retain until displaced by ablation |
| F + consensus ignore | Controlled missing-label retrain | 0.667 | Reject: EFS `val_fit` fell by 0.0041 |
| F + teacher-weighted background | Controlled fine-tune | 0.657 | Reject |
| F + nnPU | Positive-unlabeled adaptation | 0.652 | Reject |

DEIM-X completed 32 epochs. Its best epoch was 28 at **0.71627 AP50 / 0.61976 AP50:95**.
BF16 eliminated the FP16 NaN failure without changing the model or training data.

### 3.2 Current clean pipeline

| Stage | Local evidence | Hidden evidence |
|---|---:|---:|
| Clean EFS WBF | 0.7497 `val_sel` | 0.7175357 |
| EFS + safe ConvNeXt-Tiny crop rescorer | **0.7640 `val_sel`** | **0.729** |
| ConvNeXt-Base replacement rescorer | 0.7419 `val_fit` vs Tiny 0.7427 | Not submitted; rejected by gate |

The safe Tiny rescorer improved `val_sel` by **+0.0143** and hidden by roughly **+0.0115**.
The transfer is real; the remaining deficit is mainly upstream detector quality.

### 3.3 Our error diagnosis

- Maximum recall is approximately 0.95.
- Precision is only approximately 0.37 at half recall.
- At confidence 0.25, 87.6% of apparent misses still have a correct proposal at confidence
  0.001.
- AP50 to AP75 falls only about 6%; localization is not the primary bottleneck.
- `corr(log10(n_train), AP50) = -0.324`; abundant classes are worse, so conventional
  long-tail rebalancing is not supported.
- Visual review found many high-confidence unmatched predictions that are real visible debris
  absent from the annotations.
- Metadata-only reranking failed. Actual crop/image features were required for a gain.
- Random annotation-empty crops are unsafe negatives. The successful rescorer uses official
  positives, teacher-vetoed safe negatives, and ignores ambiguous crops.

**Our conclusion:** for objects already proposed, the dominant problem is confidence ranking
and selective-background supervision, not box localization or raw candidate recall.

## 4. Teammate branch: 12-model audit

### 4.1 Accepted standalone models

These are the teammate dashboard's canonical local scores, not our clean-split scores.

| Rank | Model | Backbone / pretraining | Validation AP50 |
|---:|---|---|---:|
| 1 | Co-DINO | ViT-L + Objects365 | **0.716519509** |
| 2 | MI-DETR | Swin-L | **0.715463918** |
| 3 | RT-DETRv4-X | X variant | 0.704383260 |
| 4 | Mr. DETR Align | Swin-L + Objects365 | 0.690220713 |
| 5 | Stable-DINO | Swin-L | 0.671015612 |
| 6 | D-FINE-X | Objects365 | 0.624520144 |
| 7 | RF-DETR-L | Large | 0.614911187 |
| 8 | DEIMv2-L | DINOv3 ViT-S/16 | 0.587074995 |
| 9 | DEIM-D-FINE-X | Objects365 | 0.563005103 |
| 10 | Corrected DINO | ResNet-50, four-scale | 0.561574311 |
| 11 | Stable-DINO | ResNet-50 | 0.540012673 |
| 12 | RT-DETR | ResNet-50 | 0.416151944 |

DEIMv2-L has already been tested and was weak locally. The updated permission to use DINOv3
does **not** by itself justify duplicating DEIMv2-L or assuming that DINOv3 guarantees a gain.

### 4.2 Best teammate ensemble

The current canonical ensemble is **Co-DINO ViT-L + MI-DETR Swin-L**, using class-aware,
equal-weight WBF:

| Parameter | Value |
|---|---|
| IoU threshold | 0.45 |
| Skip threshold | 0.001 |
| Model weights | 1.0 / 1.0 |
| Confidence aggregation | `max` |
| Output | top 300/image; integer `xywh` |
| Canonical validation | **0.7437401061** |
| AIdea preliminary hidden | **0.7534597** |

Fusion decision record:

| Method | Validation | Hidden | Decision |
|---|---:|---:|---|
| Equal-weight class-aware WBF | 0.7437401 | **0.7534597** | Canonical |
| Class-bucket WBF | **0.7445863** | 0.7534510 | Reject: no hidden gain; more complex |
| Cross-Model NMS | 0.7414318 | 0.7507509 | Superseded |
| Tiny-area gated RT-DETRv4 | 0.7416704 | 0.7484452 | Reject: hidden regression |
| Unrestricted three-model union | 0.7397365 | Not submitted | Reject |

The teammate result proves that WBF can beat Cross-Model NMS for this pair. It does **not**
prove that `max`, IoU 0.45, or equal weights transfer to E/F/S/X. Our previous cross-member
`max` test failed badly, so fusion parameters remain member-set-specific and must be gated on
`val_fit`.

### 4.3 Shared Audit V1

Audit scope:

- 12/12 accepted models
- 3,022 validation images
- 6,449 GT objects
- 408/408 model-class rows
- 66/66 unique model pairs

Main findings:

- Co-DINO wins AP50 on 18/34 classes.
- MI-DETR wins 9 classes; RT-DETRv4-X wins 5; Mr. DETR and Stable-DINO Swin-L win 1 each.
- RT-DETRv4-X leads recall for objects below 0.1% image area.
- D-FINE-X leads the other major size bands.
- MI-DETR leads sparse scenes with 0–5 objects/image.
- Co-DINO leads the 6–10, 11–20, and >20 objects/image slices.
- There are 854 universal failures and 399 universal misses among 6,449 GT objects.
- Objects below 1% image area are 29.6% of GT, but 515/854 (**60.3%**) of universal failures
  and 315/399 (**78.9%**) of universal misses.
- For the most extreme below-0.1% band, 140/224 are universal failures and 124/224 are
  universal misses.

**Teammate conclusion:** small-object scale is the dominant error shared by diverse detector
families. Resolution should be tested before more broad fusion tuning, long-tail methods,
counterfactual augmentation, or an internal MoE.

## 5. Unified diagnosis

The audits answer different conditional questions:

| Question | Evidence | Answer |
|---|---|---|
| If a correct candidate exists, why is AP still low? | Our low-threshold proposal audit and rescorer experiments | Bad confidence ordering, false-positive pressure, and selective/missing annotation supervision |
| Which objects have no correct candidate from any strong model? | Teammate 12-model intersection audit | Disproportionately tiny objects |
| Is localization the main problem? | Our AP50/AP75 gap | No |
| Is ordinary long-tail balancing the main problem? | Our abundance/AP correlation and failed RFS/copy-paste | No |
| Does adding arbitrary models solve it? | H, modified F, unrestricted teammate union | No |
| Does learned visual evidence improve ranking? | Safe crop rescorer | Yes |
| Is higher resolution now justified? | Teammate size-conditioned universal misses | Yes, as a controlled experiment |

There is no conflict between "recall reaches 0.95" and "tiny objects dominate universal
misses." The first is an aggregate upper envelope over all candidates and objects. The second
isolates the much smaller set that remains missed by every audited detector. Aggregate recall
can be high while the remaining misses are highly concentrated in one size band.

## 6. Updated competition-rule boundary

The organizer now permits models under research/non-commercial licences, including DINOv3,
for non-commercial competition research, provided all licence and disclosure requirements are
met. This supersedes the old blanket DINOv3 ban recorded in `CLAUDE.md`.

Operational consequences:

- DINOv3 may be used as a fine-tuned backbone or offline distillation teacher.
- The exact model, checkpoint, source, licence, external pretraining data, and use mode must be
  recorded in `12_ensemble/ATTRIBUTIONS.md` before submission.
- Permission does not establish efficacy. Teammate DEIMv2-L + DINOv3 scored only 0.5871.
- RF-DETR XL/2XL PML eligibility needs a fresh licence preflight under the new organizer rule;
  the teammate dashboard's old "license excluded" state may be stale.

## 7. Joint execution plan

### Phase 0 — completed 2026-08-19

- [x] Re-evaluated `best_stg2.pth`: **0.7163 AP50 / 0.6198 AP50:95**, reproducing the
  epoch-28 result.
- [x] Dumped X for full clean validation, `val_fit`, `val_sel`, and test with exact official
  IDs. X scores **0.7033 fit / 0.7332 sel**.
- [x] Fitted temperature on `val_fit` only: **T=0.5254**; ECE improved 0.0171 -> 0.0024.
- [x] Size audit: proposal recall@0.50 is **0.8936** below 0.1% image area, **0.9204** at
  0.1--1%, and **0.9731** at >=1%. Corresponding macro AP50-like ranking is 0.3811,
  0.6587, and 0.8962. Our split independently confirms the tiny-object bottleneck.
- [x] Ran every X-containing clean subset at the frozen EFS settings. EFSX led at 0.7454;
  FSX was second at 0.7428. E still earns its slot.
- [x] Ran a bounded 60-configuration `val_fit` grid over EFSX/FSX, IoU 0.55--0.75,
  `avg`/`max`, and three X weights. `max` failed to transfer; `avg` remained correct.
- [x] Froze **EFSX**, temperature normalization, IoU **0.65**, skip **0.001**, weights
  **1.1/0.9/1.0/1.1**, `conf_type=avg`.
- [x] Frozen `val_fit`: **0.74629**, versus EFS **0.73119** (**+0.01510**).
- [x] One-time untouched `val_sel`: **0.76351**, versus EFS **0.74967** (**+0.01384**).
- [x] Paired 400-sample bootstrap on `val_sel`: mean delta **+0.0139**, 95% CI
  **[+0.0060, +0.0232]**, P(delta>0) **99.5%**.
- [x] Applied the already-frozen safe ConvNeXt-Tiny rescorer to EFSX without retraining or
  retuning. It improved `val_fit` **0.74629 -> 0.75287** (**+0.00658**) and the one-time
  `val_sel` confirmation **0.76351 -> 0.77358** (**+0.01007**); `val_sel` AP50:95 improved
  **0.66444 -> 0.67081**. Settings remain alpha 0.5, background suppression on, and no
  class reassignment.

**Verdict:** X passes both the solo and frozen-fusion gates. It is now a canonical member, and
the existing safe Tiny rescorer also transfers positively to EFSX. Do not retune either EFSX
or the rescorer on `val_sel`.

### Phase 1 — controlled resolution experiment

Use the best verified DEIM-X checkpoint as the starting point and change **resolution only**.

1. Start with 1024 rather than 1280 because each RTX 3090 has 24 GB.
2. Keep official train/validation IDs, taxonomy, loss, model, seed, optimizer family, and
   evaluation unchanged.
3. Use BF16, gradient accumulation, and the largest safe physical batch. Record effective
   batch and learning-rate scaling.
4. Select checkpoints by frozen clean validation, reporting both overall AP50 and tiny-object
   recall/AP.
5. Do not call the run successful merely because tiny recall rises; overall AP50 and downstream
   fusion must not regress materially.

**Resolution hypothesis:** retaining more pixels will recover candidates for the tiny objects
that dominate universal misses. It is not a claim that train/test image-size domain shift
exists; that earlier hypothesis was already rejected.

### Phase 2 — ranking after the detector improves

1. Recalibrate every retained member after any resolution/model change.
2. Rebuild the fused candidate dump on `val_fit`.
3. Apply the already-frozen safe Tiny rescorer as the first ranking baseline.
4. If the member set changes substantially, test whether the rescorer's gain survives before
   any retraining or alpha sweep.
5. Never enable class reassignment; it failed repeatedly.
6. Never treat annotation-unmatched crops as automatic background.

### Phase 3 — cross-team fusion when artifacts are available

The teammate's Co-DINO + MI-DETR WBF is already stronger on hidden test than our current EFS
pipeline. Once prediction dumps and manifests are available:

1. Verify taxonomy, image IDs, coordinate convention, preprocessing, and confidence semantics.
2. Reproduce each standalone and the canonical two-model WBF locally.
3. Measure whether our DEIM-X or high-resolution DEIM-X finds unique true positives,
   especially below 1% area.
4. Add X only if it clears a frozen complementarity gate. The failed tiny-area RT-DETRv4
   branch proves that a model can lead tiny recall yet still hurt hidden fusion.
5. Test our crop rescorer on the reproduced teammate fusion using `val_fit`; do not assume its
   EFS gain transfers.

### Phase 4 — new large model only if still justified

Priority after the resolution and cross-team gates:

1. **DINOv3-backed detector only with a controlled reason.** Do not duplicate the failed
   DEIMv2-L. DEIMv2-X remains a candidate, not the default.
2. **RF-DETR XL/2XL licence re-audit.** Consider only after the new organizer permission is
   documented and only with a schedule corrected for RF-DETR-L's early peak.
3. **DINOv3 offline teacher.** Consider for feature distillation or ambiguous-region vetoing,
   not automatic pseudo-label creation.
4. Avoid another low-resolution detector whose likely errors overlap with the existing pool.

#### Model Y launched 2026-08-19 by explicit team decision

- **Architecture:** official DEIMv2-X, DINOv3 ViT-S+ backbone, 50.3M parameters.
- **Controlled reason:** test a larger DINOv3 representation, not duplicate the teammate's
  failed DEIMv2-L. The public X checkpoint reports 57.8 COCO AP versus 56.0 for L.
- **Protocol:** exact official clean train/val IDs, 640 input, 32 epochs, BF16, physical batch
  8, seed 42, public full COCO checkpoint, GPU 1.
- **Smoke evidence:** full checkpoint/backbone load passed; validation forward pass passed;
  training passed 300 optimizer steps with finite losses and about 13.1 GB peak allocated.
- **Gate:** solo clean AP50 first, followed by `val_fit` calibration/member ablation. It is not
  added to EFSX automatically and must not open `val_sel` unless it improves the frozen fit
  pipeline.

## 8. What is settled and should not be repeated

- Do not use non-official training images.
- Do not optimize ICC19; the competition metric is NAMR34 macro AP50.
- Do not retry ordinary class-balanced sampling, RFS, or rare-class copy-paste as the primary
  solution.
- Do not use arbitrary random-background crop labels.
- Do not use crop-rescorer class reassignment.
- Do not use the failed metadata-only reranker.
- Do not append a detector without rerunning member-set ablation.
- Do not place Cross-Model NMS after our EFS WBF; it reduced our local score.
- Do not universalize WBF `avg` or `max`: `avg` won for our EFS cross-member fusion, while
  `max` won for the teammate's Co-DINO/MI-DETR pair and for our intra-member DETR TTA.
- Do not duplicate teammate DEIMv2-L merely because DINOv3 is now legal.
- Do not infer hidden improvement from a tiny-object local gain without end-to-end fusion
  validation; the teammate's tiny-area RT-DETRv4 branch regressed on hidden test.

## 9. Immediate next action

Phase 0 is complete and independently confirms the tiny-object weakness. The next authorized
experiment is **Phase 1: a controlled DEIM-X 640 -> 1024 fine-tune**. Keep EFSX and its
`val_sel` result frozen. Resolution is the only intended model change; judge the run on overall
AP50, tiny-object recall/AP, and downstream frozen-fusion value.

## 10. Paper references supporting the chosen direction

- Zong et al., **DETRs with Collaborative Hybrid Assignments Training (Co-DETR)**,
  ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/papers/Zong_DETRs_with_Collaborative_Hybrid_Assignments_Training_ICCV_2023_paper.pdf
- Nan et al., **MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism**,
  CVPR 2025: https://arxiv.org/abs/2503.01463
- Huang et al., **Real-Time Object Detection Meets DINOv3 (DEIMv2)**, 2025:
  https://arxiv.org/abs/2509.20787
- Simeoni et al., **DINOv3**, 2025/2026: https://openreview.net/pdf?id=2NlGyqNjns
- Chen et al., **Vision Transformer Adapter for Dense Predictions**, ICLR 2023:
  https://arxiv.org/abs/2205.08534
- Yao et al., **Frozen-DETR: Enhancing DETR with Image Understanding from Frozen Foundation
  Models**, 2024/2025: https://arxiv.org/abs/2410.19635

#!/usr/bin/env python3
"""Background-Recalibrated Quality Focal Loss for sparsely-annotated detection.

Registered as `BackgroundRecalibratedQFL`; a drop-in for RTMDet's `QualityFocalLoss`.

THE PROBLEM IT SOLVES (measured on our data, not hypothetical):
71% of detector F's false positives on class 26 (anthropogenic_fragment) sit on
regions with ZERO ground-truth overlap, and inspection shows they are real,
visible, unlabelled plastic. 52 val images contain a confident fragment detection
and no labelled fragment at all. Standard QFL supervises every unmatched anchor
toward 0 for all classes, so a detector that correctly finds an unlabelled object
is punished, and it learns to suppress the very class it is scored on.

Labelled and unlabelled fragments are statistically indistinguishable by size
(median 7.88 vs 7.61 % of image diagonal, Mann-Whitney p=0.461), so no appearance
rule can separate them -- the only available fix is to stop training on them.

MODES
  ignore : negatives whose max predicted prob exceeds `tau` are removed from the
           loss entirely (weight -> 0). This is the positive-unlabeled framing
           (arXiv:2002.04672): unlabelled regions are *unlabeled*, not negative.
           Cannot destabilise training -- it only ever removes gradient.
  flip   : those anchors are instead supervised as POSITIVES of their argmax
           class, with quality target = their own current confidence, i.e. the
           "treat as easy positive" recalibration of BRL (arXiv:2002.05274).
           Stronger, but self-referential: it can entrench an early mistake.

Start with `ignore`. Only try `flip` if `ignore` shows signal.

WARMUP IS MANDATORY, not a nicety. A freshly re-initialised 34-class head puts high
max-probability on almost every anchor, so with tau=0.5 a unit test recalibrated
180/200 anchors -- i.e. BRL would delete nearly all background gradient and the model
would never learn background at all. `warmup_iters` keeps the loss exactly equal to
plain QFL until the head has converged enough for "confident" to mean something.
RTMDet at batch 32 runs ~424 iters/epoch on our 13,577 images, so the default 2000
is ~5 epochs.

Reference: Zhang et al., "Solving Missing-Annotation Object Detection with
Background Recalibration Loss", ICASSP 2020 (arXiv:2002.05274).
"""
import torch

from mmdet.models.losses.gfocal_loss import QualityFocalLoss
from mmdet.registry import MODELS


@MODELS.register_module()
class BackgroundRecalibratedQFL(QualityFocalLoss):
    """QualityFocalLoss that stops punishing confident predictions on unlabelled regions.

    Args:
        tau (float): confidence above which an unmatched anchor is treated as a
            missing annotation rather than background. Our fragment false
            positives sit at 0.55-0.70, so 0.5 catches most of them.
        mode (str): 'ignore' (drop from loss) or 'flip' (supervise as positive).
        warmup_iters (int): behave as plain QFL for this many forward passes.
        max_recalib_frac (float): safety valve. If more than this fraction of
            negatives would be recalibrated in one batch, skip recalibration for
            that batch -- that only happens when the head is not yet confident-
            meaningful, and silently deleting most of the background signal is
            how this loss destroys a run.
    """

    def __init__(self, *args, tau: float = 0.5, mode: str = 'ignore',
                 warmup_iters: int = 2000, max_recalib_frac: float = 0.10,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ('ignore', 'flip'), mode
        self.tau = tau
        self.mode = mode
        self.warmup_iters = warmup_iters
        self.max_recalib_frac = max_recalib_frac
        self._iter = 0
        self._n_recalib = 0        # cumulative, for logging/sanity

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        # RTMDet passes target as (labels, assign_metrics); anything else (the
        # one-hot soft-target path) is left untouched.
        if self.training:
            self._iter += 1
        if (isinstance(target, (tuple, list)) and len(target) == 2
                and self._iter > self.warmup_iters):
            label, score = target
            bg = pred.size(1)                       # FG ids 0..C-1, BG id == C
            with torch.no_grad():
                prob = pred.sigmoid()
                conf, arg = prob.max(dim=1)
                neg = label >= bg
                hit = neg & (conf > self.tau)       # confident, but unmatched
                n_neg = int(neg.sum())
                # safety valve: refuse to recalibrate a large share of negatives
                if n_neg and int(hit.sum()) > self.max_recalib_frac * n_neg:
                    hit = torch.zeros_like(hit)
            if hit.any():
                self._n_recalib += int(hit.sum())
                if self.mode == 'ignore':
                    weight = (torch.ones_like(conf) if weight is None
                              else weight.clone().float())
                    weight[hit] = 0.0
                else:                                # flip: easy positive
                    label = label.clone()
                    score = score.clone().float()
                    label[hit] = arg[hit]
                    score[hit] = conf[hit]
                target = (label, score)
        return super().forward(pred, target, weight=weight,
                               avg_factor=avg_factor,
                               reduction_override=reduction_override)

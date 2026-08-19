"""Non-negative positive-unlabeled classification for RTMDet.

RTMDet has no separate RPN objectness head.  We therefore derive the
probability of "any marine object" from its 34 sigmoid class logits and apply
the nnPU risk to that binary probability.  Official positive anchors retain
the normal Quality Focal Loss (including their competing-class negatives),
while background anchors are treated as unlabeled rather than certain
negatives.

This is an RTMDet adaptation of Eq. 6 in Yang et al., "Object Detection as a
Positive-Unlabeled Problem" (2020), not a verbatim Faster R-CNN port.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.dense_heads import RTMDetSepBNHead
from mmdet.registry import MODELS


@MODELS.register_module()
class PURTMDetHead(RTMDetSepBNHead):
    """RTMDet head with an any-object non-negative PU classification risk."""

    def __init__(self, *args, pu_weight: float = 1.0,
                 prior_momentum: float = 0.9,
                 prior_threshold: float = 0.5,
                 prior_min: float = 1e-4,
                 prior_max: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.pu_weight = float(pu_weight)
        self.prior_momentum = float(prior_momentum)
        self.prior_threshold = float(prior_threshold)
        self.prior_min = float(prior_min)
        self.prior_max = float(prior_max)
        self.register_buffer('pu_positive_prior', torch.tensor(0.01))

    @staticmethod
    def _object_probability(logits: Tensor) -> tuple[Tensor, Tensor]:
        """Return P(any class) and log P(background), stably."""
        log_p_background = F.logsigmoid(-logits).sum(dim=1)
        p_object = -torch.expm1(log_p_background)
        return p_object.clamp(1e-6, 1 - 1e-6), log_p_background

    def _nnpu_objectness(self, logits: Tensor, positive: Tensor,
                         unlabeled: Tensor) -> Tensor:
        p_object, log_p_background = self._object_probability(logits)
        valid = positive | unlabeled
        if self.training and valid.any():
            estimate = (p_object[valid].detach() >= self.prior_threshold).float().mean()
            estimate = estimate.clamp(self.prior_min, self.prior_max)
            self.pu_positive_prior.mul_(self.prior_momentum).add_(
                estimate * (1.0 - self.prior_momentum))

        if not positive.any() or not unlabeled.any():
            return logits.sum() * 0

        pi = self.pu_positive_prior.detach().clamp(self.prior_min, self.prior_max)
        positive_as_positive = -torch.log(p_object[positive]).mean()
        positive_as_negative = -log_p_background[positive].mean()
        unlabeled_as_negative = -log_p_background[unlabeled].mean()
        negative_risk = torch.relu(
            unlabeled_as_negative - pi * positive_as_negative)
        return pi * positive_as_positive + negative_risk

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            labels: Tensor, label_weights: Tensor,
                            bbox_targets: Tensor, assign_metrics: Tensor,
                            stride: List[int]):
        assert stride[0] == stride[1]
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        assign_metrics = assign_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)

        positive = (labels >= 0) & (labels < self.num_classes)
        unlabeled = (labels == self.num_classes) & (label_weights > 0)

        # Preserve QFL and competing-class supervision on official positives,
        # but do not declare every unassigned anchor a true negative.
        positive_weights = label_weights * positive.to(label_weights.dtype)
        loss_cls = self.loss_cls(
            cls_score, (labels, assign_metrics), positive_weights,
            avg_factor=1.0)
        normalizer = assign_metrics.sum().detach().clamp(min=1.0)
        loss_cls = loss_cls + self.pu_weight * normalizer * self._nnpu_objectness(
            cls_score, positive, unlabeled)

        pos_inds = positive.nonzero().squeeze(1)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_bbox_weight = assign_metrics[pos_inds]
            loss_bbox = self.loss_bbox(
                pos_bbox_pred, pos_bbox_targets, weight=pos_bbox_weight,
                avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        return (loss_cls, loss_bbox, assign_metrics.sum(),
                pos_bbox_weight.sum())

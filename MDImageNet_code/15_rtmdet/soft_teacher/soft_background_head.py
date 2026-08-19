"""RTMDet head with soft negative weights in teacher-supported regions.

All E/F/S-agreed, annotation-unmatched boxes arrive through COCO ``iscrowd``
as ``gt_instances_ignore``.  Official assignment is unchanged.  Priors that
remain background but lie inside a teacher-supported region retain a small,
non-zero negative classification weight instead of being treated as either a
hard negative or a hard pseudo-positive.  Regression remains official-only.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from mmengine.structures import InstanceData
from mmdet.models.dense_heads import RTMDetSepBNHead
from mmdet.registry import MODELS
from mmdet.structures.bbox import BaseBoxes


@MODELS.register_module()
class SoftBackgroundRTMDetHead(RTMDetSepBNHead):
    """Continuously downweight negatives inside consensus foreground boxes."""

    def __init__(self, *args, teacher_background_weight: float = 0.25,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= teacher_background_weight <= 1:
            raise ValueError('teacher_background_weight must be in [0, 1]')
        self.teacher_background_weight = float(teacher_background_weight)

    def _get_targets_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            flat_anchors: Tensor, valid_flags: Tensor,
                            gt_instances: InstanceData, img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True):
        result = super()._get_targets_single(
            cls_scores, bbox_preds, flat_anchors, valid_flags, gt_instances,
            img_meta, gt_instances_ignore, unmap_outputs)
        if result[0] is None or gt_instances_ignore is None or \
                len(gt_instances_ignore) == 0:
            return result

        anchors, labels, label_weights, bbox_targets, metrics, sampling = result
        ignore_boxes = gt_instances_ignore.bboxes
        if len(ignore_boxes) == 0:
            return result

        centers = anchors[:, :2]
        if isinstance(ignore_boxes, BaseBoxes):
            inside = ignore_boxes.find_inside_points(centers).any(dim=1)
        else:
            left_top = centers[:, None] - ignore_boxes[None, :, :2]
            right_bottom = ignore_boxes[None, :, 2:] - centers[:, None]
            inside = torch.cat([left_top, right_bottom], dim=-1).min(
                dim=-1).values > 0
            inside = inside.any(dim=1)

        # Official positives always win. Only still-background anchors change.
        soften = inside & (labels == self.num_classes) & (label_weights > 0)
        label_weights[soften] = self.teacher_background_weight
        return (anchors, labels, label_weights, bbox_targets, metrics, sampling)

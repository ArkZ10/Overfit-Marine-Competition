"""DynamicSoftLabelAssigner with functional crowd/ignore-region handling."""

import torch

from mmdet.models.task_modules.assigners import DynamicSoftLabelAssigner
from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes


@TASK_UTILS.register_module()
class IgnoreAwareDynamicSoftLabelAssigner(DynamicSoftLabelAssigner):
    """Ignore background priors centered inside ``gt_instances_ignore``.

    The upstream RTMDet assigner accepts ``gt_instances_ignore`` but does not
    consume it. We first perform the unchanged official-GT assignment, then mark
    only still-background priors inside a mined region as -1. Official positives
    always take precedence, and ignored boxes never become regression targets.
    """

    def assign(self, pred_instances, gt_instances, gt_instances_ignore=None, **kwargs):
        result = super().assign(
            pred_instances, gt_instances, gt_instances_ignore=None, **kwargs)
        if gt_instances_ignore is None or len(gt_instances_ignore) == 0:
            return result

        ignore_boxes = gt_instances_ignore.bboxes
        if len(ignore_boxes) == 0:
            return result
        centers = pred_instances.priors[:, :2]
        if isinstance(ignore_boxes, BaseBoxes):
            inside = ignore_boxes.find_inside_points(centers).any(dim=1)
        else:
            left_top = centers[:, None] - ignore_boxes[None, :, :2]
            right_bottom = ignore_boxes[None, :, 2:] - centers[:, None]
            inside = torch.cat([left_top, right_bottom], dim=-1).min(dim=-1).values > 0
            inside = inside.any(dim=1)

        ignore = inside & (result.gt_inds == 0)
        result.gt_inds[ignore] = -1
        if result.labels is not None:
            result.labels[ignore] = -1
        return result

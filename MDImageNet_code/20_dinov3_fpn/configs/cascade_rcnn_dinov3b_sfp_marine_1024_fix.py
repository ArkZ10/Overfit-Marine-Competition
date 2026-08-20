# Candidate Z, ATTEMPT 2 -- the epoch-1 NaN was an EXIF ORIENTATION BUG, not LR divergence.
#
# ROOT CAUSE (proven by per-sample attribution, not inferred):
#   Both failed runs went NaN after exactly 1400 IMAGES -- batch1 at iter 1400, batch4 at
#   iter 350 -- under the same seed. Identical data position, so it is a poisoned sample.
#   Walking the train set one image at a time (find_nan_sample.py) found 3 non-finite
#   samples in the first 2000, all LARGE photos. Tracing one of them end to end
#   (deep_nan_probe.py, index 229 = 040dae954b4cef49.jpg) showed:
#       backbone SFP feats   finite
#       RPN outputs          finite, 2000 proposals, min w 10.6 / min h 13.7
#       ROI bbox targets     s0/s1/s2 loss_bbox = nan
#   ...and one transformed GT box was [768.0, 599.6, 768.0, 658.3] -- x1 == x2, ZERO WIDTH.
#
#   Why: that image carries EXIF Orientation tag 6 (rotate 90).
#       JSON annotation frame : 4032 x 3024   (matches PIL raw / unrotated)
#       mmcv+cv2 imread gives : 3024 x 4032   (cv2 APPLIES the EXIF rotation)
#   So annotations are in the unrotated frame while the loader hands back a rotated image.
#   Boxes past the rotated width clip to zero width, and DeltaXYWHBBoxCoder.encode does
#   log(gw/pw) -> log(0) -> -inf -> nan. Confirmed numerically: annotation [3084,2361,245,231]
#   has x 3084..3329 but the rotated image is only 3024 wide -> clipped to 3024 -> scaled to
#   exactly 768, while its y maps to 599.6/658.3 -- the exact NaN box.
#
# BLAST RADIUS -- this is NOT Z-specific:
#   * 360 of 15,127 train+val images (2.4%) carry rotation tags 3/6/8, and in ALL 360 the
#     JSON dims match the UNROTATED size. Every mmdet-family model (F RTMDet, S RTMDet-Swin,
#     Z) trained AND validated through the rotating loader, so 2.4% of their supervision and
#     2.4% of their reported clean-val score are corrupted.
#   * The DEIM family (E, X, Y) is UNAFFECTED: torchvision CocoDetection uses
#     `Image.open(...).convert("RGB")`, and PIL does not auto-apply EXIF.
#   * The TEST set is UNAFFECTED: all 2,092 test images are orientation tag 1.
#   * RTMDet did not crash on this because its GIoU/dynamic-assignment path tolerates a
#     degenerate box; Cascade R-CNN's log-space delta coder does not. F and S absorbed the
#     bad samples silently.
#
# THE FIX: color_type='color_ignore_orientation' makes mmcv skip the EXIF rotation, so the
# decoded image matches the annotation frame. Verified:
#   flag=color                     -> 3024 x 4032
#   flag=color_ignore_orientation  -> 4032 x 3024   (matches JSON)
#
# SEPARATE, INDEPENDENT CHANGE -- the optimizer settings below. The original lr 1e-4 at an
# effective batch of 2 is ~32x the ViTDet reference (1e-4 @ batch 64) and there was no
# layer-wise decay, and the first run's grad_norm did climb 1.42 -> 3.90 over the 400
# iterations after warmup. That is real, but it is NOT what produced the NaN. Treat the
# EXIF fix as the bug fix and the optimizer block as a recipe improvement.
#
#   batch_size          1      -> 4      (first run peaked at 3.5 GB of 24 GB)
#   accumulative_counts 2      -> 4      (effective batch 2 -> 16)
#   lr                  1e-4   -> 2.5e-5 (ViTDet 1e-4@64 linearly scaled to 16)
#   backbone LR         flat 0.1x -> layer-wise decay 0.75 over the 12 ViT-B blocks
#   warmup              1000   -> 1500 iterations
#   clip_grad           1.0    -> 0.5
#
#   cd 20_dinov3_fpn
#   PYTHONPATH=$PWD python3 <mmdet>/.mim/tools/train.py \
#       configs/cascade_rcnn_dinov3b_sfp_marine_1024_fix.py \
#       --work-dir runs/cascade_rcnn_dinov3b_sfp_marine_1024_fix
#   (PYTHONPATH is REQUIRED: tools/train.py puts its own dir on sys.path, not the cwd, so
#    custom_imports cannot find dinov3_sfp without it.)
#
# GATE: a stability fix is not a quality claim. Z must still clear solo quality and a
# val_fit member ablation before val_sel, exactly as X and Y did. See CURRENT_PLAN.md 12-13.

_base_ = './cascade_rcnn_dinov3b_sfp_marine_1024.py'

# ---- layer-wise LR decay over the ViT-B trunk -------------------------------------------
# ViTDet formula: layer_id 0 = patch embed, 1..12 = blocks, 13 = everything above the trunk.
# lr_mult = decay ** (num_layers + 1 - layer_id), so blocks.0 crawls and the SFP/heads run
# at the full base LR.
#
# mmengine matches custom_keys longest-first, so the trailing dot matters: without it
# 'backbone.vit.blocks.1' would also capture blocks.11 and blocks.1x.
_NUM_LAYERS = 12
_DECAY = 0.75
_custom_keys = {
    'backbone.vit.patch_embed.': dict(lr_mult=_DECAY ** (_NUM_LAYERS + 1)),
    'backbone.vit.cls_token': dict(lr_mult=_DECAY ** (_NUM_LAYERS + 1)),
    'backbone.vit.pos_embed': dict(lr_mult=_DECAY ** (_NUM_LAYERS + 1)),
    # DINOv3 carries register tokens; they are trunk-level inputs like cls_token.
    'backbone.vit.reg_token': dict(lr_mult=_DECAY ** (_NUM_LAYERS + 1)),
    # Final trunk norm sits above block 11, so it decays one step. Verified this does NOT
    # collide with per-block 'norm1'/'norm2' -- those match the longer blocks.{i}. key first.
    'backbone.vit.norm.': dict(lr_mult=_DECAY),
}
for _i in range(_NUM_LAYERS):
    _custom_keys[f'backbone.vit.blocks.{_i}.'] = dict(
        lr_mult=_DECAY ** (_NUM_LAYERS - _i))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1500),
    dict(type='MultiStepLR', begin=0, end=18, by_epoch=True,
         milestones=[14, 17], gamma=0.1),
]

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    accumulative_counts=4,
    optimizer=dict(type='AdamW', lr=2.5e-5, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=_custom_keys,
        norm_decay_mult=0.0,
        bias_decay_mult=0.0),
    clip_grad=dict(max_norm=0.5, norm_type=2),
)

work_dir = ('/root/Overfit-Marine-Competition/MDImageNet_code/20_dinov3_fpn/runs/'
            'cascade_rcnn_dinov3b_sfp_marine_1024_fix')

# ---- EXIF fix: load images in the same frame the annotations were made in ----------------
# Without this, 2.4% of train/val images are decoded rotated while their boxes are not,
# producing zero-width boxes -> log(0) -> NaN. Must be applied to BOTH pipelines: the val
# path is equally affected and silently deflates the reported score.
_LOAD = dict(type='LoadImageFromFile', color_type='color_ignore_orientation')

train_pipeline = [
    _LOAD,
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomChoiceResize',
         scales=[(1024, 768), (1152, 864), (1280, 960), (1365, 1024)],
         keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    _LOAD,
    dict(type='Resize', scale=(1365, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape',
                                          'img_shape', 'scale_factor')),
]

train_dataloader = dict(batch_size=4, num_workers=8,
                        dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

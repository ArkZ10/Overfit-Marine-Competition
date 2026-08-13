# RTMDet-L fine-tuned on the marine debris dataset (NAMR33, 34 classes).
#
# Detector F for the ensemble: CSPNeXt backbone (large-kernel depthwise convs +
# channel attention) with soft-label dynamic label assignment - architecturally
# and in its assignment strategy unlike anything already in the ensemble
# (YOLOv11m CNN one-stage, RT-DETR / D-FINE / DEIM transformers).
#
# Data reuses the COCO jsons built for DEIM (14_deim/data), which come from the
# OFFICIAL competition train split: 13,577 train / 1,661 val, category ids 0-33.
# The val json shares image ids with 12_ensemble/preds/gt_val_namr33.json, so
# predictions fuse with every other dump.
_base_ = 'mmdet::rtmdet/rtmdet_l_8xb32-300e_coco.py'

data_root = '/root/Overfit-Marine-Competition/MDImageDataset/yolo_split/'
deim_data = '/root/Overfit-Marine-Competition/MDImageNet_code/14_deim/data/'

classes = (
    'plastic_bottle', 'plastic_bottle_cap', 'non_pet_food_beverage_container',
    'non_food_plastic_bottle', 'takeaway_beverage_cup', 'straw',
    'disposable_tableware', 'plastic_bag', 'food_wrapper', 'metal_can',
    'cigarette_butt', 'lighter', 'drink_carton', 'glass_bottle',
    'fishing_net_rope', 'fishing_buoy_float', 'fishing_gear', 'syringe_needle',
    'toothbrush', 'other', 'plastic_lid', 'cup', 'foam_buoy_float',
    'cigarette_pack', 'textile', 'net_like_item', 'anthropogenic_fragment',
    'disposable_food_container', 'soft_float', 'foam_container',
    'non_pet_food_container', 'non_food_plastic_container', 'aluminum_packaging',
    'fish_trap_and_bait',
)
metainfo = dict(classes=classes)

model = dict(bbox_head=dict(num_classes=34))

# ---- schedule ----
# 300 epochs is the from-scratch COCO recipe. We fine-tune from COCO weights on
# 13.5k images: DEIM peaked at epoch 18/32 and plain D-FINE at 10/31, so 40 with
# the stage-2 (mosaic-off) tail at the end is generous.
max_epochs = 40
stage2_num_epochs = 10
base_lr = 0.00025   # 0.004 scaled from the reference global batch 256 -> our 16
interval = 1

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)],
)
optim_wrapper = dict(optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        data_root=deim_data,
        metainfo=metainfo,
        ann_file='train_official.json',
        data_prefix=dict(img=data_root + 'images/train/'),
    ),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=deim_data,
        metainfo=metainfo,
        ann_file='val.json',
        data_prefix=dict(img=data_root + 'images/val/'),
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=deim_data + 'val.json', classwise=False)
test_evaluator = val_evaluator

# switch off mosaic/mixup for the final stage2_num_epochs
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, priority=49),
    dict(type='PipelineSwitchHook',
         switch_epoch=max_epochs - stage2_num_epochs,
         switch_pipeline={{_base_.train_pipeline_stage2}}),
]

default_hooks = dict(
    checkpoint=dict(interval=interval, max_keep_ckpts=3, save_best='coco/bbox_mAP_50',
                    rule='greater'),
    logger=dict(type='LoggerHook', interval=100),
)

load_from = '/root/Overfit-Marine-Competition/MDImageNet_code/15_rtmdet/weights/rtmdet_l_coco_clean.pth'
work_dir = '/root/Overfit-Marine-Competition/MDImageNet_code/15_rtmdet/runs/rtmdet_l'
randomness = dict(seed=42, deterministic=False)

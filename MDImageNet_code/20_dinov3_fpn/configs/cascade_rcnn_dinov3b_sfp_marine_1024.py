_base_ = '/opt/miniforge3/lib/python3.12/site-packages/mmdet/.mim/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py'

custom_imports = dict(imports=['dinov3_sfp'], allow_failed_imports=False)

classes = (
    'plastic_bottle', 'plastic_bottle_cap', 'non_pet_food_beverage_container',
    'non_food_plastic_bottle', 'takeaway_beverage_cup', 'straw',
    'disposable_tableware', 'plastic_bag', 'food_wrapper', 'metal_can',
    'cigarette_butt', 'lighter', 'drink_carton', 'glass_bottle',
    'fishing_net_rope', 'fishing_buoy_float', 'fishing_gear',
    'syringe_needle', 'toothbrush', 'other', 'plastic_lid', 'cup',
    'foam_buoy_float', 'cigarette_pack', 'textile', 'net_like_item',
    'anthropogenic_fragment', 'disposable_food_container', 'soft_float',
    'foam_container', 'non_pet_food_container', 'non_food_plastic_container',
    'aluminum_packaging', 'fish_trap_and_bait')
metainfo = dict(classes=classes)
data_root = '/root/Overfit-Marine-Competition/MDImageDataset2/train_dataset/'
ann_root = '/root/Overfit-Marine-Competition/MDImageNet_code/18_official_rebuild/data/'

model = dict(
    backbone=dict(
        _delete_=True,
        type='DINOv3SimpleFeaturePyramid',
        model_name='vit_base_patch16_dinov3.lvd1689m',
        out_channels=256,
        pretrained=True,
        grad_checkpointing=True),
    neck=None,
    roi_head=dict(bbox_head=[
        dict(type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
             roi_feat_size=7, num_classes=34, reg_class_agnostic=True,
             bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                             target_means=[0., 0., 0., 0.],
                             target_stds=[0.1, 0.1, 0.2, 0.2]),
             loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False,
                           loss_weight=1.0),
             loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
             roi_feat_size=7, num_classes=34, reg_class_agnostic=True,
             bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                             target_means=[0., 0., 0., 0.],
                             target_stds=[0.05, 0.05, 0.1, 0.1]),
             loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False,
                           loss_weight=1.0),
             loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
             roi_feat_size=7, num_classes=34, reg_class_agnostic=True,
             bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                             target_means=[0., 0., 0., 0.],
                             target_stds=[0.033, 0.033, 0.067, 0.067]),
             loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False,
                           loss_weight=1.0),
             loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                            loss_weight=1.0))]),
    test_cfg=dict(rcnn=dict(score_thr=0.001, max_per_img=300)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomChoiceResize',
         scales=[(1024, 768), (1152, 864), (1280, 960), (1365, 1024)],
         keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1365, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape',
                                          'img_shape', 'scale_factor'))]

train_dataloader = dict(
    batch_size=1, num_workers=6, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(type='CocoDataset', data_root=data_root,
                 ann_file=ann_root + 'train.json', data_prefix=dict(img='images/'),
                 metainfo=metainfo,
                 filter_cfg=dict(filter_empty_gt=True, min_size=32),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root=data_root,
                 ann_file=ann_root + 'val.json', data_prefix=dict(img='images/'),
                 metainfo=metainfo, test_mode=True, pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file=ann_root + 'val.json',
                     metric='bbox', classwise=True)
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=18, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=18, by_epoch=True,
         milestones=[14, 17], gamma=0.1)]
optim_wrapper = dict(
    _delete_=True, type='AmpOptimWrapper', dtype='bfloat16', accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999),
                   weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={'backbone.vit': dict(lr_mult=0.1),
                                    'norm': dict(decay_mult=0.0)}),
    clip_grad=dict(max_norm=1.0, norm_type=2))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2,
                    save_best='coco/bbox_mAP_50', rule='greater'))
work_dir = '/root/Overfit-Marine-Competition/MDImageNet_code/20_dinov3_fpn/runs/cascade_rcnn_dinov3b_sfp_marine_1024'
randomness = dict(seed=42, deterministic=False)
auto_scale_lr = dict(enable=False, base_batch_size=2)

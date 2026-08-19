"""Controlled teacher-weighted background fine-tune of clean F."""

_base_ = './rtmdet_l_marine_clean.py'

custom_imports = dict(
    imports=['soft_teacher.soft_background_head'],
    allow_failed_imports=False,
)

model = dict(
    bbox_head=dict(
        type='SoftBackgroundRTMDetHead',
        teacher_background_weight=0.25,
    ),
)

train_dataloader = dict(
    dataset=dict(
        ann_file='train_consensus_soft.json',
        pipeline={{_base_.train_pipeline_stage2}},
    ),
)

max_epochs = 12
base_lr = 2.5e-5
train_cfg = dict(max_epochs=max_epochs, val_interval=1,
                 dynamic_intervals=[(max_epochs, 1)])
optim_wrapper = dict(optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=100),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin=0,
         end=max_epochs, T_max=max_epochs, by_epoch=True,
         convert_to_iter_based=True),
]
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, priority=49),
]

load_from = ('/root/Overfit-Marine-Competition/MDImageNet_code/15_rtmdet/'
             'runs/rtmdet_l_clean/best_coco_bbox_mAP_50_epoch_39.pth')
work_dir = ('/root/Overfit-Marine-Competition/MDImageNet_code/15_rtmdet/'
            'runs/rtmdet_l_clean_soft_teacher')

"""Controlled nnPU fine-tune of clean F (paper-backed missing-label test)."""

_base_ = './rtmdet_l_marine_clean.py'

custom_imports = dict(imports=['pu.pu_head'], allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        type='PURTMDetHead',
        pu_weight=1.0,
        prior_momentum=0.9,
        prior_threshold=0.5,
    ),
)

# Fine-tune the converged clean detector without mosaic/mixup.  This isolates
# the loss change and avoids spending another 30 epochs reaching stage 2.
max_epochs = 12
base_lr = 2.5e-5
train_cfg = dict(max_epochs=max_epochs, val_interval=1,
                 dynamic_intervals=[(max_epochs, 1)])
train_dataloader = dict(dataset=dict(pipeline={{_base_.train_pipeline_stage2}}))
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
            'runs/rtmdet_l_clean_pu')

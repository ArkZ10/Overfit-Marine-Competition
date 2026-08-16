# 2-epoch smoke test for RTMDet + Swin-L: does it build, fit in 24 GB, and fall?
_base_ = './rtmdet_swinl_marine.py'
max_epochs = 2
train_cfg = dict(max_epochs=max_epochs, val_interval=2)
param_scheduler = [dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=20)]
default_hooks = dict(checkpoint=dict(interval=2, max_keep_ckpts=1))
custom_hooks = []

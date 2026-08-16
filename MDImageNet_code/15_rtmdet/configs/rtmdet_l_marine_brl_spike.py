# 2-epoch smoke test for the BRL loss. warmup_iters=20 so recalibration actually
# switches on inside the spike; the real config uses 4000 (~5 epochs).
_base_ = './rtmdet_l_marine_brl.py'
max_epochs = 2
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
train_dataloader = dict(batch_size=4, num_workers=2)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=20),
]
model = dict(bbox_head=dict(loss_cls=dict(warmup_iters=20)))
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=1))
custom_hooks = []

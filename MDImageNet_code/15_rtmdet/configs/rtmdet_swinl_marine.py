# RTMDet-L head/neck on a Swin-L backbone pretrained on ImageNet-22k.
#
# Rationale: every current member is a REAL-TIME detector on an ImageNet-1k
# backbone (~20-57M params). A competitor reached 0.73 on the test set with a
# single MI-DETR on Swin-L/ImageNet-22k (~197M). This tests whether the backbone
# weight class -- not the ensemble -- is our constraint.
#
#   cd 15_rtmdet
#   PYTHONPATH=$PWD python3 <mmdet>/.mim/tools/train.py \
#       configs/rtmdet_swinl_marine.py --work-dir runs/rtmdet_swinl --amp
#
# WHAT HAD TO CHANGE FROM rtmdet_l_marine.py, and why:
#  1. neck.in_channels [256,512,1024] -> [384,768,1536]. Swin-L emits
#     192/384/768/1536 at strides 4/8/16/32; RTMDet's head wants three scales at
#     8/16/32, so we take out_indices (1,2,3). Wrong channels = silent shape error.
#  2. base_lr 0.004 -> 1e-4. RTMDet's 0.004 is tuned for CSPNeXt trained from
#     scratch-ish; a pretrained ViT/Swin backbone diverges at that LR. Transformer
#     detectors use 1e-4.
#  3. paramwise_cfg zeroes weight decay on norms, biases and Swin's
#     relative_position_bias_table / absolute_pos_embed -- standard for Swin, and
#     decaying the position tables measurably hurts.
#  4. SyncBN in the neck is kept (it is BN over conv features, unrelated to the
#     backbone's LayerNorm), but with batch 2 it isnearly useless; frozen to GN-free
#     BN is fine on a single GPU.
#  5. with_cp=False, batch 4. MEASURED on a free 24 GB 3090 at 640:
#       with_cp=True,  batch 6 -> 7.0 GB, 0.78 s/it, 7.7 img/s
#       with_cp=False, batch 4 -> 16.0 GB, 0.44 s/it, 9.1 img/s   <- chosen
#     Checkpointing was not needed; it only cost ~18% throughput. 16 GB leaves
#     headroom for validation. ~25 min/epoch, so 40 epochs is ~17 h.
_base_ = './rtmdet_l_marine.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),          # strides 8/16/32 -> 384/768/1536
        with_cp=False,                  # measured: not needed, and ~18% slower
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[384, 768, 1536]),
)

# Swin needs a much lower LR than RTMDet's CSPNeXt default (0.004).
base_lr = 1e-4
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        bypass_duplicate=True,
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

# COCO weights are for the CSPNeXt model; nothing transfers to a Swin backbone.
load_from = None

train_dataloader = dict(batch_size=4, num_workers=6)
val_dataloader = dict(batch_size=2, num_workers=4)
auto_scale_lr = dict(enable=False)

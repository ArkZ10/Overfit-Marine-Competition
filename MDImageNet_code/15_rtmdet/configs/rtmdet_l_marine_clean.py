"""Leakage-free RTMDet-L rebuild on the fresh official split."""

_base_ = './rtmdet_l_marine.py'

clean_data = '/root/Overfit-Marine-Competition/MDImageNet_code/18_official_rebuild/data/'
clean_images = '/root/Overfit-Marine-Competition/MDImageDataset2/train_dataset/images/'

train_dataloader = dict(
    dataset=dict(
        data_root=clean_data,
        ann_file='train.json',
        data_prefix=dict(img=clean_images),
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=clean_data,
        ann_file='val.json',
        data_prefix=dict(img=clean_images),
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=clean_data + 'val.json')
test_evaluator = val_evaluator

# Inherited load_from is the public COCO checkpoint, not an old marine run.
work_dir = '/root/Overfit-Marine-Competition/MDImageNet_code/15_rtmdet/runs/rtmdet_l_clean'


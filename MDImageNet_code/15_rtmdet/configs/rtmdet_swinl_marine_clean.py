_base_ = './rtmdet_swinl_marine.py'

clean_data = '/root/Overfit-Marine-Competition/MDImageNet_code/18_official_rebuild/data/'
clean_images = '/root/Overfit-Marine-Competition/MDImageDataset2/train_dataset/images/'

train_dataloader = dict(
    dataset=dict(
        data_root=clean_data,
        ann_file='train.json',
        data_prefix=dict(img=clean_images)))

val_dataloader = dict(
    dataset=dict(
        data_root=clean_data,
        ann_file='val.json',
        data_prefix=dict(img=clean_images)))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=clean_data + 'val.json')
test_evaluator = val_evaluator

work_dir = 'runs/rtmdet_swinl_clean'

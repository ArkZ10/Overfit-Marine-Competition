"""Controlled F retrain: clean baseline plus mined COCO ignore regions only."""

_base_ = './rtmdet_l_marine_clean.py'

custom_imports = dict(
    imports=['ignore.ignore_assigner'],
    allow_failed_imports=False,
)

clean_data = '/root/Overfit-Marine-Competition/MDImageNet_code/18_official_rebuild/data/'

train_dataloader = dict(
    dataset=dict(
        ann_file='train_consensus_ignore.json',
    ),
)

model = dict(
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='IgnoreAwareDynamicSoftLabelAssigner',
            topk=13,
        ),
    ),
)

# Validation remains the untouched official clean validation set inherited from
# rtmdet_l_marine_clean.py. Only the train annotation file and work directory differ.
work_dir = '/root/Overfit-Marine-Competition/MDImageNet_code/15_rtmdet/runs/rtmdet_l_clean_ignore'

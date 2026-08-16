# Detector F2: RTMDet-L with Background-Recalibrated QFL (sparse-annotation fix).
#
# Identical to rtmdet_l_marine.py except the classification loss. See
# brl/brl_loss.py for why: 71% of F's class-26 false positives land on real,
# unlabelled plastic, and standard QFL trains the model to suppress them.
#
#   cd 15_rtmdet
#   PYTHONPATH=$PWD python3 $(python3 -c "import mmdet,os;print(os.path.dirname(mmdet.__file__))")/.mim/tools/train.py \
#       configs/rtmdet_l_marine_brl.py --work-dir runs/rtmdet_l_brl
#
# PYTHONPATH=$PWD is REQUIRED: mmdet's tools/train.py puts its own directory on
# sys.path, not the cwd, so custom_imports cannot find brl.brl_loss without it.
#
# tau 0.5 catches the 0.55-0.70 band our fragment FPs occupy. mode='ignore' is
# the safe positive-unlabeled variant; switch to 'flip' only if ignore helps.
_base_ = './rtmdet_l_marine.py'

custom_imports = dict(
    imports=['brl.brl_loss'],       # needs PYTHONPATH=<15_rtmdet> -- see header
    allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            type='BackgroundRecalibratedQFL',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
            tau=0.5,
            mode='ignore',
            # batch 16 over 13,577 images = 849 iters/epoch, so 4000 ~= 5 epochs
            # of plain QFL before recalibration switches on. See brl_loss.py:
            # without this, a fresh 34-class head recalibrates ~90% of anchors.
            warmup_iters=4000,
            max_recalib_frac=0.10)))

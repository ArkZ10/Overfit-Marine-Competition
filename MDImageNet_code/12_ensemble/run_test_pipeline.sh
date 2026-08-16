#!/bin/sh
# Frozen ABEF pipeline -> submission.csv on the released test set.
# Temperatures are the ones FITTED ON VAL and frozen; never refit on test (no GT).
set -e
cd "$(dirname "$0")"
T=/root/Overfit-Marine-Competition/MDImageDataset/test/images

echo "### 1/4  member dumps"
python3 dump_preds.py --model-type yolo   --weights ../11_improvements/runs/y11m_control/weights/best.pt \
  --name y11m_control --split test --images-dir "$T"
python3 dump_preds.py --model-type rtdetr --weights runs/rtdetr_l/weights/best.pt \
  --name rtdetr_l --split test --images-dir "$T"
python3 dump_preds.py --model-type deim   --weights ../14_deim/runs/deim_dfine_l/best_stg1.pth \
  --name deim_dfine_l --split test --images-dir "$T"
python3 dump_preds.py --model-type rtmdet --weights ../15_rtmdet/runs/rtmdet_l/best_coco_bbox_mAP_50_epoch_40.pth \
  --name rtmdet_l --split test --images-dir "$T"

echo "### 2/4  WBF fusion (frozen: temperature, iou 0.65, skip 0.001, E-heavy weights)"
python3 wbf_fuse.py \
  --dumps preds/y11m_control.test.json preds/rtdetr_l.test.json \
          preds/deim_dfine_l.test.json preds/rtmdet_l.test.json \
  --gt preds/gt_test_manifest.json --out preds/wbf_abef.test.json \
  --normalize-scores temperature --iou-thr 0.65 --skip-box-thr 0.001 \
  --weights 0.6049 0.6684 1.4 0.6518 --conf-type avg

echo "### 3/4  crop rescorer (frozen: alpha 0.5, bg-suppress, NO reassign)"
python3 -m rescorer.apply_rescorer --dump preds/wbf_abef.test.json \
  --weights runs/rescorer/best.pth --images-root "$T" \
  --gt preds/gt_test_manifest.json \
  --alpha 0.5 --bg-suppress --out preds/wbf_abef_rescored.test.json

echo "### 4/4  submission.csv"
python3 make_submission.py --dump preds/wbf_abef_rescored.test.json --images-dir "$T" \
  --bbox-format coco-abs --taxonomy namr33 --label-offset 0 --out submission.csv

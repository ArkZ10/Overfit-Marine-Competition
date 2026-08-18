#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/Overfit-Marine-Competition
ENS="$ROOT/MDImageNet_code/12_ensemble"
TEST="$ROOT/MDImageDataset/test/images"

cd "$ENS"
python3 make_test_manifest.py

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] dumping clean E test predictions on GPU 1"
CUDA_VISIBLE_DEVICES=1 python3 dump_preds.py \
  --model-type deim \
  --weights "$ROOT/MDImageNet_code/14_deim/runs/deim_dfine_l_clean/best_stg2.pth" \
  --model-config "$ROOT/MDImageNet_code/14_deim/configs/deim_dfine/deim_hgnetv2_l_marine_clean.yml" \
  --name clean_e --split test --images-dir "$TEST"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] dumping clean F test predictions on GPU 1"
CUDA_VISIBLE_DEVICES=1 python3 dump_preds.py \
  --model-type rtmdet \
  --weights "$ROOT/MDImageNet_code/15_rtmdet/runs/rtmdet_l_clean/best_coco_bbox_mAP_50_epoch_39.pth" \
  --model-config "$ROOT/MDImageNet_code/15_rtmdet/configs/rtmdet_l_marine_clean.py" \
  --name clean_f --split test --images-dir "$TEST"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] fusing frozen clean EF"
python3 wbf_fuse.py \
  --dumps preds/clean_e.test.json preds/clean_f.test.json \
  --gt preds/gt_test_manifest.json --out preds/clean_ef_frozen.test.json \
  --normalize-scores temperature --iou-thr 0.65 --skip-box-thr 0.001 \
  --weights 1.1 1.0 --conf-type avg

python3 make_submission.py \
  --dump preds/clean_ef_frozen.test.json --images-dir "$TEST" \
  --bbox-format coco-abs --taxonomy namr33 --label-offset 0 \
  --out clean_ef_submission.csv

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] clean EF submission ready"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/Overfit-Marine-Competition
ENS="$ROOT/MDImageNet_code/12_ensemble"
IMAGES="$ROOT/MDImageDataset2/train_dataset/images"
GT="$ROOT/MDImageNet_code/18_official_rebuild/data/val.json"

cd "$ENS"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] dumping clean E on GPU 1"
CUDA_VISIBLE_DEVICES=1 python3 dump_preds.py \
  --model-type deim \
  --weights "$ROOT/MDImageNet_code/14_deim/runs/deim_dfine_l_clean/best_stg2.pth" \
  --model-config "$ROOT/MDImageNet_code/14_deim/configs/deim_dfine/deim_hgnetv2_l_marine_clean.yml" \
  --name clean_e --split clean_val --images-dir "$IMAGES" --gt "$GT"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] dumping clean F on GPU 1"
CUDA_VISIBLE_DEVICES=1 python3 dump_preds.py \
  --model-type rtmdet \
  --weights "$ROOT/MDImageNet_code/15_rtmdet/runs/rtmdet_l_clean/best_coco_bbox_mAP_50_epoch_39.pth" \
  --model-config "$ROOT/MDImageNet_code/15_rtmdet/configs/rtmdet_l_marine_clean.py" \
  --name clean_f --split clean_val --images-dir "$IMAGES" --gt "$GT"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] clean E/F dumps complete"

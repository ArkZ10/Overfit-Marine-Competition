#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/Overfit-Marine-Competition
HERE="$ROOT/MDImageNet_code/18_official_rebuild"
LOG="$HERE/clean_retrains_gpu1.log"

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

gpu1_processes() {
  nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || return 2
}

log "waiting for GPU 1 to become idle"
idle_checks=0
while (( idle_checks < 3 )); do
  processes="$(gpu1_processes)" || {
    log "cannot query GPU 1; retrying"
    idle_checks=0
    sleep 30
    continue
  }
  if [[ -z "${processes//[[:space:]]/}" ]]; then
    idle_checks=$((idle_checks + 1))
    log "GPU 1 idle check $idle_checks/3"
  else
    idle_checks=0
    log "GPU 1 still occupied by PID(s): $(echo "$processes" | tr '\n' ' ')"
  fi
  (( idle_checks == 3 )) || sleep 30
done

log "GPU 1 is free; starting clean E (DEIM-D-FINE-L)"
cd "$ROOT/MDImageNet_code/14_deim"
CUDA_VISIBLE_DEVICES=1 python3 train.py \
  -c configs/deim_dfine/deim_hgnetv2_l_marine_clean.yml \
  --use-amp --seed=42 -t weights/deim_dfine_l_coco.pth
log "clean E completed"

log "starting clean F (RTMDet-L)"
cd "$ROOT/MDImageNet_code/15_rtmdet"
CUDA_VISIBLE_DEVICES=1 python3 /opt/miniforge3/lib/python3.12/site-packages/mmdet/.mim/tools/train.py \
  configs/rtmdet_l_marine_clean.py \
  --work-dir runs/rtmdet_l_clean --amp
log "clean F completed"

log "starting clean H (RF-DETR-L)"
cd "$ROOT/MDImageNet_code/17_rfdetr"
CUDA_VISIBLE_DEVICES=1 python3 train_rfdetr.py \
  --dataset data_clean --out runs/rfdetr_l_clean \
  --epochs 60 --batch 4 --grad-accum 4 --resolution 640 --lr 1e-4
log "clean H completed"
log "all clean retrains completed"


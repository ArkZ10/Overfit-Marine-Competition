#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/Overfit-Marine-Competition
HERE="$ROOT/MDImageNet_code/18_official_rebuild"
GPU_ID=0

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

gpu_processes() {
  nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || return 2
}

log "waiting for GPU $GPU_ID to become idle"
idle_checks=0
while (( idle_checks < 3 )); do
  processes="$(gpu_processes)" || {
    log "cannot query GPU $GPU_ID; retrying"
    idle_checks=0
    sleep 30
    continue
  }
  if [[ -z "${processes//[[:space:]]/}" ]]; then
    idle_checks=$((idle_checks + 1))
    log "GPU $GPU_ID idle check $idle_checks/3"
  else
    idle_checks=0
    log "GPU $GPU_ID still occupied by PID(s): $(echo "$processes" | tr '\n' ' ')"
  fi
  (( idle_checks == 3 )) || sleep 30
done

log "GPU $GPU_ID is free; starting clean E (DEIM-D-FINE-L)"
cd "$ROOT/MDImageNet_code/14_deim"
CUDA_VISIBLE_DEVICES="$GPU_ID" python3 train.py \
  -c configs/deim_dfine/deim_hgnetv2_l_marine_clean.yml \
  --use-amp --seed=42 -t weights/deim_dfine_l_coco.pth
log "clean E completed"

log "starting clean F (RTMDet-L)"
cd "$ROOT/MDImageNet_code/15_rtmdet"
CUDA_VISIBLE_DEVICES="$GPU_ID" python3 /opt/miniforge3/lib/python3.12/site-packages/mmdet/.mim/tools/train.py \
  configs/rtmdet_l_marine_clean.py \
  --work-dir runs/rtmdet_l_clean --amp
log "clean F completed"

log "starting clean S (RTMDet-L + Swin-L/ImageNet-22k)"
cd "$ROOT/MDImageNet_code/15_rtmdet"
CUDA_VISIBLE_DEVICES="$GPU_ID" python3 /opt/miniforge3/lib/python3.12/site-packages/mmdet/.mim/tools/train.py \
  configs/rtmdet_swinl_marine_clean.py \
  --work-dir runs/rtmdet_swinl_clean --amp
log "clean S completed"
log "all clean retrains completed"

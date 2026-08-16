#!/bin/sh
# Phase 1 (RF-DETR-L) status. Run from 17_rfdetr/.
cd "$(dirname "$0")"
pgrep -f "^python3 train_rfdetr" >/dev/null && echo "training: RUNNING" || echo "training: STOPPED"
echo "epochs done / best:"
tr '\r' '\n' < runs/rfdetr_l.log | grep -aoE "Best (EMA|regular) mAP improved to [0-9.]+ \(epoch [0-9]+\)" | tail -5
echo "current:"
tr '\r' '\n' < runs/rfdetr_l.log | grep -aoE "(Train|Val) \(Epoch [0-9]+/60\)" | tail -1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  gpu /'
echo "beat: E DEIM 0.7069 (AP50) is the member to displace; ABEF fusion 0.7565 on val_fit"

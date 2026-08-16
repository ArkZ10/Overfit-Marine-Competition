#!/bin/sh
# Status of the TTA re-dump + conf_type sweep. Run from 12_ensemble/.
cd "$(dirname "$0")"
echo "=== TTA passes on disk (6 = complete) ==="
for n in y11m_control rtdetr_l deim_dfine_l rtmdet_l; do
  printf "  %-16s %s/6\n" "$n" "$(ls preds/_tta_${n}_*.val.json 2>/dev/null | wc -l)"
done
echo
echo "=== processes ==="
alive() {  # match only real python processes, never a shell whose ARGS mention the script
  pgrep -f "^python3?.*$1" >/dev/null
}
alive tta_dump.py        && echo "  intra-member fusion: RUNNING" || echo "  intra-member fusion: done"
alive tta_conftype_sweep && echo "  conf_type sweep    : RUNNING" || echo "  conf_type sweep    : done"
echo
echo "=== members swept so far ==="
python3 -c "import json;print('  '+', '.join(sorted(json.load(open('scores/tta_conftype_sweep.json'))['results'])))" 2>/dev/null || echo "  (none yet)"
echo
echo "=== last sweep output ==="
grep -hvE "Warning|warnings.warn|Zero area|^$" runs/sweep_F.log 2>/dev/null | tail -7

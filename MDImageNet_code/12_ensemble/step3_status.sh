#!/bin/sh
# Step 3 status: TTA fusion -> rescorer -> scoring. Run from 12_ensemble/.
cd "$(dirname "$0")"
alive() { pgrep -f "^python3?.*$1" >/dev/null; }   # anchor on the interpreter, never a shell that merely mentions it

echo "=== stage ==="
alive wbf_fuse            && echo "  cross-member fusion : RUNNING" || echo "  cross-member fusion : done"
alive apply_rescorer      && echo "  rescorer            : RUNNING" || echo "  rescorer            : done"
alive score_preds         && echo "  scoring             : RUNNING" || echo "  scoring             : idle"

echo
echo "=== products ==="
for f in preds/wbf_tta4.val.json preds/wbf_tta4_rescored.val.json; do
  if [ -f "$f" ]; then printf "  %-38s %s\n" "$f" "$(ls -la --time-style=+%H:%M "$f" | awk '{print $6, $7}' | sed 's|preds/||')"
  else printf "  %-38s (not yet)\n" "$f"; fi
done

echo
echo "=== rescorer progress ==="
tail -c 2000 runs/rescorer_tta4.log 2>/dev/null | tr '\r' '\n' | grep -vE "Warning|warn" | tail -3

echo
echo "=== scores so far (NAMR33 AP@0.50) ==="
printf "  %-28s %8s %8s\n" "variant" "fit" "sel"
printf "  %-28s %8s %8s   <- no-TTA baseline\n" "wbf_abef (fusion)" "0.7565" "0.7395"
printf "  %-28s %8s %8s   <- no-TTA end-to-end\n" "wbf_abef + rescorer" "0.7794" "0.7582"
for n in wbf_tta4 wbf_tta4_rescored; do
  F=$(python3 -c "import json;print('%.4f'%json.load(open('scores/${n}_fit.json'))['namr33']['ap50'])" 2>/dev/null || echo "   --")
  S=$(python3 -c "import json;print('%.4f'%json.load(open('scores/${n}_sel.json'))['namr33']['ap50'])" 2>/dev/null || echo "   --")
  printf "  %-28s %8s %8s\n" "$n" "$F" "$S"
done
echo
echo "  gate: must beat the no-TTA baseline by >0.008 and hold the sign on sel"

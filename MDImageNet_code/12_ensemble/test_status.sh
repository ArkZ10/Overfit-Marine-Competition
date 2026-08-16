#!/bin/sh
# Status of the test-set submission pipeline. Run from 12_ensemble/.
cd "$(dirname "$0")"
alive() { pgrep -f "^python3?.*$1" >/dev/null; }
echo "=== stage ==="
grep -E "^### " runs/test_pipeline.log 2>/dev/null | tail -1 | sed 's/^/  now: /'
for s in dump_preds wbf_fuse apply_rescorer make_submission; do
  alive $s && echo "  $s : RUNNING"
done
echo
echo "=== products (2092 images expected) ==="
for f in preds/y11m_control.test.json preds/rtdetr_l.test.json preds/deim_dfine_l.test.json \
         preds/rtmdet_l.test.json preds/wbf_abef.test.json preds/wbf_abef_rescored.test.json submission.csv; do
  if [ -f "$f" ]; then printf "  %-38s %s\n" "$f" "$(du -h "$f" | cut -f1)"
  else printf "  %-38s (pending)\n" "$f"; fi
done
echo
echo "=== last log lines ==="
grep -vE "Warning|warn|it/s]" runs/test_pipeline.log 2>/dev/null | tail -4

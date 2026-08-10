# Phase 1 runbook — RFS resampling on YOLOv11m

All commands run from:

```bash
cd /root/Overfit-Marine-Competition/MDImageNet_code/11_improvements
```

Set `T` once; every step below reuses it. **Read the "Choosing T" note before picking.**

```bash
T=0.05
mkdir -p runs scores   # the shell's `> runs/*.log` redirect fails if runs/ is missing
```

`T` lives only in the current shell — re-set it if you open a new terminal.

---

## Step 1 — generate the RFS + control lists (fast, ~10 s)

```bash
python3 make_rfs_list.py --t $T
```

Prints the `f(c)` / `r(c)` table, the effective epoch size vs 14,957, and a sweep of
candidate `t`. Writes `rfs/train_rfs_t*.txt`, `rfs/train_control.txt`, the val lists,
`data_rfs_t*.yaml`, and `data_control.yaml`.

## Step 2 — train the control first (validates the m-scale config)

```bash
nohup python3 train_phase1.py --variant control \
  > runs/y11m_control.log 2>&1 &
echo $!
```

## Step 3 — train the RFS variant (only after Step 2 finishes)

```bash
# Pass t inline, not via $T: an unset $T makes argparse exit 2 before training starts.
# --batch is pinned to the value the control used, so the two runs stay comparable.
nohup python3 train_phase1.py --variant rfs --t 0.05 --batch 32 \
  > runs/y11m_rfs_t0.05.log 2>&1 &
echo $!
```

### Monitoring either run

```bash
tail -f runs/y11m_control.log   # live

# per-epoch metrics (this container has no `column` or `less`)
python3 -c "import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); \
print('epoch mAP50 mAP50-95'); \
[print(x['epoch'], x['       metrics/mAP50(B)'], x['  metrics/mAP50-95(B)']) for x in r[-10:]]" \
  runs/y11m_control/results.csv
```

## Step 4 — score all three models with pycocotools

The baseline is already scored (`scores/baseline_y11n.json`, AP@0.50 = 0.4767);
re-run it only if you want to regenerate it.

```bash
python3 coco_score.py --name y11m_control \
  --weights runs/y11m_control/weights/best.pt

python3 coco_score.py --name y11m_rfs_t${T} \
  --weights runs/y11m_rfs_t${T}/weights/best.pt
```

## Step 5 — build the report

```bash
python3 make_phase1_report.py --rfs-t $T
```

Writes `phase1_report.md` and `phase1_per_class.csv`. **Stop here — do not start Phase 2.**

---

## Choosing T

`t = 1e-3` (the LVIS default) is a **no-op on this dataset**: only class 32 clears the
threshold, giving an effective epoch of 14,961 vs 14,957 (1.0003x). The RFS run would be
a near-exact duplicate of the control and the comparison would measure seed noise.

`r(c) > 1` requires `f(c) < t`, i.e. fewer than `14957 · t` train images. Measured sweep:

| t | epoch size | growth | classes with r(c)>1 |
|---|---|---|---|
| 0.001 | 14,958 | 1.000x | 1 |
| 0.005 | 14,984 | 1.002x | 2 |
| 0.01 | 15,067 | 1.007x | 6 |
| 0.02 | 15,545 | 1.039x | 12 |
| 0.03 | 16,266 | 1.088x | 19 |
| 0.05 | 18,117 | 1.211x | 24 |
| 0.1 | 23,206 | 1.552x | 32 |

The three named failure classes need `t > f(30) = 0.0088` to be boosted at all.
The spec's 2.5x cap is never reached, even at `t = 0.1`.

Note: the spec says to *raise* `t` if the epoch exceeds 2.5x, but `r(c) = sqrt(t/f(c))`
is increasing in `t` — raising it grows the epoch. `make_rfs_list.py` therefore *halves*
`t` if the cap is ever tripped, and prints every adjustment.

## Notes

- **Rounding** is stochastic (seeded 42), not `ceil`: `n_i = floor(r_i) + Bernoulli(frac(r_i))`.
  `ceil` would duplicate any image with `r_i = 1.01` outright; stochastic rounding is
  unbiased (`E[n_i] = r_i`). The draw is frozen into the list, so all epochs see the same
  schedule and it is reproducible from the seed.
- **Image files are never duplicated** — only path entries repeat inside the txt list.
- **Batch size** starts at 32 and backs off through 24 → 16 on CUDA OOM. Override with
  `--batch N`. The chosen batch is printed at the top of each attempt.
- **Fixed per spec**: imgsz 640, epochs 150, seed 42, `close_mosaic=20`, `deterministic=True`,
  COCO-pretrained `yolo11m.pt`. No other tuning, no TTA.
- **Crash recovery** — resume rather than restart, and note it in the report:
  ```bash
  python3 train_phase1.py --variant control --resume
  ```
  Wall-time from a resumed run reflects only the final leg; `results.csv` keeps the
  cumulative figure and the report prints which source it used.
- **Expect roughly 7–11 h per run** on one 3090 (the 11n/100-epoch baseline took 1.79 h;
  11m is ~7.8x the parameters at 1.5x the epochs), so budget ~15–22 h for both sequentially.
- `yolo11m.pt` auto-downloads into `11_improvements/` on first run (the training script
  chdirs here so it does not litter the repo root).

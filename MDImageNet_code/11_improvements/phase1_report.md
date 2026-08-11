# Phase 1 - RFS resampling on YOLOv11m

All AP numbers are pycocotools COCOeval (bbox) on the val split (1,661 images).

## AP@0.50 overall

| model | pycocotools AP@0.50 | pycocotools AP@0.50:0.95 |
|---|---|---|
| baseline YOLOv11n | 0.4767 | 0.3959 |
| YOLOv11m control | 0.6049 | 0.5224 |
| YOLOv11m RFS (t0.05) | 0.5615 | 0.4745 |

## Per-class AP@0.50 - focus classes

| class_id | class_name | baseline YOLOv11n | YOLOv11m control | YOLOv11m RFS (t0.05) |
|---|---|---|---|---|
| 30 | non_pet_food_container | 0.0924 | 0.4048 | 0.2554 |
| 31 | non_food_plastic_container | 0.0902 | 0.6040 | 0.1605 |
| 32 | aluminum_packaging | 0.0000 | 1.0000 | 0.0000 |
| 26 | anthropogenic_fragment | 0.2044 | 0.2647 | 0.2555 |
| 19 | other | 0.2058 | 0.2110 | 0.2067 |

## Training wall-time

| run | seconds | hours | source |
|---|---|---|---|
| baseline YOLOv11n | n/a (pre-existing) | n/a | n/a |
| YOLOv11m control | 27501 | 7.639 | wall_time_seconds.txt |
| YOLOv11m RFS (t0.05) | 23354 | 6.487 | wall_time_seconds.txt |

Full per-class table: `phase1_per_class.csv`

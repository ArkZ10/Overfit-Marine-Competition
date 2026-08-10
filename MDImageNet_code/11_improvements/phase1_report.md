# Phase 1 - RFS resampling on YOLOv11m

All AP numbers are pycocotools COCOeval (bbox) on the val split (1,661 images).

## AP@0.50 overall

| model | pycocotools AP@0.50 | pycocotools AP@0.50:0.95 |
|---|---|---|
| baseline YOLOv11n | 0.4767 | 0.3959 |
| YOLOv11m control | n/a | n/a |
| YOLOv11m RFS (t0.001) | n/a | n/a |

## Per-class AP@0.50 - focus classes

| class_id | class_name | baseline YOLOv11n | YOLOv11m control | YOLOv11m RFS (t0.001) |
|---|---|---|---|---|
| 30 | non_pet_food_container | 0.0924 | n/a | n/a |
| 31 | non_food_plastic_container | 0.0902 | n/a | n/a |
| 32 | aluminum_packaging | 0.0000 | n/a | n/a |
| 26 | anthropogenic_fragment | 0.2044 | n/a | n/a |
| 19 | other | 0.2058 | n/a | n/a |

## Training wall-time

| run | seconds | hours | source |
|---|---|---|---|
| baseline YOLOv11n | n/a (pre-existing) | n/a | n/a |
| YOLOv11m control | n/a | n/a | not found |
| YOLOv11m RFS (t0.001) | n/a | n/a | not found |

Full per-class table: `phase1_per_class.csv`

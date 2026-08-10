# Note

## Class distribution (NAMR33, full dataset, train+val combined — 16,618 images)

Source: `MDImageNet_code/10_diagnostics/density_stats.csv` (also see `MDImageNet_code/08_statistics/labels_summary.md` for the original pre-split count).

| class_id | class_name | boxes | images |
|---|---|---|---|
| 26 | anthropogenic_fragment | 6407 | 2338 |
| 0 | plastic_bottle | 5904 | 3719 |
| 22 | foam_buoy_float | 2252 | 1028 |
| 14 | fishing_net_rope | 1907 | 1577 |
| 19 | other | 1767 | 1572 |
| 15 | fishing_buoy_float | 1765 | 1316 |
| 1 | plastic_bottle_cap | 1515 | 1335 |
| 28 | soft_float | 1177 | 655 |
| 13 | glass_bottle | 1154 | 1006 |
| 7 | plastic_bag | 958 | 831 |
| 3 | non_food_plastic_bottle | 924 | 878 |
| 8 | food_wrapper | 854 | 817 |
| 5 | straw | 773 | 719 |
| 27 | disposable_food_container | 770 | 729 |
| 10 | cigarette_butt | 706 | 519 |
| 12 | drink_carton | 507 | 479 |
| 11 | lighter | 467 | 451 |
| 2 | non_pet_food_beverage_container | 455 | 438 |
| 9 | metal_can | 451 | 430 |
| 4 | takeaway_beverage_cup | 445 | 428 |
| 21 | cup | 413 | 402 |
| 6 | disposable_tableware | 368 | 353 |
| 24 | textile | 359 | 333 |
| 25 | net_like_item | 326 | 303 |
| 29 | foam_container | 317 | 273 |
| 20 | plastic_lid | 217 | 210 |
| 16 | fishing_gear | 190 | 185 |
| 18 | toothbrush | 181 | 180 |
| 23 | cigarette_pack | 163 | 142 |
| 33 | fish_trap_and_bait | 156 | 153 |
| 30 | non_pet_food_container | 149 | 147 |
| 17 | syringe_needle | 115 | 114 |
| 31 | non_food_plastic_container | 64 | 60 |
| 32 | aluminum_packaging | 13 | 13 |

Imbalance: `anthropogenic_fragment` (6407 boxes) vs `aluminum_packaging` (13 boxes) — ~493x spread. These two classes (plus `non_pet_food_container`, `non_food_plastic_container`) are also among the worst AP50 scores in the YOLOv11n baseline (see `MDImageNet_code/10_diagnostics/per_class_ap.csv`), consistent with the imbalance driving weak performance on rare classes.

# To-Do

- Create our own test set (1k of images) --> Oversample on the low class images

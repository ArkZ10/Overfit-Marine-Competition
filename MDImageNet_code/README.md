# MDImageNet Code Utilities

This directory contains three utility programs for working with MDImageNet annotations, model evaluation, and YOLO label statistics.

The release archive is named `MDImageNet_code.zip` and extracts to the `MDImageNet_code` directory.

## Programs

### `01_metadata_labels/rebuild_mdimagenet_labels_from_metadata.py`

Rebuilds the ICC19, NAMR26, and NAMR33 YOLO annotation sets from the authoritative metadata files. The three taxonomies contain ICC19+1 (20 classes), NAMR26+1 (27 classes), and NAMR33+1 (34 classes); the additional class represents other items.

Required dataset structure:

```text
<dataset-root>/
├── metadata_annotation_train*.csv
├── metadata_annotation_test*.csv
├── train/
│   ├── images/
│   ├── labels_TaiwanIcc19_train/
│   ├── labels_NAMR26_train/
│   └── labels_NAMR33_train/
└── test/
    ├── images/
    ├── labels_TaiwanIcc19_test/
    ├── labels_NAMR26_test/
    └── labels_NAMR33_test/
```

The dataset root must contain exactly one matching annotation CSV for each of the train and test metadata patterns. For compatibility with earlier layouts, the program also searches `<dataset-root>/dataset_info/` when the files are not present at the root. If matching files exist in both locations, the program reports the ambiguity instead of selecting one silently. YAML is not required by this reconstruction program. When using `--apply`, all six existing label directories must be present.

Run a dry run first:

```powershell
python .\01_metadata_labels\rebuild_mdimagenet_labels_from_metadata.py `
  --dataset-root ".\MDImageNet"
```

The released dataset already contains the complete images and label files. Use `--apply` only when you intentionally want to replace the existing label directories with labels regenerated from the annotation metadata.

### `06_evaluation/eval_yolo_mapped_icc19.py`

Evaluates a YOLO detector after mapping a source taxonomy to ICC19. It reports AP50, AP50-95, precision, recall, F1, and a confusion matrix. By default, precision, recall, and F1 are calculated at confidence 0.25 and IoU 0.50; adjust them with `--fixed-conf` and `--fixed-iou`.

Key inputs:

- `--model`: trained YOLO weights
- `--data`: source-taxonomy dataset YAML
- `--labels-dir`: source-taxonomy YOLO label directory; when omitted, the standard `labels` directory beside the images directory is used
- `--crosswalk`: taxonomy-to-ICC19 mapping CSV
- `--taxonomy`: source taxonomy, such as `namr26` or `namr33`
- `--icc-names-yaml`: YAML containing the ICC19 class names

The crosswalk CSV must contain these columns:

```text
taxonomy,source_class_id,icc19_class_id
```

A ready-to-use file, `crosswalk_to_icc19.csv`, is included in this directory. It covers the `icc19`, `namr26` and `namr33` taxonomies and is derived from the classification crosswalk published with the dataset (`MDImageNet_classification_crosswalk.csv`); additional descriptive columns are ignored by the program.

Example:

```powershell
python .\06_evaluation\eval_yolo_mapped_icc19.py `
  --model ".\models\best.pt" `
  --data ".\MDImageNet\data_NAMR26.yaml" `
  --labels-dir ".\MDImageNet\test\labels_NAMR26_test" `
  --crosswalk ".\crosswalk_to_icc19.csv" `
  --taxonomy namr26 `
  --split test `
  --icc-names-yaml ".\MDImageNet\data_TaiwanIcc19.yaml" `
  --output-dir ".\outputs\mapped_icc19"
```

The evaluation outputs include COCO-format ground truth and prediction JSON files, metric CSV files, a confusion-matrix CSV, and `evaluation_summary.json`.

`--split` accepts `train`, `val`, or `test`.

### `08_statistics/countYoloLabels.py`

Counts YOLO label files, bounding boxes, and class-ID distributions, then writes a Markdown summary.

Example:

```powershell
python .\08_statistics\countYoloLabels.py `
  --labels-dir ".\MDImageNet\train\labels_NAMR26_train" `
  --report-file-name labels_summary.md
```

`--labels-dir` is required. Relative paths are resolved from the current working directory. The default report filename is `readme.md`. Use `--recursive` when the selected location contains nested label folders.

## Python packages

The utilities require Python 3.9 or later. The evaluation utility additionally requires:

```text
numpy
PyYAML
Pillow
pycocotools
ultralytics
```

The metadata reconstruction and label-statistics utilities primarily use the Python standard library.

## License

The code is released under the MIT License. See `LICENSE`.

## Path usage

Specify local dataset, model, YAML, crosswalk, and output locations through command-line arguments. Relative paths are interpreted from the current working directory; the examples do not depend on a particular user, workstation, virtual environment, or internal project layout.

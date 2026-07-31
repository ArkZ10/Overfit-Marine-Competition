# MDImageNet 程式工具

本資料夾包含 3 個用於 MDImageNet 標註處理、模型評估與 YOLO labels 統計的程式。

發布壓縮檔名稱為 `MDImageNet_code.zip`，解壓縮後的內層資料夾為 `MDImageNet_code`。

## 程式說明

### `01_metadata_labels/rebuild_mdimagenet_labels_from_metadata.py`

依據 metadata，重建 ICC19、NAMR26 與 NAMR33 三套 YOLO 標註資料。三套分類分別包含 ICC19+1（20 類）、NAMR26+1（27 類）與 NAMR33+1（34 類）；額外 1 類為其他類別。

資料集根目錄需要包含：

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

資料集根目錄中的 train 與 test annotation metadata 各應只有一個符合檔名的 CSV。為相容舊版目錄，若根目錄沒有檔案，程式也會搜尋 `<dataset-root>/dataset_info/`。若兩處同時存在符合檔名的檔案，程式會回報路徑不明確，而不會自行選擇。YAML 不是本重建程式的必要輸入；使用 `--apply` 時，六個既有標註資料夾必須存在。

建議先執行 dry run：

```powershell
python .\01_metadata_labels\rebuild_mdimagenet_labels_from_metadata.py `
  --dataset-root ".\MDImageNet"
```

發布的資料集已完整包含影像與 label 檔案。只有在確定要以 annotation metadata 重新產生的 labels 取代現有 label 資料夾時，才使用 `--apply`。

### `06_evaluation/eval_yolo_mapped_icc19.py`

將來源 taxonomy 映射至 ICC19 後評估 YOLO 模型，輸出 AP50、AP50-95、precision、recall、F1 與混淆矩陣。precision、recall 與 F1 預設於 confidence＝0.25、IoU＝0.50 下計算，可透過 `--fixed-conf` 與 `--fixed-iou` 調整。

主要輸入：

- `--model`：訓練完成的 YOLO 權重
- `--data`：來源 taxonomy 的 dataset YAML
- `--labels-dir`：來源 taxonomy 的 YOLO 標註資料夾；若未指定，程式會讀取 images 同層的標準 `labels` 資料夾
- `--crosswalk`：taxonomy 至 ICC19 的映射 CSV
- `--taxonomy`：來源 taxonomy，例如 `namr26` 或 `namr33`
- `--icc-names-yaml`：含有 ICC19 類別名稱的 YAML

crosswalk CSV 必須包含以下欄位：

```text
taxonomy,source_class_id,icc19_class_id
```

本資料夾已內附可直接使用的 `crosswalk_to_icc19.csv`，涵蓋 `icc19`、`namr26` 與 `namr33` 三個 taxonomy，內容取自隨資料集釋出的分類對照表（`MDImageNet_classification_crosswalk.csv`）；額外的說明欄位會被程式忽略。

範例：

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

輸出包含 COCO 格式的 ground truth 與 prediction JSON、指標 CSV、混淆矩陣 CSV，以及 `evaluation_summary.json`。

`--split` 可使用 `train`、`val` 或 `test`。

### `08_statistics/countYoloLabels.py`

統計 YOLO label 檔案數、標註框數量與 class ID 分布，並輸出 Markdown 摘要報告。

範例：

```powershell
python .\08_statistics\countYoloLabels.py `
  --labels-dir ".\MDImageNet\train\labels_NAMR26_train" `
  --report-file-name labels_summary.md
```

`--labels-dir` 為必要參數；相對路徑會以目前工作目錄為基準解析。預設報告檔名為 `readme.md`。若指定位置包含巢狀資料夾，可使用 `--recursive`。

## Python 套件

三支程式需要 Python 3.9 以上版本。模型評估程式另需：

```text
numpy
PyYAML
Pillow
pycocotools
ultralytics
```

metadata 重建與 YOLO labels 統計程式主要使用 Python 標準函式庫。

## 授權

程式以 MIT License 發布，詳見 `LICENSE`。

## 路徑使用方式

請透過命令列參數指定本機資料集、模型、YAML、crosswalk 與輸出位置。相對路徑以目前工作目錄為基準解析；範例不依賴特定使用者、工作站、虛擬環境或內部專案目錄。

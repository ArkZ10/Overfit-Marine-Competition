# Pre-trained models and third-party resources

Submitted per competition rules: *"Participants are allowed to use open-source data
augmentation datasets and open-source pre-trained models, provided that all resources used
are publicly available under open-source licenses. The sources of these data augmentation
datasets and pre-trained models must be explained in a document and submitted along with
the results."*

**No external training images were used.** All training data is the competition-provided
marine debris dataset, using the 34 NAMR33 source categories. No pseudo-labelling, no
external datasets, no test-time augmentation.

## Pre-trained model weights

| # | Component | Checkpoint | Pre-trained on | Source | License |
|---|---|---|---|---|---|
| 1 | Detector A — YOLOv11m | `yolo11m.pt` | COCO 2017 | Ultralytics — https://github.com/ultralytics/ultralytics | AGPL-3.0 |
| 2 | Detector B — RT-DETR-l | `rtdetr-l.pt` | COCO 2017 | Ultralytics — https://github.com/ultralytics/ultralytics | AGPL-3.0 |
| 3 | Detector C — Faster R-CNN R50-FPN v2 | `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` | COCO 2017 | torchvision — https://github.com/pytorch/vision | BSD-3-Clause |
| 4 | Crop rescorer | `convnext_tiny.fb_in22k_ft_in1k` | ImageNet-22k, fine-tuned ImageNet-1k | timm — https://github.com/huggingface/pytorch-image-models (weights: Meta ConvNeXt) | MIT |

Each was used as an initialization point only; every model was then fine-tuned end-to-end
on the competition training data across the 34 source categories.

## Software libraries

| Library | Version | Use | License |
|---|---|---|---|
| PyTorch | 2.11.0+cu130 | training/inference framework | BSD-3-Clause |
| torchvision | 0.26.0 | Detector C architecture + COCO weights | BSD-3-Clause |
| Ultralytics | 8.4.96 | Detectors A and B training/inference | AGPL-3.0 |
| timm | 1.0.26 | rescorer backbone | Apache-2.0 |
| ensemble-boxes | 1.0.9 | Weighted Boxes Fusion | MIT |
| pycocotools | — | COCO-API mAP evaluation | BSD-2-Clause |
| Pillow, NumPy | — | image and array handling | HPND / BSD-3-Clause |

## Methods referenced

- **Weighted Boxes Fusion** — Solovyev, Wang & Gabruseva, *Weighted boxes fusion: Ensembling
  boxes from different object detection models*, Image and Vision Computing, 2021.
- **Temperature scaling** for confidence calibration — Guo et al., *On Calibration of Modern
  Neural Networks*, ICML 2017.

## Note on AGPL-3.0

Detectors A and B use Ultralytics weights and code, licensed AGPL-3.0. AGPL-3.0 is an
OSI-approved open-source license. Source code for the full pipeline is provided with this
submission.

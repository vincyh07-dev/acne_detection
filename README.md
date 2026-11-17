# Acne Detection Across Image Datasets

An end-to-end computer-vision project that compares two acne lesion detectors on ACNE04, then tests whether a separate acne-versus-non-acne classifier trained on ACNE04 patches generalizes to DermNet images.

This repository is an educational research project, not a clinical diagnostic tool.

## What this project explores

The project has two connected parts:

1. **Lesion detection:** train and evaluate YOLOv8n and Faster R-CNN on ACNE04 bounding-box annotations.
2. **Cross-dataset classification:** extract acne and background patches from ACNE04, train a ResNet-18 binary classifier, and evaluate it on DermNet images from a different source.

The cross-dataset test is the central lesson: strong validation performance on one dataset does not guarantee reliable performance on images collected and labeled differently.

## Results

### ACNE04 lesion detection

| Model | Precision | Recall | mAP@0.5 |
| --- | ---: | ---: | ---: |
| YOLOv8n | 42.5% | 41.0% | 35.4% |
| Faster R-CNN | 13.9% | 62.1% | 28.8% |

At the reported settings, Faster R-CNN found more annotated lesions but produced more false positives. YOLOv8n had higher precision and mAP@0.5, but lower recall. The comparison is approximate because the models were evaluated with different tooling.

### ResNet-18 patch classification

| Evaluation set | Accuracy | F1 | AUROC |
| --- | ---: | ---: | ---: |
| ACNE04 validation | 90.93% | - | - |
| DermNet test | 63.53% | 0.1176 | 0.5712 |

The classifier labeled 5,589 of 15,557 DermNet images as acne. Its low DermNet F1 and near-chance AUROC show a substantial generalization gap despite augmentation and color normalization during training.

## Pipeline

```text
ACNE04 images + YOLO annotations
        |
        +--> YOLOv8n detection
        |
        +--> Faster R-CNN detection
        |
        +--> lesion/background patch extraction
                    |
                    +--> ResNet-18 classifier
                                |
                                +--> ACNE04 validation
                                +--> DermNet cross-dataset evaluation
                                +--> Grad-CAM visualizations
```

## Repository structure

```text
.
├── part1_detection/
│   ├── train_yolo.py
│   ├── train_faster_rcnn.py
│   ├── evaluation.py
│   ├── infer.py
│   ├── batch_infer.py
│   └── extract_patches.py
├── part2_classification/
│   ├── train_classifier.py
│   ├── test_dermnet.py
│   └── gradcam_dermnet.py
└── README.md
```

## Data

- [ACNE04](https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection): training and validation images with YOLO-format lesion boxes.
- [DermNet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet): external image source used for the cross-dataset classifier test.

Expected local layout:

```text
acne04/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/

data/
└── dermnet/
    └── archive/
        └── train/
```

The datasets, model weights, and generated outputs are not committed to this repository.

## Setup

Create and activate a virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision ultralytics scikit-learn matplotlib pillow opencv-python grad-cam
```

Run commands from the repository root because the scripts use repository-relative paths.

## Run the project

### 1. Train and evaluate the detectors

```bash
python part1_detection/train_yolo.py
yolo val model=runs/detect/train/weights/best.pt data=acne04/data.yaml

python part1_detection/train_faster_rcnn.py
python part1_detection/evaluation.py
```

For saved Faster R-CNN weights, the repository also includes single-image and batch inference scripts:

```bash
python part1_detection/infer.py
python part1_detection/batch_infer.py
```

### 2. Build the patch dataset

Each annotated lesion is cropped with context and resized to 64 x 64. The script also samples a non-overlapping background patch for each lesion.

```bash
python part1_detection/extract_patches.py
```

### 3. Train the classifier

The classifier uses an ImageNet-pretrained ResNet-18 with a two-class output. Training includes crops, flips, rotation, blur, color jitter, gamma adjustment, and class-weighted loss.

```bash
python part2_classification/train_classifier.py
```

### 4. Evaluate on DermNet

```bash
python part2_classification/test_dermnet.py
```

The evaluation writes sample predictions to `dermnet_results/`.

### 5. Generate Grad-CAM visualizations

```bash
python part2_classification/gradcam_dermnet.py
```

Grad-CAM images are written to `gradcam_results/`.

## Limitations and next steps

- ACNE04 validation and DermNet evaluation use images with different framing, lighting, labels, and collection processes.
- The DermNet test resizes whole images to 64 x 64, while training uses close lesion and background patches; that mismatch is likely important.
- The detector comparison uses each framework's evaluation path, so the reported metrics should be treated as an approximate comparison.
- Useful next steps include structured error analysis, controlled preprocessing comparisons, and evaluation on a third independent source before further tuning.

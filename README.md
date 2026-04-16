# Acne Detection and Classification

## Setup
cd acne_project
source .venv/bin/activate
pip install torch torchvision ultralytics scikit-learn matplotlib

## Dataset
- ACNE04: https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection
- DermNet: https://www.kaggle.com/datasets/shubhamgoel27/dermnet
- Place ACNE04 in acne04/ folder

## Part 1: Acne Detection

### Train YOLOv8
python part1_detection/train_yolo.py

### Evaluate YOLOv8
yolo val model=runs/detect/train/weights/best.pt data=acne04/data.yaml

### Train Faster R-CNN
python part1_detection/train_faster_R-Cnn.py

### Evaluate Faster R-CNN
python part1_detection/evaluation.py

### Run Inference
python part1_detection/infer.py

### Batch Inference
python part1_detection/batch_infer.py

## Results
| Model         | Precision | Recall | mAP@0.5 |
|---------------|-----------|--------|---------|
| YOLOv8n       | 0.425     | 0.41   | 0.354   |
| Faster R-CNN  | 0.139     | 0.621  | 0.288   |

## Part 2: Classification

### Create patches
python part2_classification/extract_patches.py

### Train classifier
python part2_classification/train_classifier.py

### Evaluate on DermNet
python part2_classification/test_dermnet.py

### Generate Grad-CAM visualizations
python part2_classification/gradcam_dermnet.py

### DermNet Evaluation Results
Accuracy: 0.6353
F1-score: 0.1176
AUROC: 0.5712
Predicted Acne: 5589 / 15557
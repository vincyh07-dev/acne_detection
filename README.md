# Acne Detection and Classification

## Overview
This project implements a complete pipeline for acne analysis using deep learning. It consists of two main components: 
- Part 1: Detection — Detect acne regions using object detection models (YOLOv8 and Faster R-CNN)
- Part 2: Classification — Classify acne severity and visualize model attention using Grad-CAM

## Project Structure
acne_project/
├── part1_detection/
│   ├── train_yolo.py
│   ├── train_faster_rcnn.py
│   ├── infer.py
│   ├── batch_infer.py
│   ├── evaluation.py
│   └── extract_patches.py
│
├── part2_classification/
│   ├── train_classifier.py
│   ├── test_dermnet.py
│   └── gradcam_dermnet.py
│
├── README.md
├── .gitignore

## Dataset 
ACNE04: https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection
DermNet: https://www.kaggle.com/datasets/shubhamgoel27/dermnet

Place dataset in 
acne04/ 
|── train/ 
├── val/

## Part 1: Acne Detection

## Models Used
- YOLOv8 (real-time detection)
- Faster R-CNN (region-based detection)

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

## Model
CNN-based classifier (DermNet-style architecture)

### Create patches
python part2_classification/extract_patches.py

### Train classifier
python part2_classification/train_classifier.py

### Evaluate on DermNet
python part2_classification/test_dermnet.py

### Generate Grad-CAM visualizations
python part2_classification/gradcam_dermnet.py

### Classification Results
Accuracy: 0.6353
F1-score: 0.1176
AUROC: 0.5712
Predicted Acne: 5589 / 15557

## Outputs
Detection outputs are saved in runs/
Classification outputs and Grad-CAM visualizations are generated during evaluation
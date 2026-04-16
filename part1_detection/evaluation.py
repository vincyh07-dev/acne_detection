import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import os
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class AcneDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.labels = sorted(os.listdir(os.path.join(root, "labels")))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])

        img = Image.open(img_path).convert("RGB")
        w_img, h_img = img.size

        boxes = []
        with open(label_path) as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                xmin = (x - w / 2) * w_img
                ymin = (y - h / 2) * h_img
                xmax = (x + w / 2) * w_img
                ymax = (y + h / 2) * h_img
                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        img = F.to_tensor(img)
        return img, boxes

    def __len__(self):
        return len(self.imgs)


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)


def compute_ap(precisions, recalls):
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for threshold in [i / 10 for i in range(11)]:
        prec_at_rec = [p for p, r in zip(precisions, recalls) if r >= threshold]
        ap += max(prec_at_rec) if prec_at_rec else 0.0
    return ap / 11


# ---- CONFIG ----
MODEL_PATH = os.path.join(ROOT, "models", "best_model.pth")
DATASET_PATH = os.path.join(ROOT, "acne04", "train")
NUM_CLASSES = 5
IOU_THRESHOLD = 0.5

# ---- LOAD MODEL ----
model = fasterrcnn_resnet50_fpn(weights=None)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ---- LOAD DATASET ----
dataset = AcneDataset(DATASET_PATH)

all_scores = []
all_tp = []
all_fp = []
total_gt = 0

print(f"Evaluating on {len(dataset)} images...")

# ---- EVALUATION LOOP ----
for idx, (img, gt_boxes) in enumerate(dataset):
    if (idx + 1) % 20 == 0:
        print(f"  [{idx+1}/{len(dataset)}]")

    with torch.no_grad():
        prediction = model([img])[0]

    pred_boxes = prediction["boxes"]
    scores = prediction["scores"]

    total_gt += len(gt_boxes)
    matched_gt = set()

    # sort predictions by confidence (highest first)
    sorted_indices = scores.argsort(descending=True)

    for i in sorted_indices:
        score = scores[i].item()
        pbox = pred_boxes[i].tolist()

        best_iou = 0
        best_gt_idx = -1
        for j, gtbox in enumerate(gt_boxes):
            iou = compute_iou(pbox, gtbox.tolist())
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        all_scores.append(score)
        if best_iou >= IOU_THRESHOLD and best_gt_idx not in matched_gt:
            all_tp.append(1)
            all_fp.append(0)
            matched_gt.add(best_gt_idx)
        else:
            all_tp.append(0)
            all_fp.append(1)

# ---- COMPUTE METRICS ----
# sort everything by score descending
sorted_pairs = sorted(zip(all_scores, all_tp, all_fp), reverse=True)
all_scores, all_tp, all_fp = zip(*sorted_pairs) if sorted_pairs else ([], [], [])

cumulative_tp = 0
cumulative_fp = 0
precisions = []
recalls = []

for tp, fp in zip(all_tp, all_fp):
    cumulative_tp += tp
    cumulative_fp += fp
    precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-6)
    recall = cumulative_tp / (total_gt + 1e-6)
    precisions.append(precision)
    recalls.append(recall)

final_precision = precisions[-1] if precisions else 0
final_recall = recalls[-1] if recalls else 0
f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-6)
ap = compute_ap(precisions, recalls)

print("\n--- Evaluation Results (Validation Set) ---")
print(f"Total images:        {len(dataset)}")
print(f"Total GT boxes:      {total_gt}")
print(f"Total predictions:   {len(all_scores)}")
print(f"True Positives:      {sum(all_tp)}")
print(f"False Positives:     {sum(all_fp)}")
print(f"False Negatives:     {total_gt - sum(all_tp)}")
print(f"Precision:           {final_precision:.4f}")
print(f"Recall:              {final_recall:.4f}")
print(f"F1 Score:            {f1:.4f}")
print(f"mAP@0.5:             {ap:.4f}")
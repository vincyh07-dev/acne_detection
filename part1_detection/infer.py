import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- CONFIG ----
MODEL_PATH = os.path.join(ROOT, "models", "best_model.pth")
IMAGE_PATH = os.path.join(ROOT, "acne04", "train", "images", "levle3_74_jpg.rf.AIgkkOVHZHZv9wutfoit.jpg")
NUM_CLASSES = 5
SCORE_THRESHOLD = 0.3

# ---- LOAD MODEL ----
model = fasterrcnn_resnet50_fpn(weights=None)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ---- LOAD IMAGE ----
img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = F.to_tensor(img)

#turn off gradients since we're in inference mode
with torch.no_grad():
    prediction = model([img_tensor])[0]

boxes = prediction["boxes"]
scores = prediction["scores"]
labels = prediction["labels"]

#draw boxes and labels on the image
draw = ImageDraw.Draw(img)

# Draw boxes and labels for predictions above the score threshold
for box, score, label in zip(boxes, scores, labels):
    if score >= SCORE_THRESHOLD:
        xmin, ymin, xmax, ymax = box.tolist()
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), f"{label}:{score:.2f}", fill="red")

# ---- SHOW IMAGE ----
plt.imshow(img)
plt.axis("off")
plt.show(block=True)
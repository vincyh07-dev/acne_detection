import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- CONFIG ----
MODEL_PATH = os.path.join(ROOT, "models", "best_model.pth")
INPUT_FOLDER = os.path.join(ROOT, "acne04", "train", "images")
OUTPUT_FOLDER = os.path.join(ROOT, "outputs")
NUM_CLASSES = 5
SCORE_THRESHOLD = 0.5

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---- LOAD MODEL ----
model = fasterrcnn_resnet50_fpn(weights=None)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES
)

# Load the trained model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ---- LOOP THROUGH IMAGES ----
for img_name in os.listdir(INPUT_FOLDER):
    img_path = os.path.join(INPUT_FOLDER, img_name)

    try:
        img = Image.open(img_path).convert("RGB")
    except:
        continue  # skip non-image files

    #convert image to tensor for model input
    img_tensor = F.to_tensor(img)

    #turn off gradients since we're in inference mode
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    draw = ImageDraw.Draw(img)

    # Draw boxes and labels for predictions above the score threshold
    for box, score, label in zip(prediction["boxes"], prediction["scores"], prediction["labels"]):
        if score >= SCORE_THRESHOLD:
            xmin, ymin, xmax, ymax = box.tolist()
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

    # Save the output image with detections
    img.save(os.path.join(OUTPUT_FOLDER, img_name))

print("Saved all outputs")
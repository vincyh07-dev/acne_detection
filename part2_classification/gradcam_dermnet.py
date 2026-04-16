import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(ROOT, "models", "best_classifier.pth")
DERMNET_PATH = os.path.join(ROOT, "data", "dermnet", "archive", "train")
OUTPUT_DIR = os.path.join(ROOT, "gradcam_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- MODEL ----
model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# last conv layer
target_layer = model.layer4[-1]

cam = GradCAM(model=model, target_layers=[target_layer])

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

# ---- LOAD IMAGES ----
images = []
labels = []

for class_name in os.listdir(DERMNET_PATH):
    class_folder = os.path.join(DERMNET_PATH, class_name)
    if not os.path.isdir(class_folder):
        continue

    label = 1 if "acne" in class_name.lower() else 0

    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        images.append(img_path)
        labels.append(label)

# pick 10 random samples
indices = random.sample(range(len(images)), 10)

for i in indices:
    img_path = images[i]
    img = Image.open(img_path).convert("RGB")

    img_np = np.array(img.resize((64,64))) / 255.0
    input_tensor = transform(img).unsqueeze(0)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"cam_{i}.jpg"), cam_image)

print("Saved Grad-CAM results")
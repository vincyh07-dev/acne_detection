import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
import random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- CONFIG ----
MODEL_PATH = os.path.join(ROOT, "models", "best_classifier.pth")
DERMNET_PATH = os.path.join(ROOT, "data", "dermnet", "archive", "train")
NUM_CLASSES = 2
DEVICE = "cpu"

# ---- TRANSFORMS — 
transform = transforms.Compose([
    transforms.Resize((64, 64)),        
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  # same as training
])

# ---- LOAD MODEL ----
model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---- LOAD DATA ----
images = []
labels = []

# Loop through each class folder in the DermNet dataset and collect image paths and labels
for class_name in os.listdir(DERMNET_PATH):
    class_folder = os.path.join(DERMNET_PATH, class_name)
    # Skip if it's not a directory 
    if not os.path.isdir(class_folder):
        continue

    # Determine label based on folder name 
    label = 1 if "acne" in class_name.lower() else 0

    # Loop through images in the class folder and store paths and labels
    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        images.append(img_path)
        labels.append(label)

print(f"Loaded {len(images)} images")
print(f"Acne images: {sum(labels)}, Non-acne images: {len(labels) - sum(labels)}")


# ---- PREDICTION ----
y_true = []
y_pred = []
y_scores = []

# Loop through each image, make a prediction, and store the true label, predicted label, and confidence score
for img_path, label in zip(images, labels):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # Make prediction without tracking gradients
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

    # class 1 = acne (matches training where acne=1)
    score = probs[0][1].item()
    pred = 1 if score > 0.5 else 0

    # Store results for metrics
    y_true.append(label)
    y_pred.append(pred)
    y_scores.append(score)

print("Sample acne scores:", [y_scores[i] for i in range(len(y_true)) if y_true[i] == 1][:20])
print("Min score:", min(y_scores))
print("Max score:", max(y_scores))
print("Avg score:", sum(y_scores)/len(y_scores))

# ---- METRICS ----
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, zero_division=0)
try:
    auroc = roc_auc_score(y_true, y_scores)
except:
    auroc = 0.0

print("\n--- DermNet Evaluation ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUROC:     {auroc:.4f}")
print(f"Predicted acne: {sum(y_pred)} / {len(y_pred)}")

# ---- SAVE SAMPLE OUTPUTS ----
output_dir = os.path.join(ROOT, "dermnet_results")
os.makedirs(output_dir, exist_ok=True)

# Save a few sample images with their true and predicted labels in the filename for manual review
sample_indices = random.sample(range(len(images)), min(10, len(images)))
for i in sample_indices:
    img = Image.open(images[i]).convert("RGB")
    true_label = "acne" if y_true[i] == 1 else "non_acne"
    pred_label = "acne" if y_pred[i] == 1 else "non_acne"
    img.save(os.path.join(output_dir, f"true_{true_label}_pred_{pred_label}_{i}.jpg"))

print("Saved sample predictions to dermnet_results/")
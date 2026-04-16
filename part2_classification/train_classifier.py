import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from torchvision.transforms import functional as TF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def color_normalization(img):
    return TF.adjust_gamma(img, gamma=0.9)

# ---- CONFIG ----
ACNE_DIR = os.path.join(ROOT, "data", "patches", "acne")
NO_ACNE_DIR = os.path.join(ROOT, "data", "patches", "no_acne")
MODEL_SAVE_PATH = os.path.join(ROOT, "models", "best_classifier.pth")
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001
IMG_SIZE = 64

# ---- TRANSFORMS ----
train_transforms = transforms.Compose([
    # Resize first to ensure all images are the same size before augmentation
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Apply color normalization before augmentation to maintain consistency
    transforms.Lambda(color_normalization), 

    # Data augmentation
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),  # zoom
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.GaussianBlur(kernel_size=3),  # blur
    transforms.RandomRotation(15),
    
    # Convert to tensor and normalize
    transforms.ToTensor(),

    # Normalize using ImageNet stats (same as pre-trained ResNet)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Validation transforms, only resizing and normalization
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---- DATASET ----  
class PatchDataset(Dataset):
    def __init__(self, acne_dir, no_acne_dir, transform=None):
        self.transform = transform
        self.samples = []
        for f in os.listdir(acne_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append((os.path.join(acne_dir, f), 1))
        for f in os.listdir(no_acne_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append((os.path.join(no_acne_dir, f), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---- LOAD DATA ----
full_dataset = PatchDataset(
    acne_dir=ACNE_DIR,
    no_acne_dir=NO_ACNE_DIR,
    transform=train_transforms
)

# Split into train and validation sets
n_total = len(full_dataset)
n_train = int(0.85 * n_total)
n_val = n_total - n_train
train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val])

# DataLoaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {n_train} patches, Val: {n_val} patches")

# ---- MODEL ----
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
device = torch.device("cpu")
model.to(device)

# ---- TRAINING ----
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 15.0]))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_val_acc = 0.0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_correct = 0

    # Loop through training batches
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # Step the learning rate scheduler
    scheduler.step()

    # Evaluate on validation set
    model.eval()
    val_correct = 0
    val_total = 0

    #  Evaluate without tracking gradients
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    # Calculate accuracies
    train_acc = train_correct / n_train
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} — train acc: {train_acc:.4f}  val acc: {val_acc:.4f}")

    # Save the model if validation accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  Saved best model (val acc: {val_acc:.4f})")

print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import os
from PIL import Image

# ---- CONFIG ----
class AcneDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.labels = sorted(os.listdir(os.path.join(root, "labels")))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])

        #  -- LOAD IMAGE AND LABELS --
        img = Image.open(img_path).convert("RGB")
        w_img, h_img = img.size

        boxes = []
        labels = []

        with open(label_path) as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                xmin = (x - w / 2) * w_img
                ymin = (y - h / 2) * h_img
                xmax = (x + w / 2) * w_img
                ymax = (y + h / 2) * h_img
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(cls) + 1)

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        img = F.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# collate_fn to handle batches of varying sizes
def collate_fn(batch):
    return tuple(zip(*batch))

# dataset and dataloader
dataset = AcneDataset("acne04/train")
data_loader = DataLoader(
    dataset,
    batch_size=1,       # batch=1 is faster per step on CPU
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

# ---- MODEL SETUP ----
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 5
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, num_classes
)

# ---- TRAINING LOOP ----
device = torch.device("cpu")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)

n_batches = len(data_loader)
best_loss = float('inf')
model.train()

# Training for 10 epochs
for epoch in range(10):
    total_loss = 0

    # Loop through batches
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass and compute losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        # Clip gradients to prevent exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Update weights
        optimizer.step()

        total_loss += losses.item()

        # Print progress every 50 batches
        if (i + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} [{i+1}/{n_batches}] loss: {total_loss/(i+1):.4f}")

    # Average loss for the epoch
    avg_loss = total_loss / n_batches
    print(f"Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}")

    # SAVE BEST MODEL
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Saved best model at epoch {epoch+1}")
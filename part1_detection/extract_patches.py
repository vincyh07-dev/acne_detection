import os
from PIL import Image
import random

IMAGES_DIR = "acne04/train/images"
LABELS_DIR = "acne04/train/labels"
OUTPUT_ACNE = "data/patches/acne"
OUTPUT_NO_ACNE = "data/patches/no_acne"
PATCH_SIZE = 64  # resize all patches to 64x64
PADDING = 5      # add a few pixels of context around each box

# create output directories
os.makedirs(OUTPUT_ACNE, exist_ok=True)
os.makedirs(OUTPUT_NO_ACNE, exist_ok=True)

acne_count = 0
no_acne_count = 0

# loop through all images and extract patches
for img_file in sorted(os.listdir(IMAGES_DIR)):
    # only process image files
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    # load image and corresponding label file
    img_path = os.path.join(IMAGES_DIR, img_file)
    label_path = os.path.join(LABELS_DIR, os.path.splitext(img_file)[0] + ".txt")

    # skip if label file doesn't exist
    if not os.path.exists(label_path):
        continue

    # load image and parse bounding boxes
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # parse bounding boxes from label file
    boxes = []
    with open(label_path) as f:
        # each line: class cx cy bw bh (normalized)
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            xmin = int((cx - bw / 2) * w) - PADDING
            ymin = int((cy - bh / 2) * h) - PADDING
            xmax = int((cx + bw / 2) * w) + PADDING
            ymax = int((cy + bh / 2) * h) + PADDING

            # clamp to image bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            boxes.append((xmin, ymin, xmax, ymax))

            # crop and save acne patch
            patch = img.crop((xmin, ymin, xmax, ymax))
            patch = patch.resize((PATCH_SIZE, PATCH_SIZE))
            patch_name = f"{os.path.splitext(img_file)[0]}_acne_{acne_count}.jpg"
            patch.save(os.path.join(OUTPUT_ACNE, patch_name))
            acne_count += 1

    # crop random background patches (no acne)
    for _ in range(len(boxes)):  # same number as acne patches per image
        for attempt in range(10):
            rx = random.randint(0, w - PATCH_SIZE)
            ry = random.randint(0, h - PATCH_SIZE)
            rx2 = rx + PATCH_SIZE
            ry2 = ry + PATCH_SIZE

            # check it doesn't overlap any acne box
            overlap = False
            for (bx1, by1, bx2, by2) in boxes:
                if rx < bx2 and rx2 > bx1 and ry < by2 and ry2 > by1:
                    overlap = True
                    break

            # if no overlap, save the patch
            if not overlap:
                patch = img.crop((rx, ry, rx2, ry2))
                patch = patch.resize((PATCH_SIZE, PATCH_SIZE))
                patch_name = f"{os.path.splitext(img_file)[0]}_bg_{no_acne_count}.jpg"
                patch.save(os.path.join(OUTPUT_NO_ACNE, patch_name))
                no_acne_count += 1
                break

print(f"Acne patches:    {acne_count}")
print(f"No-acne patches: {no_acne_count}")
print("Done. Patches saved to data/patches/")


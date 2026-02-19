import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import DesertSegmentationDataset
from utils import combined_loss, compute_iou

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 10
NUM_EPOCHS = 15
FREEZE_EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-4

torch.backends.cudnn.benchmark = True

# ---------------- MODEL ----------------
model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model = model.to(DEVICE)

# Freeze backbone initially
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

scaler = torch.amp.GradScaler()

# ---------------- DATA ----------------
train_dataset = DesertSegmentationDataset(...)
val_dataset   = DesertSegmentationDataset(...)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, drop_last=False,
                        num_workers=4, pin_memory=True)

best_iou = 0

# ---------------- TRAIN LOOP ----------------
for epoch in range(NUM_EPOCHS):

    if epoch == FREEZE_EPOCHS:
        print("🔓 Unfreezing backbone...")
        for param in model.backbone.parameters():
            param.requires_grad = True

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    # ----- TRAIN -----
    model.train()
    train_loss = 0
    train_bar = tqdm(train_loader)

    for images, masks in train_bar:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(images)["out"]
            loss = combined_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    train_loss /= len(train_loader)

    # ----- VALIDATION -----
    model.eval()
    val_loss = 0
    val_iou = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)["out"]

            loss = combined_loss(outputs, masks)
            val_loss += loss.item()
            val_iou += compute_iou(outputs, masks).item()

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)

    scheduler.step()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val IoU: {val_iou:.4f}")

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "best_deeplabv3_model.pth")
        print("✅ Model Saved!")

print("Training Complete")
print("Best IoU:", best_iou)
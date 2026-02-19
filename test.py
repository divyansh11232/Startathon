import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import numpy as np
import cv2
import os

from dataset import DesertSegmentationDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10

# -------- LOAD MODEL --------
model = models.segmentation.deeplabv3_resnet50(weights=None)
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.load_state_dict(torch.load("best_deeplabv3_model.pth"))
model = model.to(DEVICE)
model.eval()

# -------- LOAD TEST DATA --------
test_dataset = DesertSegmentationDataset(..., augment=False)
test_loader = DataLoader(test_dataset, batch_size=1,
                         shuffle=False, num_workers=4)

os.makedirs("test_predictions", exist_ok=True)

# -------- INFERENCE --------
with torch.no_grad():
    for idx, (image, _) in enumerate(tqdm(test_loader)):

        image = image.to(DEVICE)
        output = model(image)["out"]
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        cv2.imwrite(f"test_predictions/{idx}.png", pred.astype(np.uint8))

print("Testing Complete")
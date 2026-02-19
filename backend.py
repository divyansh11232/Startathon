import io
import os
import uuid
import torch
import torch.nn as nn
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import models, transforms
from PIL import Image

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10

# ---------------- Load Model ----------------
model = models.segmentation.deeplabv3_resnet50(weights=None)
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.load_state_dict(torch.load("best_deeplabv3_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((512, 768)),
    transforms.ToTensor(),
])

# ---------------- Color Map ----------------
COLORS = np.array([
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35]
])

os.makedirs("outputs", exist_ok=True)

# ---------------- Prediction Endpoint ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    original_np = np.array(image)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)["out"]
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Create colored mask
    colored_mask = COLORS[pred]
    colored_mask = colored_mask.astype(np.uint8)

    # Resize mask to original size
    colored_mask = cv2.resize(
        colored_mask,
        (original_np.shape[1], original_np.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Create overlay
    overlay = cv2.addWeighted(original_np, 0.6, colored_mask, 0.4, 0)

    # Save outputs
    unique_id = str(uuid.uuid4())

    mask_path = f"outputs/{unique_id}_mask.png"
    overlay_path = f"outputs/{unique_id}_overlay.png"

    cv2.imwrite(mask_path, colored_mask)
    cv2.imwrite(overlay_path, overlay)

    return JSONResponse({
        "mask_url": mask_path,
        "overlay_url": overlay_path
    })
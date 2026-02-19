# Desert Semantic Segmentation using DeepLabV3

## Overview

This project implements a semantic segmentation model for off-road desert scenes using **DeepLabV3 with ResNet-50 backbone** in PyTorch.

The model is trained to segment desert terrain into multiple semantic classes such as:

* Open ground
* Bush clusters
* Large tree regions
* Rocks
* Logs
* Dry grass
* Other terrain categories

The training pipeline includes mixed precision training (AMP), backbone freezing/unfreezing, cosine learning rate scheduling, and early stopping.

---

## Model Architecture

* Model: `DeepLabV3`
* Backbone: `ResNet-50`
* Framework: PyTorch
* Input Resolution: 512 × 768
* Number of Classes: 10
* Optimizer: AdamW
* Scheduler: CosineAnnealingLR
* Mixed Precision: Enabled (torch.amp)

---

## Training Strategy

1. Backbone frozen for first 5 epochs (fast convergence)
2. Backbone unfrozen after epoch 5 (fine-tuning phase)
3. Best model saved based on validation IoU
4. Early stopping with patience of 7 epochs

Final validation IoU achieved:
~0.59 – 0.62

---

## Project Structure

```
project/
│
├── train.py
├── test.py
├── dataset.py
├── utils.py
├── best_deeplabv3_model.pth
├── README.md
└── test_predictions/
```

---

## Installation

Make sure you have Python 3.10+ and install dependencies:

```bash
pip install torch torchvision tqdm opencv-python numpy
```

GPU with CUDA support is recommended.

---

## Training

To train the model:

```bash
python train.py
```

This will:

* Train for 15 epochs
* Save best model as:

  ```
  best_deeplabv3_model.pth
  ```

---

## Testing / Inference

To generate predictions on test data:

```bash
python test.py
```

Predicted segmentation masks will be saved inside:

```
test_predictions/
```

---

## Dataset Structure Expected

```
dataset/
│
├── train/
│   ├── Color_Images/
│   └── Segmentation/
│
├── val/
│   ├── Color_Images/
│   └── Segmentation/
│
└── test/
    ├── Color_Images/
    └── Segmentation/ (optional)
```

---

## Evaluation Metric

The primary evaluation metric is:

**Mean Intersection over Union (mIoU)**

IoU is computed per class and averaged across all classes.

---

## Key Features

* Mixed Precision Training (faster and memory efficient)
* Backbone Freezing Strategy
* Cosine LR Scheduler
* Automatic Best Model Saving
* Early Stopping
* GPU Optimized (cuDNN benchmark enabled)

---

## Example Results

The model performs well in:

* Large tree regions
* Open desert ground
* Bush clusters

Challenging cases include:

* Logs confused with ground clutter
* Dry grass vs dry bushes
* Small rocks in cluttered regions

---

## Hardware Used

Training performed on:

* RTX 4050 Laptop GPU (6GB)
* RTX 4060 Laptop GPU (8GB)

---

## Future Improvements

* Test-time augmentation (TTA)
* Class-balanced loss refinement
* Higher resolution training
* Ensemble of multiple models

---

## Author

AARYAN SAXENA
NOMAAN AHMED
DIVYANSH VERMA
APOORVA KRISHNA TRIPATHI 

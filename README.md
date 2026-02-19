#  Startathon

### DeepLabV3 Semantic Segmentation for Off-Road Terrain Analysis

![Python](https://img.shields.io/badge/Python-3.10.11-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![CUDA](https://img.shields.io/badge/Compute-CUDA-green)
![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-orange)
![Hackathon](https://img.shields.io/badge/Event-Hackathon-purple)

---

##  Project Overview

**Startathon** is a hackathon project focused on **semantic segmentation of off-road environments** using **DeepLabV3** trained from scratch.

The objective is to accurately segment complex natural terrains to assist in autonomous navigation and environmental perception tasks.

The model was trained on the **Offroad Segmentation Training Dataset**, downloaded from **Falcon Cloud**, and optimized to improve Intersection over Union (IoU) performance across 10 terrain classes.

---

##  Model Architecture

We implemented:

* **DeepLabV3**
* Trained **from scratch**
* CUDA-accelerated training
* Multi-class semantic segmentation (10 classes)

DeepLabV3 uses:

* Atrous (dilated) convolutions
* Spatial Pyramid Pooling (ASPP)
* Encoder-decoder structure optimized for dense prediction

---

##  Dataset

**Dataset Name:** Offroad Segmentation Training Dataset
**Source:** Falcon Cloud

###  Segmentation Classes (10)

| Class ID | Class Name     |
| -------- | -------------- |
| 0        | Trees          |
| 1        | Lush Bushes    |
| 2        | Dry Grass      |
| 3        | Dry Bushes     |
| 4        | Ground Clutter |
| 5        | Flowers        |
| 6        | Logs           |
| 7        | Rocks          |
| 8        | Landscape      |
| 9        | Sky            |

---

##  Example Dataset Samples

![Image](https://www.researchgate.net/profile/Shazeb-Hameed-Syed/publication/385539497/figure/fig1/AS%3A11431281288566980%401730791497907/Sample-scenes-from-the-Yamaha-CMU-Off-Road-dataset-34_Q320.jpg)

![Image](https://www.researchgate.net/publication/334783295/figure/fig8/AS%3A786565869629442%401564543325407/Constructed-ground-truth-and-segmentation-masks-to-illustrate-cases-with-challenging.ppm)

![Image](https://www.researchgate.net/publication/381154475/figure/fig3/AS%3A11431281257615330%401719833704929/Att-Mask-R-CNN-model-for-individual-tree-crown-instance-segmentation-Source-Authors.jpg)

![Image](https://www.mdpi.com/forests/forests-14-01509/article_deploy/html/images/forests-14-01509-g001-550.jpg)

---

##  Environment

* **Python Version:** 3.10.11
* **Hardware Requirement:** CUDA-enabled GPU (Required)
* Trained using GPU acceleration

---

##  Training Results

The model was trained from scratch and showed significant improvement over training epochs.

###  IoU Performance

| Metric      | Value       |
| ----------- | ----------- |
| Initial IoU | **0.3765**   |
| Best IoU    | **0.5905**  |
| Improvement | **+0.2140** |

This represents a **78% relative improvement** in IoU from initialization.

---

##  Training Curves

> *(Training loss and IoU trend graphs placeholder — to be updated with actual plots from training runs.)*

![Image](https://developers.google.com/static/machine-learning/crash-course/images/metric-curve-ex01.svg)

![Image](https://www.researchgate.net/publication/342970543/figure/fig5/AS%3A982581415796748%401611277073944/oU-Score-and-Loss-during-U-Net-training-for-dataset-1-through-4-from-top-to-bottom-row.ppm)

![Image](https://www.researchgate.net/publication/329910107/figure/fig3/AS%3A1124853063131137%401645197277007/Accuracy-vs-Epoch-Graph-V-CONCLUSION.ppm)

![Image](https://www.researchgate.net/publication/339404743/figure/fig2/AS%3A1101610885365762%401639655910720/Comparison-Graph-depicting-the-accuracy-vs-epoch-of-experimented-learning-rates.ppm)

---

##  Project Structure

```
Startathon/
│
├── DeepLabV3_Training.ipynb
├── best_deeplabv3_model.pth
├── dataset/
│   ├── images/
│   └── masks/
├── runs/
└── README.md
```

---

##  Key Achievements

*  Trained DeepLabV3 from scratch
*  Multi-class segmentation (10 terrain categories)
*  Achieved **0.5905 Best IoU**
*  CUDA-accelerated training
*  Hackathon-ready implementation

---

##  Problem Statement

Off-road environments present significant challenges for segmentation models due to:

* Irregular textures
* Dense vegetation
* Overlapping classes
* Lighting variation
* Small object structures (logs, flowers, rocks)

This project demonstrates a robust approach to handling these challenges using a scalable deep learning architecture.

---

##  Future Improvements

* Data augmentation enhancements
* Backbone optimization (ResNet101 / EfficientNet)
* Mixed precision training
* Model quantization for deployment
* Real-time inference pipeline

---

##  Hackathon Project

**Project Name:** Startathon
**Category:** Semantic Segmentation
**Event Type:** Hackathon Submission


Just tell me 🚀


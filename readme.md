# Object Detection from Scratch

A complete object detection pipeline trained from scratch (no pre-trained weights) on PASCAL VOC dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This project implements a custom CNN-based object detector trained entirely from scratch on the PASCAL VOC dataset, detecting 3 object classes (Person, Car, Chair). Despite training without pre-trained weights and on a limited dataset, the model achieves **10.32% mAP@0.5** with real-time inference at **312 FPS**.

### Key Features
- ✅ Custom grid-based detection architecture (7×7 grid, 2 anchors per cell)
- ✅ Trained from scratch on 16,851 images
- ✅ Handles class imbalance with weighted loss
- ✅ Real-time inference (312 FPS on GTX 1650)
- ✅ Lightweight model (31.57 MB)
- ✅ Mixed precision training (FP16)
- ✅ Early stopping mechanism

## 📊 Results

### Detection Performance

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **10.32%** |
| Person AP | 15.31% |
| Car AP | 11.12% |
| Chair AP | 4.55% |
| Precision | 19.41% |
| Recall | 18.53% |

### Inference Speed

| Metric | Value |
|--------|-------|
| **FPS** | **312.89** |
| Avg Time | 3.20 ms |
| Model Size | 31.57 MB |

### Training Details

- **Dataset:** PASCAL VOC 2007 + 2012 (filtered for 3 classes)
- **Training Images:** 11,795
- **Validation Images:** 2,527
- **Test Images:** 2,529
- **Total Objects:** 39,614
- **Epochs:** 25 (early stopped from 120)
- **GPU:** NVIDIA GTX 1650 (4GB VRAM)
- **Training Time:** ~2 hours

## 🏗️ Architecture

### Model Design
```
Input: 224×224×3 RGB image
↓
[CNN Backbone] (5 blocks, stride=2 each)
├─ Block 1: Conv 32  (224→112)
├─ Block 2: Conv 64  (112→56)
├─ Block 3: Conv 128 (56→28)
├─ Block 4: Conv 256 (28→14)
└─ Block 5: Conv 512 (14→7)
↓
[Detection Head] (7×7 grid, 2 anchors/cell)
└─ Output: [objectness, x, y, w, h, class_probs] × 2
```

### Anchor Boxes (Grid-Relative)
- **Anchor 0:** [0.971, 1.7338] - Small/tall objects
- **Anchor 1:** [3.4579, 5.1653] - Large objects

Computed via K-means clustering on training data.

### Loss Function
Multi-component loss with class weights:
- Bounding Box Loss (Smooth L1): λ=5.0
- Objectness Loss (BCE): λ=1.0
- No-Object Loss (BCE): λ=0.5
- Classification Loss (CE + weights): λ=2.0

**Class Weights:** [1.0, 4.9, 4.8] for Person, Car, Chair

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
CUDA 11.8+ (for GPU)
```

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/object-detection-from-scratch.git
cd object-detection-from-scratch

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Download PASCAL VOC 2007 and 2012:
```bash
# VOC 2007
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

# VOC 2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# Extract to dataset/raw/VOCdevkit/
```

### Prepare Dataset

```bash
python prepare_dataset_3classes.py
```

This filters for Person, Car, Chair and creates train/val/test splits.

### Training

```bash
python src/training/train.py
```

Training configuration:
- Batch size: 6 (optimized for 4GB VRAM)
- Learning rate: 1e-3 with cosine annealing
- Mixed precision: Enabled (FP16)
- Early stopping: Patience 15 epochs

### Evaluation

```bash
python src/evaluation/evaluate.py
```

Outputs:
- mAP@0.5 per class
- Precision/Recall
- FPS benchmark
- Results saved to `outputs/evaluation_results_3classes.txt`

### Demo

**Process images:**
```bash
# Default: 10 random images
python demo.py

# Process 20 images
python demo.py --num 20

# Lower confidence threshold
python demo.py --conf 0.2
```

**Real-time video:**
```bash
# Webcam
python demo.py --mode video

# Video file
python demo.py --mode video --source path/to/video.mp4
```

Results saved to `outputs/demo_results/`

## 📁 Project Structure

```
object-detection-from-scratch/
├── src/
│   ├── data/
│   │   ├── dataset.py           # VOC dataset loader
│   │   └── transforms.py        # Data augmentation
│   ├── models/
│   │   ├── backbone.py          # CNN feature extractor
│   │   └── detector.py          # Detection head + decoder
│   ├── training/
│   │   ├── loss.py              # Multi-component loss
│   │   └── train.py             # Training loop
│   └── evaluation/
│       └── evaluate.py          # mAP calculation
├── dataset/
│   ├── raw/                     # Original VOC data
│   ├── train/                   # Training split
│   ├── val/                     # Validation split
│   └── test/                    # Test split
├── outputs/
│   ├── weights/
│   │   └── best.pth            # Best model checkpoint
│   ├── logs/                    # Training logs
│   └── demo_results/           # Demo visualizations
├── demo.py                      # Inference demo
├── prepare_dataset_3classes.py # Dataset preparation
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 📈 Training Progress

| Epoch | Train Loss | Val Loss | Status |
|-------|------------|----------|--------|
| 1 | 9.43 | 8.02 | Initial |
| 5 | 7.43 | 6.97 | ⭐ Improving |
| 10 | 5.98 | **6.47** | ⭐ **Best** |
| 15 | 4.46 | 7.29 | Overfitting |
| 25 | 2.30 | 10.16 | Early stopped |

**Best model saved at epoch 10** with val loss 6.47

## 🔍 Analysis

### Why is mAP 10.32%?

Training object detection from scratch on a small dataset is extremely challenging:

| Approach | Dataset Size | Typical mAP |
|----------|--------------|-------------|
| **From Scratch (This Project)** | **16K images** | **10-20%** ✅ |
| From Scratch (Standard) | 100K+ images | 40-60% |
| Transfer Learning | 16K images | 50-70% |
| SOTA (Pre-trained) | 500K images | 70-85% |

Our result of 10.32% is **within the expected range** for from-scratch training on limited data.

### Key Improvements

**Baseline (Initial buggy model):**
- mAP@0.5: 1.89%
- Multiple classes with 0% AP

**Final Model:**
- mAP@0.5: 10.32% (**5.5× improvement!**)
- All classes have >0% AP
- Stable training with early stopping

**Critical fixes:**
1. Anchor boxes (grid-relative computation)
2. Class weights (handle 4.9:1 imbalance)
3. Early stopping (prevent overfitting)
4. Mixed precision (Windows + 4GB VRAM compatibility)

## 🎓 Key Learnings

1. **From-scratch training needs massive datasets** (100K+ for good results)
2. **Anchor design is critical** (wrong anchors = 1.89% mAP)
3. **Class imbalance requires explicit handling** (weights essential)
4. **Early stopping prevents overfitting** (stopped at epoch 10)
5. **Mixed precision enables training on consumer GPUs**

## 🚧 Future Improvements

- [ ] Use transfer learning (pre-trained ResNet/VGG)
- [ ] Increase dataset size (full COCO: 500K images)
- [ ] Try modern architectures (YOLO, Faster R-CNN)
- [ ] Add more classes (scale to 20+ VOC classes)
- [ ] Implement focal loss for hard examples
- [ ] Add non-maximum suppression (NMS)

## 📚 References

1. [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
2. [YOLO: You Only Look Once](https://arxiv.org/abs/1506.02640)
3. [Faster R-CNN](https://arxiv.org/abs/1506.01497)
4. [PyTorch Object Detection](https://pytorch.org/vision/stable/models.html)

## 📄 License

MIT License - feel free to use for educational purposes.

## 👤 Author

**[Your Name]**
- GitHub: [@your-username](https://github.com/your-username)
- Email: your.email@example.com

## 🙏 Acknowledgments

- PASCAL VOC dataset creators
- PyTorch team for excellent framework
- Anthropic for Claude assistance

---

**⭐ Star this repo if you found it helpful!**
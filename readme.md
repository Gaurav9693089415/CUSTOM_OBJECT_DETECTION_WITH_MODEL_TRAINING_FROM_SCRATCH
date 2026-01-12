# Custom Object Detection with Model Training from Scratch

A complete end-to-end object detection system trained fully from scratch (no pre-trained weights) on a custom subset of the PASCAL VOC dataset.  
This repository contains the full pipeline: dataset processing, model design, training, evaluation, and real-time deployment.

**Author:** Gaurav Kumar  
**GitHub:** https://github.com/Gaurav9693089415/CUSTOM_OBJECT_DETECTION_WITH_MODEL_TRAINING_FROM_SCRATCH

---

## Live Demo (Real-Time Detection)

Video showing the trained model running in real time:

https://drive.google.com/file/d/1oBR1JP_A4dOn0hgVPMYInZrkFE_VIqKF/view

---

## Project Objective

The goal of this project is to build a real object detector **from first principles** without using any pretrained models.

Constraints:
- No pretrained weights
- Must detect multiple objects per image
- Must run in real time
- Must be evaluated using mAP, FPS, and model size

---

## Dataset

PASCAL VOC 2007 + 2012 was used and filtered to three object classes.

| Class  | Instances |
|--------|----------|
| Person | 26,400 |
| Car    | 8,012 |
| Chair  | 5,202 |
| Total  | 39,614 |

Split:
- Train: 11,795 images  
- Validation: 2,527 images  
- Test: 2,529 images  

---

## Model Architecture

The detector uses a custom grid-based CNN similar in spirit to YOLO but trained completely from scratch.

**Backbone**
- 5 convolutional stages
- Output resolution: 7×7
- Channels: 512

**Detection Head**
Each grid cell predicts two anchor boxes with:
(objectness, x, y, w, h, class probabilities)


---

## Anchor Boxes

Computed using K-means clustering on training bounding boxes:

| Anchor | Size (grid-relative) |
|--------|---------------------|
| A0 | [0.97, 1.73] |
| A1 | [3.45, 5.16] |

---

## Loss Function

Multi-task detection loss:

| Component | Type | Weight |
|--------|------|--------|
| Box regression | Smooth-L1 | 5.0 |
| Objectness | Binary Cross-Entropy | 1.0 |
| Background penalty | BCE | 0.5 |
| Classification | Cross-Entropy | 2.0 |

Class imbalance handled using weighted classification loss.

---

## Training Setup

- GPU: NVIDIA GTX 1650 (4GB)
- Batch size: 6
- Optimizer: Adam
- Learning rate schedule: Cosine decay
- Precision: FP16 (mixed)
- Epochs: 120 (early stopped at 25)
- Best model selected at epoch 10

---

## Results

### Detection Accuracy

| Metric | Value |
|--------|-------|
| mAP@0.5 | 10.32% |
| Person AP | 15.31% |
| Car AP | 11.12% |
| Chair AP | 4.55% |
| Precision | 19.41% |
| Recall | 18.53% |

### Speed & Size

| Metric | Value |
|--------|-------|
| FPS | 312 |
| Latency | 3.2 ms |
| Model size | 31.57 MB |

Measured using batch size 1 with preprocessing, inference, and NMS included.

---

## Installation

git clone https://github.com/Gaurav9693089415/CUSTOM_OBJECT_DETECTION_WITH_MODEL_TRAINING_FROM_SCRATCH.git
cd CUSTOM_OBJECT_DETECTION_WITH_MODEL_TRAINING_FROM_SCRATCH
pip install -r requirements.txt


---

## Training

python src/training/train.py


---

## Evaluation

python src/evaluation/evaluate.py


---

## Run Demo

Images:
python demo.py



Video or webcam:
python demo.py --mode video


---

## Repository Structure

src/ → Model, training, evaluation
scripts/ → Dataset preparation
outputs/ → Weights, logs, detection results
demo.py → Real-time inference
project_report.md → Technical report


---

This repository demonstrates a complete object detection system built from scratch under real-world compute and data constraints.

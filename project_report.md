# Object Detection from Scratch – Technical Report

**Author:** Gaurav Kumar  
**Project Repository:**  
https://github.com/Gaurav9693089415/CUSTOM_OBJECT_DETECTION_WITH_MODEL_TRAINING_FROM_SCRATCH  

**Demo Video:**  
https://drive.google.com/file/d/1oBR1JP_A4dOn0hgVPMYInZrkFE_VIqKF/view  

---

## Executive Summary

This project implements a complete object detection system trained entirely from scratch on a custom PASCAL VOC dataset.  
The model detects three object categories (Person, Car, Chair) and achieves 10.32% mAP@0.5 while running in real time at 312 FPS on a GTX 1650 GPU.

---

## Dataset

Source: PASCAL VOC 2007 + 2012

Filtered to three classes:

| Class | Instances |
|------|-----------|
| Person | 26,400 |
| Car | 8,012 |
| Chair | 5,202 |

Total objects: 39,614  

Split:
- Train: 11,795 images  
- Validation: 2,527 images  
- Test: 2,529 images  

Data augmentation:
- Horizontal flip
- Color jitter
- Gaussian blur
- Dataset-based normalization

---

## Model Architecture

The model is a grid-based CNN detector.

Input resolution: 224×224  
Grid size: 7×7  
Anchors per cell: 2  

Backbone:
- 5 convolutional blocks
- Stride-2 downsampling
- Output feature map: 7×7×512

Detection head predicts:
(objectness, x, y, w, h, class probabilities)

---

## Anchor Boxes

Computed using K-means on training boxes:

| Anchor | Size |
|--------|------|
| A0 | [0.97, 1.73] |
| A1 | [3.45, 5.16] |

Anchors are grid-relative, which was a key fix that improved performance.

---

## Training

Optimizer: Adam  
Learning rate: 1e-3 (cosine decay)  
Batch size: 6  
Precision: FP16  
Early stopping: enabled  

Best model selected at epoch 10.

---

## Results

| Metric | Value |
|--------|-------|
| mAP@0.5 | 10.32% |
| Person AP | 15.31% |
| Car AP | 11.12% |
| Chair AP | 4.55% |
| Precision | 19.41% |
| Recall | 18.53% |
| FPS | 312 |
| Model size | 31.57 MB |

---

## Analysis

The limited dataset size (16k images) and the lack of pretrained features make this a very difficult learning problem.  
The resulting mAP is within the expected range for from-scratch training.

The model performs best on Person due to more data and larger object size.  
Chair is the hardest class due to small size, occlusion, and visual diversity.

Common failure modes:
- Small or distant objects
- Occluded chairs
- Overlapping people
- Extreme viewpoints

These are visible in the demo video.

---

## Trade-Offs

| Aspect | Result |
|--------|-------|
| Accuracy | Low (10.32% mAP) |
| Speed | Very high (312 FPS) |
| Model size | Small (31.57 MB) |

The system prioritizes speed and deployability over accuracy, which is suitable for real-time applications.

---

## Conclusion

This project demonstrates that a complete object detector can be built, trained, evaluated, and deployed without relying on pretrained models.  
It provides a full pipeline from raw data to real-time inference and serves as a strong foundation for larger-scale detection systems.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from src.data.dataset import VOCDetectionDataset
from src.data.transforms import get_val_transforms
from src.models.detector import GridDetector


class Evaluator:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # ✅ UPDATED: 3 classes only
        self.class_names = ['person', 'car', 'chair']
        
        # Load model
        self.model = GridDetector(num_classes=3, grid_size=7, anchors_per_cell=2)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
        print(f"   Device: {self.device}")
    
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes [xmin, ymin, xmax, ymax]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)
    
    def compute_ap(self, precisions, recalls):
        """
        Compute Average Precision using 11-point interpolation (VOC style)
        """
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        return ap
    
    def evaluate_map(self, dataset, iou_threshold=0.5, conf_threshold=0.3):
        """
        Calculate proper mAP@0.5 per class with PR curves
        """
        # Collect all predictions and ground truths per class
        all_detections = defaultdict(list)
        all_ground_truths = defaultdict(int)
        
        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc="Collecting detections"):
                img, gt_boxes, gt_labels, num_objs = dataset[idx]
                
                # Get predictions
                img_batch = img.unsqueeze(0).to(self.device)
                predictions = self.model(img_batch)
                detections = self.model.decode_predictions(predictions, conf_threshold=conf_threshold)[0]
                
                pred_boxes = detections['boxes'].cpu().numpy()
                pred_labels = detections['labels'].cpu().numpy()
                pred_scores = detections['scores'].cpu().numpy()
                
                # Ground truth (filter valid objects)
                gt_boxes_valid = gt_boxes[:num_objs].numpy()
                gt_labels_valid = gt_labels[:num_objs].numpy()
                
                # Count ground truths per class
                for gt_label in gt_labels_valid:
                    all_ground_truths[gt_label] += 1
                
                # Track which ground truths have been matched
                gt_matched = [False] * len(gt_boxes_valid)
                
                # Sort predictions by confidence (descending)
                sorted_indices = np.argsort(-pred_scores)
                
                for idx in sorted_indices:
                    pred_box = pred_boxes[idx]
                    pred_label = pred_labels[idx]
                    pred_score = pred_scores[idx]
                    
                    # Find best matching ground truth
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes_valid, gt_labels_valid)):
                        if gt_matched[gt_idx]:
                            continue
                        
                        if pred_label != gt_label:
                            continue
                        
                        iou = self.compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    # Determine if this is a true positive
                    is_tp = (best_iou >= iou_threshold and best_gt_idx != -1)
                    
                    if is_tp:
                        gt_matched[best_gt_idx] = True
                    
                    # Store detection result
                    all_detections[pred_label].append((pred_score, is_tp))
        
        # Compute AP for each class
        aps = []
        
        for class_id in range(3):  # ✅ 3 classes
            if class_id not in all_ground_truths or all_ground_truths[class_id] == 0:
                continue
            
            detections = all_detections.get(class_id, [])
            n_gt = all_ground_truths[class_id]
            
            if len(detections) == 0:
                aps.append(0.0)
                continue
            
            # Sort by score descending
            detections = sorted(detections, key=lambda x: x[0], reverse=True)
            
            # Compute precision and recall at each detection
            tp = np.array([d[1] for d in detections], dtype=float)
            fp = 1 - tp
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / n_gt
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # Compute AP using 11-point interpolation
            ap = self.compute_ap(precisions, recalls)
            aps.append(ap)
        
        # Mean AP across classes
        mAP = np.mean(aps) if aps else 0.0
        
        # Also compute overall precision/recall for reference
        total_tp = sum([sum([d[1] for d in all_detections[c]]) for c in all_detections])
        total_detections = sum([len(all_detections[c]) for c in all_detections])
        total_gt = sum(all_ground_truths.values())
        
        precision = total_tp / total_detections if total_detections > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        
        return {
            'mAP@0.5': mAP,
            'per_class_AP': {self.class_names[i]: aps[i] for i in range(len(aps))},
            'Precision': precision,
            'Recall': recall,
            'total_TP': int(total_tp),
            'total_detections': total_detections,
            'total_GT': total_gt
        }
    
    def measure_fps(self, dataset, num_samples=100):
        """Measure inference speed (FPS)"""
        times = []
        
        with torch.no_grad():
            for idx in tqdm(range(min(num_samples, len(dataset))), desc="Measuring FPS"):
                img, _, _, _ = dataset[idx]
                img_batch = img.unsqueeze(0).to(self.device)
                
                # Warmup
                if idx < 10:
                    _ = self.model(img_batch)
                    continue
                
                start = time.time()
                _ = self.model(img_batch)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        fps = 1 / avg_time
        
        return fps, avg_time


def main():
    # Configuration
    checkpoint_path = 'outputs/weights/best.pth'  # ✅ Use best model
    test_img_dir = 'dataset/test/images'
    test_ann_dir = 'dataset/test/annotations'
    
    # Load test dataset
    test_dataset = VOCDetectionDataset(
        img_dir=test_img_dir,
        ann_dir=test_ann_dir,
        transform=get_val_transforms(224),
        max_objects=30
    )
    
    print(f"\n{'='*70}")
    print(f"EVALUATION - 3 CLASSES (VOC-style mAP@0.5)")
    print(f"{'='*70}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: person, car, chair")
    print(f"{'='*70}\n")
    
    # Initialize evaluator
    evaluator = Evaluator(checkpoint_path)
    
    # Evaluate mAP
    print("\n--- mAP Evaluation ---")
    map_results = evaluator.evaluate_map(test_dataset, iou_threshold=0.5, conf_threshold=0.3)
    
    # Measure FPS
    print("\n--- Speed Benchmark ---")
    fps, avg_time = evaluator.measure_fps(test_dataset, num_samples=100)
    
    # Model size
    model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS (3 CLASSES)")
    print(f"{'='*70}")
    print(f"mAP@0.5 (11-pt): {map_results['mAP@0.5']:.4f} ({map_results['mAP@0.5']*100:.2f}%)")
    print(f"\nPer-class AP:")
    for cls_name, ap in map_results['per_class_AP'].items():
        print(f"  {cls_name:10s}: {ap:.4f} ({ap*100:.2f}%)")
    print(f"\nOverall Precision: {map_results['Precision']:.4f}")
    print(f"Overall Recall:    {map_results['Recall']:.4f}")
    print(f"Total TP / Det / GT: {map_results['total_TP']} / {map_results['total_detections']} / {map_results['total_GT']}")
    print(f"\nFPS:           {fps:.2f}")
    print(f"Avg Time:      {avg_time*1000:.2f} ms")
    print(f"Model Size:    {model_size_mb:.2f} MB")
    print(f"{'='*70}\n")
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    results_path = 'outputs/evaluation_results_3classes.txt'
    with open(results_path, 'w') as f:
        f.write(f"EVALUATION RESULTS - 3 CLASSES (VOC-style mAP@0.5)\n")
        f.write(f"{'='*70}\n")
        f.write(f"mAP@0.5 (11-point interpolation): {map_results['mAP@0.5']:.4f} ({map_results['mAP@0.5']*100:.2f}%)\n\n")
        f.write(f"Per-class AP:\n")
        for cls_name, ap in map_results['per_class_AP'].items():
            f.write(f"  {cls_name:10s}: {ap:.4f} ({ap*100:.2f}%)\n")
        f.write(f"\nOverall Precision: {map_results['Precision']:.4f}\n")
        f.write(f"Overall Recall:    {map_results['Recall']:.4f}\n")
        f.write(f"Total TP / Detections / GT: {map_results['total_TP']} / {map_results['total_detections']} / {map_results['total_GT']}\n")
        f.write(f"\nFPS:           {fps:.2f}\n")
        f.write(f"Avg Time:      {avg_time*1000:.2f} ms\n")
        f.write(f"Model Size:    {model_size_mb:.2f} MB\n")
    
    print(f"💾 Results saved to {results_path}")


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """
    Combined loss for grid-based object detection.
    ✅ UPDATED: 3 classes (person, car, chair) with optimized weights
    """
    def __init__(self, num_classes=3, grid_size=7, anchors_per_cell=2):
        super().__init__()
        
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.anchors_per_cell = anchors_per_cell
        
        # Loss weights
        self.lambda_coord = 5.0
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5
        self.lambda_class = 2.0
        
        # ✅ UPDATED: Class weights for 3 classes
        # person: 28,075 (71%) → weight 1.0
        # car:     5,677 (14%) → weight 4.9
        # chair:   5,862 (15%) → weight 4.8
        # Imbalance: ~5:1 (much better!)
        self.register_buffer(
            "class_weights",
            torch.tensor([1.0, 4.9, 4.8], dtype=torch.float32)
        )
        
        # ✅ OPTIMIZED: Anchors computed from YOUR 3-class dataset
        self.register_buffer(
            "anchors",
            torch.tensor([[0.971, 1.7338], [3.4579, 5.1653]], dtype=torch.float32)
        )
    
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])
        
        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        
        union = area1 + area2 - inter
        
        return inter / (union + 1e-6)
    
    def assign_targets(self, predictions, target_boxes, target_labels, num_objs):
        """Assign ground truth objects to grid cells and anchors"""
        device = predictions.device
        B, G, _, A, _ = predictions.shape
        
        # Initialize target tensors
        obj_mask = torch.zeros(B, G, G, A, device=device)
        noobj_mask = torch.ones(B, G, G, A, device=device)
        target_obj = torch.zeros(B, G, G, A, device=device)
        target_bbox = torch.zeros(B, G, G, A, 4, device=device)
        target_cls = torch.zeros(B, G, G, A, device=device, dtype=torch.long)
        
        for b in range(B):
            n_obj = num_objs[b].item()
            
            for obj_idx in range(n_obj):
                # Ground truth box and label
                gt_box = target_boxes[b, obj_idx]
                gt_label = target_labels[b, obj_idx].item()
                
                # Convert to center format
                gt_cx = (gt_box[0] + gt_box[2]) / 2
                gt_cy = (gt_box[1] + gt_box[3]) / 2
                gt_w = gt_box[2] - gt_box[0]
                gt_h = gt_box[3] - gt_box[1]
                
                # Skip invalid boxes
                if gt_w <= 0 or gt_h <= 0:
                    continue
                
                # Find responsible grid cell
                grid_i = int(gt_cy * G)
                grid_j = int(gt_cx * G)
                
                # Clamp to valid range
                grid_i = min(max(grid_i, 0), G - 1)
                grid_j = min(max(grid_j, 0), G - 1)
                
                # Find best anchor by IoU
                gt_w_grid = gt_w * G
                gt_h_grid = gt_h * G
                
                best_iou = 0
                best_anchor = 0
                for a in range(A):
                    anchor_w = self.anchors[a, 0]
                    anchor_h = self.anchors[a, 1]
                    
                    inter = min(gt_w_grid, anchor_w) * min(gt_h_grid, anchor_h)
                    union = gt_w_grid * gt_h_grid + anchor_w * anchor_h - inter
                    iou = inter / (union + 1e-6)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor = a
                
                # Mark as responsible
                obj_mask[b, grid_i, grid_j, best_anchor] = 1
                noobj_mask[b, grid_i, grid_j, best_anchor] = 0
                
                # Target objectness
                target_obj[b, grid_i, grid_j, best_anchor] = 1.0
                
                # Target bbox (relative to grid cell)
                tx = gt_cx * G - grid_j
                ty = gt_cy * G - grid_i
                
                safe_w = torch.clamp(gt_w_grid, min=0.01)
                safe_h = torch.clamp(gt_h_grid, min=0.01)
                
                tw = torch.log(safe_w / (self.anchors[best_anchor, 0] + 1e-6))
                th = torch.log(safe_h / (self.anchors[best_anchor, 1] + 1e-6))
                
                target_bbox[b, grid_i, grid_j, best_anchor] = torch.tensor(
                    [tx, ty, tw, th], 
                    device=device,
                    dtype=torch.float32
                )
                
                # Target class
                target_cls[b, grid_i, grid_j, best_anchor] = gt_label
        
        return obj_mask, noobj_mask, target_obj, target_bbox, target_cls
    
    def forward(self, predictions, target_boxes, target_labels, num_objs):
        """
        predictions: (B, G, G, A, 5 + num_classes)
        target_boxes: (B, max_objects, 4)
        target_labels: (B, max_objects)
        num_objs: (B,)
        """
        device = predictions.device
        B, G, _, A, _ = predictions.shape
        
        # Split predictions
        pred_obj = predictions[..., 0]
        pred_bbox = predictions[..., 1:5]
        pred_cls = predictions[..., 5:]
        
        # Assign targets
        obj_mask, noobj_mask, target_obj, target_bbox, target_cls = self.assign_targets(
            predictions, target_boxes, target_labels, num_objs
        )
        
        # 1. Objectness loss
        loss_obj = F.binary_cross_entropy_with_logits(
            pred_obj[obj_mask == 1],
            target_obj[obj_mask == 1],
            reduction='sum'
        ) if obj_mask.sum() > 0 else torch.tensor(0.0, device=device)
        
        loss_noobj = F.binary_cross_entropy_with_logits(
            pred_obj[noobj_mask == 1],
            target_obj[noobj_mask == 1],
            reduction='sum'
        ) if noobj_mask.sum() > 0 else torch.tensor(0.0, device=device)
        
        # 2. Bounding box loss
        loss_bbox = F.smooth_l1_loss(
            pred_bbox[obj_mask == 1],
            target_bbox[obj_mask == 1],
            reduction='sum'
        ) if obj_mask.sum() > 0 else torch.tensor(0.0, device=device)
        
        # 3. Classification loss (with class weights)
        pred_cls_flat = pred_cls.reshape(-1, self.num_classes)
        target_cls_flat = target_cls.reshape(-1)
        obj_mask_flat = obj_mask.reshape(-1) == 1
        
        loss_cls = F.cross_entropy(
            pred_cls_flat[obj_mask_flat],
            target_cls_flat[obj_mask_flat],
            weight=self.class_weights,
            reduction='sum'
        ) if obj_mask_flat.sum() > 0 else torch.tensor(0.0, device=device)
        
        # Normalize by number of positive anchors
        num_pos = obj_mask.sum().clamp(min=1)
        
        total_loss = (
            self.lambda_coord * loss_bbox / num_pos +
            self.lambda_obj * loss_obj / num_pos +
            self.lambda_noobj * loss_noobj / num_pos +
            self.lambda_class * loss_cls / num_pos
        )
        
        return total_loss, {
            'loss_bbox': loss_bbox.item() / (num_pos.item() + 1e-6),
            'loss_obj': loss_obj.item() / (num_pos.item() + 1e-6),
            'loss_noobj': loss_noobj.item() / (num_pos.item() + 1e-6),
            'loss_cls': loss_cls.item() / (num_pos.item() + 1e-6)
        }
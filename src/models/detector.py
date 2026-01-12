import torch
import torch.nn as nn
from src.models.backbone import CNNBackbone


class GridDetector(nn.Module):
    """
    Grid-based object detector (from scratch).
    Output: Grid predictions for multiple objects.
    
    ✅ FIXED: Correct grid-relative anchors
    """
    def __init__(self, num_classes=4, grid_size=7, anchors_per_cell=2):
        super().__init__()
        
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.anchors_per_cell = anchors_per_cell
        
        # Backbone
        self.backbone = CNNBackbone()
        
        # Detection head
        # Each cell predicts: [objectness, x, y, w, h, class_probs] per anchor
        out_channels = anchors_per_cell * (5 + num_classes)
        
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1)
        )
        
        # ✅ OPTIMIZED: Anchors computed from YOUR 3-class dataset (person, car, chair)
        # These are anchors in GRID CELL coordinates (not image coordinates)
        # Anchor 0: Small/tall objects (person, chair) - 3.4% of image
        # Anchor 1: Large objects (car, large persons) - 36.5% of image
        self.register_buffer(
            "anchors",
            torch.tensor([[0.971, 1.7338], [3.4579, 5.1653]], dtype=torch.float32)
        )
    
    def forward(self, x):
        """
        x: (B, 3, 224, 224)
        Returns: (B, grid_size, grid_size, anchors_per_cell, 5 + num_classes)
        """
        features = self.backbone(x)  # (B, 512, 7, 7)
        predictions = self.detection_head(features)  # (B, out_channels, 7, 7)
        
        B = x.size(0)
        
        # Reshape to grid format
        predictions = predictions.view(
            B, 
            self.anchors_per_cell, 
            5 + self.num_classes, 
            self.grid_size, 
            self.grid_size
        )
        
        # Permute to (B, grid_size, grid_size, anchors_per_cell, 5 + num_classes)
        predictions = predictions.permute(0, 3, 4, 1, 2).contiguous()
        
        return predictions
    
    def decode_predictions(self, predictions, conf_threshold=0.5):
        """
        Convert grid predictions to bounding boxes.
        predictions: (B, grid_size, grid_size, anchors_per_cell, 5 + num_classes)
        Returns: List of detections per image
        """
        device = predictions.device
        B, G, _, A, _ = predictions.shape
        
        detections = []
        
        for b in range(B):
            boxes = []
            scores = []
            labels = []
            
            for i in range(G):
                for j in range(G):
                    for a in range(A):
                        pred = predictions[b, i, j, a]
                        
                        objectness = torch.sigmoid(pred[0])
                        
                        if objectness < conf_threshold:
                            continue
                        
                        # Decode bbox (relative to grid cell)
                        tx, ty = torch.sigmoid(pred[1]), torch.sigmoid(pred[2])
                        tw, th = torch.exp(pred[3]), torch.exp(pred[4])
                        
                        # ✅ FIXED: Anchors are already in grid coordinates, no division needed
                        # Convert to image coordinates [0, 1]
                        cx = (j + tx) / G
                        cy = (i + ty) / G
                        w = tw * self.anchors[a, 0] / G
                        h = th * self.anchors[a, 1] / G
                        
                        # Convert to [xmin, ymin, xmax, ymax]
                        xmin = cx - w / 2
                        ymin = cy - h / 2
                        xmax = cx + w / 2
                        ymax = cy + h / 2
                        
                        # Clamp to [0, 1]
                        xmin = torch.clamp(xmin, 0, 1)
                        ymin = torch.clamp(ymin, 0, 1)
                        xmax = torch.clamp(xmax, 0, 1)
                        ymax = torch.clamp(ymax, 0, 1)
                        
                        # Class prediction
                        class_probs = torch.softmax(pred[5:], dim=0)
                        class_score, class_id = class_probs.max(0)
                        
                        final_score = objectness * class_score
                        
                        boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
                        scores.append(final_score.item())
                        labels.append(class_id.item())
            
            detections.append({
                'boxes': torch.tensor(boxes) if boxes else torch.empty((0, 4)),
                'scores': torch.tensor(scores) if scores else torch.empty((0,)),
                'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.empty((0,), dtype=torch.long)
            })
        
        return detections
import os
import torch
import cv2
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VOCDetectionDataset(Dataset):
    """
    Multi-object detection dataset for PASCAL VOC format.
    ✅ UPDATED: 3 classes only (person, car, chair)
    """
    def __init__(self, img_dir, ann_dir, transform=None, max_objects=30):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.max_objects = max_objects
        
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        # ✅ UPDATED: 3 classes only
        self.class_map = {
            "person": 0,
            "car": 1,
            "chair": 2
        }
        self.num_classes = len(self.class_map)
    
    def __len__(self):
        return len(self.images)
    
    def parse_annotation(self, xml_path):
        """
        Parse VOC XML and return all bounding boxes + classes.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.class_map:
                continue
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Normalize to [0, 1]
            boxes.append([
                xmin / width,
                ymin / height,
                xmax / width,
                ymax / height
            ])
            labels.append(self.class_map[class_name])
        
        return boxes, labels
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name.replace('.jpg', '.xml'))
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Parse annotations
        boxes, labels = self.parse_annotation(ann_path)
        
        # Safety clip: handle images with too many objects
        if len(boxes) > self.max_objects:
            boxes = boxes[:self.max_objects]
            labels = labels[:self.max_objects]
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        # Pad to max_objects (important for batching)
        num_objs = len(boxes)
        
        # Create fixed-size tensors
        target_boxes = torch.zeros((self.max_objects, 4), dtype=torch.float32)
        target_labels = torch.full((self.max_objects,), -1, dtype=torch.long)
        
        if num_objs > 0:
            target_boxes[:num_objs] = torch.tensor(boxes, dtype=torch.float32)
            target_labels[:num_objs] = torch.tensor(labels, dtype=torch.long)
        
        return image, target_boxes, target_labels, num_objs
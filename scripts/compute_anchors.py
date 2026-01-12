import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.cluster import KMeans
import xml.etree.ElementTree as ET
from tqdm import tqdm


def parse_voc_boxes(ann_dir, grid_size=7):
    """Extract all box widths and heights RELATIVE TO GRID CELL"""
    boxes = []
    
    for xml_file in tqdm(os.listdir(ann_dir), desc=f"Processing {ann_dir}"):
        if not xml_file.endswith('.xml'):
            continue
        
        xml_path = os.path.join(ann_dir, xml_file)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size = root.find('size')
            if size is None:
                continue
                
            img_w = float(size.find('width').text)
            img_h = float(size.find('height').text)
            
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Normalize to [0, 1] relative to IMAGE
                w_img = (xmax - xmin) / img_w
                h_img = (ymax - ymin) / img_h
                
                # ✅ CRITICAL: Convert to GRID-RELATIVE coordinates
                # Since grid is 7x7, each cell represents 1/7 of image
                # So a box that's 0.3 of image = 0.3 * 7 = 2.1 grid cells
                w_grid = w_img * grid_size
                h_grid = h_img * grid_size
                
                # Filter out invalid boxes
                if w_grid > 0 and h_grid > 0:
                    boxes.append([w_grid, h_grid])
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue
    
    return np.array(boxes)


def compute_optimal_anchors(boxes, n_anchors=2):
    """K-means clustering to find optimal anchor sizes"""
    kmeans = KMeans(n_clusters=n_anchors, random_state=42, n_init=10)
    kmeans.fit(boxes)
    
    anchors = kmeans.cluster_centers_
    # Sort by area
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    
    return anchors


def main():
    ann_dirs = [
        'dataset/train/annotations',
        'dataset/val/annotations'
    ]
    
    grid_size = 7  # Your grid size
    
    all_boxes = []
    for ann_dir in ann_dirs:
        if not os.path.exists(ann_dir):
            print(f"Warning: {ann_dir} not found, skipping...")
            continue
            
        print(f"\nProcessing {ann_dir}...")
        boxes = parse_voc_boxes(ann_dir, grid_size=grid_size)
        if len(boxes) > 0:
            all_boxes.append(boxes)
            print(f"  Found {len(boxes)} boxes")
    
    if len(all_boxes) == 0:
        print("Error: No boxes found!")
        return
    
    all_boxes = np.vstack(all_boxes)
    print(f"\n{'='*70}")
    print(f"Total boxes analyzed: {len(all_boxes)}")
    print(f"Box width range (grid cells): {all_boxes[:, 0].min():.3f} - {all_boxes[:, 0].max():.3f}")
    print(f"Box height range (grid cells): {all_boxes[:, 1].min():.3f} - {all_boxes[:, 1].max():.3f}")
    print(f"{'='*70}")
    
    # Compute optimal anchors
    print("\n🔍 Computing optimal anchors using K-means...")
    anchors = compute_optimal_anchors(all_boxes, n_anchors=2)
    
    print("\n" + "="*70)
    print("✅ OPTIMAL ANCHORS (GRID-RELATIVE):")
    print("="*70)
    for i, (w, h) in enumerate(anchors):
        print(f"Anchor {i}: width={w:.4f}, height={h:.4f} (grid cells)")
        print(f"           aspect_ratio={w/h:.3f}, area={w*h:.3f}")
    
    print("\n" + "="*70)
    print("📝 UPDATE YOUR CODE:")
    print("="*70)
    
    # Format as Python code
    anchors_formatted = [[float(f"{w:.4f}"), float(f"{h:.4f}")] for w, h in anchors]
    
    print("\n1️⃣  In src/models/detector.py (around line 32):")
    print("   Replace the anchors line with:")
    print(f"   self.register_buffer('anchors', torch.tensor({anchors_formatted}, dtype=torch.float32))")
    
    print("\n2️⃣  In src/training/loss.py (around line 27):")
    print("   Replace the anchors line with:")
    print(f"   self.register_buffer('anchors', torch.tensor({anchors_formatted}, dtype=torch.float32))")
    
    print("\n" + "="*70)
    print("⚠️  CRITICAL NOTES:")
    print("="*70)
    print("• These anchors are GRID-RELATIVE (not image-relative)")
    print("• For grid_size=7, these represent actual grid cell dimensions")
    print("• Do NOT divide by grid_size in your code!")
    print("• Your model will now predict accurate bounding boxes")
    print("="*70)
    
    # Analysis
    print("\n📊 Anchor Analysis:")
    print(f"   Anchor 0 (small):  covers ~{anchors[0][0]*anchors[0][1]/(grid_size*grid_size)*100:.1f}% of image")
    print(f"   Anchor 1 (large):  covers ~{anchors[1][0]*anchors[1][1]/(grid_size*grid_size)*100:.1f}% of image")
    
    # Convert back to image-relative for understanding
    print("\n🔄 For reference (image-relative [0,1] scale):")
    for i, (w, h) in enumerate(anchors):
        w_img = w / grid_size
        h_img = h / grid_size
        print(f"   Anchor {i}: width={w_img:.4f}, height={h_img:.4f} (image scale)")


if __name__ == "__main__":
    main()
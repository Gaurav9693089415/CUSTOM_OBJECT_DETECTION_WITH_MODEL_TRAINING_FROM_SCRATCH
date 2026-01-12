"""
Object Detection Demo Script
Runs detection on multiple test images and saves visualizations
"""

import torch
import cv2
import numpy as np
import os
import random
from pathlib import Path
import argparse

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.detector import GridDetector
import albumentations as A


class ObjectDetector:
    """Object detector with visualization"""
    
    def __init__(self, checkpoint_path='outputs/weights/best.pth', conf_threshold=0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        
        # Class configuration
        self.class_names = ['person', 'car', 'chair']
        self.colors = {
            0: (255, 100, 100),    # person - blue-ish
            1: (100, 255, 100),    # car - green-ish
            2: (100, 100, 255)     # chair - red-ish
        }
        
        print(f"🔄 Loading model from: {checkpoint_path}")
        
        # Load model
        self.model = GridDetector(num_classes=3, grid_size=7, anchors_per_cell=2)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Epoch: {checkpoint['epoch'] + 1}")
        print(f"   Device: {self.device}")
        print(f"   Confidence threshold: {conf_threshold}")
    
    def preprocess(self, image):
        """Preprocess image for model"""
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transformed = transform(image=image)
        img_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
        
        return img_tensor.unsqueeze(0)
    
    def detect(self, image):
        """Run detection on image"""
        h, w = image.shape[:2]
        
        # Preprocess
        img_tensor = self.preprocess(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(img_tensor)
            detections = self.model.decode_predictions(
                predictions, 
                conf_threshold=self.conf_threshold
            )[0]
        
        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        labels = detections['labels'].cpu().numpy()
        
        # Scale boxes to original image size
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
            boxes = boxes.astype(int)
        
        return boxes, scores, labels
    
    def draw_detections(self, image, boxes, scores, labels):
        """Draw bounding boxes on image"""
        img_vis = image.copy()
        
        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box
            
            # Ensure valid coordinates
            xmin = max(0, min(xmin, img_vis.shape[1] - 1))
            xmax = max(0, min(xmax, img_vis.shape[1] - 1))
            ymin = max(0, min(ymin, img_vis.shape[0] - 1))
            ymax = max(0, min(ymax, img_vis.shape[0] - 1))
            
            color = self.colors[label]
            class_name = self.class_names[label]
            
            # Draw bounding box
            cv2.rectangle(img_vis, (xmin, ymin), (xmax, ymax), color, 3)
            
            # Prepare label text
            label_text = f"{class_name}: {score:.2f}"
            font_scale = 0.8
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Draw label background
            cv2.rectangle(
                img_vis,
                (xmin, ymin - text_h - 10),
                (xmin + text_w + 10, ymin),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img_vis,
                label_text,
                (xmin + 5, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return img_vis


def demo_images(num_images=10, conf_threshold=0.3):
    """Run detection on sample test images"""
    
    print("\n" + "="*70)
    print("OBJECT DETECTION DEMO - IMAGE MODE")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = ObjectDetector(
        checkpoint_path='outputs/weights/best.pth',
        conf_threshold=conf_threshold
    )
    
    # Setup paths
    test_img_dir = Path('dataset/test/images')
    output_dir = Path('outputs/demo_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all test images
    all_images = list(test_img_dir.glob('*.jpg'))
    
    if not all_images:
        print(f"❌ No images found in {test_img_dir}")
        return
    
    # Sample random images
    num_samples = min(num_images, len(all_images))
    sample_images = random.sample(all_images, num_samples)
    
    print(f"📸 Processing {num_samples} sample images...")
    print(f"   Input: {test_img_dir}")
    print(f"   Output: {output_dir}\n")
    
    # Statistics
    total_detections = 0
    class_counts = {name: 0 for name in detector.class_names}
    
    # Process each image
    for i, img_path in enumerate(sample_images, 1):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️  [{i}/{num_samples}] Could not load {img_path.name}")
            continue
        
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        boxes, scores, labels = detector.detect(image_rgb)
        
        # Update statistics
        total_detections += len(boxes)
        for label in labels:
            class_counts[detector.class_names[label]] += 1
        
        # Draw detections
        img_vis = detector.draw_detections(image, boxes, scores, labels)
        
        # Save result
        output_path = output_dir / f'detected_{img_path.name}'
        cv2.imwrite(str(output_path), img_vis)
        
        # Print result
        det_summary = []
        for label, score in zip(labels, scores):
            det_summary.append(f"{detector.class_names[label]}({score:.2f})")
        
        det_str = ", ".join(det_summary) if det_summary else "None"
        print(f"✅ [{i:2d}/{num_samples}] {img_path.name:30s} | Size: {w}×{h:3d} | Detections: {det_str}")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 DETECTION SUMMARY")
    print("="*70)
    print(f"Total images processed: {num_samples}")
    print(f"Total detections: {total_detections}")
    print(f"\nDetections by class:")
    for class_name, count in class_counts.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"   {class_name:10s}: {count:3d} ({percentage:5.1f}%)")
    
    print(f"\n💾 Results saved to: {output_dir.absolute()}")
    print("="*70 + "\n")
    
    # Instructions
    print("💡 To view results:")
    print(f"   Windows: start {output_dir}")
    print(f"   Or open: {output_dir.absolute()}\n")


def demo_video(video_source=0, conf_threshold=0.3):
    """Run detection on video/webcam"""
    
    print("\n" + "="*70)
    print("OBJECT DETECTION DEMO - VIDEO MODE")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = ObjectDetector(
        checkpoint_path='outputs/weights/best.pth',
        conf_threshold=conf_threshold
    )
    
    # Open video source
    if isinstance(video_source, int):
        print(f"📹 Opening webcam (index {video_source})...")
    else:
        print(f"📹 Opening video: {video_source}...")
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("❌ Cannot open video source")
        print("   Try: python demo.py --mode video --source 0")
        print("   Or specify a video file path")
        return
    
    print("✅ Video source opened")
    print("\n💡 Controls:")
    print("   'q' or 'ESC' - Quit")
    print("   'p' - Pause/Resume")
    print("   's' - Save current frame\n")
    
    # Setup
    paused = False
    frame_count = 0
    save_count = 0
    output_dir = Path('outputs/demo_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("📹 End of video or read error")
                break
            
            frame_count += 1
            
            # Convert and detect
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, scores, labels = detector.detect(frame_rgb)
            
            # Draw detections
            frame_vis = detector.draw_detections(frame, boxes, scores, labels)
        else:
            # Keep showing last frame when paused
            frame_vis = frame_vis.copy()
        
        # Add info overlay
        info_text = f"Frame: {frame_count} | Detections: {len(boxes)}"
        if paused:
            info_text += " | PAUSED"
        
        cv2.putText(
            frame_vis,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        # Show frame
        cv2.imshow('Object Detection Demo - Press Q to quit', frame_vis)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("\n👋 Quitting...")
            break
        elif key == ord('p'):  # Pause
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"⏸️  {status}")
        elif key == ord('s'):  # Save frame
            save_path = output_dir / f'video_frame_{save_count:04d}.jpg'
            cv2.imwrite(str(save_path), frame_vis)
            save_count += 1
            print(f"💾 Saved frame to: {save_path}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("📊 VIDEO DEMO SUMMARY")
    print("="*70)
    print(f"Total frames processed: {frame_count}")
    print(f"Frames saved: {save_count}")
    if save_count > 0:
        print(f"Saved to: {output_dir.absolute()}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Object Detection Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on 10 random test images
  python demo.py --mode images
  
  # Run on 20 images with lower confidence
  python demo.py --mode images --num 20 --conf 0.2
  
  # Run on webcam
  python demo.py --mode video
  
  # Run on video file
  python demo.py --mode video --source path/to/video.mp4
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='images',
        choices=['images', 'video'],
        help='Demo mode: images or video (default: images)'
    )
    
    parser.add_argument(
        '--num',
        type=int,
        default=10,
        help='Number of images to process in image mode (default: 10)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.3,
        help='Confidence threshold for detections (default: 0.3)'
    )
    
    parser.add_argument(
        '--source',
        default=0,
        help='Video source: webcam index (0) or video file path (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Run demo
    try:
        if args.mode == 'images':
            demo_images(num_images=args.num, conf_threshold=args.conf)
        else:
            # Handle video source
            try:
                video_source = int(args.source)
            except ValueError:
                video_source = args.source
            
            demo_video(video_source=video_source, conf_threshold=args.conf)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
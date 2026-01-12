import os
import shutil
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ============================================================
# CONFIGURATION - 3 CLASSES VERSION
# ============================================================
VOC_ROOT = "dataset/raw/VOCdevkit"
OUT_ROOT = "dataset"

# ✅ 3 CLASSES ONLY
TARGET_CLASSES = {"person", "car", "chair"}

MAX_IMAGES = None  # Use all available

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

SEED = 42
random.seed(SEED)

# ============================================================
# CLASS MAPPING (for model training)
# ============================================================
CLASS_MAP = {
    "person": 0,
    "car": 1,
    "chair": 2
}

# ============================================================
# FUNCTIONS
# ============================================================

def filter_and_save_xml(src_xml, dst_xml):
    """
    Filter XML to keep only target classes and save.
    Returns True if any target class objects remain.
    """
    tree = ET.parse(src_xml)
    root = tree.getroot()
    
    keep = False
    for obj in list(root.findall("object")):
        cls = obj.find("name").text
        if cls not in TARGET_CLASSES:
            root.remove(obj)
        else:
            keep = True
    
    if keep:
        tree.write(dst_xml)
    
    return keep


def collect_samples(year):
    """
    Collect all samples from a VOC year that contain target classes.
    """
    base = os.path.join(VOC_ROOT, year)
    img_dir = os.path.join(base, "JPEGImages")
    ann_dir = os.path.join(base, "Annotations")
    
    if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
        print(f"⚠️  Warning: {year} not found, skipping...")
        return []
    
    samples = []
    xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]
    
    for xml in tqdm(xml_files, desc=f"Collecting {year}"):
        xml_path = os.path.join(ann_dir, xml)
        img_path = os.path.join(img_dir, xml.replace(".xml", ".jpg"))
        
        if not os.path.exists(img_path):
            continue
        
        # Quick check: does this XML contain target classes?
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            has_target = any(
                obj.find("name").text in TARGET_CLASSES
                for obj in root.findall("object")
            )
            
            if has_target:
                samples.append((img_path, xml_path))
        except Exception as e:
            print(f"⚠️  Error parsing {xml}: {e}")
            continue
    
    return samples


def make_dirs():
    """Create output directory structure."""
    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUT_ROOT}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUT_ROOT}/{split}/annotations", exist_ok=True)


def save_class_map():
    """Save class mapping for reference."""
    class_map_path = os.path.join(OUT_ROOT, "class_map.txt")
    with open(class_map_path, 'w') as f:
        f.write("CLASS MAPPING FOR MODEL (3 CLASSES):\n")
        f.write("="*50 + "\n")
        for cls_name, cls_id in sorted(CLASS_MAP.items(), key=lambda x: x[1]):
            f.write(f"{cls_id}: {cls_name}\n")
    print(f"\n💾 Class mapping saved to: {class_map_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*70)
    print("PREPARING DATASET - 3 CLASSES (FAST VERSION)")
    print("="*70)
    print(f"Target classes: {sorted(TARGET_CLASSES)}")
    print(f"Class mapping: {CLASS_MAP}")
    print(f"Split ratio: Train={SPLIT_RATIO['train']}, Val={SPLIT_RATIO['val']}, Test={SPLIT_RATIO['test']}")
    print("="*70)
    
    # Collect samples from both VOC years
    print(f"\n📂 Collecting samples from VOC dataset...")
    samples = []
    samples += collect_samples("VOC2007")
    samples += collect_samples("VOC2012")
    
    print(f"\n✅ Total samples found: {len(samples)}")
    
    # Shuffle for random split
    random.shuffle(samples)
    
    # Calculate split sizes
    n = len(samples)
    n_train = int(n * SPLIT_RATIO["train"])
    n_val = int(n * SPLIT_RATIO["val"])
    
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:]
    }
    
    print(f"\n📊 Split sizes:")
    for split_name, split_samples in splits.items():
        print(f"   {split_name:5s}: {len(split_samples):5d} images")
    
    # Create directories
    make_dirs()
    
    # Process each split
    split_stats = {}
    
    for split, items in splits.items():
        print(f"\n📝 Processing {split}...")
        saved = 0
        class_counts = {cls: 0 for cls in TARGET_CLASSES}
        
        for img_path, xml_path in tqdm(items, desc=f"  Copying {split}"):
            out_xml = f"{OUT_ROOT}/{split}/annotations/{os.path.basename(xml_path)}"
            
            # Filter and save XML
            keep = filter_and_save_xml(xml_path, out_xml)
            
            if keep:
                # Copy image
                shutil.copy(
                    img_path,
                    f"{OUT_ROOT}/{split}/images/{os.path.basename(img_path)}"
                )
                saved += 1
                
                # Count objects per class
                try:
                    tree = ET.parse(out_xml)
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        cls_name = obj.find('name').text
                        if cls_name in TARGET_CLASSES:
                            class_counts[cls_name] += 1
                except:
                    pass
            else:
                # Remove XML if no target objects
                if os.path.exists(out_xml):
                    os.remove(out_xml)
        
        split_stats[split] = {
            'images': saved,
            'class_counts': class_counts
        }
        
        print(f"   ✅ {split}: {saved} images saved")
        print(f"   Class distribution:")
        for cls_name in sorted(TARGET_CLASSES):
            print(f"      {cls_name:10s}: {class_counts[cls_name]:5d}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ DATASET PREPARED SUCCESSFULLY (3 CLASSES)!")
    print("="*70)
    
    total_images = sum(s['images'] for s in split_stats.values())
    total_objects = sum(
        sum(s['class_counts'].values()) for s in split_stats.values()
    )
    
    print(f"\n📊 Final Statistics:")
    print(f"   Total images: {total_images}")
    print(f"   Total objects: {total_objects}")
    
    print(f"\n📁 Split breakdown:")
    for split in ['train', 'val', 'test']:
        stats = split_stats[split]
        print(f"   {split.capitalize():5s}: {stats['images']:5d} images")
    
    print(f"\n🏷️  Overall class distribution:")
    overall_counts = {cls: 0 for cls in TARGET_CLASSES}
    for stats in split_stats.values():
        for cls, count in stats['class_counts'].items():
            overall_counts[cls] += count
    
    total_objs = sum(overall_counts.values())
    for cls_name in sorted(TARGET_CLASSES):
        count = overall_counts[cls_name]
        pct = (count / total_objs * 100) if total_objs > 0 else 0
        print(f"   {cls_name:10s}: {count:5d} ({pct:5.2f}%)")
    
    # Calculate imbalance
    max_count = max(overall_counts.values())
    min_count = min(overall_counts.values())
    imbalance = max_count / min_count if min_count > 0 else 0
    print(f"\n⚖️  Class imbalance ratio: {imbalance:.1f}:1")
    
    if imbalance > 10:
        print(f"   ⚠️  Moderate imbalance - will use class weights")
    elif imbalance > 5:
        print(f"   ✅ Good balance - class weights will help")
    else:
        print(f"   ✅ Excellent balance!")
    
    # Save class mapping
    save_class_map()
    
    print("\n" + "="*70)
    print("🎯 READY FOR TRAINING (3 CLASSES - FAST VERSION)!")
    print("="*70)


if __name__ == "__main__":
    main()
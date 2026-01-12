import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download(
    "vijayabhaskar96/pascal-voc-2007-and-2012"
)

print("Downloaded to:", path)

# Target directory
target_root = "dataset/raw"

os.makedirs(target_root, exist_ok=True)

# Copy VOCdevkit to our project structure
for item in os.listdir(path):
    if item == "VOCdevkit":
        shutil.copytree(
            os.path.join(path, item),
            os.path.join(target_root, item),
            dirs_exist_ok=True
        )

print("VOC dataset placed in dataset/raw/VOCdevkit")

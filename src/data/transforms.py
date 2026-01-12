import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size=224):
    """
    Training augmentations (stable, production-ready).
    Only bbox-safe augmentations to prevent tensor size mismatches.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        
        # Color augmentations (no geometric changes)
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='albumentations', 
        label_fields=['labels'],
        min_visibility=0.3
    ))


def get_val_transforms(img_size=224):
    """
    Validation transforms (no augmentation).
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))
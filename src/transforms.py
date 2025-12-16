"""
Image preprocessing and augmentation pipelines
for Diabetic Retinopathy fundus images.

Used for:
- Self-supervised BYOL pretraining
- Supervised fine-tuning
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


# --------------------------------------------------
# Utility: Crop black borders (fundus images)
# IMPORTANT: must accept **kwargs for Albumentations
# --------------------------------------------------
def crop_black(image, threshold=10, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(mask)
    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]


# --------------------------------------------------
# Base preprocessing (used everywhere)
# --------------------------------------------------
def base_preprocess(image_size=512):
    return A.Compose([
        A.Lambda(image=crop_black),
        A.Resize(height=image_size, width=image_size),
        A.CenterCrop(height=image_size, width=image_size),
    ])


# --------------------------------------------------
# Strong augmentation for BYOL (self-supervised)
# --------------------------------------------------
def byol_augment(image_size=512):
    return A.Compose([
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.5, 1.0),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.4),
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
            p=0.8
        ),
        A.GaussNoise(p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.OneOf([
            A.ToGray(p=1.0),
            A.Solarize(p=1.0),
        ], p=0.1),
        A.Normalize(),
        ToTensorV2(),
    ])


# --------------------------------------------------
# Mild augmentation for supervised fine-tuning
# --------------------------------------------------
def labeled_augment(image_size=512):
    return A.Compose([
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.8, 1.0),
            p=0.5
        ),
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.1,
            hue=0.05,
            p=0.4
        ),
        A.OneOf([
            A.CLAHE(p=1.0),
            A.NoOp()
        ], p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])


# --------------------------------------------------
# Validation / inference preprocessing (no aug)
# --------------------------------------------------
def val_transform(image_size=512):
    return A.Compose([
        A.Lambda(image=crop_black),
        A.Resize(height=image_size, width=image_size),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(),
        ToTensorV2(),
    ])

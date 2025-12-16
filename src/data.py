"""
Dataset loaders for Diabetic Retinopathy project.

Handles:
- Labeled dataset (5 DR stages)
- Unlabeled dataset (for BYOL / pseudo-labeling)
"""

import os
import cv2
import torch
from torch.utils.data import Dataset

from src.transforms import (
    base_preprocess,
    labeled_augment,
    byol_augment,
    val_transform
)


# --------------------------------------------------
# Label mapping (folder name → class index)
# --------------------------------------------------
CLASS_TO_IDX = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferate_DR": 4
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


# --------------------------------------------------
# Utility: read image safely
# --------------------------------------------------
def read_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# --------------------------------------------------
# Labeled Dataset
# --------------------------------------------------
class LabeledDRDataset(Dataset):
    """
    Dataset for labeled DR images.
    Expects directory structure:
        root/
          ├── No_DR/
          ├── Mild/
          ├── Moderate/
          ├── Severe/
          └── Proliferate_DR/
    """

    def __init__(self, root_dir, transform_type="train", image_size=512):
        self.root_dir = root_dir
        self.samples = []

        # choose transform
        if transform_type == "train":
            self.transform = labeled_augment(image_size)
        else:
            self.transform = val_transform(image_size)

        # collect image paths + labels
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, class_idx))

        if len(self.samples) == 0:
            raise RuntimeError("No labeled images found!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = read_image(path)
        image = base_preprocess()(image=image)["image"]
        image = self.transform(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long)


# --------------------------------------------------
# Unlabeled Dataset (BYOL / pseudo-labeling)
# --------------------------------------------------
class UnlabeledDRDataset(Dataset):
    """
    Dataset for unlabeled DR images.
    Expects:
        root/
          ├── img1.jpg
          ├── img2.jpg
          └── ...
    """

    def __init__(self, root_dir, image_size=512, byol=True):
        self.root_dir = root_dir
        self.byol = byol

        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(self.image_paths) == 0:
            raise RuntimeError("No unlabeled images found!")

        self.base_transform = base_preprocess(image_size)

        if byol:
            self.augment = byol_augment(image_size)
        else:
            self.augment = val_transform(image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        image = read_image(path)
        image = self.base_transform(image=image)["image"]

        if self.byol:
            view1 = self.augment(image=image)["image"]
            view2 = self.augment(image=image)["image"]
            return view1, view2
        else:
            image = self.augment(image=image)["image"]
            return image

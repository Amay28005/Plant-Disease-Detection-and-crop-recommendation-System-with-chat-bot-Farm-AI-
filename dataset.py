"""
dataset.py
----------
Loads an ImageFolder-style dataset where every sub-folder name is a class label.

Expected structure:
    DATA_DIR/
        Apple___Apple_scab/
            img001.jpg
            img002.jpg
        Apple___Black_rot/
            ...
        Tomato___Early_blight/
            ...
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import config


# ─────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def get_val_transforms():
    """No augmentation for validation and test — only resize + normalise."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


# ─────────────────────────────────────────────
#  DATASET BUILDER
# ─────────────────────────────────────────────

def build_datasets():
    """
    Loads all images, applies an 80/10/10 train/val/test split,
    and returns three dataset objects with the correct transforms.
    """
    # Load full dataset with training transforms first (split needs same length)
    full_dataset = datasets.ImageFolder(config.DATA_DIR)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)

    total       = len(full_dataset)
    n_train     = int(total * config.TRAIN_RATIO)
    n_val       = int(total * config.VAL_RATIO)
    n_test      = total - n_train - n_val

    generator   = torch.Generator().manual_seed(42)      # reproducible split
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Assign correct transforms to each subset
    train_ds.dataset = datasets.ImageFolder(config.DATA_DIR, transform=get_train_transforms())
    val_ds.dataset   = datasets.ImageFolder(config.DATA_DIR, transform=get_val_transforms())
    test_ds.dataset  = datasets.ImageFolder(config.DATA_DIR, transform=get_val_transforms())

    print(f"[Dataset]  Total: {total} | Classes: {num_classes}")
    print(f"           Train: {n_train} | Val: {n_val} | Test: {n_test}")

    return train_ds, val_ds, test_ds, class_names, num_classes


def build_loaders(train_ds, val_ds, test_ds):
    """Wraps datasets in DataLoaders."""
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,        # faster CPU→GPU transfer
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )
    return train_loader, val_loader, test_loader

"""Data loading, balanced subsetting, splitting, transforms and DataLoaders.

Key ideas
---------
* The raw dataset is imbalanced, so we sample the SAME number of images per class.
* Validation is carved out of the train pool (never from test) and does not overlap
  with the training subset.
* The official test split is touched ONLY for final evaluation.
"""
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config


# ------------------------------------------------------------------ loading
def load_oct_dataset(dataset_id: str = config.HF_DATASET_ID):
    """Download (first run) and return the DatasetDict with 'train' and 'test'."""
    return load_dataset(dataset_id)


def get_class_names(dataset) -> list:
    """Read the real label names from the dataset instead of assuming them."""
    try:
        return list(dataset["train"].features["label"].names)
    except Exception:
        return config.CLASS_NAMES


def class_counts(split) -> dict:
    """Count images per label in a split (cheap: reads labels only)."""
    return dict(sorted(Counter(split["label"]).items()))


# ------------------------------------------------------------------ splitting
def make_balanced_indices(labels, per_class: int, exclude=None, seed: int = config.SEED):
    """Pick `per_class` random indices for every class.

    Parameters
    ----------
    labels : list of int labels for the whole split
    per_class : how many images to take from each class
    exclude : set of indices that must NOT be picked (used so val != train)
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    exclude = exclude or set()
    chosen = []
    for cls in sorted(set(labels.tolist())):
        idx = [i for i in np.where(labels == cls)[0] if i not in exclude]
        if len(idx) < per_class:
            raise ValueError(
                f"Class {cls} has only {len(idx)} images available, need {per_class}."
            )
        chosen.extend(rng.choice(idx, size=per_class, replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def make_splits(dataset, train_per_class=config.TRAIN_PER_CLASS,
                val_per_class=config.VAL_PER_CLASS, seed=config.SEED):
    """Return (train_idx, val_idx) as disjoint lists of indices into dataset['train']."""
    labels = dataset["train"]["label"]
    train_idx = make_balanced_indices(labels, train_per_class, seed=seed)
    val_idx = make_balanced_indices(
        labels, val_per_class, exclude=set(train_idx), seed=seed + 1
    )
    assert not set(train_idx) & set(val_idx), "Data leakage: train/val overlap!"
    return train_idx, val_idx


# ------------------------------------------------------------------ transforms
def get_transforms(train: bool, pretrained: bool = True):
    """Build the image pipeline.

    Augmentation is deliberately mild: OCT scans have a fixed anatomical orientation,
    so no vertical flips and no large rotations.
    """
    mean, std = (
        (config.IMAGENET_MEAN, config.IMAGENET_STD)
        if pretrained
        else ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    )
    size = config.IMAGE_SIZE
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ------------------------------------------------------------------ dataset wrapper
class OCTDataset(Dataset):
    """Wraps a Hugging Face split (optionally a subset of indices) for PyTorch.

    Corrupted/unreadable images are replaced by a black image so one bad file
    never crashes a long training run (the brief asks for this).
    """

    def __init__(self, hf_split, indices=None, transform=None):
        self.hf_split = hf_split
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.hf_split[self.indices[i]]
        try:
            image = row["image"].convert("RGB")  # pretrained nets need 3 channels
        except Exception:
            image = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE))
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


def get_dataloaders(dataset, pretrained: bool = True,
                    batch_size: int = config.BATCH_SIZE,
                    train_per_class=config.TRAIN_PER_CLASS,
                    val_per_class=config.VAL_PER_CLASS):
    """Build train / val / test DataLoaders."""
    train_idx, val_idx = make_splits(dataset, train_per_class, val_per_class)

    train_ds = OCTDataset(dataset["train"], train_idx, get_transforms(True, pretrained))
    val_ds = OCTDataset(dataset["train"], val_idx, get_transforms(False, pretrained))
    test_ds = OCTDataset(dataset["test"], None, get_transforms(False, pretrained))

    kw = dict(batch_size=batch_size, num_workers=config.NUM_WORKERS,
              pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


def preprocess_pil_image(image: Image.Image, pretrained: bool = True) -> torch.Tensor:
    """Turn one PIL image into a (1, 3, H, W) tensor for inference (used by the app)."""
    return get_transforms(train=False, pretrained=pretrained)(image.convert("RGB")).unsqueeze(0)

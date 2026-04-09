"""
Custom PyTorch Dataset and 70/15/15 split for Brain Tumor MRI.
Class mapping: glioma=0, meningioma=1, notumor=2, pituitary=3.
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

import config
from transforms import get_train_transforms, get_val_test_transforms


def find_class_dirs(root: Path, class_names: List[str]) -> Tuple[Path, List[str]]:
    """Locate dataset root and which class dirs exist. Returns (data_root, present_classes)."""
    candidate = root / config.DATASET_SUBDIR
    if candidate.exists():
        root = candidate
    present = []
    for c in class_names:
        if (root / c).exists() or (root / "Training" / c).exists() or (root / "Testing" / c).exists():
            present.append(c)
    if not present:
        for name in os.listdir(root):
            d = root / name
            if d.is_dir() and name not in ("Training", "Testing"):
                for c in class_names:
                    if c.lower() == name.lower():
                        present.append(c)
                        break
    return root, present


def collect_samples(data_root: Path, class_names: List[str]) -> List[Tuple[str, int]]:
    """Return list of (file_path, label_index). Checks data_root and data_root/Training, data_root/Testing if present."""
    samples = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    roots_to_scan = [data_root]
    for sub in ("Training", "Testing"):
        p = data_root / sub
        if p.exists():
            roots_to_scan.append(p)
    for root in roots_to_scan:
        for label_idx, class_name in enumerate(class_names):
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for f in class_dir.iterdir():
                if f.suffix.lower() in exts:
                    samples.append((str(f.resolve()), label_idx))
    return samples


def get_splits(samples: List[Tuple[str, int]], train_ratio: float, val_ratio: float,
               test_ratio: float, random_state: int):
    """Split into train / val / test. Sum of ratios should be 1.0."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    train, rest = train_test_split(samples, train_size=train_ratio, random_state=random_state, stratify=[s[1] for s in samples])
    val_size = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(rest, train_size=val_size, random_state=random_state, stratify=[s[1] for s in rest])
    return train, val, test


class BrainTumorDataset(Dataset):
    """Dataset of (image_path, label). Applies transform on __getitem__."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataloaders(
    data_root: Optional[Path] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 224,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders with 70/15/15 split and consistent class mapping."""
    data_root = data_root or config.get_data_root()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}. Download the dataset first.")

    class_names = config.CLASS_NAMES
    root, present = find_class_dirs(data_root, class_names)
    if len(present) < 4:
        raise FileNotFoundError(f"Expected 4 class dirs in {root}. Found: {present}")

    samples = collect_samples(root, class_names)
    if not samples:
        raise FileNotFoundError(f"No images found under {root}")

    train_s, val_s, test_s = get_splits(
        samples,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.TEST_RATIO,
        config.RANDOM_STATE,
    )

    train_t = get_train_transforms()   # heavy augmentation (train only)
    val_t = get_val_test_transforms()  # clean pipeline (val + test)

    train_ds = BrainTumorDataset(train_s, transform=train_t)
    val_ds = BrainTumorDataset(val_s, transform=val_t)
    test_ds = BrainTumorDataset(test_s, transform=val_t)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

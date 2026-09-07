from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None
    class_names: list[str]
    class_weights: torch.Tensor


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(18),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def _class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    weights = counts.sum() / (counts.clamp_min(1) * num_classes)
    return weights


def create_data_bundle(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataBundle:
    train_tf, eval_tf = build_transforms(image_size)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_path = data_dir / "test"
    test_ds = datasets.ImageFolder(test_path, transform=eval_tf) if test_path.exists() else None

    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError("Train and validation class mappings differ")
    if test_ds is not None and train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError("Train and test class mappings differ")

    class_weights = _class_weights(train_ds.targets, len(train_ds.classes))
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    return DataBundle(
        train_loader=DataLoader(train_ds, shuffle=True, **common),
        val_loader=DataLoader(val_ds, shuffle=False, **common),
        test_loader=DataLoader(test_ds, shuffle=False, **common) if test_ds else None,
        class_names=train_ds.classes,
        class_weights=class_weights,
    )

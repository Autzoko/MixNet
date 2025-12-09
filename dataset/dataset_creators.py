import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List, Tuple
from pathlib import Path
import torchvision.transforms.functional as TF

from dataset.meta.UltrasoundSample import load_ultrasound_metadata
from dataset.augmentation.simple_augmentation import SimpleAugmentation
from dataset.augmentation.contrastive_augmentation import ContrastiveAugmentation
from dataset.dataset.ultrasound_segmentation import UltrasoundSegmentationDataset
from dataset.dataset.ultrasound_contrastive import UltrasoundContrastiveDataset

def create_segmentation_dataloaders(
        train_meta_path: str,
        val_meta_path: str,
        image_size: Tuple[int, int] = (256, 256),
        batch_size: int = 4,
        num_workers: int = 4,
        augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_samples = load_ultrasound_metadata(train_meta_path)
    val_samples = load_ultrasound_metadata(val_meta_path)

    print(f"Loaded {len(train_samples)} training samples.")
    print(f"Loaded {len(val_samples)} validation samples.")

    train_augment = SimpleAugmentation() if augment_train else None

    train_dataset = UltrasoundSegmentationDataset(
        samples=train_samples,
        image_size=image_size,
        augment=train_augment,
        normalize=True,
        return_label=True,
        return_domain=True,
    )

    val_dataset = UltrasoundSegmentationDataset(
        samples=val_samples,
        image_size=image_size,
        augment=None,
        normalize=True,
        return_label=True,
        return_domain=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader

def create_contrastive_dataloaders(
        source_meta_path: str,
        target_meta_path: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        batch_size: int = 8,
        num_workers: int = 4,
) -> Dict[str, DataLoader]:
    
    view_transform = ContrastiveAugmentation()

    source_samples = load_ultrasound_metadata(source_meta_path)
    print(f"Loaded {len(source_samples)} source samples.")

    source_base_dataset = UltrasoundSegmentationDataset(
        samples=source_samples,
        image_size=image_size,
        augment=None,
        normalize=True,
        return_label=True,
        return_domain=True,
    )
    source_contrastive_dataset = UltrasoundContrastiveDataset(
        base_dataset=source_base_dataset,
        view_transform=view_transform,
        return_label=True,
        return_domain=True,
    )

    source_loader = DataLoader(
        source_contrastive_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    loaders = {'source': source_loader}

    if target_meta_path is not None:
        target_samples = load_ultrasound_metadata(target_meta_path)
        print(f"Loaded {len(target_samples)} target samples.")

        target_base_dataset = UltrasoundSegmentationDataset(
            samples=target_samples,
            image_size=image_size,
            augment=None,
            normalize=True,
            return_label=True,
            return_domain=True,
        )
        target_contrastive_dataset = UltrasoundContrastiveDataset(
            base_dataset=target_base_dataset,
            view_transform=view_transform,
            return_label=True,
            return_domain=True,
        )

        target_loader = DataLoader(
            target_contrastive_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        loaders['target'] = target_loader

    return loaders
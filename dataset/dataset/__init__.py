import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List, Tuple
from pathlib import Path
import torchvision.transforms.functional as TF

from dataset.meta.UltrasoundSample import UltrasoundSample


class UltrasoundSegmentationDataset(Dataset):
    """
    General Ultrasound Segmentation Dataset.

    Supports:
     - Labeled/Unlabeled data
     - Flexible augmentations
     - Automatic normalization and resizing
     - Can return labels and domains
    """

    def __init__(
            self,
            samples: List[UltrasoundSample],
            image_size: Tuple[int, int] = (256, 256),
            augment: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
            normalize: bool = True,
            return_label: bool = True,
            return_domain: bool = True,
    ):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.return_label = return_label
        self.return_domain = return_domain

    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_array(self, path: str) -> np.ndarray:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix == '.npz':
            data = np.load(path)
            arr = data['arr_0'] if 'arr_0' in data else data[list(data.files)[0]]
        else:
            arr = np.load(path)

        return arr
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 1e-6:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        return img
    
    def _resize_tensor(
            self,
            tensor: torch.Tensor,
            size: Tuple[int, int],
            is_mask: bool = False
    ) -> torch.Tensor:
        mode = 'nearest' if is_mask else 'bilinear'
        align_corners = None if is_mask else False
        
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=size,
            mode=mode,
            align_corners=align_corners
        )
        tensor = tensor.squeeze(0)

        return tensor
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        
        img_np = self._load_array(sample.image_path)
        
        if self.normalize:
            img_np = self._normalize_image(img_np)

        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=0)  # (H, W) -> (C, H, W)
        elif img_np.ndim == 3:
            pass  # (C, H, W)
        else:
            raise ValueError(f"Unsupported image shape: {img_np.shape}")
        
        image = torch.from_numpy(img_np).float()

        if image.shape[1:] != self.image_size:
            image = self._resize_tensor(image, self.image_size, is_mask=False)
        
        mask = None
        if self.mask_path is not None:
            mask_np = self._load_array(sample.mask_path)
            mask_np = mask_np.astype(np.float32)

            if mask_np.ndim == 2:
                mask_np = np.expand_dims(mask_np, axis=0)  # (H, W) -> (1, H, W)

            mask = torch.from_numpy(mask_np).float()

            if mask.shape[1:] != self.image_size:
                mask = self._resize_tensor(mask, self.image_size, is_mask=True)

            mask = (mask > 0.5).float()  # Binarize mask

        if self.augment is not None:
            aug_data = {'image': image, 'mask': mask}
            aug_data = self.augment(aug_data)
            image = aug_data['image']
            mask = aug_data.get('mask', mask)

        output = {
            'id': sample.id,
            'image': image,
            'mask': mask
        }

        if self.return_label and sample.label is not None:
            output['label'] = torch.tensor(sample.label, dtype=torch.long)
        
        if self.return_domain and sample.domain is not None:
            output['domain'] = torch.tensor(sample.domain, dtype=torch.long)

        return output
            

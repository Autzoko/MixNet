from typing import List, Tuple, Dict, Any
import torch
import torch.nn.functional as F

import torchvision.transforms.functional as TF

import numpy as np


class SimpleAugmentation:
    """
    Simple Augmentation: Random horizontal flip and random rotation.
    """

    def __init__(
            self,
            flip_prob: float = 0.5,
            rotate_prob: float = 0.5,
            rotate_angles: List[int] = [0, 90, 180, 270],
            brightness_range: Tuple[float, float] = (0.8, 1.2),
            contrast_range: Tuple[float, float] = (0.8, 1.2)
    ):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_angles = rotate_angles
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = data['image']
        mask = data.get('mask', None)

        if torch.rand(1).item() < self.flip_prob:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        if torch.rand(1).item() < self.flip_prob:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)

        if torch.rand(1).item() < self.rotate_prob:
            angle = np.random.choice(self.rotate_angles)
            angle = int(angle)
            if angle != 0:
                image = TF.rotate(image, angle)
                if mask is not None:
                    mask = TF.rotate(mask, angle)
        
        brightness_factor = np.random.uniform(*self.brightness_range)
        image = TF.adjust_brightness(image, brightness_factor)

        contrast_factor = np.random.uniform(*self.contrast_range)
        image = TF.adjust_contrast(image, contrast_factor)

        data['image'] = image
        if mask is not None:
            data['mask'] = mask

        return data
    

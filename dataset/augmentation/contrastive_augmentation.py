import numpy as np
import torch
from typing import Tuple
import torchvision.transforms.functional as TF
class ContrastiveAugmentation:
    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.8,
        brightness_range: Tuple[float, float] = (0.6, 1.4),
        contrast_range: Tuple[float, float] = (0.6, 1.4),
        gamma_range: Tuple[float, float] = (0.7, 1.3),
        noise_std: float = 0.02,
    ):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.noise_std = noise_std
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.flip_prob:
            image = TF.hflip(image)
        if torch.rand(1).item() < self.flip_prob:
            image = TF.vflip(image)
        
        if torch.rand(1).item() < self.rotate_prob:
            angle = np.random.uniform(-30, 30)
            image = TF.rotate(image, angle)
        
        brightness_factor = np.random.uniform(*self.brightness_range)
        image = TF.adjust_brightness(image, brightness_factor)
        
        contrast_factor = np.random.uniform(*self.contrast_range)
        image = TF.adjust_contrast(image, contrast_factor)
        
        gamma = np.random.uniform(*self.gamma_range)
        image = TF.adjust_gamma(image, gamma)
        
        if self.noise_std > 0:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
            image = torch.clamp(image, 0, 1)
        
        return image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Callable, Dict, Any

from dataset.dataset.ultrasound_segmentation import UltrasoundSegmentationDataset


class UltrasoundContrastiveDataset(Dataset):
    """
    Ultrasound Dataset for Contrastive Learning.

    Based on given UltrasoundSegmentationDataset, each sample returns two different augmented views
    for SimCLR, SupCon, etc. contrastive learning methods.
    """

    def __init__(
            self,
            base_dataset: UltrasoundSegmentationDataset,
            view_transform: Callable[[torch.Tensor], torch.Tensor],
            return_label: bool = True,
            return_domain: bool = True,
    ):
        self.base_dataset = base_dataset
        self.view_transform = view_transform
        self.return_label = return_label
        self.return_domain = return_domain

    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        image = sample['image']

        view1 = self.view_transform(image)
        view2 = self.view_transform(image)

        output = {
            'id': sample['id'],
            'view1': view1,
            'view2': view2,
        }

        if self.return_label and 'label' in sample:
            output['label'] = sample['label']
        
        if self.return_domain and 'domain' in sample:
            output['domain'] = sample['domain']

        return output
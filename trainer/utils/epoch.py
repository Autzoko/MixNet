import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Tuple, Dict
from tqdm import tqdm

from trainer.loss.ultrasam import UltraSAMLoss


def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: UltraSAMLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        max_grad_norm: float = 1.0,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Train for one epoch with UltraSAM Loss.
    
    Args:
        model: The segmentation model
        dataloader: Training data loader
        criterion: UltraSAM loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        avg_loss: Average total loss
        avg_dice_score: Average Dice score (metric)
        loss_components: Dictionary of average loss components
    """
    model.train()

    total_loss = 0.0
    total_dice_loss = 0.0
    total_focal_loss = 0.0
    total_iou_loss = 0.0
    total_seg_loss = 0.0
    
    total_dice_score = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        images = batch['image'].to(device)  # (B, C, H, W)
        masks = batch['mask'].to(device)    # (B, 1, H, W)

        optimizer.zero_grad()
        
        # ============================================================
        # Forward pass
        # ============================================================
        outputs = model(images)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]  # (B, 1, H, W)
            
            if len(outputs) > 1 and isinstance(outputs[1], dict):
                pred_iou = outputs[1].get('iou', None)
            else:
                pred_iou = None
                
        elif isinstance(outputs, dict):
            logits = outputs['logits']
            pred_iou = outputs.get('iou', None)
        else:
            # 如果直接是 tensor
            logits = outputs
            pred_iou = None
        
        loss, loss_dict = criterion(logits, masks, pred_iou)
        
        # ============================================================
        # Backward pass
        # ============================================================
        loss.backward()
        
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()

        with torch.no_grad():
            pred_mask = torch.sigmoid(logits)
            pred_binary = (pred_mask > 0.5).float()
            
            # Compute Dice score
            intersection = (pred_binary * masks).sum()
            union = pred_binary.sum() + masks.sum()
            dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)

        total_loss += loss_dict['total']
        total_dice_loss += loss_dict['dice']
        total_focal_loss += loss_dict['focal']
        total_iou_loss += loss_dict['iou']
        total_seg_loss += loss_dict['seg']
        total_dice_score += dice_score.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss_dict["total"]:.4f}',
            'dice': f'{dice_score.item():.4f}',
            'focal': f'{loss_dict["focal"]:.4f}',
        })

    avg_loss = total_loss / num_batches
    avg_dice_score = total_dice_score / num_batches
    
    loss_components = {
        'total': avg_loss,
        'dice': total_dice_loss / num_batches,
        'focal': total_focal_loss / num_batches,
        'iou': total_iou_loss / num_batches,
        'seg': total_seg_loss / num_batches,
    }

    return avg_loss, avg_dice_score, loss_components


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: UltraSAMLoss,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Validate for one epoch with UltraSAM Loss.
    
    Args:
        model: The segmentation model
        dataloader: Validation data loader
        criterion: UltraSAM loss function
        device: Device to validate on
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average total loss
        avg_dice_score: Average Dice score (metric)
        loss_components: Dictionary of average loss components
    """
    model.eval()

    total_loss = 0.0
    total_dice_loss = 0.0
    total_focal_loss = 0.0
    total_iou_loss = 0.0
    total_seg_loss = 0.0
    
    total_dice_score = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        outputs = model(images)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
            if len(outputs) > 1 and isinstance(outputs[1], dict):
                pred_iou = outputs[1].get('iou', None)
            else:
                pred_iou = None
        elif isinstance(outputs, dict):
            logits = outputs['logits']
            pred_iou = outputs.get('iou', None)
        else:
            logits = outputs
            pred_iou = None
        
        loss, loss_dict = criterion(logits, masks, pred_iou)
        
        pred_mask = torch.sigmoid(logits)
        pred_binary = (pred_mask > 0.5).float()
        
        intersection = (pred_binary * masks).sum()
        union = pred_binary.sum() + masks.sum()
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        
        total_loss += loss_dict['total']
        total_dice_loss += loss_dict['dice']
        total_focal_loss += loss_dict['focal']
        total_iou_loss += loss_dict['iou']
        total_seg_loss += loss_dict['seg']
        total_dice_score += dice_score.item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_dict["total"]:.4f}',
            'dice': f'{dice_score.item():.4f}',
        })

    avg_loss = total_loss / num_batches
    avg_dice_score = total_dice_score / num_batches
    
    loss_components = {
        'total': avg_loss,
        'dice': total_dice_loss / num_batches,
        'focal': total_focal_loss / num_batches,
        'iou': total_iou_loss / num_batches,
        'seg': total_seg_loss / num_batches,
    }

    return avg_loss, avg_dice_score, loss_components
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Tuple
from tqdm import tqdm

from trainer.utils.dice import dice_bce_loss, dice_coefficient



def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        
        logits, aux = model(images)

        loss = dice_bce_loss(logits, masks, bce_weight=1.0, dice_weight=1.0)
        dice = dice_coefficient(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}'
        })

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches

    return avg_loss, avg_dice


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    for batch in pbar:
        images = batch['images'].to(device)
        masks = batch['mask'].to(device)

        logits, aux = model(images)

        loss = dice_bce_loss(logits, masks, bce_weight=1.0, dice_weight=1.0)
        dice = dice_coefficient(logits, masks)

        total_loss += loss.item()
        total_dice += dice.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}'
        })

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches

    return avg_loss, avg_dice

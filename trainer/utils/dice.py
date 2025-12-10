import torch
import torch.nn.functional as F

def dice_coefficient(
        logits: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)

    probs = probs.view(-1)
    targets = targets.view(-1)

    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum()

    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    dice = torch.clamp(dice, 0.0, 1.0)

    return dice

def dice_bce_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0
) -> torch.Tensor:
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_coefficient(logits, targets)
    dice_loss = 1.0 - dice

    total_loss = bce_weight * bce_loss + dice_loss

    return total_loss
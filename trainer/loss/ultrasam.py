import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6, smooth=1.0):
        super().__init__()
        self.eps = epsilon
        self.smth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        B = pred.size(0)

        pred_flat = pred.view(B, -1)
        target_flat = target.view(B, -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smth) / (union + self.smth + self.eps)

        dice_loss = 1.0 - dice.mean()

        return dice_loss
    


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        p_t = target * pred + (1 - target) * (1 - pred)

        focal_weight = (1 - p_t) ** self.gamma

        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)

        focal_loss = alpha_weight * focal_weight * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

class IoULoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def compute_iou(self, pred: torch.Tensor, target: torch.Tensor):
        B = pred.size(0)
        pred_flat = pred.view(B, -1)
        target_flat = target.view(B, -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

        iou = (intersection + self.epsilon) / (union + self.epsilon)

        return iou
    
    def forward(self, pred_iou, pred_mask, target_mask):
        gt_iou = self.compute_iou(pred_mask, target_mask)
        iou_loss = F.l1_loss(pred_iou, gt_iou)

        return iou_loss
    


class UltraSAMLoss(nn.Module):
    def __init__(
            self,
            lambda_dice=1.0,
            lambda_focal=20.0,
            lambda_iou=1.0,
            dice_epsilon=1e-6,
            dice_smooth=1.0,
            focal_alpha=0.25,
            focal_gamma=2.0,
    ):
        super().__init__()

        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.lambda_iou = lambda_iou

        self.dice_loss = SoftDiceLoss(epsilon=dice_epsilon, smooth=dice_smooth)
        self.focal_loss = SigmoidFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.iou_loss = IoULoss(epsilon=dice_epsilon)

    def forward(self, logits, mask_gt, pred_iou=None):
        pred_mask = torch.sigmoid(logits)
        dice_loss = self.dice_loss(pred_mask, mask_gt)
        focal_loss = self.focal_loss(pred_mask, mask_gt)
        seg_loss = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss

        if pred_iou is not None:
            iou_loss = self.iou_loss(pred_iou, pred_mask.detach(), mask_gt)
            total_loss = seg_loss + self.lambda_iou * iou_loss
        else:
            iou_loss = torch.tensor(0.0, device=logits.device)
            total_loss = seg_loss

        loss_dict = {
            'dice': dice_loss.item(),
            'focal': focal_loss.item(),
            'iou': iou_loss.item(),
            'seg': seg_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict



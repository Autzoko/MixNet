import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceCELoss(nn.Module):
    """
    二分类版 Dice + BCE With Logits
    对标 MONAI 的 DiceCELoss(to_onehot_y=True, softmax=True, include_background=False) 的二值情况
    """
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0, epsilon: float = 1e-6, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.eps = epsilon
        self.smooth = smooth

    def dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        probs/targets: (B, 1, H, W) ∈ [0,1]
        """
        B = probs.size(0)
        probs_flat = probs.view(B, -1)
        targets_flat = targets.view(B, -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)
        dice_loss = 1.0 - dice.mean()
        return dice_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 1, H, W) 原始 logit
        targets: (B, 1, H, W) ∈ {0,1}
        """
        probs = torch.sigmoid(logits)

        dice = self.dice_loss(probs, targets)
        ce = F.binary_cross_entropy_with_logits(logits, targets)

        loss = self.dice_weight * dice + self.ce_weight * ce
        return loss
    

class FocalTverskyLoss2D(nn.Module):
    """
    Focal Tversky Loss for binary segmentation.
    和你 BUSBRA + SegMamba 脚本里的 FocalTverskyLoss 一致，只是维度上兼容 2D/3D。
    Args:
        alpha: 控制 FN 权重（偏大则更惩罚漏检；适合小病灶）
        beta:  控制 FP 权重
        gamma: focal 指数，越大越关注难样本
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 1, H, W)
        targets: (B, 1, H, W) ∈ {0,1}
        """
        probs = torch.sigmoid(logits)

        # 展平到 (B, N)
        B = probs.size(0)
        probs_flat = probs.view(B, -1)
        targets_flat = targets.view(B, -1)

        # Tversky 组成
        TP = (probs_flat * targets_flat).sum(dim=1)
        FP = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        FN = ((1.0 - probs_flat) * targets_flat).sum(dim=1)

        tversky = (TP + self.eps) / (TP + self.alpha * FN + self.beta * FP + self.eps)

        loss = (1.0 - tversky) ** self.gamma
        return loss.mean()
    

class SegMambaLoss(nn.Module):
    """
    仿 SegMamba 的组合 loss:
      total = λ_seg * (Dice + CE) + λ_tversky * FocalTversky

    适用于二分类分割 (BUSI/BUSBRA)，
    输入 logits 形状为 (B, 1, H, W)，mask 为 (B, 1, H, W)。
    """
    def __init__(
        self,
        lambda_seg: float = 1.0,        # DiceCE 总权重
        lambda_tversky: float = 0.3,    # FocalTversky 权重（可仿照你 SegMamba 脚本 lambda_tversky=0.3）
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_tversky = lambda_tversky

        self.dicece = BinaryDiceCELoss(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            epsilon=eps,
            smooth=1.0
        )
        self.tversky = FocalTverskyLoss2D(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            eps=eps
        )

    def forward(self, logits: torch.Tensor, mask_gt: torch.Tensor):
        """
        logits: (B, 1, H, W)
        mask_gt: (B, 1, H, W) ∈ {0,1}
        """
        seg_loss = self.dicece(logits, mask_gt)
        tversky_loss = self.tversky(logits, mask_gt)

        total_loss = self.lambda_seg * seg_loss + self.lambda_tversky * tversky_loss

        loss_dict = {
            "seg": seg_loss.item(),
            "tversky": tversky_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict
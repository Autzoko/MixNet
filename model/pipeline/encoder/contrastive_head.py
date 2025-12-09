import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalContrastiveHead(nn.Module):
    """
    Global Contrastive Head for representation learning.
    Input: (B, C, H, W)
    Output: (B, D)
    """

    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)  # (B, C)
        x = self.proj(x)  # (B, D)
        return x
    

class LocalContrastiveHead(nn.Module):
    """
    Local Contrastive Head for dense representation learning.
    Input: (B, C, H, W)
    Output: (B, N_patches, D)
    """

    def __init__(self, in_dim, patch_size=8, out_dim=128):
        super().__init__()

        self.patch_size = patch_size
        self.pool = nn.AdaptiveAvgPool2d(patch_size)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        patch_size = self.patch_size

        n_h = H // patch_size
        n_w = W // patch_size

        if n_h * patch_size != H or n_w * patch_size != W:
            pad_h = (patch_size - H % patch_size) % patch_size
            pad_w = (patch_size - W % patch_size) % patch_size

            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, H, W = x.shape
            n_h = H // patch_size
            n_w = W // patch_size

        x = x.view(B, C, n_h, patch_size, n_w, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, n_h * n_w, C, patch_size * patch_size)
        x = x.mean(-1)  # (B, N_patches, C)

        x = self.proj(x)  # (B, N_patches, D)

        return x
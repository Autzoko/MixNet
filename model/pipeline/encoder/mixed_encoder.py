import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Optional
import math

from model.pipeline.utils.mamba import MambaBlock, MambaLayer
from model.pipeline.utils.attention import WindowAttentionBlock
from model.pipeline.utils.norm import LayerNorm2d
from model.pipeline.utils.conv import ConvStem, RDCNNBlock
from model.pipeline.utils.gsc import GSC
from model.pipeline.utils.mlp import MlpChannel

from model.pipeline.encoder.contrastive_head import GlobalContrastiveHead, LocalContrastiveHead

class MTBlock(nn.Module):
    """
    Mamba + Transformer (Window Attention) Mixed Encoder Block.
    Input: (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, dim, window_size=8, num_heads=8, state_dim=16, use_mamba=True, use_attn=False, mlp_ratio: float = 2.0):
        super().__init__()

        self.use_attn = use_attn

        self.gsc = GSC(dim)
        self.mamba_layer = MambaLayer(dim, state_dim=state_dim, use_mamba=use_mamba)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MlpChannel(dim, mlp_hidden)
        self.mlp_norm = nn.InstanceNorm2d(dim, affine=True)
        
        if use_attn:
            self.window_attn = WindowAttentionBlock(dim, window_size=window_size, num_heads=num_heads)

    def forward(self, x):
        x = self.gsc(x)

        x = self.mamba_layer(x)

        identity = x

        x = self.mlp_norm(x)
        x = self.mlp(x) + identity

        if self.use_attn:
            x = self.window_attn(x)

            
        return x

    

class DownSampleLayer(nn.Module):
    """
    Downsampling Layer: 2x downsampling + Channel doubling.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = LayerNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.conv(x))
    

class HybridEncoder(nn.Module):
    """
    Hybrid Encoder: CNN + Mamba + Window Attention.
    """

    def __init__(
            self,
            in_ch: int = 1,
            base_ch: int = 32,
            depths: List[int] = [2, 2, 3],
            window_sizes: List[int] = [8, 8, 16],
            num_heads: int = 8,
            state_dim: int = 16,
            contrast_dim: int = 128,
            enable_local_contrast: bool = False,
            local_patch_size: int = 8
    ):
        super().__init__()

        self.enable_local_contrast = enable_local_contrast

        self.stem = ConvStem(in_ch=in_ch, base_ch=base_ch)
        
        # STAGE 1
        self.stage1_blocks = nn.ModuleList([
            RDCNNBlock(base_ch) for _ in range(depths[0])
        ])
        self.down1 = DownSampleLayer(base_ch, base_ch * 2)

        # STAGE 2
        stage2_dim = base_ch * 2
        self.stage2_blocks = nn.ModuleList([
            MTBlock(stage2_dim, window_size=window_sizes[1], num_heads=num_heads, state_dim=state_dim)
            for _ in range(depths[1])
        ])
        self.down2 = DownSampleLayer(stage2_dim, base_ch * 4)

        # STAGE 3
        stage3_dim = base_ch * 4
        self.stage3_blocks = nn.ModuleList([
            MTBlock(stage3_dim, window_size=window_sizes[2], num_heads=num_heads, state_dim=state_dim)
            for _ in range(depths[2])
        ])

        self.global_head = GlobalContrastiveHead(
            in_dim=stage3_dim,
            hidden_dim=512,
            out_dim=contrast_dim
        )

        if enable_local_contrast:
            self.local_head = LocalContrastiveHead(
                in_dim=stage3_dim,
                patch_size=local_patch_size,
                out_dim=contrast_dim
            )
        else:
            self.local_head = None

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Feed forward function.
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_ch, H, W)

        Returns:
            features: multi-scale features [f1, f2, f3]
                - f1: (B, base_ch, H, W) - Stage 1 output
                - f2: (B, base_ch*2, H/2, W/2) - Stage 2 output
                - f3: (B, base_ch*4, H/4, W/4) - Stage 3 output
            global_embed: global contrastive learning embedding (B, D)
            local_embed: local contrastive learning embedding (B, N_patches, D) or None
        """

        features = []

        # STEM
        x = self.stem(x)

        # STAGE 1
        for block in self.stage1_blocks:
            x = block(x)
        f1 = x
        features.append(f1)

        x = self.down1(x)

        # STAGE 2
        for block in self.stage2_blocks:
            x = block(x)
        f2 = x
        features.append(f2)

        x = self.down2(x)

        # STAGE 3
        for block in self.stage3_blocks:
            x = block(x)
        f3 = x
        features.append(f3)

        # Global Contrastive Head
        global_embed = self.global_head(f3)

        local_embed = None
        if self.enable_local_contrast and self.local_head is not None:
            local_embed = self.local_head(f3)

        return features, global_embed, local_embed



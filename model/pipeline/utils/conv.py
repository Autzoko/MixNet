import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Optional
import math

from model.pipeline.utils.norm import LayerNorm2d

class ConvStem(nn.Module):
    """
    Input convolutional stem for processing image data.
    Input: (B, in_ch, H, W)
    Output: (B, base_ch, H, W)
    """

    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = LayerNorm2d(base_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
class RDCNNBlock(nn.Module):
    """
    Residual Dilated CNN Block, for high-resolution feature extraction.
    Input: (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, channels):
        super().__init__()

        # First conv block
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = LayerNorm2d(channels)
        self.act1 = nn.GELU()

        # Deeper dilated conv
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = LayerNorm2d(channels)
        self.act2 = nn.GELU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.dwconv(out)
        out = self.pwconv(out)
        out = self.norm2(out)
        out = self.act2(out)

        return out + identity

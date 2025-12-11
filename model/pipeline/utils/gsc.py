import torch
import torch.nn as nn
import torch.nn.functional as F

class GSC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.act = nn.ReLU(inplace=True)

        self.proj2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(in_channels, affine=True)
        self.act2 = nn.ReLU(inplace=True)

        self.proj3 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm2d(in_channels, affine=True)
        self.act3 = nn.ReLU(inplace=True)

        self.proj4 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm2d(in_channels, affine=True)
        self.act4 = nn.ReLU(inplace=True)


    def forward(self, x: torch.Tensor):
        identity = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.act(x1)

        x1 = self.proj(x1)
        x1 = self.norm2(x1)
        x1 = self.act2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.act3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.act4(x)

        return x + identity
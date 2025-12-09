import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock2d(nn.Module):
    """
    Standard Convolutional Block:
    Conv -> Norm -> GELU -> Conv -> Norm -> GELU + Residual

    Args:
        - in_ch: Input channels
        - out_ch: Output channels
        - norm_type: Normalization type ('batch' or 'layer')
        _ num_groups: Number of groups for GroupNorm (if used)
    """

    def __init__(self, in_ch, out_ch, norm_type='batch', num_groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        
        if norm_type == 'batch':
            self.norm1 = nn.BatchNorm2d(out_ch)
            self.norm2 = nn.BatchNorm2d(out_ch)
        elif norm_type == 'group':
            self.norm1 = nn.GroupNorm(num_groups, out_ch)
            self.norm2 = nn.GroupNorm(num_groups, out_ch)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}. Use 'batch' or 'group'.")
        
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.act2 = nn.GELU()

        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """
        Input: (B, in_ch, H, W)
        Output: (B, out_ch, H, W)
        """
        identity = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act2(out)

        return out

class UpBlock2d(nn.Module):
    """
    Upsamplepling Block:
    Upsample + Skip Connection + Feature Fusion
    """
    def __init__(
            self,
            in_ch_low,
            in_ch_skip,
            out_ch,
            mode="bilinear",
            align_corners=False,
            norm_type='batch',
    ):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners if mode == 'bilinear' else None

        self.conv_block = ConvBlock2d(
            in_ch_low + in_ch_skip,
            out_ch,
            norm_type=norm_type
        )

    def forward(self, low_res, skip):
        """
        Args:
            - low_res: (B, in_ch_low, H_low, W_low) - deeper layer feature
            - skip: (B, in_ch_skip, H_skip, W_skip) - skip connection feature
        Returns:
            - out: (B, out_ch, H_skip, W_skip) - fused feature
        """

        _, _, H_skip, W_skip = skip.shape

        if self.mode == 'bilinear':
            up = F.interpolate(
                low_res,
                size=(H_skip, W_skip),
                mode=self.mode,
                align_corners=self.align_corners
            )
        else:
            up = F.interpolate(
                low_res,
                size=(H_skip, W_skip),
                mode=self.mode
            )

        x = torch.cat([up, skip], dim=1)  # (B, in_ch_low + in_ch_skip, H_skip, W_skip)

        out = self.conv_block(x)  # (B, out_ch, H_skip, W_skip)

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from model.pipeline.utils.unet import ConvBlock2d, UpBlock2d

class UNetDecoder2D(nn.Module):
    """
    U-Net Decoder for 2D data.
    """
    def __init__(
            self,
            enc_channels: Tuple[int, int, int] = (32, 64, 128),
            dec_channels: Tuple[int, int, int] = (128, 64, 32),
            num_classes: int = 1,
            up_mode: str = 'bilinear',
            align_corners: bool = False,
            norm_type: str = 'batch',
            use_bottleneck: bool = True,
    ):
        super().__init__()

        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.num_classes = num_classes
        self.use_bottleneck = use_bottleneck

        C1, C2, C3 = enc_channels
        D1, D2, D3 = dec_channels

        if use_bottleneck:
            self.bottleneck = ConvBlock2d(C3, D1, norm_type=norm_type)
        else:
            assert D1 == C3, "When use_bottleneck=False, dec_channels[0] must equal enc_channels[2]"
            self.bottleneck = nn.Identity()

        self.up1 = UpBlock2d(
            in_ch_low=D1,
            in_ch_skip=C2,
            out_ch=D2,
            mode=up_mode,
            align_corners=align_corners,
            norm_type=norm_type
        )

        self.up2 = UpBlock2d(
            in_ch_low=D2,
            in_ch_skip=C1,
            out_ch=D3,
            mode=up_mode,
            align_corners=align_corners,
            norm_type=norm_type
        )

        self.seg_head = nn.Conv2d(D3, num_classes, kernel_size=1)

    def forward(
            self,
            features: List[torch.Tensor],
            return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Feed forward function.
        Args:
            features: List of feature maps from the encoder [feat1, feat2, feat3]
                      where feat1 has the highest resolution.
                      - features[0]: (B, C1, H, W)
                        - features[1]: (B, C2, H/2, W/2)
                        - features[2]: (B, C3, H/4, W/4)
            return_intermediate: If True, also return intermediate feature maps.

        Returns:
            - if return_intermediate is False:
                logits: (B, num_classes, H, W) - final segmentation map
            - if return_intermediate is True:
                logits: (B, num_classes, H, W) - final segmentation map
                intermediates: dict, containing 'x3', 'x2', 'x1' feature maps
        """

        f1, f2, f3 = features  #f1: (B,C1,H,W), f2: (B,C2,H/2,W/2), f3: (B,C3,H/4,W/4)

        x3 = self.bottleneck(f3)  # (B,D1,H/4,W/4)
        x2 = self.up1(x3, f2)     # (B,D2,H/2,W/2)
        x1 = self.up2(x2, f1)     # (B,D3,H,W)

        logits = self.seg_head(x1)  # (B,num_classes,H,W)

        if return_intermediate:
            intermediates = {
                'x3': x3,
                'x2': x2,
                'x1': x1
            }
            return logits, intermediates
        else:
            return logits

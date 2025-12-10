import torch
import torch.nn as nn

from model.pipeline.utils.conv import ConvStem, RDCNNBlock
from model.pipeline.encoder.mixed_encoder import DownSampleLayer

class PlainCNNEncoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, depths=[2, 2, 3]):
        super().__init__()
        self.stem = ConvStem(in_ch=in_ch, base_ch=base_ch)
        self.stage1 = nn.Sequential(*[RDCNNBlock(base_ch) for _ in range(depths[0])])
        self.down1 = DownSampleLayer(base_ch, base_ch * 2)
        self.stage2 = nn.Sequential(*[RDCNNBlock(base_ch * 2) for _ in range(depths[1])])
        self.down2 = DownSampleLayer(base_ch * 2, base_ch * 4)
        self.stage3 = nn.Sequential(*[RDCNNBlock(base_ch * 4) for _ in range(depths[2])])

    def forward(self, x):
        feats = []
        x = self.stem(x)
        x = self.stage1(x); f1 = x; feats.append(f1)
        x = self.down1(x)
        x = self.stage2(x); f2 = x; feats.append(f2)
        x = self.down2(x)
        x = self.stage3(x); f3 = x; feats.append(f3)
        global_repr = None
        local_repr = None
        return feats, global_repr, local_repr
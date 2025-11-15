import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import DeformableConv2d, ResidualBlock, SpatialAttention, ChannelAttention, MultiConv

# ---------------------------
#  Feature encoders
# ---------------------------

class PanFeatureEncoder(nn.Module):
    """
    Top branch: SPOT Pan 1.5 m → feature maps (n channels)
    DeformableConv → BN → ReLU → 2×Residual → SpatialAttention
    """
    def __init__(self, in_ch=1, base_ch=64, n_resblocks=2):
        super().__init__()
        self.head = nn.Sequential(
            DeformableConv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.resblocks = nn.Sequential(
            *[ResidualBlock(base_ch) for _ in range(n_resblocks)]
        )
        self.spatial_att = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.head(x)
        x = self.resblocks(x)
        x = self.spatial_att(x)
        return x


class S2Spot10mEncoder(nn.Module):
    """
    Bottom branch shared encoder for S2 (Query) and SPOT@10 m (Key)
    DeformableConv → BN → ReLU → MultiConv → ChannelAttention
    """
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        self.head = nn.Sequential(
            DeformableConv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.body = MultiConv(base_ch, n_layers=3)
        self.chan_att = ChannelAttention(base_ch)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.chan_att(x)
        return x



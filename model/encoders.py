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

# ---------------------------
#  Cross-attention (Q from S2, K from SPOT@10m)
# ---------------------------

class CrossAttentionMap(nn.Module):
    """
    Compute a single-channel attention map from S2 and SPOT 10m features.

    - Q, K: (B, C, H, W)
    - scaled dot-product per pixel, then softmax over spatial positions.
      (Keeps the "scaled dot-product + softmax" spirit while yielding
       a 1×H×W attention map.)
    """
    def __init__(self, temperature: Optional[float] = None):
        super().__init__()
        self.temperature = temperature  # if None → sqrt(C) computed on the fly

    def forward(self, q, k):
        b, c, h, w = q.shape
        temp = self.temperature or math.sqrt(c)

        # elementwise dot across channels → similarity map
        sim = (q * k).sum(dim=1, keepdim=True) / temp  # (B,1,H,W)

        # softmax over spatial positions
        sim_flat = sim.view(b, 1, -1)
        att_flat = F.softmax(sim_flat, dim=-1)
        att = att_flat.view(b, 1, h, w)  # (B,1,H,W)

        return att  # "attention map" in your diagram



# ---------------------------
#  Super-resolution head
# ---------------------------

class SRHead(nn.Module):
    """
    DeformableConv → BN → ReLU → 2×Residual → PixelShuffle×scale → Conv(out_ch)
    """
    def __init__(self, in_ch, base_ch=64, out_ch=4, scale=4):
        super().__init__()
        self.pre = nn.Sequential(
            DeformableConv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualBlock(base_ch),
            ResidualBlock(base_ch),
        )

        # one PixelShuffle with r=scale
        self.upconv = nn.Conv2d(base_ch, base_ch * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pre(x)
        x = self.body(x)
        x = self.upconv(x)
        x = self.pixel_shuffle(x)
        x = self.final(x)
        return x

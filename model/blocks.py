import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
#  Basic building blocks
# ---------------------------

class DeformableConv2d(nn.Module):
    """
    Placeholder for deformable conv.
    Swap with mmcv.ops.DeformConv2d or similar if you like.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size, padding=padding),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size, padding=padding),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class SpatialAttention(nn.Module):
    """
    Simple spatial attention: conv over [avg_pool, max_pool] across channels.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * att


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    """
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.mlp(y).view(b, c, 1, 1)
        return x * y


class MultiConv(nn.Module):
    """
    Simple 'multi conv' block: stack of Conv+BN+ReLU layers.
    """
    def __init__(self, ch, n_layers=3, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        layers = []
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(ch, ch, kernel_size, padding=padding),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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

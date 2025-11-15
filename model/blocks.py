import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

# ---------------------------
#  Basic building blocks
# ---------------------------


class DeformableConv2d(nn.Module):
    """Thin wrapper around :func:`torchvision.ops.deform_conv2d`.

    The module learns both the offsets (and optional modulation mask) and the
    convolution weights so that it can be used as a drop-in replacement for a
    standard ``nn.Conv2d``.  ``torchvision`` ships the CUDA/C++ optimized
    kernels, so this wrapper only needs to prepare the inputs for the op.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        deformable_groups: int = 1,
        bias: bool = True,
        modulation: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size: Tuple[int, int] = kernel_size
        self.stride: Tuple[int, int] = stride
        self.padding: Tuple[int, int] = padding
        self.dilation: Tuple[int, int] = dilation
        self.deformable_groups = deformable_groups
        self.modulation = modulation

        offset_channels = 2 * deformable_groups * kernel_size[0] * kernel_size[1]
        self.offset_conv = nn.Conv2d(
            in_ch,
            offset_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        if modulation:
            mask_channels = deformable_groups * kernel_size[0] * kernel_size[1]
            self.mask_conv = nn.Conv2d(
                in_ch,
                mask_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            )
        else:
            self.mask_conv = None

        self.weight = nn.Parameter(
            torch.empty(out_ch, in_ch, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        if self.mask_conv is not None:
            nn.init.constant_(self.mask_conv.weight, 0.0)
            nn.init.constant_(self.mask_conv.bias, 0.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = None
        if self.mask_conv is not None:
            mask = torch.sigmoid(self.mask_conv(x))
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


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
    DeformableConv → BN → ReLU → 2×Residual → (optional PixelShuffle) → Conv(out_ch)
    """

    def __init__(self, in_ch, base_ch=64, out_ch=4, scale=1):
        super().__init__()
        self.scale = scale

        self.pre = nn.Sequential(
            DeformableConv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualBlock(base_ch),
            ResidualBlock(base_ch),
        )

        if scale > 1:
            self.upconv = nn.Conv2d(
                base_ch, base_ch * (scale**2), kernel_size=3, padding=1
            )
            self.pixel_shuffle = nn.PixelShuffle(scale)
        else:
            self.upconv = nn.Identity()
            self.pixel_shuffle = nn.Identity()

        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pre(x)
        x = self.body(x)
        x = self.upconv(x)
        x = self.pixel_shuffle(x)
        x = self.final(x)
        return x


# ---------------------------
#  Cross-attention (Q from S2, K from SPOT@10m)
# ---------------------------

class CrossAttentionMap(nn.Module):
    """
    Transformer-style cross-attention producing a 1-channel attention map.

    Inputs:
        q, k: (B, C, H, W)

    Steps:
        1. Q = Wq(q), K = Wk(k), V = Wv(k)
        2. Flatten spatial dims → (B, HW, C)
        3. Multi-head scaled dot-product attention
        4. Project output back to (B, C, H, W)
        5. Collapse to 1-channel via a final conv (learnable)
        6. Softmax over spatial dims → attention map (B,1,H,W)
    """
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # Collapse C → 1
        self.to_map = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, q, k):
        B, C, H, W = q.shape

        # ---- Q, K, V projections ----
        Q = self.q_proj(q)  # (B, C, H, W)
        K = self.k_proj(k)
        V = self.v_proj(k)

        # ---- reshape to heads ----
        # (B, heads, H*W, head_dim)
        def reshape_heads(x):
            x = x.view(B, self.num_heads, self.head_dim, H * W)
            return x.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)

        Qh = reshape_heads(Q)
        Kh = reshape_heads(K)
        Vh = reshape_heads(V)

        # ---- scaled dot-product ----
        # attn = softmax( Q @ K^T / sqrt(d) )
        attn = (Qh @ Kh.transpose(-2, -1)) * self.scale  # (B, heads, HW, HW)
        attn = attn.softmax(dim=-1)

        # ---- output: (B, heads, HW, head_dim) ----
        out = attn @ Vh

        # ---- merge heads ----
        out = out.permute(0, 1, 3, 2).contiguous()  # (B, heads, head_dim, HW)
        out = out.view(B, C, H * W)
        out = out.view(B, C, H, W)

        # ---- output projection ----
        out = self.out_proj(out)  # (B, C, H, W)

        # ---- collapse C → 1 via conv ----
        att_map = self.to_map(out)  # (B,1,H,W)

        # ---- spatial softmax ----
        att_map = att_map.view(B, 1, -1)
        att_map = att_map.softmax(dim=-1)
        att_map = att_map.view(B, 1, H, W)

        return att_map



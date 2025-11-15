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
    """
    A lightweight PyTorch module wrapping ``torchvision.ops.deform_conv2d`` to provide
    learnable deformable 2D convolutions with optional modulation.

    This layer predicts spatial offsets (and optionally modulation masks) from the input
    feature map and applies them to a learned convolution kernel, enabling spatially
    adaptive receptive fields. It can be used as a drop-in replacement for ``nn.Conv2d``.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    kernel_size : int or tuple of int, default=3
        Size of the convolution kernel.
    stride : int or tuple of int, default=1
        Convolution stride.
    padding : int or tuple of int, default=1
        Zero-padding applied to the input.
    dilation : int or tuple of int, default=1
        Dilation factor for the convolution kernel.
    deformable_groups : int, default=1
        Number of independent deformable kernel groups.
    bias : bool, default=True
        Whether to include a learnable bias term.
    modulation : bool, default=True
        If True, predicts a per-location modulation mask in addition to offsets.

    Notes
    -----
    - Offsets are predicted via a 2D convolution producing
      ``2 * deformable_groups * kernel_h * kernel_w`` channels.
    - If modulation is enabled, an additional convolution predicts a mask in ``[0, 1]``.
    - ``torchvision`` provides optimized CUDA/C++ kernels; this wrapper prepares
      offset/mask tensors and parameters.
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
        """
        Initialize convolution weights, offsets, and optional modulation masks.

        - Kernel weights are initialized with Kaiming uniform initialization.
        - Offset and mask prediction layers are zero-initialized so the module
          initially behaves like a standard convolution.
        - If a bias term is present, it is initialized uniformly according to the
          fan-in of the kernel.
        """

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
        """
        Apply deformable convolution to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, C_out, H_out, W_out)`` after applying
            deformable convolution using learned offsets (and optional modulation).

        Notes
        -----
        - Offsets are generated from ``offset_conv(x)``.
        - If modulation is enabled, a mask in ``[0, 1]`` is generated via ``sigmoid``.
        - The underlying operation delegates to ``torchvision.ops.deform_conv2d``.
        """
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
    """
    A standard residual block consisting of two Conv–BN layers with an intermediate
    ReLU activation, wrapped in a skip connection.

    This block preserves the spatial resolution and channel count of the input and
    is commonly used to improve gradient flow and stabilize deeper architectures.

    Parameters
    ----------
    ch : int
        Number of input and output channels.
    kernel_size : int, default=3
        Size of the convolution kernels. Padding is chosen to preserve spatial size.

    Returns
    -------
    torch.Tensor
        Output tensor with the same shape as the input, computed as
        ``x + F(x)`` where ``F`` is the stacked Conv–BN–ReLU–Conv–BN subnetwork.
    """

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
    Spatial attention module that reweights feature maps based on aggregated
    channel-wise statistics.

    This block computes both the average-pooled and max-pooled spatial descriptors
    across channels, concatenates them, and feeds them through a small convolution
    to produce a spatial attention mask in the range ``[0, 1]``. The mask is then
    applied multiplicatively to the input, enhancing informative spatial regions
    while suppressing less relevant ones.

    Parameters
    ----------
    kernel_size : int, default=7
        Kernel size of the convolution used to generate the attention mask.
        Padding is set automatically to preserve spatial resolution.

    Forward Inputs
    --------------
    x : torch.Tensor
        Input tensor of shape ``(B, C, H, W)``.

    Forward Outputs
    ---------------
    torch.Tensor
        Output tensor of the same shape as the input, computed as
        ``x * attention_mask``.

    Notes
    -----
    - The attention mask is derived from two complementary descriptors:
      mean-pooled and max-pooled activations over channels.
    - This module refines spatial saliency without changing tensor dimensionality.
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
    Channel attention module following the Squeeze-and-Excitation (SE) mechanism.

    This block adaptively recalibrates channel-wise feature responses by explicitly
    modeling inter-channel dependencies. It performs global average pooling to
    produce a compact channel descriptor, processes it through a small MLP, and
    outputs a set of per-channel weights in ``[0, 1]``. These weights modulate the
    original feature map, amplifying informative channels and suppressing noisy ones.

    Parameters
    ----------
    ch : int
        Number of input and output channels.
    reduction : int, default=16
        Reduction ratio used in the bottleneck of the MLP. Controls the capacity
        and computational cost of the attention module.

    Forward Inputs
    --------------
    x : torch.Tensor
        Input tensor of shape ``(B, C, H, W)``.

    Forward Outputs
    ---------------
    torch.Tensor
        Output tensor of the same shape as the input, reweighted by learned
        channel-wise importance coefficients.

    Notes
    -----
    - Global average pooling produces a ``(B, C)`` descriptor.
    - The MLP applies a bottleneck transformation and outputs ``C`` sigmoid-activated
      weights.
    - Broadcasting the resulting weights over spatial dimensions yields a modulation
      mask that scales each channel independently.
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
    A lightweight convolutional feature extractor composed of repeated
    ``Conv2d → BatchNorm2d → ReLU`` layers.

    This block allows stacking multiple identical convolutional layers to deepen
    feature processing while preserving spatial resolution and channel count.
    It is commonly used to add local nonlinear capacity without changing tensor
    dimensionality.

    Parameters
    ----------
    ch : int
        Number of input and output channels for all convolution layers.
    n_layers : int, default=3
        Number of repeated convolutional sub-layers.
    kernel_size : int, default=3
        Convolution kernel size for each layer. Padding is set to preserve spatial size.

    Forward Inputs
    --------------
    x : torch.Tensor
        Input tensor of shape ``(B, C, H, W)``.

    Forward Outputs
    ---------------
    torch.Tensor
        Output tensor with identical shape to the input, after applying all
        stacked convolutional layers.

    Notes
    -----
    - All intermediate layers preserve spatial dimensions due to symmetric padding.
    - Increasing ``n_layers`` increases the receptive field and nonlinear expressiveness.
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
    Super-resolution reconstruction head combining deformable convolution,
    residual refinement, and optional pixel-shuffle upsampling.

    This module takes fused high-resolution features and produces the final
    super-resolved output. It first applies a deformable convolution to align
    and normalize feature geometry, refines the representation using two
    residual blocks, optionally upsamples the feature map using a
    pixel-shuffle block, and finally maps the features into the desired number
    of output channels.

    Parameters
    ----------
    in_ch : int
        Number of channels in the fused input features.
    base_ch : int, default=64
        Number of feature channels used within the SR head.
    out_ch : int, default=4
        Number of output channels (e.g., target Sentinel-2 bands).
    scale : int, default=1
        Spatial upsampling factor. If ``scale > 1``, a pixel-shuffle upsampling
        stage is applied; otherwise the block preserves spatial resolution.

    Notes
    -----
    - If ``scale > 1``, the head performs learned upsampling via
      ``Conv2d → PixelShuffle(scale)``.
    - If ``scale == 1``, the module acts as a pure refinement head without
      changing spatial resolution.
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
        """
        Forward pass of the super-resolution head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of fused features with shape ``(B, in_ch, H, W)``.

        Returns
        -------
        torch.Tensor
            Super-resolved output tensor of shape
            ``(B, out_ch, H * scale, W * scale)``.
            If ``scale == 1``, the spatial resolution is preserved.

        Workflow
        --------
        1. **DeformableConv → BN → ReLU**
           Align and normalize fused features.
        2. **ResidualBlock × 2**
           Local refinement and nonlinear enhancement.
        3. **Upsampling (optional)**
           If ``scale > 1``, expand channels and apply PixelShuffle.
        4. **Final 3×3 Conv**
           Project refined features into target spectral channels.
        """
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
    Multi-head transformer-style cross-attention module that produces a
    single-channel spatial attention map.

    This block takes two feature maps (query and key/value) with identical
    spatial and channel dimensions, projects them into Q/K/V embeddings, and
    computes multi-head scaled dot-product attention over the flattened spatial
    domain. The attended features are projected back to the original channel
    dimension, collapsed to a single channel via a learnable 1×1 convolution,
    and normalized with a spatial softmax to yield an attention map in which
    all locations sum to 1 per sample.

    Parameters
    ----------
    dim : int, default=64
        Channel dimension of the input feature maps and internal Q/K/V
        projections.
    num_heads : int, default=4
        Number of attention heads. Must evenly divide ``dim``.

    Notes
    -----
    - Inputs are expected in ``(B, C, H, W)`` format, with ``C == dim``.
    - Attention is computed over the flattened spatial dimension ``HW``,
      enabling long-range dependencies across the entire feature map.
    - The final output is a spatial probability distribution per sample,
      suitable for weighing or masking high-resolution features downstream.
    """

    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Linear projections for Q, K, V
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # Collapse C → 1
        self.to_map = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, q, k):
        """
        Compute a 1-channel spatial attention map from query and key/value features.

        Parameters
        ----------
        q : torch.Tensor
            Query feature map of shape ``(B, C, H, W)``, typically derived from
            the Sentinel-2 branch.
        k : torch.Tensor
            Key/value feature map of shape ``(B, C, H, W)``, typically derived
            from the reference (e.g., PAN) branch.

        Returns
        -------
        torch.Tensor
            Attention map of shape ``(B, 1, H, W)``, where each ``H×W`` map is a
            spatial probability distribution (values in ``[0, 1]`` that sum to 1
            per sample).

        Notes
        -----
        - Q, K, and V are obtained via learned 1×1 convolutions.
        - Multi-head attention is applied over the flattened spatial dimension
          with standard scaled dot-product attention.
        - The attended features are merged across heads, projected back to
          ``dim`` channels, collapsed to a single channel via ``to_map``, and
          normalized with a spatial softmax.
        """
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

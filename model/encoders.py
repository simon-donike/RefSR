import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import (
    DeformableConv2d,
    ResidualBlock,
    SpatialAttention,
    ChannelAttention,
    MultiConv,
)

# ---------------------------
#  Feature encoders
# ---------------------------


class PanFeatureEncoder(nn.Module):
    """
    High-resolution PAN feature encoder used in the top branch of the network.

    This encoder processes high-resolution SPOT panchromatic imagery and extracts
    rich spatial features suitable for fusion with Sentinel-2 inputs. It applies a
    deformable convolution for geometric flexibility, followed by residual
    refinement and spatial attention to emphasize salient structures (e.g., edges,
    textures, fine details).

    Parameters
    ----------
    in_ch : int, default=1
        Number of input channels (typically 1 for PAN imagery).
    base_ch : int, default=64
        Number of feature channels produced by the encoder.
    n_resblocks : int, default=2
        Number of residual refinement blocks applied after the initial convolution.

    Notes
    -----
    - The deformable convolution expands the ability to adapt to local geometry
      misalignments between modalities.
    - Residual blocks increase nonlinear representational capacity without altering
      spatial size.
    - Spatial attention enhances local regions with strong structural cues, helping
      the model focus on informative PAN features.
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
        """
        Forward pass of the PAN feature encoder.

        Parameters
        ----------
        x : torch.Tensor
            High-resolution PAN tensor of shape ``(B, in_ch, H, W)``.

        Returns
        -------
        torch.Tensor
            Refined PAN feature map of shape ``(B, base_ch, H, W)``,
            enriched with spatial attention weighting.
        """
        x = self.head(x)
        x = self.resblocks(x)
        x = self.spatial_att(x)
        return x


class S2Spot10mEncoder(nn.Module):
    """
    Shared 10 m-resolution encoder used for both Sentinel-2 (query) features and
    downsampled SPOT PAN (key) features.

    This module extracts aligned mid-resolution representations from both modalities,
    enabling consistent feature spaces for cross-attention. The encoder applies a
    deformable convolution for geometric flexibility, followed by a stack of
    convolutional layers for local feature extraction, and a channel attention module
    to emphasize spectrally relevant channels.

    Parameters
    ----------
    in_ch : int
        Number of input channels (e.g., 4 for RGB-NIR Sentinel-2, 1 for PAN-LR).
    base_ch : int, default=64
        Number of feature channels used throughout the encoder.

    Notes
    -----
    - The deformable convolution helps compensate for sub-pixel misalignments
      between S2 and SPOT grids.
    - ``MultiConv`` expands the local receptive field while preserving resolution.
    - ``ChannelAttention`` adaptively reweights spectral channels based on
      learned global context.
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
        """
        Forward pass of the shared 10 m encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, in_ch, H, W)`` at 10 m resolution.

        Returns
        -------
        torch.Tensor
            Feature map of shape ``(B, base_ch, H, W)``, aligned for use as
            query or key/value inputs to the cross-attention module.
        """
        x = self.head(x)
        x = self.body(x)
        x = self.chan_att(x)
        return x

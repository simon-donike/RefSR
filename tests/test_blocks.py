import pytest
import torch

from model.blocks import (
    ChannelAttention,
    CrossAttentionMap,
    DeformableConv2d,
    MultiConv,
    ResidualBlock,
    SpatialAttention,
    SRHead,
)


def test_deformable_conv2d_preserves_shape():
    torch.manual_seed(0)
    module = DeformableConv2d(3, 8, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 16, 16)
    out = module(x)
    assert out.shape == (2, 8, 16, 16)


def test_residual_block_skip_connection():
    block = ResidualBlock(16)
    x = torch.randn(1, 16, 8, 8)
    y = block(x)
    assert y.shape == x.shape
    assert torch.allclose(y - x, block.block(x), atol=1e-6)


def test_spatial_attention_modulates_features_but_preserves_shape():
    att = SpatialAttention(kernel_size=7)
    x = torch.randn(2, 4, 10, 10, requires_grad=True)
    out = att(x)

    # same spatial/feature shape
    assert out.shape == x.shape

    # attention should not amplify magnitude if mask ∈ [0,1]
    # (allow tiny numerical slack)
    assert torch.all(torch.abs(out) <= torch.abs(x) + 1e-6)

    # sanity check: it's actually doing something, not identity
    assert not torch.allclose(out, x)


def test_channel_attention_rescales_channels():
    att = ChannelAttention(32)
    x = torch.randn(2, 32, 6, 6)
    out = att(x)
    assert out.shape == x.shape


def test_multi_conv_stack():
    module = MultiConv(12, n_layers=2)
    x = torch.randn(3, 12, 5, 5)
    out = module(x)
    assert out.shape == x.shape


def test_sr_head_with_upsample():
    head = SRHead(in_ch=65, base_ch=32, out_ch=4, scale=2)
    x = torch.randn(1, 65, 8, 8)
    out = head(x)
    assert out.shape == (1, 4, 16, 16)


def test_cross_attention_map_properties():
    module = CrossAttentionMap(dim=32, num_heads=4)
    q = torch.randn(2, 32, 4, 4)
    k = torch.randn(2, 32, 4, 4)
    att_map = module(q, k)
    assert att_map.shape == (2, 1, 4, 4)
    flattened = att_map.view(2, -1)
    sums = flattened.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert torch.all((att_map >= 0) & (att_map <= 1))
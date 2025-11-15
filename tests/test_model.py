import torch

from model.model import PanS2FusionSR


def test_model_forward_shapes():
    # Use the same base_ch as the model's attention dimension (64)
    model = PanS2FusionSR(s2_in_ch=4, pan_in_ch=1, base_ch=64, out_ch=4)

    # LR S2 10 m, HR PAN at 2× (for this small test)
    s2_lr = torch.randn(2, 4, 32, 32)
    pan_hr = torch.randn(2, 1, 64, 64)

    sr, att = model(s2_lr, pan_hr)

    # Check that forward runs and shapes are consistent
    assert sr.shape == (2, 4, 64, 64)  # SR on HR grid
    assert att.shape == (2, 1, 32, 32)  # attention at LR (10 m) resolution


def test_model_forward_with_explicit_pan_lr():
    # Use base_ch consistent with the cross-attention dim (64)
    model = PanS2FusionSR(
        s2_in_ch=4,
        pan_in_ch=1,
        base_ch=64,
        out_ch=3,
    )

    s2_lr = torch.randn(1, 4, 16, 16)
    pan_hr = torch.randn(1, 1, 32, 32)
    pan_lr = torch.randn(1, 1, 16, 16)

    sr, att = model(s2_lr, pan_hr, pan_lr)

    # SR on the HR grid, with 3 output channels
    assert sr.shape == (1, 3, 32, 32)
    # Attention map at LR (10 m) resolution
    assert att.shape == (1, 1, 16, 16)

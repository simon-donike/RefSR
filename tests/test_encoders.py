import torch

from model.encoders import PanFeatureEncoder, S2Spot10mEncoder


def test_pan_feature_encoder_output_shape():
    encoder = PanFeatureEncoder(in_ch=1, base_ch=16, n_resblocks=1)
    x = torch.randn(2, 1, 32, 32)
    out = encoder(x)
    assert out.shape == (2, 16, 32, 32)


def test_s2_spot_encoder_output_shape():
    encoder = S2Spot10mEncoder(in_ch=4, base_ch=32)
    x = torch.randn(1, 4, 24, 24)
    out = encoder(x)
    assert out.shape == (1, 32, 24, 24)

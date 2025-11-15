import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# import model blocks
from model.encoders import PanFeatureEncoder, S2Spot10mEncoder
from model.blocks import CrossAttentionMap, SRHead


class PanS2FusionSR(nn.Module):
    """
    Reference-guided super-resolution model that fuses Sentinel-2 10 m inputs with
    high-resolution PAN features.

    Core functionality
    ------------------
    - Encodes HR PAN features and 10 m S2/PAN features separately.
    - Computes a transformer-style cross-attention map between S2 (query) and
      PAN-10 m (key/value) features.
    - Upsamples the attention map to the HR grid and fuses it with HR PAN features.
    - Reconstructs super-resolved Sentinel-2 outputs through the SR head.

    Parameters
    ----------
    s2_in_ch : int
        Number of input Sentinel-2 channels (e.g., 4 for RGB-NIR).
    pan_in_ch : int
        Number of PAN channels (typically 1).
    base_ch : int
        Feature dimension used throughout encoders and attention.
    sr_scale : int
        Logical LR→HR scaling factor (grid relationship only).
    out_ch : int
        Number of output SR channels.

    Forward Inputs
    --------------
    s2_lr : torch.Tensor
        Sentinel-2 LR tensor of shape ``(B, C_s2, H, W)``.
    spot_pan_hr : torch.Tensor
        HR PAN tensor of shape ``(B, 1, H_hr, W_hr)``.
    spot_pan_lr : torch.Tensor, optional
        10 m PAN tensor of shape ``(B, 1, H, W)``. If omitted, downsampled from HR.

    Returns
    -------
    s2_sr : torch.Tensor
        Super-resolved Sentinel-2 output, shape ``(B, out_ch, H_hr, W_hr)``.
    att_map_10m : torch.Tensor
        10 m attention map used for fusion, shape ``(B, 1, H, W)``.
    """

    def __init__(
        self,
        s2_in_ch: int = 4,  # e.g. RGBNIR
        pan_in_ch: int = 1,  # panchromatic
        base_ch: int = 64,
        sr_scale: int = 4,
        out_ch: int = 4,
    ):
        super().__init__()

        # HR PAN encoder (top branch)
        self.pan_encoder_hr = PanFeatureEncoder(
            in_ch=pan_in_ch,
            base_ch=base_ch,
        )

        # separate encoders for S2 and SPOT@10m (bottom branch)
        self.s2_encoder_10m = S2Spot10mEncoder(
            in_ch=s2_in_ch,  # for now: 4-band RFB-NIR
            base_ch=base_ch,
        )
        self.spot_encoder_10m = S2Spot10mEncoder(
            in_ch=pan_in_ch,  # 1 channel PAN
            base_ch=base_ch,
        )

        # Cross-attention
        self.cross_att = CrossAttentionMap()

        # SR head: concat [pan_hr_feats, att_map_up]
        self.sr_head = SRHead(
            in_ch=base_ch + 1,
            base_ch=base_ch,
            out_ch=out_ch,
            scale=1,  # pan_hr already at target 2.5 m grid
        )

    def forward(
        self,
        s2_lr: torch.Tensor,
        spot_pan_hr: torch.Tensor,
        spot_pan_lr: Optional[torch.Tensor] = None,
    ):
        b, _, h, w = s2_lr.shape

        # if no explicit 10 m PAN given, downsample HR pan
        if spot_pan_lr is None:
            spot_pan_lr = F.interpolate(
                spot_pan_hr,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        # --- encoders ---
        pan_hr_feats = self.pan_encoder_hr(spot_pan_hr)  # (B,C,H_hr,W_hr)
        q_s2 = self.s2_encoder_10m(s2_lr)  # (B,C,H,W)
        k_spot = self.spot_encoder_10m(spot_pan_lr)  # (B,C,H,W)

        # --- cross attention ---
        att_map_10m = self.cross_att(q_s2, k_spot)  # (B,1,H,W)

        # upsample att map to HR resolution
        att_map_hr = F.interpolate(
            att_map_10m,
            size=pan_hr_feats.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # --- fusion + SR head ---
        fused = torch.cat([pan_hr_feats, att_map_hr], dim=1)  # (B,C+1,H_hr,W_hr)
        s2_sr = self.sr_head(fused)  # (B,out_ch,H*scale,W*scale)

        return s2_sr, att_map_10m


if __name__ == "__main__":
    # simple test
    model = PanS2FusionSR(
        s2_in_ch=4,  # e.g. RGBNIR
        pan_in_ch=1,
        base_ch=64,
        sr_scale=4,
        out_ch=4,
    )

    s2 = torch.randn(2, 4, 64, 64)  # 10 m
    pan_hr = torch.randn(2, 1, 256, 256)  # 2.5 m grid (4×)
    out, att = model(s2, pan_hr)

    # -----------------------------------------------------------
    # Sanity Check
    # -----------------------------------------------------------
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    total_params = count_params(model)
    pan_encoder_params = count_params(model.pan_encoder_hr)
    s2_encoder_params = count_params(model.s2_encoder_10m)
    spot_encoder_params = count_params(model.spot_encoder_10m)
    sr_head_params = count_params(model.sr_head)
    cross_att_params = count_params(model.cross_att)

    print("\n=== PanS2FusionSR Sanity Check ===")
    print(f" Input S2 LR:         {tuple(s2.shape)}")
    print(f" Input PAN HR:        {tuple(pan_hr.shape)}")
    print("----------------------------------------------")
    print(f" Output SR S2:        {tuple(out.shape)}")
    print(f" Attention map (10m): {tuple(att.shape)}")
    print("==============================================\n")

    print("=== Parameter Summary (Trainable) ===")
    print(f" Total parameters:               {total_params:,}")
    print("----------------------------------------------")
    print(f" PAN HR encoder:                 {pan_encoder_params:,}")
    print(f" S2 10m encoder:                 {s2_encoder_params:,}")
    print(f" SPOT 10m encoder:               {spot_encoder_params:,}")
    print(f" Cross-attention module:         {cross_att_params:,}")
    print(f" Super-resolution head:          {sr_head_params:,}")
    print("==============================================\n")

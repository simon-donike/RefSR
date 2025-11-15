import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# import model blocks
from model.encoders import PanFeatureEncoder, S2Spot10mEncoder, CrossAttentionMap, SRHead


class PanS2FusionSR(nn.Module):
    """
    Inputs:
      - s2_lr:       (B, C_s2, H, W)        Sentinel-2 10 m (Query)
      - spot_pan_hr: (B, 1, H_hr, W_hr)     SPOT Pan 1.5 m
      - spot_pan_lr: optional (B, 1, H, W)  SPOT Pan downsampled to 10 m (Key)
    """

    def __init__(
        self,
        s2_in_ch: int = 4,   # e.g. RGBNIR
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
            in_ch=s2_in_ch, # for now: 4-band RFB-NIR
            base_ch=base_ch,
        )
        self.spot_encoder_10m = S2Spot10mEncoder(
            in_ch=pan_in_ch, # 1 channel PAN
            base_ch=base_ch,
        )

        # Cross-attention
        self.cross_att = CrossAttentionMap()

        # SR head: concat [pan_hr_feats, att_map_up]
        self.sr_head = SRHead(
            in_ch=base_ch + 1,
            base_ch=base_ch,
            out_ch=out_ch,
            scale=1,   # pan_hr already at target 2.5 m grid
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
        pan_hr_feats = self.pan_encoder_hr(spot_pan_hr)          # (B,C,H_hr,W_hr)
        q_s2 = self.s2_encoder_10m(s2_lr)                        # (B,C,H,W)
        k_spot = self.spot_encoder_10m(spot_pan_lr)              # (B,C,H,W)

        # --- cross attention ---
        att_map_10m = self.cross_att(q_s2, k_spot)               # (B,1,H,W)

        # upsample att map to HR resolution
        att_map_hr = F.interpolate(
            att_map_10m,
            size=pan_hr_feats.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # --- fusion + SR head ---
        fused = torch.cat([pan_hr_feats, att_map_hr], dim=1)     # (B,C+1,H_hr,W_hr)
        s2_sr = self.sr_head(fused)                              # (B,out_ch,H*scale,W*scale)

        return s2_sr, att_map_10m



if __name__ == "__main__":
    # simple test
    model = PanS2FusionSR(
        s2_in_ch=4,   # e.g. RGBNIR
        pan_in_ch=1,
        base_ch=64,
        sr_scale=4,
        out_ch=4,
    )

    s2 = torch.randn(2, 4, 64, 64)         # 10 m
    pan_hr = torch.randn(2, 1, 256, 256)   # 2.5 m grid (4×)
    out, att = model(s2, pan_hr)
    print(out.shape, att.shape)
    # Should be → torch.Size([2, 4, 256, 256])  torch.Size([2, 1, 64, 64])
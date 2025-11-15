import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# import model blocks
from model.encoders import PanFeatureEncoder, S2Spot10mEncoder, CrossAttentionMap, SRHead


# ---------------------------
#  Full model
# ---------------------------

class PanS2FusionSR(nn.Module):
    """
    Inputs:
      - s2_lr:    (B, C_s2, H, W)        Sentinel-2 10 m (Query)
      - spot_pan_hr: (B, 1, H_hr, W_hr)  SPOT Pan 1.5 m
      - spot_pan_lr (optional): (B, 1, H, W)  SPOT Pan downsampled to 10 m (Key)
        If not given, we downsample spot_pan_hr to match S2 resolution.

    Output:
      - s2_sr: (B, C_out, H*scale, W*scale)  S2 SR @ 2.5 m
    """

    def __init__(
        self,
        s2_in_ch: int = 4,      # e.g. RGBNIR
        pan_in_ch: int = 1,
        base_ch: int = 64,
        sr_scale: int = 4,
        out_ch: int = 4,        # SR S2 bands you want to predict
    ):
        super().__init__()
        # Encoders
        self.pan_encoder_hr = PanFeatureEncoder(in_ch=pan_in_ch, base_ch=base_ch)
        self.s2_spot10_encoder = S2Spot10mEncoder(in_ch=s2_in_ch, base_ch=base_ch)

        # Cross-attention
        self.cross_att = CrossAttentionMap()

        # SR head: concatenated [pan_features_hr, attention_map_up]
        self.sr_head = SRHead(in_ch=base_ch + 1, base_ch=base_ch, out_ch=out_ch, scale=sr_scale)

    def forward(self, s2_lr, spot_pan_hr, spot_pan_lr: Optional[torch.Tensor] = None):
        """
        s2_lr:      B x C_s2 x H x W
        spot_pan_hr: B x 1 x H_hr x W_hr
        spot_pan_lr: optional B x 1 x H x W
        """
        # --- resolutions ---
        b, _, h, w = s2_lr.shape

        if spot_pan_lr is None:
            # downsample HR pan to S2 resolution for Key
            spot_pan_lr = F.interpolate(
                spot_pan_hr, size=(h, w), mode="bilinear", align_corners=False
            )

        # ---------- Encoders ----------
        # HR pan → features (top branch)
        pan_hr_feats = self.pan_encoder_hr(spot_pan_hr)               # (B, C, H_hr, W_hr)

        # 10 m S2 and SPOT→ Query & Key features (bottom branch)
        q_s2 = self.s2_spot10_encoder(s2_lr)            # (B, C, H, W)  Query
        k_spot = self.s2_spot10_encoder(spot_pan_lr)    # (B, C, H, W)  Key

        # ---------- Cross-attention ----------
        att_map_10m = self.cross_att(q_s2, k_spot)      # (B, 1, H, W)

        # upsample attention map to HR resolution to match pan_hr_feats
        att_map_hr = F.interpolate(
            att_map_10m, size=pan_hr_feats.shape[-2:], mode="bilinear", align_corners=False
        )                                              # (B,1,H_hr,W_hr)

        # ---------- Fusion + SR head ----------
        fused = torch.cat([pan_hr_feats, att_map_hr], dim=1)  # (B, C+1, H_hr, W_hr)
        s2_sr = self.sr_head(fused)                           # (B, out_ch, H*scale, W*scale)

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
    # → torch.Size([2, 4, 256, 256])  torch.Size([2, 1, 64, 64])
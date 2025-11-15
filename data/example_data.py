import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class RandomPanS2Dataset(Dataset):
    """
    Minimal dummy dataset for testing the PanS2FusionSR model.
    Produces random tensors with correct shapes.
    """

    def __init__(
        self,
        n_samples=1000,
        s2_channels=4,  # RGBNIR by default
        pan_channels=1,
        lr_size=64,  # 10 m patch size
        scale=4,  # HR PAN resolution is x4
    ):
        super().__init__()
        self.n_samples = n_samples
        self.s2_channels = s2_channels
        self.pan_channels = pan_channels
        self.lr_size = lr_size
        self.hr_size = lr_size * scale
        self.scale = scale

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # S2 LR:  B x C_s2 x H x W
        s2_lr = torch.randn(self.s2_channels, self.lr_size, self.lr_size)

        # SPOT Pan HR: B x 1 x H_hr x W_hr
        spot_pan_hr = torch.randn(self.pan_channels, self.hr_size, self.hr_size)

        # SPOT Pan LR (optional input to model)
        spot_pan_lr = torch.randn(self.pan_channels, self.lr_size, self.lr_size)

        sample = {
            "s2_lr": s2_lr,
            "spot_pan_hr": spot_pan_hr,
            "spot_pan_lr": spot_pan_lr,
            "index": idx,  # optional, useful for debugging
        }
        return sample

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


# ------------------------------------------------------
# Lightning DataModule
# ------------------------------------------------------


class RandomPanS2DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        dcfg = self.cfg.data
        full = RandomPanS2Dataset(
            n_samples=dcfg.n_samples,
            s2_channels=dcfg.s2_channels,
            pan_channels=dcfg.pan_channels,
            lr_size=dcfg.lr_size,
            scale=dcfg.scale,
        )

        val_size = int(dcfg.val_split * dcfg.n_samples)
        train_size = dcfg.n_samples - val_size

        self.train_dataset, self.val_dataset = random_split(
            full,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        dcfg = self.cfg.data
        return DataLoader(
            self.train_dataset,
            batch_size=dcfg.batch_size,
            shuffle=True,
            num_workers=dcfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        dcfg = self.cfg.data
        return DataLoader(
            self.val_dataset,
            batch_size=dcfg.batch_size,
            shuffle=False,
            num_workers=dcfg.num_workers,
            pin_memory=True,
        )



# ------------------------------------------------------
# Test
# ------------------------------------------------------
if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/example_config.yaml")
    dm = RandomPanS2DataModule(config)

    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    print("S2 LR:", batch["s2_lr"].shape)
    print("SPOT HR:", batch["spot_pan_hr"].shape)
    print("SPOT LR:", batch["spot_pan_lr"].shape)

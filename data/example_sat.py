import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import rasterio

class SyntheticPanS2FromTIFF(Dataset):
    """
    Synthetic Pan+S2 dataset built from real 4-band GeoTIFFs.

    Inputs:
      - s2_lr      (4 x LR x LR)
      - spot_pan_hr (1 x HR x HR)
      - spot_pan_lr (1 x LR x LR)

    Target:
      - s2_hr      (4 x HR x HR)  <-- ground truth
    """

    def __init__(
        self,
        root,
        n_samples=1000,
        pattern="*.tif",
        s2_channels=4,
        pan_channels=1,
        lr_size=64,
        scale=4,
    ):
        super().__init__()
        self.filepaths = sorted(glob.glob(os.path.join(root, pattern)))
        if len(self.filepaths) == 0:
            raise RuntimeError(f"No TIFFs found in {root}")

        self.n_samples = n_samples
        self.s2_channels = s2_channels
        self.pan_channels = pan_channels
        self.lr_size = lr_size
        self.hr_size = lr_size * scale
        self.scale = scale

    def _read_random_hr_patch(self, path):
        with rasterio.open(path) as src:
            img = src.read(out_dtype="float32")   # (C,H,W)

        C, H, W = img.shape
        if C < self.s2_channels:
            raise RuntimeError(f"{path} has {C} bands, need {self.s2_channels}")

        if H < self.hr_size or W < self.hr_size:
            raise RuntimeError(f"{path} too small for hr_size={self.hr_size}")

        # Random crop
        y0 = np.random.randint(0, H - self.hr_size + 1)
        x0 = np.random.randint(0, W - self.hr_size + 1)
        hr = img[:self.s2_channels, y0:y0+self.hr_size, x0:x0+self.hr_size]

        return torch.from_numpy(hr)   # (4, HR, HR)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        path = self.filepaths[idx % len(self.filepaths)]

        # -----------------------
        # Ground truth (HR S2)
        # -----------------------
        s2_hr = self._read_random_hr_patch(path)          # (4, HR, HR)
        s2_hr = s2_hr / 255.0                            # Scale to [0,1]
        s2_hr = torch.clamp(s2_hr, 0.0, 1.0)

        # -----------------------
        # Synthetic LR S2 (10 m)
        # -----------------------
        s2_lr = F.interpolate(
            s2_hr.unsqueeze(0),
            size=(self.lr_size, self.lr_size),
            mode="bicubic",
            align_corners=False,
        )[0]

        # -----------------------
        # Synthetic PAN data
        # -----------------------
        pan_hr = s2_hr.mean(dim=0, keepdim=True)          # (1, HR, HR)

        pan_lr = F.interpolate(
            pan_hr.unsqueeze(0),
            size=(self.lr_size, self.lr_size),
            mode="bicubic",
            align_corners=False,
        )[0]

        return {
            "s2_lr": s2_lr,
            "spot_pan_lr": pan_lr,
            "spot_pan_hr": pan_hr,
            "s2_hr": s2_hr,        # <-- GROUND TRUTH TARGET
            "index": idx,
            "source_path": path,
        }

# ------------------------------------------------------
# Lightning DataModule
# ------------------------------------------------------


class SyntheticPanS2DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        dcfg = self.cfg.data
        full = SyntheticPanS2FromTIFF(root="/data2/simon/austria_buildings/hr_orthofoto")

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


if __name__ == "__main__":
    from omegaconf import OmegaConf
    conf = OmegaConf.load("config/example_config.yaml")

    ds_path = "/data2/simon/austria_buildings/hr_orthofoto"
    dm = SyntheticPanS2DataModule(conf)
    dm.setup()

    from tqdm import tqdm
    for i in tqdm(dm.train_dataloader()):
        pass
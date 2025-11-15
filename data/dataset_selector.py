import pytorch_lightning as pl
from torch.utils.data import DataLoader

def select_dataset(cfg):
    if cfg.data.type == "SyntheticPanS2FromTIFF":
        from data.example_sat import SyntheticPanS2FromTIFF as ds_class
    elif cfg.data.type == "RandomPanS2Dataset":
        from data.example_data import RandomPanS2Dataset as ds_class
    else:
        raise ValueError(f"Unknown dataset type: {cfg.data.type}")

    ds_train = ds_class(cfg, phase="train")
    ds_val = ds_class(cfg, phase="val")

    dm = dataset_to_datamodule(ds_train, ds_val, cfg)
    print("[INFO] Creating Dataset of type:", cfg.data.type)
    print("[INFO] Train samples:", len(ds_train), " - Val samples:", len(ds_val))
    return dm


def dataset_to_datamodule(ds_train,ds_val,cfg):
    class CustomDataModule(pl.LightningDataModule):
        def __init__(self, ds_train, ds_val, cfg):
            super().__init__()
            self.ds_train = ds_train
            self.ds_val = ds_val
            self.batch_size = cfg.data.batch_size
            self.num_workers = cfg.data.num_workers

        def train_dataloader(self):
            return DataLoader(
                self.ds_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        def val_dataloader(self):
            return DataLoader(
                self.ds_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

    return CustomDataModule(ds_train, ds_val, cfg)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("config/example_config.yaml")
    dm = select_dataset(config)

    from tqdm import tqdm
    for i in tqdm(dm.train_dataloader()):
        pass
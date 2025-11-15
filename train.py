# train.py

import pytorch_lightning as pl
from omegaconf import OmegaConf

# Set visible GPUs
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# -------------------------------------------------------
# Main
# -------------------------------------------------------


def main():
    # Get Config
    cfg = OmegaConf.load("config/example_config.yaml")

    # Get DataModule
    from data.dataset_selector import select_dataset
    datamodule = select_dataset(cfg)

    # Get Model
    from model.sr_model import PanS2System

    model = PanS2System(cfg)

    tcfg = cfg.trainer
    trainer = pl.Trainer(
        max_epochs=tcfg.max_epochs,
        val_check_interval=tcfg.val_check_interval,
        accelerator=tcfg.accelerator,
        devices=tcfg.devices,
        log_every_n_steps=tcfg.log_every_n_steps,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

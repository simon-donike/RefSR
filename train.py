# train.py

import pytorch_lightning as pl
from omegaconf import OmegaConf


# -------------------------------------------------------
# Main
# -------------------------------------------------------


def main():
    # Get Config
    cfg = OmegaConf.load("config/example_config.yaml")

    # Get Data
    if cfg.data.type == "SyntheticPanS2DataModule":
        from data.example_sat import SyntheticPanS2DataModule

        datamodule = SyntheticPanS2DataModule(cfg)
    elif cfg.data.type == "RandomPanS2DataModule":
        from data.example_data import RandomPanS2DataModule

        datamodule = RandomPanS2DataModule(cfg)
    else:
        raise ValueError(f"Unknown data type: {cfg.data.type}")

    # Get Model
    from model.sr_model import PanS2System

    model = PanS2System(cfg)

    tcfg = cfg.trainer
    trainer = pl.Trainer(
        max_epochs=tcfg.max_epochs,
        accelerator=tcfg.accelerator,
        devices=tcfg.devices,
        log_every_n_steps=tcfg.log_every_n_steps,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

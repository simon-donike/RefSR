# train.py

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf

# local imports
from model.model import PanS2FusionSR
from data.example_data import RandomPanS2DataModule

# -------------------------------------------------------
# Lightning Module
# -------------------------------------------------------

class PanS2System(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        mcfg = cfg.model
        self.model = PanS2FusionSR(
            s2_in_ch=mcfg.s2_in_ch,
            pan_in_ch=mcfg.pan_in_ch,
            base_ch=mcfg.base_ch,
            sr_scale=mcfg.sr_scale,
            out_ch=mcfg.out_ch,
        )

        self.loss_cfg = cfg.loss
        self.opt_cfg = cfg.optimizer
        self.sched_cfg = cfg.scheduler

    def compute_loss(self, batch):
        sr_pred, _ = self.model(
            batch["s2_lr"], batch["spot_pan_hr"], batch["spot_pan_lr"]
        )

        # simple pseudo-target
        target = F.interpolate(
            batch["s2_lr"],
            size=sr_pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if self.loss_cfg.type == "l1":
            return F.l1_loss(sr_pred, target, reduction=self.loss_cfg.reduction)
        elif self.loss_cfg.type == "l2":
            return F.mse_loss(sr_pred, target, reduction=self.loss_cfg.reduction)
        else:
            raise ValueError(f"Unknown loss: {self.loss_cfg.type}")

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        ocfg = self.opt_cfg

        if ocfg.name.lower() == "adam":
            opt = torch.optim.Adam(
                self.parameters(),
                lr=ocfg.lr,
                betas=tuple(ocfg.betas),
                eps=ocfg.eps,
                weight_decay=ocfg.weight_decay,
            )
        elif ocfg.name.lower() == "adamw":
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=ocfg.lr,
                betas=tuple(ocfg.betas),
                eps=ocfg.eps,
                weight_decay=ocfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer {ocfg.name}")

        # optional scheduler
        if self.sched_cfg.enable:
            if self.sched_cfg.name == "step_lr":
                sched = torch.optim.lr_scheduler.StepLR(
                    opt,
                    step_size=self.sched_cfg.step_size,
                    gamma=self.sched_cfg.gamma,
                )
            else:
                raise ValueError(f"Unknown scheduler {self.sched_cfg.name}")

            return {"optimizer": opt, "lr_scheduler": sched}

        return opt


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    cfg = OmegaConf.load("config/example_config.yaml")
    datamodule = RandomPanS2DataModule(cfg)
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

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pathlib import Path


# local imports
from model.model import PanS2FusionSR

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
        target = batch["s2_hr"]
        
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

    def on_after_backward(self):
        # save imgs
        val_loader = self.trainer.datamodule.val_dataloader()
        val_batch = next(iter(val_loader))
        sr_pred, _ = self.model(
            val_batch["s2_lr"],
            val_batch["spot_pan_hr"],
            val_batch["spot_pan_lr"],
        )
        target = F.interpolate(
            val_batch["s2_lr"],
            size=sr_pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        self.create_visualization(val_batch, sr_pred, target)

    def create_visualization(self, batch, sr_pred, target):
        if plt is None:
            self.print("Matplotlib is not installed; skipping visualization export.")
            return

        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

        idx = 0
        lr = batch["s2_lr"][idx]
        hr = target[idx]
        pan = batch["spot_pan_hr"][idx]
        sr = sr_pred[idx]

        def _to_image(tensor, grayscale=False):
            tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
            if grayscale or tensor.shape[0] == 1:
                return tensor[0].numpy()
            if tensor.shape[0] >= 3:
                return tensor[:3].permute(1, 2, 0).numpy()
            return tensor.permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        images = [
            (axes[0], _to_image(lr), "S2 LR"),
            (axes[1], _to_image(hr), "S2 HR / Target"),
            (axes[2], _to_image(pan, grayscale=True), "PAN Reference"),
            (axes[3], _to_image(sr), "SR Prediction"),
        ]

        for ax, img, title in images:
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(f"Validation Epoch {self.current_epoch}")
        out_path = logs_dir / f"val_epoch_{self.current_epoch:04d}_sample.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        self.print(f"Saved validation visualization to {out_path}")

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

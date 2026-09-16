from .WeatherInterpModel import (
    WeatherEncoder,
    WeatherDecoder,
    FiLM,
    HermiteLatentInterpolator,
    HermiteParamNetFiLM,
    WeatherHermiteModel,
    WeatherUNetBaselineModel,
    WeatherUNetResidualLinearModel,
    WeatherUNetDirectModel,
)

# Legacy Lightning wrapper is kept as-is below for compatibility.

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch Lightning is not installed. Install with: pip install lightning"
        ) from exc


class WeatherTimeInterpModule(pl.LightningModule):
    def __init__(
        self,
        latent_channels: int = 64,
        cond_dim: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_latent: float = 0.1,
        lambda_reg_d: float = 0.0,
        use_latent_loss: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = WeatherHermiteModel(
            latent_channels=latent_channels,
            cond_dim=cond_dim,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_latent = lambda_latent
        self.lambda_reg_d = lambda_reg_d
        self.use_latent_loss = use_latent_loss

    def forward(self, x0, xT, tau, cond):
        return self.model(x0, xT, tau, cond)

    def _shared_step(self, batch, stage: str):
        x0 = batch["x0"]
        xT = batch["xT"]
        xt = batch["xt"]
        tau = batch["tau"].float()
        cond = batch["cond"].float()

        x_hat, aux = self(x0, xT, tau, cond)
        loss_recon = F.mse_loss(x_hat, xt)

        loss = loss_recon
        log_dict = {f"{stage}/loss_recon": loss_recon}

        if self.use_latent_loss and self.lambda_latent > 0.0:
            z_t = self.model.encoder(xt[:, :, 1:])
            z_tau = aux["z_tau"]
            loss_latent = F.mse_loss(z_tau, z_t)
            loss = loss + self.lambda_latent * loss_latent
            log_dict[f"{stage}/loss_latent"] = loss_latent

        if self.lambda_reg_d > 0.0 and aux.get("d0") is not None and aux.get("d1") is not None:
            d0 = aux["d0"]
            d1 = aux["d1"]
            loss_reg_d = (d0.pow(2).mean() + d1.pow(2).mean()) * 0.5
            loss = loss + self.lambda_reg_d * loss_reg_d
            log_dict[f"{stage}/loss_reg_d"] = loss_reg_d

        log_dict[f"{stage}/loss"] = loss
        self.log_dict(log_dict, prog_bar=(stage == "train"), on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }

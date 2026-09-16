#!/usr/bin/env python3
"""Phase 3: Latent VFI module (deterministic predictor) on top of frozen Phase 2.

Architecture:
    x_0, x_T (input fields)
        │
        ▼ frozen encoder (from Phase 2.ae)
    z_0, z_T (latents)
        │
        ▼ trainable Latent VFI: (z_0, z_T, τ) → z_τ_pred
        │   - small UNet/MLP-Mixer over latent
        │   - τ-conditioning via AdaLN-Zero
        ▼
    z_τ_pred
        │
        ▼ frozen Phase 2 decoder + static_proj + AdaLN + scale (from Phase 2)
    x_pred = bilinear(x_0, x_T, τ) + scale · decoder(z_τ_pred + static)

Loss: L2 on z (latent target = encoder(x_target)) + λ · L1 on x (physical recon)

Logging (paper-grade):
    train/latent_l2, train/x_l1, train/x_mse_model, train/x_mse_bilinear, train/improve_vs_bil
    val/* (same)
    val_rmse_h{1,2,3,4,5}/<channel>  ← per-τ per-channel (135 metrics)
    val_rmse_h{1..5}/surface_mean
    val_rmse_h{1..5}/avg_relative_improvement
    val_rmse/aggregate_surface_mean, val_rmse/aggregate_relative_improvement

Cosine LR + warmup with clamp (no oscillation if extended).
"""
import os, sys, argparse, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
torch.set_float32_matmul_precision('high')

PHASE2_TRAINER = os.environ.get('PHASE2_TRAINER', '/tmp/train_phase2_adaln_zero.py')
_ns = {}
exec(open(PHASE2_TRAINER).read().replace("if __name__ == '__main__':", "if False:"), _ns)
Phase2AdaLNZero = _ns['Phase2AdaLNZero']
CH_27 = _ns['CH_27']

sys.path.insert(0, '/home/jovyan/dsuhoi/weather_time_interpolation')
sys.path.insert(0, '/workspace/code/wti')
from weather_time_interp.memmap_dataset import ERA5MemmapDataset


class LatentVFINet(nn.Module):
    """Small UNet on latent: (z_0, z_T, τ) → z_τ_pred.

    Input: concat(z_0, z_T) → (B, 2C, h, w)
    AdaLN-Zero modulation by τ throughout.
    Output: z_τ_pred with same shape as z_0.
    """
    def __init__(self, latent_ch=64, hidden=128, n_blocks=4, time_dim=128):
        super().__init__()
        self.latent_ch = latent_ch
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, hidden * 3 * n_blocks),                     # γ, β, gate per block
        )
        nn.init.normal_(self.time_mlp[-1].weight, std=0.01)                     # γ,β non-zero
        nn.init.zeros_(self.time_mlp[-1].bias)
        with torch.no_grad():
            for b in range(n_blocks):                                           # zero only gate slice per block
                offset = b * hidden * 3
                self.time_mlp[-1].weight[offset + 2 * hidden : offset + 3 * hidden].zero_()

        # Input proj: concat(z_0, z_T) = 2C → hidden
        self.in_proj = nn.Conv2d(2 * latent_ch, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        ) for _ in range(n_blocks)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden) for _ in range(n_blocks)])
        # Output proj: hidden → C
        self.out_proj = nn.Conv2d(hidden, latent_ch, kernel_size=1)
        # Init out_proj small so initial output ~ near-zero residual (gentle start)
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)

        self.n_blocks = n_blocks
        self.hidden = hidden

    def time_embed(self, tau):
        half = self.time_dim // 2
        device = tau.device
        freqs = torch.exp(torch.linspace(0, -math.log(10000), half, device=device))
        args = tau.view(-1, 1) * freqs.view(1, -1)
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, z_0, z_T, tau):
        # Initial guess: latent linear interpolation (residual target = z_τ - this)
        if tau.dim() == 1:
            tau_b = tau.view(-1, 1, 1, 1)
        else:
            tau_b = tau.view(-1, 1, 1, 1)
        z_lin = (1.0 - tau_b) * z_0 + tau_b * z_T                               # baseline z

        # Build feature
        h = torch.cat([z_0, z_T], dim=1)
        h = self.in_proj(h)

        t_emb = self.time_embed(tau.view(-1).float())
        mod = self.time_mlp(t_emb)                                              # (B, hidden*3*n_blocks)
        per_block = mod.view(mod.size(0), self.n_blocks, 3, self.hidden)

        for i, (block, norm) in enumerate(zip(self.blocks, self.norms)):
            γ, β, gate = per_block[:, i, 0], per_block[:, i, 1], per_block[:, i, 2]
            γ = γ.view(-1, self.hidden, 1, 1)
            β = β.view(-1, self.hidden, 1, 1)
            gate = gate.view(-1, self.hidden, 1, 1)
            h_norm = norm(h)
            h_mod = (1 + γ) * h_norm + β
            h_out = block(h_mod)
            h = h + gate * h_out                                                # gated residual

        delta_z = self.out_proj(h)
        return z_lin + delta_z                                                  # z_τ_pred = z_lin + learned residual


class Phase3LatentVFI(pl.LightningModule):
    def __init__(self, phase2_ckpt, in_channels=27,
                 hidden=128, n_blocks=4, time_dim=128,
                 lr=1e-4, weight_decay=1e-4,
                 lambda_x=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.lambda_x = lambda_x

        # Load frozen Phase 2 model (which contains frozen Phase 1 ae + trainable decoder/scale/tau_proj)
        self.phase2 = Phase2AdaLNZero.load_from_checkpoint(phase2_ckpt, map_location='cpu')
        for p in self.phase2.parameters():
            p.requires_grad = False
        self.phase2.eval()

        latent_ch = self.phase2.ae.hparams.latent_channels
        self.vfi = LatentVFINet(latent_ch=latent_ch, hidden=hidden,
                                n_blocks=n_blocks, time_dim=time_dim)

    def _encode(self, x):
        with torch.no_grad():
            x_c = self.phase2.ae._crop_lat(x)
            x_c = self.phase2.ae._pad_lon(x_c)
            return self.phase2.ae.encoder(x_c)

    def _decode_residual(self, z, tau, target_h, target_w):
        with torch.no_grad():
            return self.phase2._decode(z, tau, target_h, target_w)

    def _step(self, batch, prefix):
        x_0 = batch['x0']; x_T = batch['x1']
        x_tgt = batch['target']
        tau = batch['tau']
        if tau.dim() == 1: tau_b = tau.view(-1, 1, 1, 1)
        elif tau.dim() == 2: tau_b = tau.view(-1, 1, 1, 1)
        else: tau_b = tau
        tau_scalar = tau.view(-1) if tau.dim() <= 2 else tau.flatten()[:x_0.size(0)]
        tau_h = batch.get('tau_hour')

        x_bilinear = (1.0 - tau_b) * x_0 + tau_b * x_T

        # Encode (frozen)
        z_0 = self._encode(x_0)
        z_T = self._encode(x_T)
        z_tgt = self._encode(x_tgt)                                             # target latent

        # Predict z_τ
        z_pred = self.vfi(z_0, z_T, tau_scalar)

        # Decode predicted latent (frozen decoder)
        delta = self._decode_residual(z_pred, tau_scalar, x_tgt.size(-2), x_tgt.size(-1))
        s = self.phase2.scale.view(1, -1, 1, 1)
        x_pred = x_bilinear + s * delta

        # Losses
        latent_l2 = F.mse_loss(z_pred, z_tgt)
        x_pred_m = self.phase2.ae._apply_sst_mask(x_pred)
        x_tgt_m = self.phase2.ae._apply_sst_mask(x_tgt)
        x_bil_m = self.phase2.ae._apply_sst_mask(x_bilinear)
        x_l1 = ((x_pred_m - x_tgt_m).abs() * self.phase2.ae.lat_w).mean()
        loss = latent_l2 + self.lambda_x * x_l1

        # Diagnostics
        sq_model = (x_pred_m.float() - x_tgt_m.float()) ** 2
        sq_bil = (x_bil_m.float() - x_tgt_m.float()) ** 2
        mse_model = (sq_model * self.phase2.ae.lat_w).mean()
        mse_bil = (sq_bil * self.phase2.ae.lat_w).mean()

        self.log(f'{prefix}/loss', loss, sync_dist=True, prog_bar=True)
        self.log(f'{prefix}/latent_l2', latent_l2.detach(), sync_dist=True)
        self.log(f'{prefix}/x_l1', x_l1.detach(), sync_dist=True)
        self.log(f'{prefix}/x_mse_model', mse_model.detach(), sync_dist=True)
        self.log(f'{prefix}/x_mse_bilinear', mse_bil.detach(), sync_dist=True)
        self.log(f'{prefix}/improve_vs_bil', (mse_bil - mse_model).detach(), sync_dist=True)

        # PER-τ × PER-CHANNEL accumulators on val
        if prefix == 'val' and tau_h is not None:
            sq_model_lw = (sq_model * self.phase2.ae.lat_w).sum(dim=(-2, -1))   # (B, 27)
            sq_bil_lw = (sq_bil * self.phase2.ae.lat_w).sum(dim=(-2, -1))
            tau_h_squeezed = tau_h.squeeze(-1) if tau_h.dim() > 1 else tau_h
            for b in range(x_0.size(0)):
                h = int(tau_h_squeezed[b].item())
                if 1 <= h <= 5:
                    self._val_sq_model[h] = self._val_sq_model.get(h, 0) + sq_model_lw[b].detach()
                    self._val_sq_bil[h] = self._val_sq_bil.get(h, 0) + sq_bil_lw[b].detach()
                    self._val_count[h] = self._val_count.get(h, 0) + 1
        return loss

    def on_validation_epoch_start(self):
        self._val_sq_model = {}
        self._val_sq_bil = {}
        self._val_count = {}

    def training_step(self, batch, idx): return self._step(batch, 'train')
    def validation_step(self, batch, idx): return self._step(batch, 'val')

    def on_validation_epoch_end(self):
        if not self._val_count:
            return
        n_px = self.phase2.ae.land_mask.numel()
        all_h_model_sum = torch.zeros(self.hparams.in_channels, device=self.device)
        all_h_bil_sum = torch.zeros(self.hparams.in_channels, device=self.device)
        all_count = 0
        for h in sorted(self._val_count.keys()):
            cnt = self._val_count[h]
            if cnt == 0: continue
            sm = self._val_sq_model[h]; sb = self._val_sq_bil[h]
            if self.trainer.world_size > 1:
                cnt_t = torch.tensor(float(cnt), device=self.device)
                torch.distributed.all_reduce(sm, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(sb, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(cnt_t, op=torch.distributed.ReduceOp.SUM)
                cnt = int(cnt_t.item())
            rmse_model = (sm / (cnt * n_px)).sqrt()
            rmse_bil = (sb / (cnt * n_px)).sqrt()
            for i, ch in enumerate(CH_27):
                self.log(f'val_rmse_h{h}/{ch}', rmse_model[i].item(), rank_zero_only=True)
                self.log(f'val_bil_rmse_h{h}/{ch}', rmse_bil[i].item(), rank_zero_only=True)
            sfc_idx = [CH_27.index(c) for c in ['t2m','u10','v10','mslp','sst','tcc','tcwv']]
            self.log(f'val_rmse_h{h}/surface_mean',
                     rmse_model[sfc_idx].mean().item(), rank_zero_only=True)
            ri = (1.0 - rmse_model / rmse_bil.clamp(min=1e-8)).mean()
            self.log(f'val_rmse_h{h}/avg_relative_improvement', ri.item(), rank_zero_only=True)
            all_h_model_sum += sm
            all_h_bil_sum += sb
            all_count += cnt

        # Aggregate
        if all_count > 0:
            rmse_model_agg = (all_h_model_sum / (all_count * n_px)).sqrt()
            rmse_bil_agg = (all_h_bil_sum / (all_count * n_px)).sqrt()
            sfc_idx = [CH_27.index(c) for c in ['t2m','u10','v10','mslp','sst','tcc','tcwv']]
            self.log('val_rmse/aggregate_surface_mean',
                     rmse_model_agg[sfc_idx].mean().item(), rank_zero_only=True)
            ri = (1.0 - rmse_model_agg / rmse_bil_agg.clamp(min=1e-8)).mean()
            self.log('val_rmse/aggregate_relative_improvement', ri.item(), rank_zero_only=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay,
                                betas=(0.9, 0.95))
        total = self.trainer.estimated_stepping_batches if self.trainer else 100000
        warmup = min(500, total // 30)

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, total - warmup))       # clamp!
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))       # floor 1%
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase2_ckpt', required=True)
    p.add_argument('--years', nargs='+', type=int,
                   default=[2014, 2015, 2016, 2017, 2018, 2019])
    p.add_argument('--bs', type=int, default=8)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--max_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--n_blocks', type=int, default=4)
    p.add_argument('--lambda_x', type=float, default=1.0)
    p.add_argument('--exp_name', default='exp_dcae_phase3_latent_vfi')
    p.add_argument('--gpus', nargs='+', type=int, default=[0])
    p.add_argument('--memmap_dir', default='/tmp/wb2_0p5_cache')
    p.add_argument('--log_root', default='/home/jovyan/dsuhoi/weather_time_interpolation/logs')
    args = p.parse_args()

    train_ds = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=args.years,
        max_tau_hours=6, samples_per_date=4, train=True,
        eval_hours=[1, 2, 3, 4, 5],
        static_path='data/static_features_0p5.pt',
        stats_path='data/json_stats_0p5.nc',
        surface_stats_path='data/surface_stats_0p5.json')
    val_ds = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=[2020],
        max_tau_hours=6, samples_per_date=4, train=False,
        eval_hours=[1, 2, 3, 4, 5],
        static_path='data/static_features_0p5.pt',
        stats_path='data/json_stats_0p5.nc',
        surface_stats_path='data/surface_stats_0p5.json')

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = Phase3LatentVFI(phase2_ckpt=args.phase2_ckpt, in_channels=27,
                            hidden=args.hidden, n_blocks=args.n_blocks,
                            lr=args.lr, lambda_x=args.lambda_x)
    trainable = sum(q.numel() for q in model.parameters() if q.requires_grad)
    total = sum(q.numel() for q in model.parameters())
    print(f'params: total={total/1e6:.2f}M  trainable_vfi={trainable/1e6:.2f}M  '
          f'hidden={args.hidden} blocks={args.n_blocks}')

    out_dir = Path(args.log_root) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=out_dir, save_top_k=-1, every_n_epochs=1,
                              filename='{epoch}-{step}', save_last=True)
    logger = TensorBoardLogger(save_dir=str(out_dir / 'lightning_logs'), name='', version=0)

    if len(args.gpus) > 1:
        from pytorch_lightning.strategies import DDPStrategy
        backend = os.environ.get('DDP_BACKEND', 'nccl')
        strategy = DDPStrategy(process_group_backend=backend, find_unused_parameters=True)
    else:
        strategy = 'auto'
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator='gpu', devices=args.gpus,
        strategy=strategy, precision='bf16-mixed',
        callbacks=[ckpt_cb], logger=logger,
        log_every_n_steps=20, gradient_clip_val=1.0, val_check_interval=0.5)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()

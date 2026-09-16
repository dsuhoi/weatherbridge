"""CorrDiff (NVIDIA) + Flow Matching velocity over WeatherDCAE NoSkip 37.5M baseline (VFI).

Variant of train_corrdiff_fm_v2.py — base is WeatherDCAE NoSkip 24ch 6yr 6h ep8
(paper SOTA, see metrics/eval_0p5_2020_fast/weatherdcae_noskip_24ch_6yr_ep8.json)
instead of DC-AE Skip 14M 6yr.

Pipeline:
  1. x_base = WeatherDCAE_NoSkip(x_0, x_T, tau_vfi, static)  # FROZEN paper SOTA, 24ch
  2. delta_target = x_target[:, :24] - x_base                # CorrDiff residual signal
  3. Conditional FM (Lipman 2023, rectified):
       x_t = (1 - t)*x_init + t*delta_target
       v_target = delta_target - x_init                      # constant velocity
  4. v_theta(x_t, t_fm, x_0[:, :24], x_T[:, :24], x_base, tau_vfi) trained with MSE
  5. Inference: x_init=0 -> N-step Euler ODE -> delta_pred
       x_final = x_base + scale * delta_pred

FM velocity U-Net is identical architecture to Skip baseline version
(same hidden, n_levels, in_ch=24 -> 4*24=96 input ch concat) for fair compare.

Per-tau x per-channel val RMSE buckets (5 tau hours x 24 channels matrix).
"""
import os, sys, argparse, math, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')

sys.path.insert(0, '/workspace/code/wti')
sys.path.insert(0, '/home/jovyan/dsuhoi/weather_time_interpolation')

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
try:
    from pytorch_lightning.loggers import TensorBoardLogger
except Exception:
    TensorBoardLogger = None
from pytorch_lightning.loggers import CSVLogger
from weather_time_interp.memmap_dataset import ERA5MemmapDataset

CH_24 = ["T1000","T925","T850","T700","U1000","U925","U850","U700","V1000","V925","V850","V700",
         "Q1000","Q925","Q850","Q700","Z1000","Z925","Z850","Z700",
         "t2m","u10","v10","mslp"]
TAU_HOURS = [1, 2, 3, 4, 5]
N_CH = 24


def sinusoidal_embed(t, dim):
    half = dim // 2
    freqs = torch.exp(torch.linspace(0, -math.log(10000), half, device=t.device))
    a = t.view(-1, 1) * freqs.view(1, -1)
    return torch.cat([a.sin(), a.cos()], dim=-1)


class DualAdaLNZero(nn.Module):
    def __init__(self, dim, t_dim_fm=128, t_dim_vfi=64):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.t_dim_fm = t_dim_fm
        self.t_dim_vfi = t_dim_vfi
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim_fm + t_dim_vfi, (t_dim_fm + t_dim_vfi) * 4),
            nn.SiLU(),
            nn.Linear((t_dim_fm + t_dim_vfi) * 4, dim * 3),
        )
        nn.init.normal_(self.time_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.time_mlp[-1].bias)
        with torch.no_grad():
            self.time_mlp[-1].weight[2 * dim:].zero_()

    def forward(self, h, tau_fm, tau_vfi):
        h_n = self.norm(h)
        emb_fm = sinusoidal_embed(tau_fm.view(-1).float(), self.t_dim_fm)
        emb_vfi = sinusoidal_embed(tau_vfi.view(-1).float(), self.t_dim_vfi)
        emb = torch.cat([emb_fm, emb_vfi], dim=-1)
        gamma, beta, gate = self.time_mlp(emb).chunk(3, dim=-1)
        gamma = gamma.view(-1, h.size(1), 1, 1)
        beta = beta.view(-1, h.size(1), 1, 1)
        gate = gate.view(-1, h.size(1), 1, 1)
        return h + gate * ((1 + gamma) * h_n + beta)


def conv_blk(ic, oc, stride=1):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, stride=stride, padding=1),
        nn.GroupNorm(8, oc), nn.SiLU(),
        nn.Conv2d(oc, oc, 3, padding=1),
        nn.GroupNorm(8, oc), nn.SiLU(),
    )


class FMVelocityUNet(nn.Module):
    """Velocity U-Net. Input cat(x_t, x_0, x_T, x_base) = 4 * in_ch."""
    def __init__(self, in_ch=24, hidden=64, n_levels=3):
        super().__init__()
        self.in_conv = nn.Conv2d(4 * in_ch, hidden, 3, padding=1)
        ch = [hidden * (2 ** i) for i in range(n_levels + 1)]
        self.down = nn.ModuleList([conv_blk(ch[i], ch[i + 1], stride=2) for i in range(n_levels)])
        self.adaln_down = nn.ModuleList([DualAdaLNZero(ch[i + 1]) for i in range(n_levels)])
        self.mid = conv_blk(ch[-1], ch[-1])
        self.mid_adaln = DualAdaLNZero(ch[-1])
        self.up_conv = nn.ModuleList([
            nn.ConvTranspose2d(ch[-(i + 1)], ch[-(i + 2)], 4, stride=2, padding=1)
            for i in range(n_levels)
        ])
        self.up = nn.ModuleList([conv_blk(ch[-(i + 2)] * 2, ch[-(i + 2)]) for i in range(n_levels)])
        self.adaln_up = nn.ModuleList([DualAdaLNZero(ch[-(i + 2)]) for i in range(n_levels)])
        self.out_conv = nn.Conv2d(ch[0], in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_t, x_0, x_T, x_base, tau_fm, tau_vfi):
        h = torch.cat([x_t, x_0, x_T, x_base], dim=1)
        h = self.in_conv(h)
        skips = [h]
        for blk, ada in zip(self.down, self.adaln_down):
            h = blk(h)
            h = ada(h, tau_fm, tau_vfi)
            skips.append(h)
        h = self.mid(h)
        h = self.mid_adaln(h, tau_fm, tau_vfi)
        skips = skips[:-1]
        for up_c, blk, ada, skip in zip(self.up_conv, self.up, self.adaln_up, skips[::-1]):
            h = up_c(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            h = blk(torch.cat([h, skip], dim=1))
            h = ada(h, tau_fm, tau_vfi)
        return self.out_conv(h)


def _load_frozen_base(ckpt_path):
    """Load WeatherHermiteLightningModule baseline (any architecture) via Lightning wrapper."""
    from trainer_weather_hermite import WeatherHermiteLightningModule
    lit = WeatherHermiteLightningModule.load_from_checkpoint(
        ckpt_path, map_location='cpu', strict=False)
    inner = lit.model
    n_params = sum(p.numel() for p in inner.parameters())
    print(f'  Base loaded via Lightning wrapper: type={type(inner).__name__} '
          f'params={n_params/1e6:.2f}M')
    return inner


class CorrDiffFMVFI(pl.LightningModule):
    """CorrDiff + FM correction over frozen 24ch WeatherDCAE base (NoSkip 37.5M)."""

    def __init__(self, base_ckpt, in_channels=24,
                 hidden=64, n_levels=3,
                 static_features_path='data/static_features_0p5.pt',
                 lr=2e-4, weight_decay=1e-4,
                 scale_init=0.10, n_ode_steps=5):
        super().__init__()
        self.save_hyperparameters()
        print(f'[init] loading frozen base from {base_ckpt}')
        self.base = _load_frozen_base(base_ckpt)
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

        self.velocity = FMVelocityUNet(in_ch=in_channels, hidden=hidden, n_levels=n_levels)
        self.scale = nn.Parameter(torch.full((in_channels,), float(scale_init)))
        self.n_ode_steps = n_ode_steps

        static = torch.load(static_features_path, weights_only=False).float()
        lsm = static[0]
        self.register_buffer('static_feats', static, persistent=False)
        self.register_buffer('land_mask', (lsm > 0.5).float(), persistent=False)
        H = lsm.shape[0]
        lat = torch.linspace(89.75, -89.75, H, dtype=torch.float32)
        lw = torch.cos(torch.deg2rad(lat))
        lw = lw / lw.sum() * H
        self.register_buffer('lat_w', lw.view(1, 1, -1, 1), persistent=False)

        self.val_buckets = {h: {'sq_model': torch.zeros(in_channels),
                                'sq_base': torch.zeros(in_channels),
                                'count': torch.tensor(0.0)} for h in TAU_HOURS}

    def setup(self, stage=None):
        for h in TAU_HOURS:
            for k, v in self.val_buckets[h].items():
                self.val_buckets[h][k] = v.to(self.device)

    @torch.no_grad()
    def _baseline_pred(self, x_0_24, x_T_24, tau_vfi, static):
        """WeatherDCAE base forward on 24ch; returns 24ch x_base."""
        if static is None:
            B = x_0_24.size(0)
            static = self.static_feats.unsqueeze(0).expand(B, -1, -1, -1)
        out = self.base(x_0_24, x_T_24, tau_vfi.view(-1), None, static)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def _step(self, batch, prefix):
        x_0 = batch['x0'][:, :N_CH]
        x_T = batch['x1'][:, :N_CH]
        x_tgt = batch['target'][:, :N_CH]
        tau_vfi = batch['tau']
        if tau_vfi.dim() > 1:
            tau_vfi = tau_vfi.view(-1)
        static = batch.get('static')

        x_base = self._baseline_pred(x_0, x_T, tau_vfi, static)
        delta_target = x_tgt - x_base

        B = x_0.size(0)
        tau_fm = torch.rand(B, device=x_0.device)
        x_init = torch.randn_like(delta_target)
        t_b = tau_fm.view(-1, 1, 1, 1)
        x_t = (1 - t_b) * x_init + t_b * delta_target
        v_target = delta_target - x_init

        v_pred = self.velocity(x_t, x_0, x_T, x_base, tau_fm, tau_vfi)
        fm_loss = F.mse_loss(v_pred, v_target)

        self.log(f'{prefix}/fm_loss', fm_loss, sync_dist=True, prog_bar=True)

        if prefix == 'val':
            with torch.no_grad():
                xt = torch.zeros_like(delta_target)
                N = self.n_ode_steps
                for k in range(N):
                    t_k = torch.full((B,), k / N, device=x_0.device)
                    v_k = self.velocity(xt, x_0, x_T, x_base, t_k, tau_vfi)
                    xt = xt + v_k / N
                delta_pred = xt
                s = torch.tanh(self.scale).view(1, -1, 1, 1)
                x_final = x_base + s * delta_pred

                sq_m = (x_final.float() - x_tgt.float()) ** 2
                sq_b = (x_base.float() - x_tgt.float()) ** 2

                mse_m = (sq_m * self.lat_w).mean()
                mse_b = (sq_b * self.lat_w).mean()
                self.log('val/mse_model', mse_m, sync_dist=True)
                self.log('val/mse_baseline', mse_b, sync_dist=True)
                self.log('val/improve_vs_base_rel',
                         (mse_b - mse_m) / (mse_b + 1e-12), sync_dist=True, prog_bar=True)
                self.log('val/scale_mean', self.scale.detach().abs().mean(), sync_dist=True)

                sq_m_pc = (sq_m * self.lat_w).sum(dim=(-2, -1))
                sq_b_pc = (sq_b * self.lat_w).sum(dim=(-2, -1))
                tau_hour = batch.get('tau_hour')
                if tau_hour is not None:
                    th = tau_hour.view(-1)
                    for h in TAU_HOURS:
                        mask = (th == h)
                        if mask.any():
                            self.val_buckets[h]['sq_model'] += sq_m_pc[mask].sum(dim=0)
                            self.val_buckets[h]['sq_base'] += sq_b_pc[mask].sum(dim=0)
                            self.val_buckets[h]['count'] += int(mask.sum().item())

        return fm_loss

    def training_step(self, b, i):
        return self._step(b, 'train')

    def validation_step(self, b, i):
        return self._step(b, 'val')

    def on_validation_epoch_end(self):
        if self.trainer is not None and self.trainer.world_size > 1:
            for h in TAU_HOURS:
                for k in ['sq_model', 'sq_base', 'count']:
                    torch.distributed.all_reduce(self.val_buckets[h][k],
                                                  op=torch.distributed.ReduceOp.SUM)
        n_px = self.land_mask.numel()
        for h in TAU_HOURS:
            cnt = self.val_buckets[h]['count'].item()
            if cnt <= 0:
                continue
            rmse_m = (self.val_buckets[h]['sq_model'] / (cnt * n_px)).sqrt()
            rmse_b = (self.val_buckets[h]['sq_base'] / (cnt * n_px)).sqrt()
            for i, ch in enumerate(CH_24):
                self.log(f'val_rmse_tau{h}/{ch}', rmse_m[i].item(), rank_zero_only=True)
                self.log(f'val_rmse_tau{h}_base/{ch}', rmse_b[i].item(), rank_zero_only=True)
            self.log(f'val_rmse_tau{h}/mean_all', rmse_m.mean().item(), rank_zero_only=True)
            self.log(f'val_rmse_tau{h}_base/mean_all', rmse_b.mean().item(), rank_zero_only=True)
            sfc_idx = [CH_24.index(c) for c in ['t2m', 'u10', 'v10', 'mslp']]
            self.log(f'val_rmse_tau{h}/mean_surface',
                     torch.stack([rmse_m[i] for i in sfc_idx]).mean().item(), rank_zero_only=True)
            imp = (rmse_b.mean() - rmse_m.mean()) / (rmse_b.mean() + 1e-12) * 100.0
            self.log(f'val_rmse_tau{h}/improve_pct', imp.item(), rank_zero_only=True)
        for h in TAU_HOURS:
            for k in ['sq_model', 'sq_base']:
                self.val_buckets[h][k].zero_()
            self.val_buckets[h]['count'].zero_()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay, betas=(0.9, 0.95))
        total = self.trainer.estimated_stepping_batches if self.trainer else 100000
        warmup = min(500, total // 30)

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            p = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_ckpt', required=True,
                   help='Path to frozen WeatherDCAE NoSkip 24ch baseline ckpt')
    p.add_argument('--years', nargs='+', type=int, default=[2017, 2018, 2019])
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--max_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--n_levels', type=int, default=3)
    p.add_argument('--n_ode_steps', type=int, default=5)
    p.add_argument('--exp_name', default='exp_corrdiff_fm_weatherdcae_3yr_fibo')
    p.add_argument('--gpus', nargs='+', type=int, default=[0])
    p.add_argument('--memmap_dir', default='/tmp/wb2_0p5_cache')
    p.add_argument('--log_root', default='/workspace/code/wti/logs')
    p.add_argument('--save_top_k', type=int, default=2)
    p.add_argument('--resume', default=None)
    args = p.parse_args()

    train_ds = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=args.years, max_tau_hours=6,
        samples_per_date=4, train=True, eval_hours=[1, 2, 3, 4, 5],
        static_path='data/static_features_0p5.pt',
        stats_path='data/json_stats_0p5.nc',
        surface_stats_path='data/surface_stats_0p5.json')
    val_ds = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=[2020], max_tau_hours=6,
        samples_per_date=4, train=False, eval_hours=[1, 2, 3, 4, 5],
        static_path='data/static_features_0p5.pt',
        stats_path='data/json_stats_0p5.nc',
        surface_stats_path='data/surface_stats_0p5.json')
    print(f'train: {len(train_ds)}  val: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = CorrDiffFMVFI(
        base_ckpt=args.base_ckpt, in_channels=N_CH,
        hidden=args.hidden, n_levels=args.n_levels, lr=args.lr,
        n_ode_steps=args.n_ode_steps,
    )
    n = sum(q.numel() for q in model.parameters())
    trainable = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f'params: total={n/1e6:.2f}M  trainable={trainable/1e6:.2f}M  '
          f'(velocity U-Net + scale)')

    out_dir = Path(args.log_root) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=out_dir, save_top_k=args.save_top_k,
                              monitor='val/improve_vs_base_rel', mode='max',
                              every_n_epochs=1,
                              filename='{epoch}-{step}-imp{val/improve_vs_base_rel:.3f}',
                              auto_insert_metric_name=False,
                              save_last=True)
    try:
        logger = TensorBoardLogger(save_dir=str(out_dir / 'lightning_logs'),
                                    name='', version=0)
    except Exception:
        logger = CSVLogger(save_dir=str(out_dir / 'lightning_logs'),
                           name='', version=0)
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
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == '__main__':
    main()

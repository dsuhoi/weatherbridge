#!/usr/bin/env python3
"""12h interpolation with HOLDOUT protocol (odd-hours + boundary evens train, middle evens held out).

Train tau subset:  {1, 2, 3, 5, 7, 9, 10, 11}  (8 hours, "seen")
Eval ALL tau:      {1..11}; split into seen vs unseen={4, 6, 8}.

Trainer already supports seen_hours/val_seen/val_unseen via `seen_hours=` kwarg.
Logs:
  test/seen/mean_rmse_norm     — RMSE on train hours {1,2,3,5,7,9,10,11}
  test/unseen/mean_rmse_norm   — RMSE on held-out hours {4,6,8}
  test/hour_<h>/mean_rmse_norm — per-hour breakdown for h=0..12

Key fix vs train_0p5_memmap_12h_v3.py:
  - HOURS_PER_TAU_UNIT hardcoded at 6.0 in memmap_dataset → with max_tau=12, tau goes
    up to 11/6 ≈ 1.83, breaking bilinear (1-tau)*x0 + tau*xT residual blending.
    We wrap the dataset to recompute  tau = tau_hour / delta_t_hours  (= h/12 ∈ [0,1]).

Env vars (same as v3):
  YEARS, MAX_EPOCHS, BATCH_SIZE, LR, MODEL_TYPE, LATENT_CHANNELS, OUT_DIR, NUM_WORKERS,
  KEEP_24CH (=1 to drop sst/tcc/tcwv), VAL_YEARS (default: 2020), VAL_EVERY_N_EPOCHS,
  TRAIN_HOURS (default: "1 2 3 5 7 9 10 11"), EVAL_HOURS (default: "1 2 3 4 5 6 7 8 9 10 11"),
  EXTRA (model-specific kwargs as "k1=v1,k2=v2,...").
"""
import os, sys, time
sys.path.insert(0, os.environ.get("WTI_ROOT", "/workspace/code/wti"))
import torch
torch.set_float32_matmul_precision('high')

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import WeatherHermiteLightningModule, ERA5WeatherHermiteDataset


DELTA_T = 12.0  # 12h window


class TauRescaleWrapper(Dataset):
    """Recompute tau = tau_hour / delta_t (memmap_dataset hardcodes /6 via HOURS_PER_TAU_UNIT).

    Works whether the base is `ERA5WeatherHermiteDataset` (grouped, tau is a vector
    over hours) or a plain `ERA5MemmapDataset` (scalar tau).
    """
    def __init__(self, base, delta_t: float = DELTA_T):
        self.base = base
        self.delta_t = float(delta_t)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        out = self.base[idx]
        if "tau_hour" in out:
            tau_hour = out["tau_hour"].float()
            new_tau = tau_hour / self.delta_t
            out["tau"] = new_tau.view_as(out["tau"]) if out["tau"].shape == new_tau.shape else new_tau
        return out


class TruncateChannelsWrapper(Dataset):
    """Drop sst, tcc, tcwv (last 3 of 27 channels) → 24-channel input/target."""
    def __init__(self, base, n_keep: int = 24):
        self.base = base
        self.n_keep = n_keep
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        out = self.base[idx]
        # ERA5WeatherHermiteDataset adapter outputs "x0","xT","target"
        for key in ("x0", "xT", "x1", "target"):
            if key in out and isinstance(out[key], torch.Tensor) and out[key].dim() >= 3:
                out[key] = out[key][..., :self.n_keep, :, :].contiguous()
        return out


def parse_extra():
    e = os.environ.get("EXTRA", "")
    if not e:
        return {}
    out = {}
    for kv in e.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            k = k.strip(); v = v.strip()
            try: v = int(v)
            except ValueError:
                try: v = float(v)
                except ValueError:
                    if v.lower() == "true": v = True
                    elif v.lower() == "false": v = False
            out[k] = v
    return out


def parse_int_list_env(name: str, default: str):
    return [int(x) for x in os.environ.get(name, default).split()]


def build_loader(years, train: bool, train_hours, eval_hours, samples_per_date,
                 batch_size, num_workers, keep_24ch: bool, delta_t: float,
                 memmap_dir: str, static_path: str, stats_path: str, surface_stats_path: str):
    ds_base = ERA5MemmapDataset(
        memmap_dir=memmap_dir, years=years,
        max_tau_hours=int(delta_t), samples_per_date=samples_per_date,
        train=train,
        train_hours=train_hours if train else None,
        eval_hours=eval_hours if not train else None,
        static_path=static_path,
        stats_path=stats_path,
        surface_stats_path=surface_stats_path,
    )
    cg_full = ds_base.channel_groups
    if keep_24ch:
        channel_groups = {k: v for k, v in cg_full.items() if k not in ("sst", "tcc", "tcwv")}
    else:
        channel_groups = cg_full
    ds = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=delta_t)
    ds = TauRescaleWrapper(ds, delta_t=delta_t)
    if keep_24ch:
        ds = TruncateChannelsWrapper(ds, n_keep=24)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader, channel_groups, len(ds)


def main():
    years = list(map(int, os.environ.get("YEARS", "2018").split()))
    val_years = list(map(int, os.environ.get("VAL_YEARS", "2020").split()))
    max_epochs = int(os.environ.get("MAX_EPOCHS", "8"))
    batch_size = int(os.environ.get("BATCH_SIZE", "2"))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", str(batch_size)))
    lr = float(os.environ.get("LR", "1e-4"))
    model_type = os.environ.get("MODEL_TYPE", "fuxi_swinv2_residual_linear")
    latent_channels = int(os.environ.get("LATENT_CHANNELS", "256"))
    out_dir = os.environ.get("OUT_DIR", "logs/exp_0p5_12h_oddskip")
    num_workers = int(os.environ.get("NUM_WORKERS", "8"))
    val_num_workers = int(os.environ.get("VAL_NUM_WORKERS", str(min(4, num_workers))))
    keep_24ch = os.environ.get("KEEP_24CH", "1") == "1"
    val_every_n_epochs = int(os.environ.get("VAL_EVERY_N_EPOCHS", "1"))
    limit_val_batches = float(os.environ.get("LIMIT_VAL_BATCHES", "1.0"))
    samples_per_date_train = int(os.environ.get("SAMPLES_PER_DATE_TRAIN", "2"))
    samples_per_date_val   = int(os.environ.get("SAMPLES_PER_DATE_VAL", "1"))
    extra_kwargs = parse_extra()

    train_hours = parse_int_list_env("TRAIN_HOURS", "1 2 3 5 7 9 10 11")
    eval_hours  = parse_int_list_env("EVAL_HOURS",  "1 2 3 4 5 6 7 8 9 10 11")

    memmap_dir          = os.environ.get("MEMMAP_DIR",          "/tmp/wb2_0p5_cache")
    static_path         = os.environ.get("STATIC_PATH",         "data/static_features_0p5.pt")
    stats_path          = os.environ.get("STATS_PATH",          "data/json_stats_0p5.nc")
    surface_stats_path  = os.environ.get("SURFACE_STATS_PATH",  "data/surface_stats_0p5.json")

    n_surface = 4 if keep_24ch else 7
    n_total   = 24 if keep_24ch else 27

    print(f"=== TRAIN 12h ODDSKIP ({model_type}) ===")
    print(f"  train_years={years} val_years={val_years}  epochs={max_epochs}  bs={batch_size}/{val_batch_size}")
    print(f"  train_hours (SEEN)  = {train_hours}")
    print(f"  eval_hours          = {eval_hours}  → unseen = {sorted(set(eval_hours)-set(train_hours))}")
    print(f"  delta_t={DELTA_T}h, channels: {n_total} (24ch={keep_24ch}), latent={latent_channels}")
    print(f"  out={out_dir}")
    t0 = time.time()

    train_loader, channel_groups, n_train = build_loader(
        years, train=True, train_hours=train_hours, eval_hours=None,
        samples_per_date=samples_per_date_train,
        batch_size=batch_size, num_workers=num_workers,
        keep_24ch=keep_24ch, delta_t=DELTA_T, memmap_dir=memmap_dir,
        static_path=static_path, stats_path=stats_path, surface_stats_path=surface_stats_path,
    )
    print(f"train samples (grouped): {n_train}", flush=True)

    val_loader, _, n_val = build_loader(
        val_years, train=False, train_hours=None, eval_hours=eval_hours,
        samples_per_date=samples_per_date_val,
        batch_size=val_batch_size, num_workers=val_num_workers,
        keep_24ch=keep_24ch, delta_t=DELTA_T, memmap_dir=memmap_dir,
        static_path=static_path, stats_path=stats_path, surface_stats_path=surface_stats_path,
    )
    print(f"val samples (grouped): {n_val}", flush=True)

    base_kwargs = dict(
        model_type=model_type,
        channel_groups=channel_groups,
        n_pl_channels=20, n_surface_channels=n_surface, n_static_features=3,
        latent_channels=latent_channels,
        lat_weighted_loss=True,
        lat_crop=int(os.environ.get("LAT_CROP", "0")),
        lambda_residual=1.0, residual_scale_floor=0.05, residual_scale_init=0.30,
        lambda_anchor=0.5, anchor_every_n_batches=4,
        lr=lr, weight_decay=1e-5,
        use_aurora_weights=True, use_physical_scales_loss=False,
        max_tau_hours=int(DELTA_T),
        seen_hours=train_hours,        # ← split val into seen/unseen
    )
    if "fuxi" in model_type:
        base_kwargs.update(fuxi_depth=8, fuxi_num_heads=8, fuxi_window_size_h=5,
                           fuxi_window_size_w=9, fuxi_patch_size=4, fuxi_drop_path=0.1)
    elif "sdyff" in model_type:
        base_kwargs.update(sdyff_num_layers=6, sdyff_n_modes_lat=16, sdyff_n_modes_lon=32,
                           sdyff_dropout=0.1, sdyff_drop_path=0.1,
                           sdyff_inference_steps=5, sdyff_train_refine_steps=1)
    elif "modafno" in model_type:
        base_kwargs.update(modafno_depth=12, modafno_num_blocks=8, modafno_drop_rate=0.0)
    elif "dcae" in model_type:
        _boc = os.environ.get("DCAE_BLOCK_CHANNELS")
        _lpb = os.environ.get("DCAE_LAYERS_PER_BLOCK")
        block_out = tuple(int(x) for x in _boc.split(",")) if _boc else (128, 128, 256, 256)
        layers = tuple(int(x) for x in _lpb.split(",")) if _lpb else (2, 2, 2, 2)
        base_kwargs.update(block_out_channels=block_out, layers_per_block=layers)
        if os.environ.get("FREEZE_SKIP_GATES", "0") in ("1", "true", "True"):
            base_kwargs["freeze_skip_gates"] = True
    base_kwargs.update(extra_kwargs)

    model = WeatherHermiteLightningModule(**base_kwargs)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params: {n_params:.1f}M", flush=True)

    ckpt_every = int(os.environ.get("CKPT_EVERY_N_EPOCHS", "2"))
    ckpt_cb = ModelCheckpoint(dirpath=out_dir, save_last=True, save_top_k=-1, every_n_epochs=ckpt_every)
    precision = os.environ.get("PRECISION", "bf16-mixed")
    devices_env = os.environ.get("DEVICES", "1")
    devices = int(devices_env) if devices_env.isdigit() else [int(x) for x in devices_env.split(",")]
    strategy = "ddp_find_unused_parameters_true" if (isinstance(devices, list) and len(devices) > 1) or (isinstance(devices, int) and devices > 1) else "auto"
    limit_train_batches_env = os.environ.get("LIMIT_TRAIN_BATCHES", "1.0")
    limit_train_batches = float(limit_train_batches_env)
    trainer = pl.Trainer(
        max_epochs=max_epochs, log_every_n_steps=int(os.environ.get("LOG_EVERY_N_STEPS", "20")),
        devices=devices, accelerator="gpu", precision=precision, strategy=strategy,
        check_val_every_n_epoch=val_every_n_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=int(os.environ.get("SANITY_VAL_STEPS", "0")),
        enable_checkpointing=True, enable_progress_bar=False,
        callbacks=[ckpt_cb], default_root_dir=out_dir,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=os.environ.get("RESUME_CKPT") or None)
    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()

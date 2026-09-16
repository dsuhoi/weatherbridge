#!/usr/bin/env python3
"""Universal 0.5° training using memmap dataset on local SSD.

Env vars:
    YEARS           e.g. "2014 2015 2016 2017 2018 2019" (train years)
    MAX_EPOCHS=8
    BATCH_SIZE=1
    LR=1e-4
    MODEL_TYPE     fuxi_swinv2_residual_linear | sdyff_dyffusion_residual_linear | modafno_residual_linear
    LATENT_CHANNELS=256
    OUT_DIR=logs/exp_<name>
    NUM_WORKERS=4
    EXTRA          extra model kwargs as comma-separated key=value pairs
"""
import os, sys, time
sys.path.insert(0, "/home/jovyan/dsuhoi/weather_time_interpolation")
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import WeatherHermiteLightningModule, ERA5WeatherHermiteDataset


def env(k, default, cast=str):
    v = os.environ.get(k, default)
    return cast(v) if not callable(cast) else cast(v)


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


def main():
    years = list(map(int, os.environ.get("YEARS", "2018").split()))
    max_epochs = int(os.environ.get("MAX_EPOCHS", "8"))
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    lr = float(os.environ.get("LR", "1e-4"))
    model_type = os.environ.get("MODEL_TYPE", "fuxi_swinv2_residual_linear")
    latent_channels = int(os.environ.get("LATENT_CHANNELS", "256"))
    out_dir = os.environ.get("OUT_DIR", "logs/exp_0p5_universal")
    num_workers = int(os.environ.get("NUM_WORKERS", "4"))
    extra_kwargs = parse_extra()

    print(f"=== TRAIN 0.5° ({model_type}) ===")
    print(f"  years={years} (train), test/val=none")
    print(f"  max_epochs={max_epochs}, batch_size={batch_size}, lr={lr}, num_workers={num_workers}")
    print(f"  latent_channels={latent_channels}, out_dir={out_dir}")
    print(f"  extra: {extra_kwargs}")
    t0 = time.time()

    ds_base = ERA5MemmapDataset(
        memmap_dir="/tmp/wb2_0p5_cache",
        years=years,
        max_tau_hours=6,
        samples_per_date=4,
        train=True,
        train_hours=[1, 3, 5],
        static_path="data/static_features_0p5.pt",
        stats_path="data/json_stats_0p5.nc",
        surface_stats_path="data/surface_stats_0p5.json",
    )
    channel_groups = ds_base.channel_groups
    ds_train = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=6.0)
    print(f"train samples: {len(ds_train)} (over {len(years)} year(s))", flush=True)

    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0)

    base_kwargs = dict(
        model_type=model_type,
        channel_groups=channel_groups,
        n_pl_channels=20,
        n_surface_channels=7,
        n_static_features=3,
        latent_channels=latent_channels,
        lat_weighted_loss=True,
        lat_crop=int(os.environ.get("LAT_CROP", "0")),
        lambda_residual=1.0,
        residual_scale_floor=0.05,
        residual_scale_init=0.30,
        lambda_anchor=0.5,
        anchor_every_n_batches=4,
        lr=lr,
        weight_decay=1e-5,
        use_aurora_weights=True,
        use_physical_scales_loss=False,
        max_tau_hours=6,
    )
    if "fuxi" in model_type:
        base_kwargs.update(dict(
            fuxi_depth=8, fuxi_num_heads=8,
            fuxi_window_size_h=5, fuxi_window_size_w=9,
            fuxi_patch_size=4, fuxi_drop_path=0.1,
        ))
    elif "sdyff" in model_type:
        base_kwargs.update(dict(
            sdyff_num_layers=6, sdyff_n_modes_lat=16, sdyff_n_modes_lon=32,
            sdyff_dropout=0.1, sdyff_drop_path=0.1,
            sdyff_inference_steps=5, sdyff_train_refine_steps=1,
        ))
    elif "modafno" in model_type:
        base_kwargs.update(dict(
            modafno_depth=12, modafno_num_blocks=8, modafno_drop_rate=0.0,
        ))
    elif "dcae" in model_type:
        _boc = os.environ.get("DCAE_BLOCK_CHANNELS")
        _lpb = os.environ.get("DCAE_LAYERS_PER_BLOCK")
        if _boc:
            base_kwargs["block_out_channels"] = tuple(int(x) for x in _boc.split(","))
        if _lpb:
            base_kwargs["layers_per_block"] = tuple(int(x) for x in _lpb.split(","))
        if os.environ.get("FREEZE_SKIP_GATES", "0") in ("1", "true", "True"):
            base_kwargs["freeze_skip_gates"] = True
    base_kwargs.update(extra_kwargs)

    model = WeatherHermiteLightningModule(**base_kwargs)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params: {n_params:.1f}M", flush=True)

    # Save every 2 epochs (and last.ckpt) to save NFS space.
    ckpt_every = int(os.environ.get("CKPT_EVERY_N_EPOCHS", "2"))
    ckpt_cb = ModelCheckpoint(dirpath=out_dir, save_last=True, save_top_k=-1, every_n_epochs=ckpt_every)
    # S-DYff/SHT uses cuFFT which restricts fp16 to power-of-two signal sizes;
    # 720 is not a power of 2 → use bf16 for sdyff models. bf16 also fine for others on A100.
    precision = os.environ.get("PRECISION", "16-mixed")
    if "sdyff" in model_type and precision == "16-mixed":
        precision = "bf16-mixed"
        print(f"  (auto-switched to bf16-mixed for {model_type} — cuFFT fp16 doesn't support non-power-of-2 sizes)")
    devices_env = os.environ.get("DEVICES", "1")
    devices = int(devices_env) if devices_env.isdigit() else [int(x) for x in devices_env.split(",")]
    strategy = "ddp_find_unused_parameters_true" if (isinstance(devices, list) and len(devices) > 1) or (isinstance(devices, int) and devices > 1) else "auto"
    trainer = pl.Trainer(
        max_epochs=max_epochs, log_every_n_steps=50,
        devices=devices, accelerator="gpu", precision=precision, strategy=strategy,
        limit_val_batches=0, num_sanity_val_steps=0,
        enable_checkpointing=True, enable_progress_bar=False,
        callbacks=[ckpt_cb], default_root_dir=out_dir,
    )
    trainer.fit(model, train_dataloaders=loader,
                ckpt_path=os.environ.get("RESUME_CKPT") or None)
    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()

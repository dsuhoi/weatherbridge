#!/usr/bin/env python3
"""Hydra 1.3 entry point for WTI training.

Composes data + model + trainer groups from `conf/` and dispatches to one
of three training paths based on `model.kind`:

  - "hermite"     : WeatherHermiteLightningModule + memmap dataset (default)
  - "atm_vfi"     : PixelAttentionVFI Lightning class (custom)
  - "corrdiff_fm" : CorrDiffFMVFI Lightning class with frozen base ckpt

Examples:

    python train.py                                  # defaults: dcae NoSkip 6h
    python train.py model=fuxi trainer=default
    python train.py model=corrdiff_fm \\
        model.base_ckpt=logs/.../last.ckpt trainer=ddp_2gpu
    python train.py +legacy=train_dcae_skip_24ch_6yr
    python train.py model=weatherdcae trainer=smoke   # smoke test
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# Repo root on sys.path so `weather_time_interp`, `trainer_weather_hermite`,
# `train_corrdiff_fm_weatherdcae`, and `scripts/train_atm_vfi_12h_oddskip`
# can be imported.
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _apply_env_vars(env_vars: Dict[str, Any]) -> None:
    """Apply model-level env-var overrides in one block before model init.

    `trainer_weather_hermite.py` reads SDYFF_NLAT/NLON/LAT_CROP and
    MODAFNO_INP_H/W/NATIVE_H/W at module-instantiation time via os.environ.
    We never mutate env vars after this point.
    """
    if not env_vars:
        return
    for k, v in env_vars.items():
        os.environ[str(k)] = str(v)
        print(f"  [env] {k}={v}")


def _build_dataset(cfg: DictConfig, *, val: bool):
    """Instantiate ERA5MemmapDataset for train or val split.

    Returns (raw_dataset, wrapped_dataset, channel_groups, n_pl, n_surface).
    `wrapped_dataset` is the grouped ERA5WeatherHermiteDataset, optionally
    rescaled (for delta_t != 6h) and truncated (for keep_24ch).
    """
    import torch
    from torch.utils.data import Dataset

    from weather_time_interp.memmap_dataset import ERA5MemmapDataset
    from trainer_weather_hermite import ERA5WeatherHermiteDataset

    years = list(cfg.data.val_years) if val else list(cfg.data.years)
    sample_pd = cfg.data.samples_per_date_val if val else cfg.data.samples_per_date_train

    base = ERA5MemmapDataset(
        memmap_dir=str(cfg.data.memmap_dir),
        years=years,
        max_tau_hours=int(cfg.data.max_tau_hours),
        samples_per_date=int(sample_pd),
        train=not val,
        train_hours=list(cfg.data.train_hours) if not val else None,
        eval_hours=list(cfg.data.eval_hours) if val else None,
        static_path=str(cfg.data.static_path),
        stats_path=str(cfg.data.stats_path),
        surface_stats_path=str(cfg.data.surface_stats_path),
    )

    cg_full = base.channel_groups
    if bool(cfg.data.keep_24ch):
        channel_groups = {k: v for k, v in cg_full.items() if k not in ("sst", "tcc", "tcwv")}
    else:
        channel_groups = cg_full

    delta_t = float(cfg.data.delta_t_hours)
    wrapped: Dataset = ERA5WeatherHermiteDataset(base, delta_t_hours=delta_t)

    # tau rescale wrapper if delta_t != 6h (memmap_dataset hardcodes
    # HOURS_PER_TAU_UNIT=6, breaking long-window bilinear residual blends).
    if abs(delta_t - 6.0) > 1e-6:
        wrapped = _TauRescaleWrapper(wrapped, delta_t=delta_t)

    # 24ch truncation wrapper.
    if bool(cfg.data.keep_24ch):
        wrapped = _TruncateChannelsWrapper(wrapped, n_keep=24)

    n_surface = int(cfg.data.n_surface_channels)
    n_pl = int(cfg.data.n_pl_channels)
    return base, wrapped, channel_groups, n_pl, n_surface


class _TauRescaleWrapper:
    def __init__(self, base, delta_t: float):
        import torch as _torch
        self._torch = _torch
        self.base = base
        self.delta_t = float(delta_t)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]
        if "tau_hour" in out:
            tau_hour = out["tau_hour"].float()
            new_tau = tau_hour / self.delta_t
            out["tau"] = new_tau.view_as(out["tau"]) if out["tau"].shape == new_tau.shape else new_tau
        return out


class _TruncateChannelsWrapper:
    def __init__(self, base, n_keep: int = 24):
        import torch as _torch
        self._torch = _torch
        self.base = base
        self.n_keep = int(n_keep)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        import torch
        out = self.base[idx]
        for key in ("x0", "xT", "x1", "target"):
            if key in out and isinstance(out[key], torch.Tensor) and out[key].dim() >= 3:
                out[key] = out[key][..., : self.n_keep, :, :].contiguous()
        return out


# --------------------------------------------------------------------------- #
def _train_hermite(cfg: DictConfig) -> str:
    """Dispatch path for kind=hermite. Returns the run output directory."""
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    try:
        from lightning.pytorch.loggers import TensorBoardLogger
    except Exception:
        TensorBoardLogger = None
    from torch.utils.data import DataLoader

    from trainer_weather_hermite import WeatherHermiteLightningModule

    _apply_env_vars(OmegaConf.to_container(cfg.model.env_vars, resolve=True) or {})

    base_train, ds_train, channel_groups, n_pl, n_surface = _build_dataset(cfg, val=False)
    print(f"[data] train samples={len(ds_train)}", flush=True)

    if cfg.data.val_years and len(list(cfg.data.val_years)) > 0:
        try:
            _, ds_val, _, _, _ = _build_dataset(cfg, val=True)
            print(f"[data] val samples={len(ds_val)}", flush=True)
        except FileNotFoundError as e:
            print(f"[data] val skipped (missing memmap): {e}")
            ds_val = None
    else:
        ds_val = None

    train_loader = DataLoader(
        ds_train,
        batch_size=int(cfg.trainer.batch_size),
        shuffle=True,
        num_workers=int(cfg.trainer.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.trainer.num_workers) > 0,
    )
    val_loader = (
        DataLoader(
            ds_val,
            batch_size=int(cfg.trainer.val_batch_size),
            shuffle=False,
            num_workers=int(cfg.trainer.val_num_workers),
            pin_memory=True,
            persistent_workers=int(cfg.trainer.val_num_workers) > 0,
        )
        if ds_val is not None
        else None
    )

    # Build WeatherHermiteLightningModule kwargs.
    m = cfg.model
    def _get(key: str, default=None):
        """Defensive access — model YAMLs may omit non-default fields."""
        try:
            v = m[key]
        except Exception:
            return default
        return default if v is None else v

    base_kwargs: Dict[str, Any] = dict(
        model_type=str(m.model_type),
        channel_groups=channel_groups,
        n_pl_channels=n_pl,
        n_surface_channels=n_surface,
        n_static_features=int(cfg.data.n_static_features),
        latent_channels=int(_get("latent_channels", 256)),
        lat_weighted_loss=bool(_get("lat_weighted_loss", True)),
        lat_crop=int(_get("lat_crop", 0)),
        lambda_residual=float(_get("lambda_residual", 1.0)),
        residual_scale_floor=float(_get("residual_scale_floor", 0.05)),
        residual_scale_init=float(_get("residual_scale_init", 0.30)),
        lambda_anchor=float(_get("lambda_anchor", 0.5)),
        anchor_every_n_batches=int(_get("anchor_every_n_batches", 4)),
        lr=float(_get("lr", 1e-4)),
        weight_decay=float(_get("weight_decay", 1e-5)),
        use_aurora_weights=bool(_get("use_aurora_weights", True)),
        use_physical_scales_loss=bool(_get("use_physical_scales_loss", False)),
        max_tau_hours=int(cfg.data.max_tau_hours),
    )
    seen_hours = _get("seen_hours")
    if seen_hours is not None:
        base_kwargs["seen_hours"] = list(seen_hours)
    mt = str(m.model_type)
    if "fuxi" in mt:
        base_kwargs.update(
            fuxi_depth=int(_get("fuxi_depth", 8)),
            fuxi_num_heads=int(_get("fuxi_num_heads", 8)),
            fuxi_window_size_h=int(_get("fuxi_window_size_h", 5)),
            fuxi_window_size_w=int(_get("fuxi_window_size_w", 9)),
            fuxi_patch_size=int(_get("fuxi_patch_size", 4)),
            fuxi_drop_path=float(_get("fuxi_drop_path", 0.1)),
        )
    elif "sdyff" in mt:
        base_kwargs.update(
            sdyff_num_layers=int(_get("sdyff_num_layers", 6)),
            sdyff_n_modes_lat=int(_get("sdyff_n_modes_lat", 16)),
            sdyff_n_modes_lon=int(_get("sdyff_n_modes_lon", 32)),
            sdyff_dropout=float(_get("sdyff_dropout", 0.1)),
            sdyff_drop_path=float(_get("sdyff_drop_path", 0.1)),
            sdyff_inference_steps=int(_get("sdyff_inference_steps", 5)),
            sdyff_train_refine_steps=int(_get("sdyff_train_refine_steps", 1)),
        )
    elif "modafno" in mt:
        base_kwargs.update(
            modafno_depth=int(_get("modafno_depth", 12)),
            modafno_num_blocks=int(_get("modafno_num_blocks", 8)),
            modafno_drop_rate=float(_get("modafno_drop_rate", 0.0)),
        )
    elif "dcae" in mt:
        boc = _get("block_out_channels")
        if boc is not None:
            base_kwargs["block_out_channels"] = tuple(int(x) for x in boc)
        lpb = _get("layers_per_block")
        if lpb is not None:
            base_kwargs["layers_per_block"] = tuple(int(x) for x in lpb)
        if bool(_get("freeze_skip_gates", False)):
            base_kwargs["freeze_skip_gates"] = True
        if bool(_get("direct_prediction", False)):
            base_kwargs["direct_prediction"] = True

    model = WeatherHermiteLightningModule(**base_kwargs)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] type={mt} params={n_params:.2f}M", flush=True)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir),
        save_last=True,
        save_top_k=int(cfg.trainer.save_top_k),
        every_n_epochs=int(cfg.trainer.ckpt_every_n_epochs),
    )

    # Logger: CSV + TB if available.
    loggers: List[Any] = [CSVLogger(save_dir=str(out_dir / "lightning_logs"), name="", version=0)]
    if TensorBoardLogger is not None:
        try:
            loggers.append(TensorBoardLogger(save_dir=str(out_dir / "lightning_logs_tb"), name="", version=0))
        except Exception:
            pass

    precision = str(cfg.trainer.precision)
    if "sdyff" in mt and precision == "16-mixed":
        precision = "bf16-mixed"
        print("  (sdyff: auto-switched 16-mixed -> bf16-mixed for SHT cuFFT)")

    devices = cfg.trainer.devices
    if isinstance(devices, str) and devices.isdigit():
        devices_resolved: Any = int(devices)
    elif isinstance(devices, str):
        devices_resolved = [int(x) for x in devices.split(",")]
    else:
        devices_resolved = int(devices) if isinstance(devices, int) else list(devices)
    n_dev = devices_resolved if isinstance(devices_resolved, int) else len(devices_resolved)
    strategy = str(cfg.trainer.strategy)
    if strategy == "auto" and isinstance(n_dev, int) and n_dev > 1:
        strategy = "ddp_find_unused_parameters_true"

    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        devices=devices_resolved,
        accelerator=str(cfg.trainer.accelerator),
        precision=precision,
        strategy=strategy,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        num_sanity_val_steps=int(cfg.trainer.sanity_val_steps),
        check_val_every_n_epoch=int(cfg.trainer.val_every_n_epochs) if val_loader is not None else 1,
        enable_checkpointing=True,
        enable_progress_bar=bool(cfg.trainer.enable_progress_bar),
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        callbacks=[ckpt_cb],
        logger=loggers,
        default_root_dir=str(out_dir),
    )

    resume = cfg.trainer.resume_ckpt
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=str(resume) if resume else None)
    return str(out_dir)


def _train_atm_vfi(cfg: DictConfig) -> str:
    """Dispatch path for kind=atm_vfi."""
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    # Import the model class from the legacy script.
    legacy_path = REPO / "scripts" / "train_atm_vfi_12h_oddskip.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("_atmvfi_mod", str(legacy_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    PixelAttentionVFI = mod.PixelAttentionVFI
    TauRescaleAnd24chWrapper = mod.TauRescaleAnd24chWrapper

    from weather_time_interp.memmap_dataset import ERA5MemmapDataset

    delta_t = float(cfg.data.delta_t_hours)
    window_hours = int(cfg.data.max_tau_hours)

    train_base = ERA5MemmapDataset(
        memmap_dir=str(cfg.data.memmap_dir),
        years=list(cfg.data.years),
        max_tau_hours=window_hours,
        samples_per_date=int(cfg.data.samples_per_date_train),
        train=True,
        train_hours=list(cfg.data.train_hours),
        static_path=str(cfg.data.static_path),
        stats_path=str(cfg.data.stats_path),
        surface_stats_path=str(cfg.data.surface_stats_path),
    )
    val_base = ERA5MemmapDataset(
        memmap_dir=str(cfg.data.memmap_dir),
        years=list(cfg.data.val_years),
        max_tau_hours=window_hours,
        samples_per_date=int(cfg.data.samples_per_date_val),
        train=False,
        eval_hours=list(cfg.data.eval_hours),
        static_path=str(cfg.data.static_path),
        stats_path=str(cfg.data.stats_path),
        surface_stats_path=str(cfg.data.surface_stats_path),
    )
    train_ds = TauRescaleAnd24chWrapper(train_base, delta_t=delta_t, n_keep=24)
    val_ds = TauRescaleAnd24chWrapper(val_base, delta_t=delta_t, n_keep=24)
    print(f"[atm_vfi] train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.trainer.batch_size), shuffle=True,
        num_workers=int(cfg.trainer.num_workers), pin_memory=True,
        persistent_workers=int(cfg.trainer.num_workers) > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.trainer.val_batch_size), shuffle=False,
        num_workers=int(cfg.trainer.val_num_workers), pin_memory=True,
        persistent_workers=int(cfg.trainer.val_num_workers) > 0,
    )

    model = PixelAttentionVFI(
        in_channels=24,
        hidden=int(cfg.model.hidden),
        n_levels=int(cfg.model.n_levels),
        static_features_path=str(cfg.data.static_path),
        lr=float(cfg.model.lr),
        train_taus=tuple(int(x) for x in cfg.data.train_hours),
        eval_taus=tuple(int(x) for x in cfg.data.eval_hours),
        delta_t=delta_t,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[atm_vfi] params={n_params:.2f}M", flush=True)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir), save_last=True, save_top_k=int(cfg.trainer.save_top_k),
        every_n_epochs=int(cfg.trainer.ckpt_every_n_epochs),
        filename="{epoch}-{step}",
    )
    logger = CSVLogger(save_dir=str(out_dir / "lightning_logs"), name="", version=0)

    devices = cfg.trainer.devices
    devices_resolved = int(devices) if isinstance(devices, int) else (
        [int(x) for x in devices.split(",")] if isinstance(devices, str) and "," in devices
        else int(devices) if isinstance(devices, str) and devices.isdigit()
        else list(devices)
    )
    n_dev = devices_resolved if isinstance(devices_resolved, int) else len(devices_resolved)
    strategy = "auto" if (isinstance(n_dev, int) and n_dev == 1) else "ddp_find_unused_parameters_true"

    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=devices_resolved,
        strategy=strategy,
        precision=str(cfg.trainer.precision),
        callbacks=[ckpt_cb],
        logger=logger,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        check_val_every_n_epoch=int(cfg.trainer.val_every_n_epochs),
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        num_sanity_val_steps=int(cfg.trainer.sanity_val_steps),
        enable_progress_bar=bool(cfg.trainer.enable_progress_bar),
    )
    resume = cfg.trainer.resume_ckpt
    trainer.fit(model, train_loader, val_loader, ckpt_path=str(resume) if resume else None)
    return str(out_dir)


def _train_corrdiff_fm(cfg: DictConfig) -> str:
    """Dispatch path for kind=corrdiff_fm."""
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    try:
        from pytorch_lightning.loggers import TensorBoardLogger
    except Exception:
        TensorBoardLogger = None

    base_ckpt = cfg.model.base_ckpt
    if not base_ckpt or str(base_ckpt) in {"???", "None"}:
        raise ValueError("kind=corrdiff_fm requires model.base_ckpt to be set "
                         "(frozen WeatherDCAE NoSkip 24ch baseline path).")

    # Import CorrDiffFMVFI from the appropriate root-level script.
    is_12h = int(cfg.data.max_tau_hours) >= 12
    script_name = "train_corrdiff_fm_weatherdcae_12h.py" if is_12h else "train_corrdiff_fm_weatherdcae.py"
    legacy_path = REPO / script_name
    import importlib.util
    spec = importlib.util.spec_from_file_location("_corrdiff_mod", str(legacy_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    CorrDiffFMVFI = mod.CorrDiffFMVFI

    from weather_time_interp.memmap_dataset import ERA5MemmapDataset

    delta_t = float(cfg.data.delta_t_hours)

    train_ds = ERA5MemmapDataset(
        memmap_dir=str(cfg.data.memmap_dir),
        years=list(cfg.data.years),
        max_tau_hours=int(cfg.data.max_tau_hours),
        samples_per_date=int(cfg.data.samples_per_date_train),
        train=True,
        eval_hours=list(cfg.data.eval_hours),
        static_path=str(cfg.data.static_path),
        stats_path=str(cfg.data.stats_path),
        surface_stats_path=str(cfg.data.surface_stats_path),
    )
    val_ds = ERA5MemmapDataset(
        memmap_dir=str(cfg.data.memmap_dir),
        years=list(cfg.data.val_years),
        max_tau_hours=int(cfg.data.max_tau_hours),
        samples_per_date=int(cfg.data.samples_per_date_val),
        train=False,
        eval_hours=list(cfg.data.eval_hours),
        static_path=str(cfg.data.static_path),
        stats_path=str(cfg.data.stats_path),
        surface_stats_path=str(cfg.data.surface_stats_path),
    )
    print(f"[corrdiff_fm] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=int(cfg.trainer.batch_size), shuffle=True,
                              num_workers=int(cfg.trainer.num_workers), pin_memory=True,
                              persistent_workers=int(cfg.trainer.num_workers) > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.trainer.val_batch_size), shuffle=False,
                            num_workers=int(cfg.trainer.val_num_workers), pin_memory=True,
                            persistent_workers=int(cfg.trainer.val_num_workers) > 0)

    model = CorrDiffFMVFI(
        base_ckpt=str(base_ckpt),
        in_channels=24,
        hidden=int(cfg.model.hidden),
        n_levels=int(cfg.model.n_levels),
        lr=float(cfg.model.lr),
        n_ode_steps=int(cfg.model.n_ode_steps),
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir),
        save_top_k=int(cfg.trainer.save_top_k) if int(cfg.trainer.save_top_k) >= 0 else 2,
        monitor="val/improve_vs_base_rel", mode="max",
        every_n_epochs=int(cfg.trainer.ckpt_every_n_epochs),
        filename="{epoch}-{step}",
        save_last=True,
    )
    if TensorBoardLogger is not None:
        try:
            logger = TensorBoardLogger(save_dir=str(out_dir / "lightning_logs"), name="", version=0)
        except Exception:
            logger = CSVLogger(save_dir=str(out_dir / "lightning_logs"), name="", version=0)
    else:
        logger = CSVLogger(save_dir=str(out_dir / "lightning_logs"), name="", version=0)

    devices = cfg.trainer.devices
    devices_resolved = int(devices) if isinstance(devices, int) else (
        [int(x) for x in devices.split(",")] if isinstance(devices, str) and "," in devices
        else int(devices) if isinstance(devices, str) and devices.isdigit()
        else list(devices)
    )
    n_dev = devices_resolved if isinstance(devices_resolved, int) else len(devices_resolved)
    strategy = "auto" if (isinstance(n_dev, int) and n_dev == 1) else "ddp_find_unused_parameters_true"

    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=devices_resolved,
        strategy=strategy,
        precision=str(cfg.trainer.precision),
        callbacks=[ckpt_cb],
        logger=logger,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        num_sanity_val_steps=int(cfg.trainer.sanity_val_steps),
        enable_progress_bar=bool(cfg.trainer.enable_progress_bar),
    )
    resume = cfg.trainer.resume_ckpt
    trainer.fit(model, train_loader, val_loader, ckpt_path=str(resume) if resume else None)
    return str(out_dir)


# --------------------------------------------------------------------------- #
@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("=" * 70)
    print("WTI training — Hydra entry point")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg, resolve=False))
    print("-" * 70)

    # Set seed early.
    try:
        import lightning.pytorch as pl
        pl.seed_everything(int(cfg.seed))
    except Exception:
        pass

    # Resolve out_dir relative to repo root.
    if not Path(cfg.out_dir).is_absolute():
        with hydra.utils.open_dict(cfg) if False else _allow_struct_unset(cfg):
            cfg.out_dir = str(REPO / cfg.out_dir)

    t0 = time.time()
    kind = str(cfg.model.kind)
    if kind == "hermite":
        out_dir = _train_hermite(cfg)
    elif kind == "atm_vfi":
        out_dir = _train_atm_vfi(cfg)
    elif kind == "corrdiff_fm":
        out_dir = _train_corrdiff_fm(cfg)
    else:
        raise ValueError(f"Unknown model.kind={kind}")

    # Persist hparams.yaml for downstream eval.py auto-load.
    hparams_out = Path(out_dir) / "hparams.yaml"
    try:
        with open(hparams_out, "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        print(f"[done] hparams.yaml -> {hparams_out}")
    except Exception as e:
        print(f"[warn] could not write hparams.yaml: {e}")

    dt = (time.time() - t0) / 60.0
    print(f"\n=== DONE in {dt:.1f} min — outputs in {out_dir} ===")


from contextlib import contextmanager

@contextmanager
def _allow_struct_unset(cfg: DictConfig):
    """Temporarily allow setting fields not in the schema (Hydra struct mode)."""
    was_struct = OmegaConf.is_struct(cfg)
    if was_struct:
        OmegaConf.set_struct(cfg, False)
    try:
        yield cfg
    finally:
        if was_struct:
            OmegaConf.set_struct(cfg, True)


if __name__ == "__main__":
    main()

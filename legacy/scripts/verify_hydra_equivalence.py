#!/usr/bin/env python3
"""Hydra vs direct-Lightning equivalence harness for WeatherDCAE NoSkip 24ch.

Builds the same Lightning module + dataloaders TWO ways:

  A) "legacy" — direct programmatic construction (mirrors the pre-Hydra
     cloud.ru `scripts/train_0p5_memmap.py` invocation but with the actual
     ckpt hparams: n_surface_channels=4, lr=5e-5, lat_crop=-8, etc.).

  B) "hydra" — `python train.py +legacy=train_weatherdcae_noskip_24ch_6yr ...`
     composed via Hydra (this is the new entry point).

Both use:
  - data: WB-2 0.5° memmap, years=[2017,2018,2019] (only years on fibo).
  - trainer: max_epochs=1, limit_train_batches=64, limit_val_batches=8.
  - seed=42, single GPU, bf16-mixed.

The script writes loss curves + final-state hashes to ``out_dir`` so a
follow-up cell can diff them. Run twice: once with --mode=legacy, once with
--mode=hydra. Both write a JSON of ``train_loss[step], val_loss[epoch],
param_max_abs, param_l2``.

Usage:
    python verify_hydra_equivalence.py --mode legacy --out out/legacy.json
    python verify_hydra_equivalence.py --mode hydra  --out out/hydra.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _build_module(channel_groups, n_pl=20, n_surface=4):
    """Build WeatherHermiteLightningModule with the noskip-24ch-6yr hparams.

    These mirror the cloud.ru ep8 ckpt's hyper_parameters (extracted from
    weatherdcae_noskip_24ch_6yr_ep8.ckpt). lr=5e-5 was the real cloud.ru lr.
    """
    from trainer_weather_hermite import WeatherHermiteLightningModule
    return WeatherHermiteLightningModule(
        model_type="dcae_adaln_residual_linear",
        channel_groups=channel_groups,
        n_pl_channels=n_pl,
        n_surface_channels=n_surface,
        n_static_features=3,
        latent_channels=256,
        lat_weighted_loss=True,
        lat_crop=-8,
        lambda_residual=1.0,
        residual_scale_floor=0.05,
        residual_scale_init=0.30,
        lambda_anchor=0.5,
        anchor_every_n_batches=4,
        lr=5e-5,
        weight_decay=1e-5,
        use_aurora_weights=True,
        use_physical_scales_loss=False,
        max_tau_hours=6,
        block_out_channels=(128, 128, 256, 256),
        layers_per_block=(2, 2, 2, 2),
    )


def run_legacy(memmap_dir: str, stats_path: str, surface_stats_path: str,
               static_path: str, out_dir: Path, seed: int = 42):
    """Direct-Lightning training (no Hydra) — programmatic dataset + module."""
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, Dataset
    from weather_time_interp.memmap_dataset import ERA5MemmapDataset
    from trainer_weather_hermite import ERA5WeatherHermiteDataset

    pl.seed_everything(seed)

    train_base = ERA5MemmapDataset(
        memmap_dir=memmap_dir,
        years=[2017, 2018, 2019],
        max_tau_hours=6,
        samples_per_date=4,
        train=True,
        train_hours=[1, 3, 5],
        static_path=static_path,
        stats_path=stats_path,
        surface_stats_path=surface_stats_path,
    )
    val_base = ERA5MemmapDataset(
        memmap_dir=memmap_dir,
        years=[2020],
        max_tau_hours=6,
        samples_per_date=1,
        train=False,
        eval_hours=[1, 2, 3, 4, 5],
        static_path=static_path,
        stats_path=stats_path,
        surface_stats_path=surface_stats_path,
    )
    cg_full = train_base.channel_groups
    cg_24 = {k: v for k, v in cg_full.items() if k not in ("sst", "tcc", "tcwv")}

    ds_train = ERA5WeatherHermiteDataset(train_base, delta_t_hours=6.0)
    ds_val = ERA5WeatherHermiteDataset(val_base, delta_t_hours=6.0)

    class _Truncate24(Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            out = self.base[i]
            for k in ("x0", "xT", "x1", "target"):
                if k in out and isinstance(out[k], torch.Tensor) and out[k].dim() >= 3:
                    out[k] = out[k][..., :24, :, :].contiguous()
            return out
    ds_train = _Truncate24(ds_train)
    ds_val = _Truncate24(ds_val)

    train_loader = DataLoader(ds_train, batch_size=4, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=0,
                            pin_memory=True)

    model = _build_module(cg_24)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[legacy] params={n_params:.2f}M, cg keys={list(cg_24.keys())}", flush=True)

    ckpt_cb = ModelCheckpoint(dirpath=str(out_dir), save_last=True, save_top_k=-1,
                              every_n_epochs=1)
    trainer = pl.Trainer(
        max_epochs=1,
        log_every_n_steps=1,
        devices=1,
        accelerator="gpu",
        precision="bf16-mixed",
        strategy="auto",
        limit_train_batches=64,
        limit_val_batches=8,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        enable_progress_bar=False,
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb],
        default_root_dir=str(out_dir),
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return _collect_metrics(model, trainer, out_dir)


def run_hydra(memmap_dir: str, stats_path: str, surface_stats_path: str,
              static_path: str, out_dir: Path, seed: int = 42):
    """Invoke `train.py +legacy=train_weatherdcae_noskip_24ch_6yr` directly.

    This emits its own ckpt + lightning_logs into out_dir. We then load the
    same metrics from the CSV logger output.
    """
    import subprocess
    cmd = [
        sys.executable, str(REPO / "train.py"),
        "+legacy=train_weatherdcae_noskip_24ch_6yr",
        "trainer=smoke",
        "trainer.max_epochs=1",
        "trainer.limit_train_batches=64",
        "trainer.limit_val_batches=8",
        "trainer.batch_size=4",
        "trainer.val_batch_size=4",
        "trainer.log_every_n_steps=1",
        "trainer.num_workers=0",
        "trainer.val_num_workers=0",
        "trainer.precision=bf16-mixed",
        "trainer.gradient_clip_val=1.0",
        f"seed={seed}",
        # lr override to match the cloud.ru ckpt hparams
        "model.lr=5e-5",
        # Available data on fibo: 2017-2019 only
        "data.years=[2017,2018,2019]",
        "data.val_years=[2020]",
        f"data.memmap_dir={memmap_dir}",
        f"data.stats_path={stats_path}",
        f"data.surface_stats_path={surface_stats_path}",
        f"data.static_path={static_path}",
        "data.samples_per_date_train=4",
        "data.samples_per_date_val=1",
        f"exp_name=verify_hydra",
        f"out_dir={out_dir}",
    ]
    print("[hydra] cmd:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH','')}"
    result = subprocess.run(cmd, cwd=str(REPO), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"hydra train.py exited {result.returncode}")
    return _parse_hydra_metrics(out_dir)


def _collect_metrics(model, trainer, out_dir: Path):
    """Collect train/val metrics from a Lightning trainer + final model state."""
    import torch
    metrics = {}
    metrics["final_train_loss"] = float(trainer.callback_metrics.get(
        "train/loss_epoch", trainer.callback_metrics.get("train/loss", float("nan"))))
    metrics["final_val_loss"] = float(trainer.callback_metrics.get(
        "val/loss", trainer.callback_metrics.get("val_loss", float("nan"))))

    # Hash parameter state
    all_params = torch.cat([p.detach().float().cpu().flatten() for p in model.parameters()])
    metrics["param_max_abs"] = float(all_params.abs().max())
    metrics["param_l2"] = float(all_params.norm())
    metrics["param_count"] = int(all_params.numel())

    # CSV log
    csv_dirs = list(Path(out_dir).rglob("metrics.csv"))
    if csv_dirs:
        import csv
        with open(csv_dirs[0]) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        metrics["csv_path"] = str(csv_dirs[0])
        metrics["csv_rows"] = len(rows)
        # Extract train loss per step
        train_losses = []
        val_losses = []
        for row in rows:
            tl = row.get("train/loss_step") or row.get("train/loss")
            vl = row.get("val/loss")
            if tl not in (None, ""):
                try:
                    train_losses.append(float(tl))
                except ValueError:
                    pass
            if vl not in (None, ""):
                try:
                    val_losses.append(float(vl))
                except ValueError:
                    pass
        metrics["train_losses"] = train_losses
        metrics["val_losses"] = val_losses
    return metrics


def _parse_hydra_metrics(out_dir: Path):
    """Parse the CSV log written by Hydra train.py."""
    import csv
    metrics = {}
    csv_paths = list(Path(out_dir).rglob("metrics.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no metrics.csv under {out_dir}")
    csv_path = csv_paths[0]
    metrics["csv_path"] = str(csv_path)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    metrics["csv_rows"] = len(rows)
    train_losses, val_losses = [], []
    for row in rows:
        tl = row.get("train/loss_step") or row.get("train/loss")
        vl = row.get("val/loss")
        if tl not in (None, ""):
            try:
                train_losses.append(float(tl))
            except ValueError:
                pass
        if vl not in (None, ""):
            try:
                val_losses.append(float(vl))
            except ValueError:
                pass
    metrics["train_losses"] = train_losses
    metrics["val_losses"] = val_losses
    metrics["final_train_loss"] = train_losses[-1] if train_losses else float("nan")
    metrics["final_val_loss"] = val_losses[-1] if val_losses else float("nan")

    # Parse ckpt for param state
    import torch
    ckpts = sorted(Path(out_dir).rglob("*.ckpt"))
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        all_params = torch.cat([v.detach().float().flatten() for v in state.values()
                                if v.dtype.is_floating_point])
        metrics["param_max_abs"] = float(all_params.abs().max())
        metrics["param_l2"] = float(all_params.norm())
        metrics["param_count"] = int(all_params.numel())
        metrics["ckpt_path"] = str(ckpts[-1])
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["legacy", "hydra"], required=True)
    ap.add_argument("--out", required=True, help="Output JSON path for metrics")
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--stats-path", default=str(REPO / "data/json_stats_0p5.nc"))
    ap.add_argument("--surface-stats-path", default=str(REPO / "data/surface_stats_0p5.json"))
    ap.add_argument("--static-path", default=str(REPO / "data/static_features_0p5.pt"))
    ap.add_argument("--out-dir", default=None, help="Lightning out_dir (logs+ckpts)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir) if args.out_dir else out_json.parent / f"run_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "legacy":
        metrics = run_legacy(args.memmap_dir, args.stats_path, args.surface_stats_path,
                             args.static_path, out_dir, seed=args.seed)
    else:
        metrics = run_hydra(args.memmap_dir, args.stats_path, args.surface_stats_path,
                            args.static_path, out_dir, seed=args.seed)

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{args.mode}] wrote {out_json}: "
          f"final_train_loss={metrics.get('final_train_loss')}, "
          f"final_val_loss={metrics.get('final_val_loss')}, "
          f"param_l2={metrics.get('param_l2')}")


if __name__ == "__main__":
    main()

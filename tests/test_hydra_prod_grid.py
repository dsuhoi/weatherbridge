"""Prod-grid smoke tests for S-DYff + ModAFNO on real 360x720 ERA5 memmap.

The default smoke test grid (16x32 synth) is too small for:

    - S-DYff: `torch_harmonics.SphericalHarmonicTransform` requires nlat/nlon
      >= the configured n_modes, and the model was tuned for 360x720.
    - ModAFNO: the AFNO blocks are configured against MODAFNO_NATIVE_H/W = 360/720;
      smaller grids exercise codepaths that are never used in production.

This module covers the two skips from `test_hydra_train_one_step.py` with a
manual real-grid forward + backward + optimizer.step + ckpt save, gated by
`@pytest.mark.prod_grid` (skipped by default; run with `pytest -m prod_grid`).

Required mounts inside the docker:

    /tmp/wb2_0p5_cache/wb2_2020.bin    (real ERA5 memmap; produced by downloader)
    /workspace/code/wti                 (this repo; mounted into the container)

Invocation:

    docker run --rm --gpus '"device=1"' --ipc host \\
      -v /home/d.sukhorukov/weather_time_interpolation:/workspace/code/wti \\
      -v /tmp/wti_cache:/tmp/wb2_0p5_cache \\
      wti-train:v1 \\
      bash -lc "cd /workspace/code/wti && pip install -q torch_harmonics==0.6.5 && \\
                pytest tests/test_hydra_prod_grid.py -m prod_grid -v"
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

PROD_MEMMAP_DIR = Path("/tmp/wb2_0p5_cache")
PROD_DATA_BIN = PROD_MEMMAP_DIR / "wb2_2020.bin"
PROD_STATS = REPO / "data" / "json_stats_0p5.nc"
PROD_SURF_STATS = REPO / "data" / "surface_stats_0p5.json"
PROD_STATIC = REPO / "data" / "static_features_0p5.pt"

pytestmark = [pytest.mark.gpu, pytest.mark.prod_grid]


def _prod_grid_available() -> bool:
    """Return True iff the production grid memmap + stats files exist."""
    return (
        PROD_DATA_BIN.exists()
        and PROD_STATS.exists()
        and PROD_SURF_STATS.exists()
        and PROD_STATIC.exists()
    )


@pytest.fixture(scope="module")
def real_memmap_dir() -> str:
    if not _prod_grid_available():
        pytest.skip(
            f"prod-grid fixture missing: {PROD_DATA_BIN} or stats. "
            "Run via the docker invocation in module docstring."
        )
    return str(PROD_MEMMAP_DIR)


def _compose_prod_cfg(model_name: str, out_dir: Path, real_memmap_dir: str):
    """Compose Hydra cfg with prod-grid data + smoke trainer."""
    import hydra
    from hydra import compose, initialize_config_dir

    conf_dir = str(REPO / "conf")
    overrides = [
        f"model={model_name}",
        "trainer=smoke",
        "trainer.batch_size=1",
        "trainer.val_batch_size=1",
        "trainer.limit_train_batches=2",
        "trainer.limit_val_batches=1",
        # Smallest real chunk: a single year, very small samples_per_date.
        "data.years=[2020]",
        "data.val_years=[2020]",
        "data.samples_per_date_train=1",
        "data.samples_per_date_val=1",
        f"data.memmap_dir={real_memmap_dir}",
        f"data.stats_path={PROD_STATS}",
        f"data.surface_stats_path={PROD_SURF_STATS}",
        f"data.static_path={PROD_STATIC}",
        f"exp_name=prod_grid_{model_name}",
        f"out_dir={out_dir}",
    ]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=conf_dir):
        cfg = compose(config_name="train", overrides=overrides)
    return cfg


def _read_8_hours_of_data(real_memmap_dir: str):
    """Read 8 consecutive hours from wb2_2020.bin into a torch tensor.

    Returns: (data[T=8, C=27, H=360, W=720] fp32, channel_names list).
    """
    import json
    import numpy as np
    import torch

    with open(Path(real_memmap_dir) / "wb2_2020.json") as f:
        meta = json.load(f)
    T, C, H, W = meta["T"], meta["n_channels"], meta["H"], meta["W"]
    arr = np.memmap(
        str(Path(real_memmap_dir) / "wb2_2020.bin"),
        dtype=np.float32,
        mode="r",
        shape=(T, C, H, W),
    )
    # 8 consecutive hours starting at t=0
    chunk = torch.from_numpy(np.ascontiguousarray(arr[0:8])).float()
    return chunk, meta["channel_names"]


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_name", ["sdyff", "modafno"])
def test_train_one_step_prod_grid(tmp_path: Path, real_memmap_dir: str, model_name: str):
    """End-to-end: Hydra train.py runs one optimizer.step on real 360x720 data."""
    import subprocess
    import sys

    out_dir = tmp_path / f"prod_grid_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO / "train.py"),
        f"model={model_name}",
        "trainer=smoke",
        "trainer.batch_size=1",
        "trainer.val_batch_size=1",
        "trainer.limit_train_batches=2",
        "trainer.limit_val_batches=1",
        "trainer.num_workers=0",
        "trainer.val_num_workers=0",
        "data.years=[2020]",
        "data.val_years=[2020]",
        "data.samples_per_date_train=1",
        "data.samples_per_date_val=1",
        f"data.memmap_dir={real_memmap_dir}",
        f"data.stats_path={PROD_STATS}",
        f"data.surface_stats_path={PROD_SURF_STATS}",
        f"data.static_path={PROD_STATIC}",
        f"exp_name=prod_grid_{model_name}",
        f"out_dir={out_dir}",
        "seed=42",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH','')}"
    result = subprocess.run(
        cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900
    )
    if result.returncode != 0:
        print("STDOUT (tail):")
        print("\n".join(result.stdout.splitlines()[-60:]))
        print("STDERR (tail):")
        print("\n".join(result.stderr.splitlines()[-60:]))
    assert result.returncode == 0, f"train.py for {model_name} exited {result.returncode}"

    ckpts = list(out_dir.rglob("*.ckpt"))
    assert ckpts, f"no ckpt written under {out_dir} for {model_name}"


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_name", ["sdyff", "modafno"])
def test_model_build_prod_grid_forward_backward(real_memmap_dir: str, model_name: str):
    """Programmatic build + manual forward/backward on a real 360x720 batch.

    Faster than the subprocess test above; isolates the model + loss.
    """
    import torch
    from hydra.utils import instantiate as _  # noqa: F401 (sanity)

    # Compose cfg (also triggers env-var resolution check)
    cfg = _compose_prod_cfg(model_name, Path("/tmp/_prod_grid_smoke"), real_memmap_dir)

    # Apply env vars (sdyff/modafno read these at module-init time)
    from train import _apply_env_vars
    from omegaconf import OmegaConf
    env_vars = OmegaConf.to_container(cfg.model.env_vars, resolve=True) or {}
    _apply_env_vars(env_vars)

    # Build the module
    from trainer_weather_hermite import WeatherHermiteLightningModule

    # Use a tiny channel_groups consistent with 24ch (skip sst/tcc/tcwv).
    channel_groups = {
        "T": [0, 1, 2, 3],
        "U": [4, 5, 6, 7],
        "V": [8, 9, 10, 11],
        "Q": [12, 13, 14, 15],
        "Z": [16, 17, 18, 19],
        "t2m": [20],
        "u10": [21],
        "v10": [22],
        "mslp": [23],
    }
    base_kwargs = dict(
        model_type=str(cfg.model.model_type),
        channel_groups=channel_groups,
        n_pl_channels=20,
        n_surface_channels=4,
        n_static_features=3,
        latent_channels=int(cfg.model.latent_channels),
        lat_weighted_loss=bool(cfg.model.lat_weighted_loss),
        lat_crop=int(cfg.model.lat_crop),
        lambda_residual=float(cfg.model.lambda_residual),
        residual_scale_floor=float(cfg.model.residual_scale_floor),
        residual_scale_init=float(cfg.model.residual_scale_init),
        lambda_anchor=float(cfg.model.lambda_anchor),
        anchor_every_n_batches=int(cfg.model.anchor_every_n_batches),
        lr=float(cfg.model.lr),
        weight_decay=float(cfg.model.weight_decay),
        use_aurora_weights=bool(cfg.model.use_aurora_weights),
        max_tau_hours=int(cfg.data.max_tau_hours),
    )
    if model_name == "sdyff":
        base_kwargs.update(
            sdyff_num_layers=int(cfg.model.sdyff_num_layers),
            sdyff_n_modes_lat=int(cfg.model.sdyff_n_modes_lat),
            sdyff_n_modes_lon=int(cfg.model.sdyff_n_modes_lon),
            sdyff_dropout=float(cfg.model.sdyff_dropout),
            sdyff_drop_path=float(cfg.model.sdyff_drop_path),
            sdyff_inference_steps=int(cfg.model.sdyff_inference_steps),
            sdyff_train_refine_steps=int(cfg.model.sdyff_train_refine_steps),
        )
    elif model_name == "modafno":
        base_kwargs.update(
            modafno_depth=int(cfg.model.modafno_depth),
            modafno_num_blocks=int(cfg.model.modafno_num_blocks),
            modafno_drop_rate=float(cfg.model.modafno_drop_rate),
        )
    model = WeatherHermiteLightningModule(**base_kwargs).cuda()
    model.train()

    # Pull a real batch (1 sample) from the memmap, truncate to 24ch.
    data, _ = _read_8_hours_of_data(real_memmap_dir)
    # data: [8, 27, 360, 720]; truncate to 24ch and build the batch dict that
    # WeatherHermiteLightningModule._shared_step expects.
    x0 = data[0:1, :24].cuda()      # [B=1, 24, 360, 720]
    xT = data[6:7, :24].cuda()      # endpoint at tau=max_tau_hours=6
    target = data[3:4, :24].cuda()  # midpoint at tau_hour=3
    tau = torch.tensor([[0.5]], device="cuda")          # tau normalized = 3/6
    tau_hour = torch.tensor([[3]], dtype=torch.long, device="cuda")
    cond = torch.tensor([[6.0]], device="cuda")         # delta_t_hours
    batch = {
        "x0": x0, "xT": xT, "target": target,
        "tau": tau, "tau_hour": tau_hour, "cond": cond,
        "static": torch.zeros(1, 3, 360, 720, device="cuda"),
    }

    # Manual AdamW so we don't need a Trainer attachment for `configure_optimizers`.
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.model.lr),
                            weight_decay=float(cfg.model.weight_decay))

    out = model.training_step(batch, 0)
    loss = out["loss"] if isinstance(out, dict) else out
    assert torch.isfinite(loss), f"non-finite training loss for {model_name}: {loss}"
    loss.backward()
    opt.step()
    opt.zero_grad()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[prod_grid:{model_name}] params={n_params:.1f}M, loss={float(loss):.4f}")

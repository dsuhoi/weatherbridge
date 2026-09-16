#!/usr/bin/env python3
"""Per-region × per-season per-channel RMSE breakdown for the 12 h paper.

Mirrors ``tools/eval/region_season_eval.py`` but with the 12 h window,
the seven regions used in the 12 h evaluation matrix (Tropics, N-Extra,
S-Extra, Arctic, Antarctic, North Atlantic, West Pacific), and fractional
land/ocean surface regimes.

Aggregation across τ: each window contributes once per region/season for
each τ ∈ ``eval_hours``; the JSON reports a single number per
(region, season, channel) that is the lat-cosine-weighted RMSE averaged
over all (window, τ) pairs that fell in that (region, season).

JSON schema::

    {
      "model_name": "...",
      "checkpoint": "...",
      "model_type": "...",
      "delta_t_hours": 12.0,
      "regions": [...],
      "seasons": ["DJF", "MAM", "JJA", "SON"],
      "channel_names": [...],
      "per_region_season": {
        "Tropics": {
          "DJF": {
            "rmse_norm_avg": <float>,
            "rmse_per_channel": {"T1000": <float>, ...},
            "n_samples": <int>
          },
          ...
        },
        ...
      },
      "tau_aggregation": "mean over τ∈[1..11]"
    }

Output path: ``metrics/region_season_12h/<model_name>.json``.

Usage example::

    python tools/eval/region_season_12h_eval.py \
        --memmap-dir /workspace/code/wti/cache/wb2_0p5_cache \
        --test-year 2020 \
        --models WeatherDCAE:logs/.../last.ckpt \
        --out-dir metrics/region_season_12h
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shlex
import sys
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.normalization import (
    STATIC_FEATURES_3,
    file_provenance,
)

# Heavy deps are loaded lazily so ``--help`` works in environments without
# xarray / the weather_time_interp package installed.
try:  # pragma: no cover - import-time guard
    from torch.utils.data import DataLoader  # type: ignore
    from weather_time_interp.eval_datasets import (  # type: ignore
        ERA5WeatherHermiteDataset,
    )
    from weather_time_interp.memmap_dataset import (  # type: ignore
        ERA5MemmapDataset,
    )
except ImportError as _imp_err:  # noqa: WPS440
    _DEFERRED_IMPORT_ERROR = _imp_err

    class _StubBase:  # noqa: D401
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise _DEFERRED_IMPORT_ERROR

    DataLoader = _StubBase  # type: ignore[assignment]
    ERA5WeatherHermiteDataset = _StubBase  # type: ignore[assignment]
    ERA5MemmapDataset = _StubBase  # type: ignore[assignment]
else:
    _DEFERRED_IMPORT_ERROR = None


# Regions for the 12 h paper. The first five are zonal bands; the last two
# are basin boxes (lat, lon). Longitude convention is [-180, 180].
REGIONS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Tropics":        {"lat": (-23.5, 23.5)},
    "N-Extra":        {"lat": (23.5, 60.0)},
    "S-Extra":        {"lat": (-60.0, -23.5)},
    "Arctic":         {"lat": (60.0, 90.0)},
    "Antarctic":      {"lat": (-90.0, -60.0)},
    "North_Atlantic": {"lat": (25.0, 60.0), "lon": (-80.0, 0.0)},
    "West_Pacific":   {"lat": (10.0, 45.0), "lon": (120.0, 180.0)},
}
SURFACE_REGIMES = ("Land", "Ocean")

SEASONS: Dict[str, Tuple[int, ...]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def region_season_evaluation_source_paths() -> dict[str, Path]:
    """Return every repository source that can affect this evaluation."""
    repo_root = Path(__file__).resolve().parents[2]
    model_root = repo_root / "weather_time_interp" / "model"
    sources = {
        "tools/eval/region_season_12h_eval.py": Path(__file__).resolve(),
        "tools/eval/capmatched_loader.py": (
            repo_root / "tools" / "eval" / "capmatched_loader.py"
        ),
        "tools/train/train_capacity_matched_6h.py": (
            repo_root / "tools" / "train" / "train_capacity_matched_6h.py"
        ),
        "tools/train/training_protocol.py": (
            repo_root / "tools" / "train" / "training_protocol.py"
        ),
        "weather_time_interp/eval_datasets.py": (
            repo_root / "weather_time_interp" / "eval_datasets.py"
        ),
        "weather_time_interp/memmap_dataset.py": (
            repo_root / "weather_time_interp" / "memmap_dataset.py"
        ),
        "weather_time_interp/normalization.py": (
            repo_root / "weather_time_interp" / "normalization.py"
        ),
        "weather_time_interp/model/weatherbridge_upr_lite_model.py": (
            model_root / "weatherbridge_upr_lite_model.py"
        ),
        "weather_time_interp/model/weatherbridge_upr_scaled_model.py": (
            model_root / "weatherbridge_upr_scaled_model.py"
        ),
        "weather_time_interp/model/weatherbridge_upr_spherical_model.py": (
            model_root / "weatherbridge_upr_spherical_model.py"
        ),
        "weather_time_interp/model/weatherbridge_flow_model.py": (
            model_root / "weatherbridge_flow_model.py"
        ),
        "weather_time_interp/model/weather_amt_model.py": (
            model_root / "weather_amt_model.py"
        ),
        "weather_time_interp/model/weather_amt_residual_model.py": (
            model_root / "weather_amt_residual_model.py"
        ),
        "weather_time_interp/model/temporal_expert_router.py": (
            model_root / "temporal_expert_router.py"
        ),
        "weather_time_interp/model/dcae_adaln_model.py": (
            model_root / "dcae_adaln_model.py"
        ),
        "weather_time_interp/model/dcae_adaln_skip_model.py": (
            model_root / "dcae_adaln_skip_model.py"
        ),
        "weather_time_interp/model/amt_upstream/feat_enc.py": (
            model_root / "amt_upstream" / "feat_enc.py"
        ),
        "weather_time_interp/model/amt_upstream/flow_utils.py": (
            model_root / "amt_upstream" / "flow_utils.py"
        ),
        "weather_time_interp/model/amt_upstream/ifrnet.py": (
            model_root / "amt_upstream" / "ifrnet.py"
        ),
        "weather_time_interp/model/amt_upstream/multi_flow.py": (
            model_root / "amt_upstream" / "multi_flow.py"
        ),
        "weather_time_interp/model/amt_upstream/raft.py": (
            model_root / "amt_upstream" / "raft.py"
        ),
        "legacy/scripts/train_atm_vfi_12h_oddskip.py": (
            repo_root
            / "legacy"
            / "scripts"
            / "train_atm_vfi_12h_oddskip.py"
        ),
        "legacy/scripts/trainer_weather_hermite.py": (
            repo_root / "legacy" / "scripts" / "trainer_weather_hermite.py"
        ),
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing region/season evaluation sources: "
            + ", ".join(sorted(missing))
        )
    return sources


def _parse_model_entries(
    value: str,
) -> list[tuple[str, str, dict[str, str]]]:
    models: list[tuple[str, str, dict[str, str]]] = []
    for entry in value.split(","):
        parts = entry.split(":", 2)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(f"invalid NAME:CHECKPOINT[:ENVS] entry: {entry}")
        environment: dict[str, str] = {}
        if len(parts) == 3 and parts[2]:
            for assignment in shlex.split(parts[2]):
                key, separator, assigned_value = assignment.partition("=")
                if (
                    not separator
                    or not key.isidentifier()
                    or key in environment
                ):
                    raise ValueError(
                        f"invalid model environment assignment: {assignment}"
                    )
                environment[key] = assigned_value
        models.append((parts[0], parts[1], environment))
    names = [name for name, _, _ in models]
    if not models or len(names) != len(set(names)):
        raise ValueError("model list must be non-empty with unique names")
    return models


@contextmanager
def _temporary_environment(
    updates: dict[str, str],
) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_paired_region_windows(
    path: Path,
    window_keys: list[tuple[int, int, int]],
    region_mse: list[np.ndarray],
    *,
    region_names: list[str],
    channel_names: list[str],
) -> dict[str, object]:
    if not window_keys or len(window_keys) != len(region_mse):
        raise ValueError("paired region windows are empty or misaligned")
    keys = np.asarray(window_keys, dtype=np.int64)
    values = np.asarray(region_mse, dtype=np.float32)
    expected_shape = (
        len(window_keys),
        len(region_names),
        len(channel_names),
    )
    if keys.shape != (len(window_keys), 3):
        raise ValueError("region window keys must have shape [N, 3]")
    if values.shape != expected_shape:
        raise ValueError(
            f"region MSE shape {values.shape} != {expected_shape}"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("region MSE must be finite and non-negative")
    index_sha256 = hashlib.sha256(
        json.dumps(
            [tuple(int(value) for value in row) for row in keys],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            year=keys[:, 0],
            t0=keys[:, 1],
            tau=keys[:, 2],
            mse_norm=values,
            region_names=np.asarray(region_names),
            channel_names=np.asarray(channel_names),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "window_index_sha256": index_sha256,
        "n_windows": len(window_keys),
    }


def _build_region_masks(
    H: int,
    W: int,
    device: torch.device,
    land_sea_fraction: torch.Tensor | None = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Build geographic and surface-regime weights plus cos-lat weight."""
    if H == 360:
        lat = torch.linspace(89.75, -89.75, H, device=device, dtype=torch.float32)
    elif H == 181:
        lat = torch.linspace(90.0, -90.0, H, device=device, dtype=torch.float32)
    else:
        lat = torch.linspace(90.0, -90.0, H, device=device, dtype=torch.float32)
    # ERA5 longitudes typically run from 0 (or -180) eastwards; assume
    # uniform spacing covering [0, 360) and emit the [-180, 180] form for
    # bbox comparisons.
    lon_0_360 = torch.linspace(0.0, 360.0, W + 1, device=device, dtype=torch.float32)[:-1]
    lon = torch.where(lon_0_360 > 180.0, lon_0_360 - 360.0, lon_0_360)

    cos_w = torch.cos(torch.deg2rad(lat)).view(1, 1, H, 1).float()

    masks: Dict[str, torch.Tensor] = {}
    for name, spec in REGIONS.items():
        lat_min, lat_max = spec["lat"]
        m_lat = ((lat >= lat_min) & (lat <= lat_max)).float().view(1, 1, H, 1)
        if "lon" in spec:
            lon_min, lon_max = spec["lon"]
            m_lon = (
                ((lon >= lon_min) & (lon <= lon_max)).float().view(1, 1, 1, W)
            )
            mask = (m_lat * m_lon).expand(1, 1, H, W).contiguous()
        else:
            mask = m_lat.expand(1, 1, H, W).contiguous()
        masks[name] = mask
    if land_sea_fraction is not None:
        land = torch.as_tensor(
            land_sea_fraction,
            device=device,
            dtype=torch.float32,
        )
        if land.shape != (H, W) or not torch.isfinite(land).all():
            raise ValueError("land-sea fraction must be finite with shape HxW")
        if float(land.min()) < -1.0e-5 or float(land.max()) > 1.0 + 1.0e-5:
            raise ValueError("land-sea fraction must lie in [0, 1]")
        land = land.clamp(0.0, 1.0).view(1, 1, H, W)
        masks["Land"] = land
        masks["Ocean"] = 1.0 - land
    return masks, cos_w


def _month_from_doy(year: int, t0: int, h: int) -> int:
    d = _dt.date(int(year), 1, 1) + _dt.timedelta(days=(int(t0) + int(h)) // 24)
    return d.month


def _load_model_safe(
    ckpt_path: str,
    device: torch.device,
    channel_groups: Dict[str, List[int]],
    *,
    static_path: str,
):
    """Load checkpoint — dispatches between WeatherHermite and ATM-VFI loaders.

    ATM-VFI checkpoints come from ``train_atm_vfi_12h_oddskip.py`` and use a
    separate ``PixelAttentionVFI`` LightningModule with the call signature
    ``model.net(x0, xT, tau)`` over 24 channels (sst/tcc/tcwv dropped).
    Everything else goes through ``WeatherHermiteLightningModule``.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    state = ckpt.get("state_dict", ckpt)
    mt = hparams.get("model_type", "")

    # Capacity-matched checkpoints explicitly record the architecture. Their
    # state also uses a net.* prefix, which the historical heuristic below
    # otherwise misclassifies as the older ATM-VFI Lightning module.
    if hparams.get("arch"):
        from tools.eval.capmatched_loader import load_capmatched_checkpoint

        del ckpt
        return load_capmatched_checkpoint(
            ckpt_path,
            device,
            static_path=static_path,
        )

    # ATM-VFI detection: the ATM-VFI checkpoint's state dict has every key
    # prefixed by ``net.`` (PixelAttentionVFINet), while WeatherHermite
    # checkpoints use ``model.``/``diffusion.``/etc.
    state_keys = list(state.keys())
    net_pref_frac = (
        sum(1 for k in state_keys if k.startswith("net.")) / max(1, len(state_keys))
    )
    is_atmvfi = (
        mt.startswith("atm_vfi")
        or "atm_vfi" in ckpt_path.lower()
        or net_pref_frac > 0.95
        or "in_channels" in hparams  # PixelAttentionVFI uses this kwarg
    )

    if is_atmvfi:
        import importlib.util
        train_mod_path = (
            Path(__file__).resolve().parents[2]
            / "legacy"
            / "scripts"
            / "train_atm_vfi_12h_oddskip.py"
        )
        spec = importlib.util.spec_from_file_location(
            "train_atm_vfi", train_mod_path
        )
        atm_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(atm_mod)  # type: ignore[union-attr]
        PixelAttentionVFI = atm_mod.PixelAttentionVFI
        model = PixelAttentionVFI.load_from_checkpoint(
            ckpt_path, map_location=device
        )
        model.to(device).eval()
        return model, "atm_vfi_pixel_attn"

    # Sniff number of encoder stages from the state dict.
    max_down = max(
        (
            int(k.split("down_blocks.")[1].split(".")[0])
            for k in state.keys()
            if "down_blocks." in k
        ),
        default=-1,
    )
    n_stages_ckpt = (max_down + 1) // 2  # 2 down_blocks per stage

    # Direct-construction path for 4-stage DC-AE Skip / NoSkip models. The
    # WeatherHermiteLightningModule wrapper forces a 3-tuple block_out_channels
    # fallback when the provided tuple has len != 3, which silently breaks
    # state-dict loading for DC-AE Skip 4-stage ckpts. Sidestep by building
    # the underlying nn.Module directly and wrapping in a small adapter that
    # mimics the LightningModule's forward signature.
    if mt == "dcae_adaln_skip_residual_linear" and n_stages_ckpt == 4:
        # 3-stage block_out_channels=(128, 128, 256) reproduces the actual
        # 4-block encoder/decoder layout in these checkpoints (extra inner
        # 128-channel stage between the input 128 and the 256-channel
        # bottleneck). The hparams claim ``(128, 128, 256, 256)`` but
        # decoder.up_blocks.4.conv1 is [128, 128], so the real boc is
        # (128, 128, 256) with the canonical 3-element block_type tuple
        # ("ResBlock", "ResBlock", "EfficientViTBlock").
        boc = (128, 128, 256)
        lpb = (2, 2, 2)
        from weather_time_interp.model.dcae_adaln_skip_model import (  # type: ignore
            WeatherDCAEAdaLNSkipModel,
        )
        core = WeatherDCAEAdaLNSkipModel(
            latent_channels=int(hparams.get("latent_channels", 256)),
            block_out_channels=boc,
            layers_per_block=lpb,
            block_type=("ResBlock", "ResBlock", "EfficientViTBlock"),
            qkv_multiscales=((), (), (5,)),
            in_channels=int(hparams.get("n_pl_channels", 20))
                        + int(hparams.get("n_surface_channels", 4)),
            out_channels=int(hparams.get("n_pl_channels", 20))
                         + int(hparams.get("n_surface_channels", 4)),
            lat_crop=int(hparams.get("lat_crop", 0)),
            residual_scale_init=float(hparams.get("residual_scale_init", 0.3)),
            residual_scale_learnable=bool(hparams.get("residual_scale_learnable", True)),
            residual_clip=hparams.get("residual_clip"),
            residual_scale_floor=float(hparams.get("residual_scale_floor", 0.0)),
            direct_prediction=bool(hparams.get("direct_prediction", False)),
            n_static_features=int(hparams.get("n_static_features", 0)),
            skip_gate_init=0.0,
            skip_lateral_rank=0,
        )
        # Strip "model." prefix from state dict keys to match nn.Module names.
        core_state = {
            k[len("model."):]: v
            for k, v in state.items()
            if k.startswith("model.")
        }
        missing, unexpected = core.load_state_dict(core_state, strict=False)
        print(
            f"  [direct-load 4stage {mt}] missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
        core.to(device).eval()

        class _Adapter(torch.nn.Module):
            """Wraps the underlying nn.Module to match the LightningModule's
            forward(x0, xT, tau, cond, static=...) signature."""
            def __init__(self, m):
                super().__init__()
                self._core = m
            def forward(self, x0, xT, tau, cond=None, static=None):
                return self._core(x0, xT, tau, cond, static=static)
        return _Adapter(core).to(device).eval(), mt

    legacy_scripts = (
        Path(__file__).resolve().parents[2] / "legacy" / "scripts"
    )
    if str(legacy_scripts) not in sys.path:
        sys.path.insert(0, str(legacy_scripts))
    from trainer_weather_hermite import WeatherHermiteLightningModule  # type: ignore

    if "dcae" in mt:
        boc = tuple(hparams.get("block_out_channels", ()))
        lpb = tuple(hparams.get("layers_per_block", ()))
        hparams["block_out_channels"] = (
            boc if len(boc) >= 3 else (128, 256, 512)
        )
        hparams["layers_per_block"] = (
            lpb[:3] if len(lpb) > 3 else (lpb or (3, 3, 3))
        )
    for k in ("channel_groups", "_class_path", "model_type_save"):
        hparams.pop(k, None)
    hparams["channel_groups"] = channel_groups
    model = WeatherHermiteLightningModule(**hparams)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, mt


def _day_picks(K: int) -> list[int]:
    if K < 1 or K > 28:
        raise ValueError("eval_days_per_month must be in [1, 28]")
    if K == 1:
        return [15]
    return sorted({
        1 + round(index * 27 / (K - 1))
        for index in range(K)
    })


def _economy_filter(ds_base: ERA5MemmapDataset, K: int) -> None:
    day_picks = _day_picks(K)
    allowed = set(day_picks)
    filt = []
    for entry in ds_base.index:
        y, t0, _, _ = entry
        doy = t0 // 24
        try:
            d = _dt.date(int(y), 1, 1) + _dt.timedelta(days=int(doy))
        except Exception:
            continue
        if d.day in allowed:
            filt.append(entry)
    ds_base.index = filt
    print(f"  economy filter: {len(filt)} entries")


def _parse_eval_hours(s: str) -> List[int]:
    return sorted({int(x) for x in s.split(",") if x.strip()})


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument(
        "--models",
        required=True,
        help="Comma-separated NAME:CKPT[:ENVS] entries.",
    )
    ap.add_argument(
        "--out-dir",
        default="metrics/region_season_12h",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--samples-per-date", type=int, default=2)
    ap.add_argument("--eval-days-per-month", type=int, default=8)
    ap.add_argument("--max-tau-hours", type=int, default=12)
    ap.add_argument(
        "--eval-hours",
        type=str,
        default="4,6,8",
        help="Comma-separated unseen τ values to aggregate over.",
    )
    ap.add_argument(
        "--keep-n-channels",
        type=int,
        default=24,
        help=(
            "Slice x0/xT/target to first N channels before forward. All 12h "
            "models in the leaderboard were trained with keep_24ch=true "
            "(sst/tcc/tcwv dropped). Pass 0 to disable slicing."
        ),
    )
    ap.add_argument("--device", default=None)
    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()

    if _DEFERRED_IMPORT_ERROR is not None:
        raise SystemExit(
            "region_season_12h_eval requires xarray and the weather_time_interp "
            f"package; original ImportError: {_DEFERRED_IMPORT_ERROR}"
        )

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")

    eval_hours = _parse_eval_hours(args.eval_hours)
    max_tau = int(args.max_tau_hours)
    expected_eval_hours = {
        6: [2, 4],
        12: [4, 6, 8],
    }.get(max_tau)
    if (
        expected_eval_hours is None
        or eval_hours != expected_eval_hours
        or args.samples_per_date != 2
        or args.eval_days_per_month != 8
        or args.keep_n_channels != 24
    ):
        raise ValueError(
            "region/season protocol requires max_tau/eval_hours 6/[2,4] "
            "or 12/[4,6,8], samples_per_date=2, "
            "eval_days_per_month=8, and keep_n_channels=24"
        )

    t_global = time.time()
    ds_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[args.test_year],
        max_tau_hours=max_tau,
        samples_per_date=args.samples_per_date,
        train=False,
        eval_hours=eval_hours,
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
    )
    if args.eval_days_per_month is not None:
        _economy_filter(ds_base, max(1, int(args.eval_days_per_month)))
    channel_groups = ds_base.channel_groups
    channel_names = list(ds_base.channel_names) + list(ds_base.surface_variables)
    keep_n = int(args.keep_n_channels) if int(args.keep_n_channels) > 0 else None
    if keep_n is not None and keep_n < len(channel_names):
        channel_names = channel_names[:keep_n]
        print(
            f"  keep_n_channels={keep_n}: "
            f"truncated channel_names to {len(channel_names)}"
        )
    print(f"  channels ({len(channel_names)}): {channel_names}")

    test_wrapped = ERA5WeatherHermiteDataset(
        ds_base, delta_t_hours=float(max_tau)
    )
    print(f"  windows: {len(test_wrapped)}")
    loader = DataLoader(
        test_wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    first_arr = list(ds_base.memmaps.values())[0]
    H = first_arr.shape[-2]
    W = first_arr.shape[-1]
    static_features = torch.load(
        args.static_path,
        map_location="cpu",
        weights_only=False,
    ).float()
    if static_features.ndim != 3 or static_features.size(0) < 1:
        raise ValueError("static feature tensor must contain land-sea fraction")
    region_masks, cos_w = _build_region_masks(
        H,
        W,
        device,
        land_sea_fraction=static_features[0],
    )
    region_names = list(region_masks)
    region_normalizers: Dict[str, float] = {}
    for rname, mask in region_masks.items():
        wsum = float((mask * cos_w).sum().item())
        if wsum < 1e-9:
            raise RuntimeError(f"empty region mask: {rname}")
        region_normalizers[rname] = wsum
    print("  region cos-weight sums: " + ", ".join(
        f"{k}={v:.3f}" for k, v in region_normalizers.items()
    ))

    grouped = getattr(test_wrapped, "_grouped_indices", None)

    # Parse model list.
    models_list = _parse_model_entries(args.models)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from tools.train.training_protocol import memmap_dataset_provenance

    evaluation_dataset_provenance = memmap_dataset_provenance(
        args.memmap_dir,
        [args.test_year],
    )
    evaluation_input_provenance = {
        "static_features": file_provenance(args.static_path),
        "pressure_level_stats": file_provenance(args.stats_path),
        "surface_stats": file_provenance(args.surface_stats_path),
    }
    evaluation_source_paths = region_season_evaluation_source_paths()
    evaluation_code_provenance = {
        name: _sha256_file(path)
        for name, path in evaluation_source_paths.items()
    }

    for name, ckpt, model_environment in models_list:
        checkpoint_path = Path(ckpt)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(ckpt)
        checkpoint_provenance = file_provenance(checkpoint_path)
        t0_m = time.time()
        with _temporary_environment(model_environment):
            model, mt = _load_model_safe(
                ckpt,
                device,
                channel_groups,
                static_path=args.static_path,
            )
        is_atmvfi = mt == "atm_vfi_pixel_attn"
        print(f"  {name}: type={mt} (atm_vfi={is_atmvfi})")

        # Accumulators: sum of weighted MSE per (region, season, channel) and
        # an integer sample count per (region, season). We accumulate over all
        # τ ∈ eval_hours; the JSON reports the mean RMSE across (window, τ).
        sum_sq: Dict[str, Dict[str, Dict[str, float]]] = {
            r: {s: {} for s in SEASONS} for r in region_names
        }
        counts: Dict[str, Dict[str, int]] = {
            r: {s: 0 for s in SEASONS} for r in region_names
        }
        paired_window_keys: list[tuple[int, int, int]] = []
        paired_region_mse: list[np.ndarray] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                x0 = batch["x0"].to(device, non_blocking=True)
                xT = batch["xT"].to(device, non_blocking=True)
                tau_hour_all = batch["tau_hour"].long()  # (B, nH, 1)
                target_all = batch["target"].to(device, non_blocking=True)
                static = batch.get("static")
                if static is not None:
                    static = static.to(device, non_blocking=True)
                # Slice channels for keep_24ch-trained models (all 12h ckpts).
                if keep_n is not None and x0.size(1) > keep_n:
                    x0 = x0[:, :keep_n].contiguous()
                    xT = xT[:, :keep_n].contiguous()
                    target_all = target_all[:, :, :keep_n].contiguous()
                B, nH = x0.size(0), tau_hour_all.size(1)
                cond = torch.full(
                    (B,), float(max_tau), device=device, dtype=torch.float32
                )

                for h_idx in range(nH):
                    tau_hour_h = tau_hour_all[:, h_idx, 0]  # (B,)
                    target_h = target_all[:, h_idx]
                    # FIXME(HOURS_PER_TAU): recompute τ from hours; see
                    # weather_time_interp/config.py (divisor is hardcoded
                    # at 6.0, wrong for 12 h windows).
                    tau_h = tau_hour_h.float().to(device) / float(max_tau)
                    if is_atmvfi:
                        # ATM-VFI's forward signature is model.net(x0, xT, tau)
                        # with no static / cond. Inputs are already 24 channels
                        # via the keep_n slice above.
                        pred = model.net(x0, xT, tau_h)
                    else:
                        out = model(x0, xT, tau_h, cond, static=static)
                        pred = out[0] if isinstance(out, tuple) else out
                    err = (pred - target_h) ** 2  # (B, C, H, W)

                    hours_per_sample = tau_hour_h.tolist()
                    for i in range(B):
                        h = int(hours_per_sample[i])
                        if h not in eval_hours:
                            continue
                        wrapped_index = batch_idx * loader.batch_size + i
                        if wrapped_index >= len(test_wrapped):
                            break
                        base_idx = (
                            grouped[wrapped_index][0]
                            if grouped is not None
                            else wrapped_index
                        )
                        year, t0, _, _ = ds_base.index[base_idx]
                        mo = _month_from_doy(int(year), int(t0), int(h))
                        season = next(
                            (s for s, mns in SEASONS.items() if mo in mns),
                            None,
                        )
                        if season is None:
                            continue
                        em = err[i : i + 1]  # (1, C, H, W)
                        region_row: list[np.ndarray] = []
                        for rname, mask in region_masks.items():
                            w = mask * cos_w
                            wsum = region_normalizers[rname]
                            wmse = (em * w).sum(dim=(0, 2, 3)) / wsum  # (C,)
                            wmse_values = (
                                wmse.detach().float().cpu().numpy()
                            )
                            region_row.append(wmse_values)
                            bucket = sum_sq[rname][season]
                            for ci, cname in enumerate(channel_names):
                                bucket.setdefault(cname, 0.0)
                                bucket[cname] += float(wmse_values[ci])
                            counts[rname][season] += 1
                        paired_window_keys.append(
                            (int(year), int(t0), int(h))
                        )
                        paired_region_mse.append(np.stack(region_row))
                if batch_idx % 25 == 0:
                    print(f"    batch {batch_idx}/{len(loader)}", flush=True)

        # Build payload.
        per_region_season: Dict[str, Dict[str, Dict[str, object]]] = {}
        for rname in region_names:
            per_region_season[rname] = {}
            for sname in SEASONS:
                n = counts[rname][sname]
                if n == 0:
                    per_region_season[rname][sname] = {
                        "rmse_norm_avg": None,
                        "rmse_per_channel": {},
                        "n_samples": 0,
                    }
                    continue
                rmse_ch = {
                    cname: float(np.sqrt(sum_sq[rname][sname][cname] / n))
                    for cname in channel_names
                    if cname in sum_sq[rname][sname]
                }
                avg = (
                    float(np.mean(list(rmse_ch.values())))
                    if rmse_ch
                    else None
                )
                per_region_season[rname][sname] = {
                    "rmse_norm_avg": avg,
                    "rmse_per_channel": rmse_ch,
                    "n_samples": int(n),
                }

        paired_path = out_dir / f"{name}.paired.npz"
        paired_artifact = _write_paired_region_windows(
            paired_path,
            paired_window_keys,
            paired_region_mse,
            region_names=region_names,
            channel_names=channel_names,
        )
        current_checkpoint_provenance = file_provenance(checkpoint_path)
        current_input_provenance = {
            "static_features": file_provenance(args.static_path),
            "pressure_level_stats": file_provenance(args.stats_path),
            "surface_stats": file_provenance(args.surface_stats_path),
        }
        current_dataset_provenance = memmap_dataset_provenance(
            args.memmap_dir,
            [args.test_year],
        )
        current_code_provenance = {
            source_name: _sha256_file(source_path)
            for source_name, source_path in evaluation_source_paths.items()
        }
        if current_checkpoint_provenance != checkpoint_provenance:
            raise RuntimeError(
                f"{checkpoint_path}: checkpoint changed during evaluation"
            )
        if current_input_provenance != evaluation_input_provenance:
            raise RuntimeError("evaluation inputs changed during evaluation")
        if current_dataset_provenance != evaluation_dataset_provenance:
            raise RuntimeError("evaluation dataset changed during evaluation")
        if current_code_provenance != evaluation_code_provenance:
            raise RuntimeError("evaluation source changed during evaluation")
        payload = {
            "schema_version": 6,
            "model_name": name,
            "checkpoint": ckpt,
            "checkpoint_sha256": checkpoint_provenance["sha256"],
            "checkpoint_provenance": checkpoint_provenance,
            "model_environment": model_environment,
            "model_type": mt,
            "delta_t_hours": float(max_tau),
            "test_year": int(args.test_year),
            "samples_per_date": int(args.samples_per_date),
            "eval_days_per_month": int(args.eval_days_per_month),
            "eval_day_picks": _day_picks(
                int(args.eval_days_per_month)
            ),
            "evaluation_index_sha256": paired_artifact[
                "window_index_sha256"
            ],
            "evaluation_input_provenance": evaluation_input_provenance,
            "evaluation_dataset_provenance": (
                evaluation_dataset_provenance
            ),
            "static_feature_names": list(STATIC_FEATURES_3),
            "regions": region_names,
            "region_types": {
                **{name: "geographic" for name in REGIONS},
                **{name: "surface_fraction" for name in SURFACE_REGIMES},
            },
            "surface_regime_definition": {
                "Land": "land_sea_fraction",
                "Ocean": "1 - land_sea_fraction",
            },
            "seasons": list(SEASONS.keys()),
            "channel_names": channel_names,
            "eval_hours": eval_hours,
            "tau_aggregation": f"mean over τ∈{eval_hours}",
            "per_region_season": per_region_season,
            "paired_region_windows": paired_artifact,
            "evaluation_code_provenance": evaluation_code_provenance,
        }
        out_path = out_dir / f"{name}.json"
        temporary = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(temporary, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, out_path)
        print(f"  saved {out_path} in {(time.time() - t0_m) / 60:.1f} min")
        del model
        torch.cuda.empty_cache()

    print(f"\n=== DONE in {(time.time() - t_global) / 60:.1f} min ===")


if __name__ == "__main__":
    main()

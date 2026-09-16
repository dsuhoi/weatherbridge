#!/usr/bin/env python3
"""Single-process batch eval for the **12h interpolation paper**.

Loads test data + climatology + stats once, then loops over checkpoints.
For each model and each interior hour τ ∈ {1, ..., 11} (with x0=t, xT=t+12),
this script computes:

  * per-channel RMSE in normalised units (paper convention),
  * per-channel RMSE in physical units (denormalised),
  * per-channel ACC (anomaly correlation, lat-weighted).

Two baselines are reported alongside the model:

  * ``bilinear``: closed-form ``(1 - τ/12)·x0 + (τ/12)·xT``,
  * ``bicubic``:  Catmull-Rom through 4 anchors ``(x_{-12}, x0, x12, x24)``
    when the surrounding hours exist in the year, otherwise falls back
    to the bilinear value for that sample (mirrors the convention used
    by the 6h eval where 2-anchor cubic collapses to linear).

JSON output schema::

    {
      "checkpoint": "...",
      "model_type": "...",
      "delta_t_hours": 12.0,
      "num_samples": N,
      "channel_names": [24 strings],
      "per_tau": {
        "1":  {"model": {"rmse_norm_T1000": ..., "rmse_phys_T1000": ...,
                          "acc_T1000": ...},
                "bilinear": {...},
                "bicubic":  {...}},
        ...
        "11": {...}
      },
      "seen_tau":   [1, 2, 3, 5, 7, 9, 10, 11],
      "unseen_tau": [4, 6, 8]
    }

Output path: ``metrics/eval_12h_2020_pre_ep10/<model_name>.json``.

This script **does not** modify ``tools/eval/batch_eval_memmap.py`` (the
6h driver) — it is a separate, parallel entrypoint.

Usage example::

    python tools/eval/batch_eval_12h_memmap.py \
        --memmap-dir /workspace/code/wti/cache/wb2_0p5_cache \
        --test-year 2020 \
        --climatology /workspace/code/wti/data/climatology_1990-2019_0p5_canonical_v2.zarr \
        --models WeatherDCAE:logs/.../last.ckpt,DCAE_Skip:logs/.../last.ckpt \
        --out-dir metrics/eval_12h_2020_pre_ep10 \
        --paper-tag 12h_2020
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.eval.climatology import climatology_time_weights
from tools.eval.climatology import validate_climatology_archive
from weather_time_interp.metrics.physical_consistency import (
    DIAGNOSTIC_COMPONENTS,
    physical_diagnostics,
    validate_physical_channel_order,
)
from weather_time_interp.normalization import (
    STATIC_FEATURES_3,
    file_provenance,
    static_feature_provenance,
    zarr_store_provenance,
)

# Heavy deps are imported lazily inside helpers / main() so this module can
# be imported (and ``--help`` shown) without xarray / torch.utils.data /
# the trainer stack present.
try:  # pragma: no cover - import-time guard
    import xarray as xr  # type: ignore
except ImportError:  # noqa: WPS440
    xr = None  # type: ignore[assignment]

try:  # pragma: no cover
    from weather_time_interp.eval_datasets import (  # type: ignore
        ERA5WeatherHermiteDataset,
    )
    from weather_time_interp.eval_runner import (  # type: ignore
        BatchContext,
        Config,
        EvaluationRunner,
        PerHourAccums,
    )
    from weather_time_interp.memmap_dataset import ERA5MemmapDataset  # type: ignore
except ImportError as _imp_err:  # noqa: WPS440
    # Stash so any attempt to actually *run* the script raises with a clear
    # message, but ``--help`` and class-import-time definitions below can
    # still work via stub base classes.
    _DEFERRED_IMPORT_ERROR: Optional[ImportError] = _imp_err  # type: ignore[assignment]

    class _StubBase:  # noqa: D401
        """Placeholder base class used when heavy deps are missing.

        Instantiating it (e.g. in ``main()``) raises the original ImportError.
        Subclassing is fine — that's all we need so module import succeeds.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise _DEFERRED_IMPORT_ERROR  # type: ignore[misc]

    BatchContext = _StubBase  # type: ignore[assignment]
    Config = _StubBase  # type: ignore[assignment]
    EvaluationRunner = _StubBase  # type: ignore[assignment]
    PerHourAccums = _StubBase  # type: ignore[assignment]
    ERA5WeatherHermiteDataset = _StubBase  # type: ignore[assignment]
    ERA5MemmapDataset = _StubBase  # type: ignore[assignment]
else:
    _DEFERRED_IMPORT_ERROR = None

# Lazy-imported inside the model entrypoint so the script can be imported
# (and ``--help`` shown) without the full trainer dependency stack:
#   from trainer_weather_hermite import WeatherHermiteLightningModule


# ---------------------------------------------------------------------------- #
# Constants                                                                    #
# ---------------------------------------------------------------------------- #


# Held-out continuous-τ values for the H2 hypothesis (Fig 9).
# These are NOT seen during training: the dataset's ``eval_hours`` argument
# decides which τ are emitted by the loader. SEEN are the τ for which the
# model received supervision at training time.
SEEN_TAU_12H: Tuple[int, ...] = (1, 2, 3, 5, 7, 9, 10, 11)
UNSEEN_TAU_12H: Tuple[int, ...] = (4, 6, 8)

# Channel names used to look up climatology variables (PL only).
# Surface variables are looked up by their literal name.
_CLIM_PL_PREFIXES = {"T": "t", "U": "u", "V": "v", "Q": "q", "Z": "z"}


# ---------------------------------------------------------------------------- #
# Helpers: bicubic 4-anchor Catmull-Rom interpolation                          #
# ---------------------------------------------------------------------------- #


def _catmull_rom_4anchor(
    x_prev: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    x_next: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Catmull-Rom cubic through 4 anchors at parameter ``tau ∈ [0, 1]``.

    Uses the standard centripetal-free Catmull-Rom form
    (the most common bicubic-in-time fallback when only one segment is
    being interpolated). Tensors have shape ``(B, C, H, W)``; ``tau`` is
    a 1-D ``(B,)`` tensor.
    """
    t = tau.float().view(-1, 1, 1, 1)
    t2 = t * t
    t3 = t2 * t
    # Catmull-Rom basis (m_i form):
    #   p(t) = 0.5 * (
    #     ( -t3 + 2 t2 -  t) * P_{-1}
    #   + (3 t3 - 5 t2     + 2) * P_0
    #   + (-3 t3 + 4 t2 +  t) * P_1
    #   + (  t3 -   t2       ) * P_2
    #   )
    b_prev = (-t3 + 2.0 * t2 - t)
    b_0 = (3.0 * t3 - 5.0 * t2 + 2.0)
    b_1 = (-3.0 * t3 + 4.0 * t2 + t)
    b_next = (t3 - t2)
    return 0.5 * (b_prev * x_prev + b_0 * x0 + b_1 * x1 + b_next * x_next)


def temporal_curvature_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitude_weights: torch.Tensor,
) -> torch.Tensor:
    """Return per-window/channel error in the discrete temporal curvature.

    ``prediction`` and ``target`` contain an equally spaced trajectory with
    shape ``(time, batch, channel, latitude, longitude)``. The metric compares
    their second temporal differences, averages over every valid centre time,
    and applies the declared spherical strip-area spatial mean.
    """
    if prediction.shape != target.shape or prediction.dim() != 5:
        raise ValueError(
            "temporal trajectories must have one matching 5-D shape"
        )
    if prediction.size(0) < 3:
        raise ValueError("temporal curvature requires at least three times")
    if latitude_weights.shape != (1, 1, prediction.size(-2), 1):
        raise ValueError("latitude weights do not match the trajectory grid")
    prediction_curvature = (
        prediction[:-2] - 2.0 * prediction[1:-1] + prediction[2:]
    )
    target_curvature = target[:-2] - 2.0 * target[1:-1] + target[2:]
    squared_error = (prediction_curvature - target_curvature).square()
    spatial_mse = (
        (squared_error * latitude_weights).sum(dim=(-2, -1))
        / float(prediction.size(-1))
    )
    return spatial_mse.mean(dim=0)


# ---------------------------------------------------------------------------- #
# Climatology lookup (subset / lightweight)                                    #
# ---------------------------------------------------------------------------- #


class _ClimatologyLookup:
    """Minimal climatology lookup, mirrors ``tools/eval/compute_acc.ClimatologyLookup``.

    Inlined here so this script does not import from the heavier ``compute_acc``
    module (which pulls the full trainer stack via its top-level import).
    Behaviour MUST stay bit-identical for the channels both modules support.
    """

    def __init__(
        self,
        path: Path,
        channel_names: Sequence[str],
        device: torch.device,
        lazy: bool = False,
        cache_dir: Optional[Path] = None,
        store_identity: str = "",
    ) -> None:
        self.device = device
        self.lazy = lazy
        ds = xr.open_zarr(str(path), consolidated=True)
        self.archive_provenance = validate_climatology_archive(ds, path)
        H = int(ds.sizes["latitude"])
        W = int(ds.sizes["longitude"])
        n_hour = int(ds.sizes["hour"])
        n_doy = int(ds.sizes["dayofyear"])
        self.n_doy = n_doy
        n_c = len(channel_names)
        self.storage_shape = (n_c, n_hour, n_doy, H, W)
        has_levels = "level" in ds.coords
        src_levels: List[int] = (
            list(map(int, ds["level"].values)) if has_levels else []
        )
        sources: List[Tuple[str, Optional[int]]] = []
        for name in channel_names:
            if (
                len(name) > 1
                and name[0] in _CLIM_PL_PREFIXES
                and name[1:].isdigit()
            ):
                if not has_levels:
                    raise ValueError(
                        f"climatology lacks 'level' dim, cannot map PL ch '{name}'"
                    )
                short = _CLIM_PL_PREFIXES[name[0]]
                lvl = int(name[1:])
                if lvl not in src_levels:
                    raise ValueError(
                        f"climatology missing level {lvl} for '{name}'"
                    )
                lvl_idx = src_levels.index(lvl)
                sources.append((short, lvl_idx))
            else:
                if name not in ds.data_vars:
                    raise ValueError(
                        f"climatology missing surface var '{name}'"
                    )
                sources.append((name, None))

        if lazy:
            self._ds = ds
            self._sources = sources
            self.clim: Optional[torch.Tensor] = None
            self._slot_cache: OrderedDict[Tuple[int, int], torch.Tensor] = (
                OrderedDict()
            )
            self._cache_dir = cache_dir
            if self._cache_dir is not None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
            channel_key = (
                f"{','.join(channel_names)}|{store_identity}"
            ).encode("utf-8")
            self._cache_key = hashlib.sha1(channel_key).hexdigest()[:10]
            return

        ds.load()
        arr = np.zeros(self.storage_shape, dtype=np.float32)
        for ci, (var_name, level_idx) in enumerate(sources):
            field = ds[var_name]
            if level_idx is not None:
                field = field.isel(level=level_idx)
            arr[ci] = field.values
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            mean_per_ch = np.nanmean(arr.reshape(n_c, -1), axis=1)
            for ci in range(n_c):
                arr[ci] = np.where(np.isnan(arr[ci]), mean_per_ch[ci], arr[ci])
        # Keep on CPU; transfer per-sample (the 0.5° array is ~tens of GiB).
        self.clim = torch.from_numpy(arr)
        self._ds = None
        self._sources = sources
        ds.close()

    def _read_lazy_slot(self, hour_idx: int, doy_idx: int) -> torch.Tensor:
        """Read one (hour, day-of-year) field for every requested channel."""
        cache_key = (hour_idx, doy_idx)
        cached = self._slot_cache.get(cache_key)
        if cached is not None:
            self._slot_cache.move_to_end(cache_key)
            return cached

        assert self._ds is not None
        _, n_hour, _, height, width = self.storage_shape
        cache_path = None
        if self._cache_dir is not None:
            cache_path = (
                self._cache_dir
                / f"climatology_{self._cache_key}_doy{doy_idx + 1:03d}.npy"
            )
        if cache_path is not None and cache_path.exists():
            cached_day = np.load(cache_path, mmap_mode="r")
            expected_shape = (n_hour, len(self._sources), height, width)
            if cached_day.shape != expected_shape:
                raise ValueError(
                    f"invalid climatology cache {cache_path}: "
                    f"{cached_day.shape} != {expected_shape}"
                )
            self._slot_cache.clear()
            for hi in range(n_hour):
                self._slot_cache[(hi, doy_idx)] = torch.from_numpy(
                    np.array(cached_day[hi], copy=True)
                )
            return self._slot_cache[cache_key]

        arr = np.empty(
            (n_hour, len(self._sources), height, width),
            dtype=np.float32,
        )
        fields: Dict[str, np.ndarray] = {}
        for var_name, _ in self._sources:
            if var_name not in fields:
                field = self._ds[var_name].isel(dayofyear=doy_idx)
                if "level" in field.dims:
                    field = field.transpose(
                        "hour", "level", "latitude", "longitude"
                    )
                else:
                    field = field.transpose("hour", "latitude", "longitude")
                fields[var_name] = np.asarray(field.values, dtype=np.float32)
        for ci, (var_name, level_idx) in enumerate(self._sources):
            values = fields[var_name]
            if level_idx is not None:
                values = values[:, level_idx]
            for hi in range(n_hour):
                slot = values[hi]
                if np.isnan(slot).any():
                    finite_mean = np.nanmean(slot)
                    fill_value = (
                        float(finite_mean) if np.isfinite(finite_mean) else 0.0
                    )
                    slot = np.nan_to_num(slot, nan=fill_value)
                arr[hi, ci] = slot

        if cache_path is not None:
            tmp_path = cache_path.with_name(
                f".{cache_path.stem}.{os.getpid()}.tmp.npy"
            )
            np.save(tmp_path, arr)
            tmp_path.replace(cache_path)

        # The evaluator walks dates chronologically. Retaining one complete
        # day minimises zarr reads while keeping the cache near 100 MiB.
        self._slot_cache.clear()
        for hi in range(n_hour):
            self._slot_cache[(hi, doy_idx)] = torch.from_numpy(arr[hi])
        return self._slot_cache[cache_key]

    def lookup(
        self,
        doy: int,
        hour_frac: float,
        year: int,
    ) -> torch.Tensor:
        """Linear interp between the 4 climatology slots (every 6h)."""
        h0_idx, h1_idx, doy0_idx, doy1_idx, w = (
            climatology_time_weights(
                day_of_year=doy,
                hour=hour_frac,
                year=year,
                climatology_days=self.n_doy,
            )
        )
        if self.lazy:
            c0 = self._read_lazy_slot(h0_idx, doy0_idx)
            c1 = self._read_lazy_slot(h1_idx, doy1_idx)
            return (1.0 - w) * c0 + w * c1
        assert self.clim is not None
        c0 = self.clim[:, h0_idx, doy0_idx]
        c1 = self.clim[:, h1_idx, doy1_idx]
        return (1.0 - w) * c0 + w * c1  # (C, H, W) on CPU


# ---------------------------------------------------------------------------- #
# 12h runner                                                                   #
# ---------------------------------------------------------------------------- #


class BatchModelRunner12h(EvaluationRunner):
    """Per-τ RMSE (norm + phys) + ACC for one model + bilinear + bicubic.

    Built on :class:`EvaluationRunner` (Phase 0 base class). The base class
    iterates the loader once; we override :meth:`_process_batch` to run the
    model once per (batch, h_idx) and collect the three accumulators.

    Channels reported in JSON: all 24 ``channel_names`` from the dataset.
    Channels included in ACC: all channels that the climatology actually
    provides (NaN-safe; ``tisr`` and any missing var are reported as 0 ACC).
    """

    def __init__(
        self,
        cfg: Config,
        *,
        model: torch.nn.Module,
        model_type: str,
        ckpt_path: str,
        climatology: Optional["_ClimatologyLookup"],
        acc_indices: Sequence[int],
        seen_tau: Sequence[int] = SEEN_TAU_12H,
        unseen_tau: Sequence[int] = UNSEEN_TAU_12H,
        keep_n_channels: Optional[int] = None,
        input_n_channels: Optional[int] = None,
        proper_rmse: bool = False,
        save_window_metrics: bool = False,
        save_physical_metrics: bool = False,
        save_temporal_metrics: bool = False,
    ) -> None:
        super().__init__(cfg)
        self.model = model
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.climatology = climatology  # None => skip ACC
        self.acc_indices = list(int(i) for i in acc_indices)
        self.seen_tau = list(seen_tau)
        self.unseen_tau = list(unseen_tau)
        # When set (e.g. 24), x0/xT/target tensors are sliced to first N channels
        # before forward + bicubic anchor reads — supports keep_24ch trained ckpts.
        self.keep_n_channels = keep_n_channels
        self.input_n_channels = input_n_channels
        self.proper_rmse = bool(proper_rmse)
        self.save_window_metrics = bool(save_window_metrics)
        self.save_physical_metrics = bool(save_physical_metrics)
        self.save_temporal_metrics = bool(save_temporal_metrics)
        if self.save_physical_metrics and not self.save_window_metrics:
            raise ValueError("physical metrics require save_window_metrics=True")
        if self.save_temporal_metrics and not self.save_window_metrics:
            raise ValueError("temporal metrics require save_window_metrics=True")

        # Accumulators (filled in run()).
        self._rmse_norm_sum_sq: Dict[str, Dict[int, Dict[str, float]]] = {}
        self._rmse_phys_sum_sq: Dict[str, Dict[int, Dict[str, float]]] = {}
        self._n_per_tau: Dict[int, int] = {}

        self._acc_sxy: Dict[str, Dict[int, torch.Tensor]] = {}
        self._acc_sxx: Dict[str, Dict[int, torch.Tensor]] = {}
        self._acc_syy: Dict[str, Dict[int, torch.Tensor]] = {}
        self._window_year: List[int] = []
        self._window_t0: List[int] = []
        self._window_tau: List[int] = []
        self._window_mse_norm: Dict[str, List[np.ndarray]] = {}
        self._window_acc: Dict[str, List[np.ndarray]] = {}
        self._window_physical: Dict[
            str,
            Dict[str, List[np.ndarray]],
        ] = {}
        self._temporal_year: List[int] = []
        self._temporal_t0: List[int] = []
        self._temporal_center_count: List[int] = []
        self._temporal_curvature_mse: Dict[str, List[np.ndarray]] = {}
        self._temporal_centers: tuple[int, ...] | None = None

    # --- EvaluationRunner abstract API ----------------------------------- #

    def method_names(self) -> List[str]:
        return ["model", "bilinear", "bicubic"]

    def predict(
        self,
        method: str,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau_h: torch.Tensor,
        *,
        batch_ctx: BatchContext,
    ) -> torch.Tensor:
        # Predictions are produced directly in ``_process_batch`` to avoid
        # re-calling the model from this method (it depends on extras the
        # base class doesn't expose). Returning a zero tensor is safe — the
        # base reduction loop is **not** used by this runner (we override
        # _process_batch wholesale).
        return torch.zeros_like(x0)

    def output_payload(self, method: str, accums: PerHourAccums) -> Dict[str, Any]:
        return {}

    # --- override _new_accums and _process_batch ------------------------- #

    def _init_accumulators(self) -> None:
        methods = self.method_names()
        tau_set = list(self.cfg.eval_hours)
        # Reset all per-method dicts. We assign fresh dicts rather than
        # mutating in-place so the method also works when the instance was
        # created via ``__new__`` (e.g. in unit tests).
        self._rmse_norm_sum_sq = {}
        self._rmse_phys_sum_sq = {}
        self._acc_sxy = {}
        self._acc_sxx = {}
        self._acc_syy = {}
        self._window_year = []
        self._window_t0 = []
        self._window_tau = []
        self._window_mse_norm = {method: [] for method in methods}
        self._window_acc = {method: [] for method in methods}
        self._window_physical = {
            method: {
                diagnostic: []
                for diagnostic in DIAGNOSTIC_COMPONENTS
            }
            for method in ("model", "bilinear")
        }
        self._temporal_year = []
        self._temporal_t0 = []
        self._temporal_center_count = []
        self._temporal_curvature_mse = {
            method: [] for method in ("model", "bilinear")
        }
        self._temporal_centers = None
        for m in methods:
            self._rmse_norm_sum_sq[m] = {tau: {} for tau in tau_set}
            self._rmse_phys_sum_sq[m] = {tau: {} for tau in tau_set}
            self._acc_sxy[m] = {
                tau: torch.zeros(
                    len(self.acc_indices), dtype=torch.float64, device=self.device
                )
                for tau in tau_set
            }
            self._acc_sxx[m] = {
                tau: torch.zeros_like(self._acc_sxy[m][tau]) for tau in tau_set
            }
            self._acc_syy[m] = {
                tau: torch.zeros_like(self._acc_sxy[m][tau]) for tau in tau_set
            }
        self._n_per_tau = {tau: 0 for tau in tau_set}

    # --- main loop ------------------------------------------------------- #

    def run(self) -> Dict[str, Any]:
        t_global = time.time()
        self._setup_data()
        if self.save_physical_metrics:
            validate_physical_channel_order(list(self.channel_names))
            if (
                self.keep_n_channels is not None
                and self.keep_n_channels < 24
            ):
                raise ValueError(
                    "physical metrics require at least 24 retained channels"
                )
        # If keep_n_channels is set, drop ACC indices that would point past the
        # sliced tensor (the climatology channels for sst/tcc/tcwv are dropped).
        # Must happen BEFORE _init_accumulators so acc accumulator dims match.
        if self.keep_n_channels is not None:
            kept = [i for i in self.acc_indices if i < self.keep_n_channels]
            if len(kept) != len(self.acc_indices):
                self.acc_indices = kept
        self._init_accumulators()
        print(f"setup done in {time.time() - t_global:.1f}s")

        # Wall-clock & sample counters for the model.
        wrapped_ds = self.test_wrapped
        base_ds = self.ds_base
        grouped_idx = getattr(wrapped_ds, "_grouped_indices", None)

        acc_idx_t = torch.tensor(
            self.acc_indices, device=self.device, dtype=torch.long
        )
        mu = self.mu_all
        sigma = self.sigma_all
        cond_value = float(self.cfg.dt_hours)
        max_tau = float(self.cfg.max_tau_hours)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.loader):
                self._process_one_batch(
                    batch=batch,
                    batch_idx=batch_idx,
                    wrapped_ds=wrapped_ds,
                    base_ds=base_ds,
                    grouped_idx=grouped_idx,
                    acc_idx_t=acc_idx_t,
                    mu=mu,
                    sigma=sigma,
                    cond_value=cond_value,
                    max_tau=max_tau,
                )

        payload = self._finalize_payload()
        print(f"\n=== DONE in {(time.time() - t_global) / 60:.1f} min ===")
        return payload

    # ------------------------------------------------------------------ #

    def _process_one_batch(
        self,
        *,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        wrapped_ds: ERA5WeatherHermiteDataset,
        base_ds: ERA5MemmapDataset,
        grouped_idx: Optional[List[List[int]]],
        acc_idx_t: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        cond_value: float,
        max_tau: float,
    ) -> None:
        device = self.device
        x0 = batch["x0"].to(device, non_blocking=True)
        xT = batch["xT"].to(device, non_blocking=True)
        tau_hour_all = batch["tau_hour"].long()  # (B, nH, 1)
        target_all = batch["target"].to(device, non_blocking=True)  # (B, nH, C, H, W)
        static = batch.get("static")
        if static is not None:
            static = static.to(device, non_blocking=True)

        model_x0 = x0
        model_xT = xT
        if self.input_n_channels is not None:
            model_x0 = x0[:, : self.input_n_channels].contiguous()
            model_xT = xT[:, : self.input_n_channels].contiguous()

        # If model was trained with keep_24ch=true the input/target tensors must
        # be sliced to the first N channels (sst/tcc/tcwv are dropped).
        if self.keep_n_channels is not None and x0.size(1) > self.keep_n_channels:
            n = self.keep_n_channels
            x0 = x0[:, :n].contiguous()
            xT = xT[:, :n].contiguous()
            target_all = target_all[:, :, :n].contiguous()
            # Models without an explicit wider input contract were trained on
            # the same retained channels as the target. DCAE-style models set
            # ``input_n_channels`` and intentionally keep their wider input.
            if self.input_n_channels is None:
                model_x0 = x0
                model_xT = xT

        B = x0.size(0)
        nH = tau_hour_all.size(1)
        cond = torch.full((B,), cond_value, device=device, dtype=torch.float32)
        channel_names = self.channel_names
        if self.keep_n_channels is not None:
            n = self.keep_n_channels
            channel_names = channel_names[:n]
            mu = mu[:, :n]
            sigma = sigma[:, :n]

        temporal_predictions: List[torch.Tensor] = []
        temporal_targets: List[torch.Tensor] = []
        for h_idx in range(nH):
            tau_hour_h = tau_hour_all[:, h_idx, 0]  # (B,) ints
            target_h = target_all[:, h_idx]  # (B, C, H, W)

            # Recompute tau from tau_hour against the 12h window: the dataset
            # stores ``tau_norm = tau_h / HOURS_PER_TAU_UNIT (=6)`` which is
            # incorrect for 12h windows (overshoots [0, 1]). FIXME(HOURS_PER_TAU):
            # ``weather_time_interp/config.py`` hardcodes the divisor at 6.0 —
            # this script sidesteps the issue by recomputing tau on the fly.
            tau_h = tau_hour_h.float().to(device) / max_tau  # (B,)

            # Predictions:
            #   - bilinear: closed-form 2-anchor cubic
            #   - bicubic:  4-anchor Catmull-Rom (falls back to bilinear when
            #               the surrounding hours don't exist in the year)
            pred_bil = _bilinear_2anchor(x0, xT, tau_h)
            pred_bic = self._bicubic_4anchor(
                x0=x0,
                xT=xT,
                tau_h=tau_h,
                tau_hour_h=tau_hour_h,
                base_ds=base_ds,
                wrapped_ds=wrapped_ds,
                grouped_idx=grouped_idx,
                batch_idx=batch_idx,
                mu=mu,
                sigma=sigma,
            )
            pred_model = self._forward_model(
                x0=model_x0,
                xT=model_xT,
                tau_h=tau_h,
                cond=cond,
                static=static,
            )
            if self.keep_n_channels is not None:
                pred_model = pred_model[:, : self.keep_n_channels].contiguous()
            if self.save_temporal_metrics:
                temporal_predictions.append(pred_model)
                temporal_targets.append(target_h)

            # Per-sample reduction. ``w_lat`` sums to one over latitude, so a
            # proper spatial mean also divides the longitude sum by W. The
            # legacy mode omits that division for old cache compatibility.
            w_lat = self.w_lat  # (1, 1, H, 1)
            spatial_scale = 1.0 / float(self.W) if self.proper_rmse else 1.0
            err_m_lat = (
                ((pred_model - target_h) ** 2 * w_lat).sum(dim=(-2, -1))
                * spatial_scale
            )
            err_b_lat = (
                ((pred_bil - target_h) ** 2 * w_lat).sum(dim=(-2, -1))
                * spatial_scale
            )
            err_c_lat = (
                ((pred_bic - target_h) ** 2 * w_lat).sum(dim=(-2, -1))
                * spatial_scale
            )

            # Physical-unit RMSE.
            tgt_phys = target_h * sigma + mu
            mdl_phys = pred_model * sigma + mu
            bil_phys = pred_bil * sigma + mu
            bic_phys = pred_bic * sigma + mu
            err_m_phys = (
                ((mdl_phys - tgt_phys) ** 2 * w_lat).sum(dim=(-2, -1))
                * spatial_scale
            )
            err_b_phys = (
                ((bil_phys - tgt_phys) ** 2 * w_lat).sum(dim=(-2, -1))
                * spatial_scale
            )
            err_c_phys = (
                ((bic_phys - tgt_phys) ** 2 * w_lat).sum(dim=(-2, -1))
                * spatial_scale
            )
            physical_values: Dict[str, Dict[str, torch.Tensor]] = {}
            if self.save_physical_metrics:
                physical_values = {
                    "model": physical_diagnostics(mdl_phys, tgt_phys),
                    "bilinear": physical_diagnostics(bil_phys, tgt_phys),
                }

            # Anomalies on ACC channels.
            tgt_phys_a = tgt_phys.index_select(1, acc_idx_t)
            mdl_phys_a = mdl_phys.index_select(1, acc_idx_t)
            bil_phys_a = bil_phys.index_select(1, acc_idx_t)
            bic_phys_a = bic_phys.index_select(1, acc_idx_t)

            tau_hours_per_sample = tau_hour_h.tolist()
            for i in range(B):
                tau_i = int(tau_hours_per_sample[i])
                if tau_i not in self._n_per_tau:
                    continue
                wrapped_index = batch_idx * self.loader.batch_size + i
                if wrapped_index >= len(wrapped_ds):
                    break
                year, t0 = _extract_year_t0(
                    wrapped_index=wrapped_index,
                    grouped_idx=grouped_idx,
                    base_ds=base_ds,
                )
                window_acc_values = {
                    method: np.full(len(self.acc_indices), np.nan, dtype=np.float32)
                    for method in self.method_names()
                }
                if self.climatology is not None:
                    ts = base_ds.time_starts[year] + timedelta(hours=int(t0 + tau_i))
                    doy = ts.timetuple().tm_yday
                    hf = ts.hour + ts.minute / 60.0
                    clim_at_t = self.climatology.lookup(
                        doy,
                        hf,
                        ts.year,
                    ).unsqueeze(0).to(device)

                    t_an = tgt_phys_a[i : i + 1] - clim_at_t
                    m_an = mdl_phys_a[i : i + 1] - clim_at_t
                    b_an = bil_phys_a[i : i + 1] - clim_at_t
                    c_an = bic_phys_a[i : i + 1] - clim_at_t

                    self._acc_sxy["model"][tau_i] += (w_lat * m_an * t_an).sum(
                        dim=(0, 2, 3)
                    ).double()
                    self._acc_sxx["model"][tau_i] += (w_lat * m_an * m_an).sum(
                        dim=(0, 2, 3)
                    ).double()
                    self._acc_syy["model"][tau_i] += (w_lat * t_an * t_an).sum(
                        dim=(0, 2, 3)
                    ).double()
                    self._acc_sxy["bilinear"][tau_i] += (w_lat * b_an * t_an).sum(
                        dim=(0, 2, 3)
                    ).double()
                    self._acc_sxx["bilinear"][tau_i] += (w_lat * b_an * b_an).sum(
                        dim=(0, 2, 3)
                    ).double()
                    self._acc_syy["bilinear"][tau_i] += (w_lat * t_an * t_an).sum(
                        dim=(0, 2, 3)
                    ).double()
                    self._acc_sxy["bicubic"][tau_i] += (w_lat * c_an * t_an).sum(
                        dim=(0, 2, 3)
                    ).double()
                    self._acc_sxx["bicubic"][tau_i] += (w_lat * c_an * c_an).sum(
                        dim=(0, 2, 3)
                    ).double()
                    self._acc_syy["bicubic"][tau_i] += (w_lat * t_an * t_an).sum(
                        dim=(0, 2, 3)
                    ).double()
                    for method, anomaly in (
                        ("model", m_an),
                        ("bilinear", b_an),
                        ("bicubic", c_an),
                    ):
                        numerator = (w_lat * anomaly * t_an).sum(dim=(0, 2, 3))
                        denominator = torch.sqrt(
                            (w_lat * anomaly * anomaly).sum(dim=(0, 2, 3))
                            * (w_lat * t_an * t_an).sum(dim=(0, 2, 3))
                            + 1e-12
                        )
                        window_acc_values[method] = (
                            (numerator / denominator).float().cpu().numpy()
                        )
                self._n_per_tau[tau_i] += 1
                if self.save_window_metrics:
                    self._window_year.append(year)
                    self._window_t0.append(t0)
                    self._window_tau.append(tau_i)
                    for method, error in (
                        ("model", err_m_lat),
                        ("bilinear", err_b_lat),
                        ("bicubic", err_c_lat),
                    ):
                        self._window_mse_norm[method].append(
                            error[i].float().cpu().numpy()
                        )
                        self._window_acc[method].append(window_acc_values[method])
                    if self.save_physical_metrics:
                        for method, diagnostics in physical_values.items():
                            for diagnostic, values in diagnostics.items():
                                self._window_physical[method][diagnostic].append(
                                    values[i].cpu().numpy().astype(np.float32)
                                )

                for ci, name in enumerate(channel_names):
                    self._rmse_norm_sum_sq["model"][tau_i].setdefault(name, 0.0)
                    self._rmse_norm_sum_sq["model"][tau_i][name] += float(
                        err_m_lat[i, ci].item()
                    )
                    self._rmse_norm_sum_sq["bilinear"][tau_i].setdefault(name, 0.0)
                    self._rmse_norm_sum_sq["bilinear"][tau_i][name] += float(
                        err_b_lat[i, ci].item()
                    )
                    self._rmse_norm_sum_sq["bicubic"][tau_i].setdefault(name, 0.0)
                    self._rmse_norm_sum_sq["bicubic"][tau_i][name] += float(
                        err_c_lat[i, ci].item()
                    )
                    self._rmse_phys_sum_sq["model"][tau_i].setdefault(name, 0.0)
                    self._rmse_phys_sum_sq["model"][tau_i][name] += float(
                        err_m_phys[i, ci].item()
                    )
                    self._rmse_phys_sum_sq["bilinear"][tau_i].setdefault(name, 0.0)
                    self._rmse_phys_sum_sq["bilinear"][tau_i][name] += float(
                        err_b_phys[i, ci].item()
                    )
                    self._rmse_phys_sum_sq["bicubic"][tau_i].setdefault(name, 0.0)
                    self._rmse_phys_sum_sq["bicubic"][tau_i][name] += float(
                        err_c_phys[i, ci].item()
                    )

        if self.save_temporal_metrics:
            self._record_temporal_metrics(
                x0=x0,
                xT=xT,
                predictions=temporal_predictions,
                targets=temporal_targets,
                tau_hour_all=tau_hour_all,
                batch_idx=batch_idx,
                wrapped_ds=wrapped_ds,
                base_ds=base_ds,
                grouped_idx=grouped_idx,
                max_tau=max_tau,
            )

        if batch_idx % 25 == 0:
            print(f"  batch {batch_idx}/{len(self.loader)}")

    # --- model forward --------------------------------------------------- #

    def _record_temporal_metrics(
        self,
        *,
        x0: torch.Tensor,
        xT: torch.Tensor,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        tau_hour_all: torch.Tensor,
        batch_idx: int,
        wrapped_ds: ERA5WeatherHermiteDataset,
        base_ds: ERA5MemmapDataset,
        grouped_idx: Optional[List[List[int]]],
        max_tau: float,
    ) -> None:
        """Record one trajectory-curvature score per anchor window."""
        width = int(round(max_tau))
        if width < 2 or float(width) != float(max_tau):
            raise ValueError("temporal metric requires an integer window width")
        if len(predictions) != tau_hour_all.size(1) or len(targets) != len(
            predictions
        ):
            raise ValueError("temporal trajectory collection is incomplete")
        tau_matrix = tau_hour_all[:, :, 0].cpu().numpy()
        if not np.all(tau_matrix == tau_matrix[:1]):
            raise ValueError(
                "temporal metric requires a common tau schedule per batch"
            )
        tau_values = [int(value) for value in tau_matrix[0]]
        if len(set(tau_values)) != len(tau_values):
            raise ValueError("temporal metric received duplicate tau values")

        prediction_by_tau = {0: x0, width: xT}
        target_by_tau = {0: x0, width: xT}
        for tau, prediction, target in zip(
            tau_values,
            predictions,
            targets,
        ):
            prediction_by_tau[tau] = prediction
            target_by_tau[tau] = target
        expected_taus = list(range(width + 1))
        if sorted(prediction_by_tau) != expected_taus:
            raise ValueError(
                "temporal metric requires every interior integer tau"
            )

        prediction_sequence = torch.stack(
            [prediction_by_tau[tau] for tau in expected_taus]
        )
        target_sequence = torch.stack(
            [target_by_tau[tau] for tau in expected_taus]
        )
        time = prediction_sequence.new_tensor(expected_taus).view(
            width + 1,
            1,
            1,
            1,
            1,
        ) / float(width)
        bilinear_sequence = (
            (1.0 - time) * x0.unsqueeze(0) + time * xT.unsqueeze(0)
        )
        scores = {
            "model": temporal_curvature_mse(
                prediction_sequence,
                target_sequence,
                self.w_lat,
            ),
            "bilinear": temporal_curvature_mse(
                bilinear_sequence,
                target_sequence,
                self.w_lat,
            ),
        }
        centers = tuple(range(1, width))
        if self._temporal_centers is None:
            self._temporal_centers = centers
        elif self._temporal_centers != centers:
            raise ValueError("temporal centre schedule changed between batches")

        for index in range(x0.size(0)):
            wrapped_index = batch_idx * self.loader.batch_size + index
            if wrapped_index >= len(wrapped_ds):
                break
            year, t0 = _extract_year_t0(
                wrapped_index=wrapped_index,
                grouped_idx=grouped_idx,
                base_ds=base_ds,
            )
            self._temporal_year.append(year)
            self._temporal_t0.append(t0)
            self._temporal_center_count.append(len(centers))
            for method, values in scores.items():
                self._temporal_curvature_mse[method].append(
                    values[index].float().cpu().numpy()
                )

    def _forward_model(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau_h: torch.Tensor,
        cond: torch.Tensor,
        static: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # ATM-VFI (``PixelAttentionVFI``) has a different forward signature:
        # ``model.net(x0, xT, tau)`` over 24ch tensors (no static / cond).
        # Mirrors the dispatch in ``tools/eval/region_season_12h_eval.py``.
        if self.model_type == "atm_vfi_pixel_attn":
            return self.model.net(x0, xT, tau_h)
        out = self.model(x0, xT, tau_h, cond, static=static)
        if isinstance(out, tuple):
            return out[0]
        return out

    # --- bicubic 4-anchor lookup ---------------------------------------- #

    def _bicubic_4anchor(
        self,
        *,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau_h: torch.Tensor,
        tau_hour_h: torch.Tensor,
        base_ds: ERA5MemmapDataset,
        wrapped_ds: ERA5WeatherHermiteDataset,
        grouped_idx: Optional[List[List[int]]],
        batch_idx: int,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Catmull-Rom through ``(x_{-W}, x0, xW, x_{2W})`` with bilinear fallback.

        The dataset only yields ``x0`` and ``xT`` per sample; we read the
        surrounding anchors directly from the memmap. If either anchor lies
        outside the year, we fall back to bilinear for that sample (same
        convention used by the 6h eval when bicubic collapses).
        """
        device = self.device
        max_tau = int(self.cfg.max_tau_hours)
        B = x0.size(0)

        # Per-sample fetch of x_{-W} / x_{2W}.
        n_pl = base_ds.in_channels
        x_prev_list: List[torch.Tensor] = []
        x_next_list: List[torch.Tensor] = []
        use_cubic_mask: List[bool] = []
        for i in range(B):
            wrapped_index = batch_idx * self.loader.batch_size + i
            if wrapped_index >= len(wrapped_ds):
                # Beyond dataset — fallback.
                x_prev_list.append(torch.zeros_like(x0[0]))
                x_next_list.append(torch.zeros_like(x0[0]))
                use_cubic_mask.append(False)
                continue
            year, t0 = _extract_year_t0(
                wrapped_index=wrapped_index,
                grouped_idx=grouped_idx,
                base_ds=base_ds,
            )
            arr = base_ds.memmaps[year]
            T = base_ds.years_T[year]
            t_prev = t0 - max_tau
            t_next = t0 + 2 * max_tau
            if t_prev < 0 or t_next >= T:
                x_prev_list.append(torch.zeros_like(x0[0]))
                x_next_list.append(torch.zeros_like(x0[0]))
                use_cubic_mask.append(False)
                continue
            # The evaluation memmap is read-only.  Materialise writable
            # arrays before handing their storage to PyTorch.
            x_prev_raw = torch.from_numpy(
                np.array(arr[t_prev], dtype=np.float32, order="C", copy=True)
            )
            x_next_raw = torch.from_numpy(
                np.array(arr[t_next], dtype=np.float32, order="C", copy=True)
            )
            x_prev_pl = (x_prev_raw[:n_pl] - base_ds.mu) / base_ds.sigma
            x_next_pl = (x_next_raw[:n_pl] - base_ds.mu) / base_ds.sigma
            x_prev_s = (
                x_prev_raw[n_pl:] - base_ds.surface_mu
            ) / base_ds.surface_sigma
            x_next_s = (
                x_next_raw[n_pl:] - base_ds.surface_mu
            ) / base_ds.surface_sigma
            x_prev_full = torch.cat([x_prev_pl, x_prev_s], dim=0)
            x_next_full = torch.cat([x_next_pl, x_next_s], dim=0)
            x_prev_full = torch.nan_to_num(x_prev_full, nan=0.0, posinf=0.0, neginf=0.0)
            x_next_full = torch.nan_to_num(x_next_full, nan=0.0, posinf=0.0, neginf=0.0)
            # Slice to keep_n_channels if the model was trained with keep_24ch.
            if self.keep_n_channels is not None:
                k = self.keep_n_channels
                x_prev_full = x_prev_full[:k]
                x_next_full = x_next_full[:k]
            x_prev_list.append(x_prev_full.to(device, non_blocking=True))
            x_next_list.append(x_next_full.to(device, non_blocking=True))
            use_cubic_mask.append(True)

        x_prev_b = torch.stack(x_prev_list, dim=0)  # (B, C, H, W)
        x_next_b = torch.stack(x_next_list, dim=0)
        pred_cubic = _catmull_rom_4anchor(x_prev_b, x0, xT, x_next_b, tau_h)
        pred_bilinear = _bilinear_2anchor(x0, xT, tau_h)

        # Mask: where we couldn't fetch anchors, use bilinear.
        mask = torch.tensor(
            use_cubic_mask, device=device, dtype=torch.float32
        ).view(-1, 1, 1, 1)
        return mask * pred_cubic + (1.0 - mask) * pred_bilinear

    # --- finalize -------------------------------------------------------- #

    def _finalize_payload(self) -> Dict[str, Any]:
        eval_tau = list(self.cfg.eval_hours)
        proper_rmse = bool(getattr(self, "proper_rmse", False))
        save_window_metrics = bool(getattr(self, "save_window_metrics", False))
        save_physical_metrics = bool(
            getattr(self, "save_physical_metrics", False)
        )
        save_temporal_metrics = bool(
            getattr(self, "save_temporal_metrics", False)
        )
        channel_names = list(self.channel_names)
        keep_n_channels = getattr(self, "keep_n_channels", None)
        samples_per_date = int(getattr(self.cfg, "samples_per_date", 1))
        eval_days_per_month = getattr(self.cfg, "eval_days_per_month", None)
        eval_days_of_month = getattr(self.cfg, "eval_days_of_month", None)
        if keep_n_channels is not None:
            channel_names = channel_names[:keep_n_channels]
        acc_channel_names = [channel_names[i] for i in self.acc_indices if i < len(channel_names)]
        per_tau_out: Dict[str, Any] = {}
        window_tau = np.asarray(self._window_tau, dtype=np.int16)

        for tau in eval_tau:
            n = self._n_per_tau.get(tau, 0)
            if n == 0:
                continue
            per_tau_out[str(tau)] = {"model": {}, "bilinear": {}, "bicubic": {}}
            for method in self.method_names():
                per_tau_out[str(tau)].setdefault(method, {})
                # RMSE (norm + phys) per channel
                for name in channel_names:
                    if name in self._rmse_norm_sum_sq[method][tau]:
                        rmse_norm = float(
                            np.sqrt(
                                self._rmse_norm_sum_sq[method][tau][name] / n
                            )
                        )
                        rmse_phys = float(
                            np.sqrt(
                                self._rmse_phys_sum_sq[method][tau][name] / n
                            )
                        )
                        per_tau_out[str(tau)][method][f"rmse_norm_{name}"] = (
                            rmse_norm
                        )
                        per_tau_out[str(tau)][method][f"rmse_phys_{name}"] = (
                            rmse_phys
                        )
                # ACC per channel (only if climatology was provided).
                if self.climatology is not None and len(self.acc_indices) > 0:
                    acc = self._acc_sxy[method][tau] / torch.sqrt(
                        self._acc_sxx[method][tau] * self._acc_syy[method][tau]
                        + 1e-12
                    )
                    for ai, name in enumerate(acc_channel_names):
                        per_tau_out[str(tau)][method][f"acc_{name}"] = float(
                            acc[ai].item()
                        )
                    per_tau_out[str(tau)][method]["acc_mean"] = float(acc.mean().item())
                if (
                    save_physical_metrics
                    and method in self._window_physical
                ):
                    tau_mask = window_tau == tau
                    for diagnostic, components in DIAGNOSTIC_COMPONENTS.items():
                        values = np.asarray(
                            self._window_physical[method][diagnostic],
                            dtype=np.float64,
                        )[tau_mask]
                        if values.size == 0:
                            continue
                        per_tau_out[str(tau)][method][
                            f"physical_{diagnostic}"
                        ] = float(values.mean())
                        for component_index, component in enumerate(components):
                            per_tau_out[str(tau)][method][
                                f"physical_{diagnostic}_{component}"
                            ] = float(values[:, component_index].mean())

        payload = {
            "schema_version": 3,
            "checkpoint": self.ckpt_path,
            "model_type": self.model_type,
            "delta_t_hours": float(self.cfg.dt_hours),
            "num_samples": int(sum(self._n_per_tau.values())),
            "n_per_tau": {str(k): int(v) for k, v in self._n_per_tau.items()},
            "years": list(self.cfg.test_years),
            "channel_names": channel_names,
            "acc_channel_names": acc_channel_names,
            "seen_tau": list(self.seen_tau),
            "unseen_tau": list(self.unseen_tau),
            "evaluation_protocol": {
                "rmse_reduction": (
                    "spherical_strip_area_weighted_spatial_mean"
                    if proper_rmse
                    else "legacy_longitude_sum"
                ),
                "samples_per_date": samples_per_date,
                "eval_days_per_month": eval_days_per_month,
                "eval_days_of_month": (
                    None
                    if eval_days_of_month is None
                    else [int(day) for day in eval_days_of_month]
                ),
                "sample_strategy": (
                    "explicit_calendar_days"
                    if eval_days_of_month is not None
                    else (
                        "all_valid_anchor_windows"
                        if eval_days_per_month is None
                        else "legacy_economy_calendar_days"
                    )
                ),
                "full_year": (
                    eval_days_per_month is None
                    and eval_days_of_month is None
                ),
                "eval_hours": list(self.cfg.eval_hours),
                "save_window_metrics": save_window_metrics,
                "save_physical_metrics": save_physical_metrics,
                "save_temporal_metrics": save_temporal_metrics,
                "latitude_grid": getattr(
                    self,
                    "latitude_grid_name",
                    "unspecified",
                ),
            },
            "per_tau": per_tau_out,
        }
        if save_window_metrics:
            index_text = "\n".join(
                f"{year},{t0},{tau}"
                for year, t0, tau in zip(
                    self._window_year,
                    self._window_t0,
                    self._window_tau,
                )
            )
            payload["evaluation_protocol"]["index_sha256"] = hashlib.sha256(
                index_text.encode("utf-8")
            ).hexdigest()
        if save_temporal_metrics:
            if not self._temporal_year or self._temporal_centers is None:
                raise ValueError("temporal metrics were requested but not recorded")
            temporal_index_text = "\n".join(
                f"{year},{t0}"
                for year, t0 in zip(
                    self._temporal_year,
                    self._temporal_t0,
                )
            )
            payload["evaluation_protocol"]["temporal_index_sha256"] = (
                hashlib.sha256(
                    temporal_index_text.encode("utf-8")
                ).hexdigest()
            )
            payload["evaluation_protocol"]["temporal_centers"] = list(
                self._temporal_centers
            )
            payload["temporal_curvature_rmse_norm"] = {
                method: float(
                    np.sqrt(
                        np.asarray(values, dtype=np.float64).mean(axis=0)
                    ).mean()
                )
                for method, values in self._temporal_curvature_mse.items()
            }
        return payload

    def window_metrics_payload(self) -> Dict[str, np.ndarray]:
        """Return paired per-window arrays used for block-bootstrap analysis."""
        if not self.save_window_metrics:
            return {}
        payload: Dict[str, np.ndarray] = {
            "year": np.asarray(self._window_year, dtype=np.int16),
            "t0": np.asarray(self._window_t0, dtype=np.int32),
            "tau": np.asarray(self._window_tau, dtype=np.int8),
            "channel_names": np.asarray(
                list(self.channel_names)[: self.keep_n_channels],
                dtype="U16",
            ) if self.keep_n_channels is not None else np.asarray(
                self.channel_names,
                dtype="U16",
            ),
            "acc_channel_names": np.asarray(
                [
                    self.channel_names[index]
                    for index in self.acc_indices
                    if index < len(self.channel_names)
                ],
                dtype="U16",
            ),
        }
        for method in self.method_names():
            payload[f"mse_norm_{method}"] = np.asarray(
                self._window_mse_norm[method],
                dtype=np.float32,
            )
            payload[f"acc_{method}"] = np.asarray(
                self._window_acc[method],
                dtype=np.float32,
            )
        if getattr(self, "save_physical_metrics", False):
            for method, diagnostics in self._window_physical.items():
                for diagnostic, values in diagnostics.items():
                    payload[f"physical_{diagnostic}_{method}"] = np.asarray(
                        values,
                        dtype=np.float32,
                    )
        if getattr(self, "save_temporal_metrics", False):
            payload["temporal_year"] = np.asarray(
                self._temporal_year,
                dtype=np.int16,
            )
            payload["temporal_t0"] = np.asarray(
                self._temporal_t0,
                dtype=np.int32,
            )
            payload["temporal_center_count"] = np.asarray(
                self._temporal_center_count,
                dtype=np.int8,
            )
            payload["temporal_centers"] = np.asarray(
                self._temporal_centers,
                dtype=np.int8,
            )
            for method, values in self._temporal_curvature_mse.items():
                payload[f"temporal_curvature_mse_{method}"] = np.asarray(
                    values,
                    dtype=np.float32,
                )
        return payload


# ---------------------------------------------------------------------------- #
# Stand-alone helpers used both inside and outside the runner                  #
# ---------------------------------------------------------------------------- #


def _bilinear_2anchor(
    x0: torch.Tensor, x1: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    tau_ = tau.float().view(-1)
    B = x0.size(0)
    if tau_.numel() == 1 and B > 1:
        tau_ = tau_.expand(B)
    tau_b = tau_.view(B, 1, 1, 1)
    return (1.0 - tau_b) * x0 + tau_b * x1


def _extract_year_t0(
    *,
    wrapped_index: int,
    grouped_idx: Optional[List[List[int]]],
    base_ds: ERA5MemmapDataset,
) -> Tuple[int, int]:
    if grouped_idx is not None:
        base_i = grouped_idx[wrapped_index][0]
    else:
        base_i = wrapped_index
    entry = base_ds.index[base_i]
    return int(entry[0]), int(entry[1])


# ---------------------------------------------------------------------------- #
# Model loading                                                                #
# ---------------------------------------------------------------------------- #


def _load_model_safe(
    ckpt_path: str,
    device: torch.device,
    channel_groups: Dict[str, List[int]],
    static_path: str = "data/static_features_0p5.pt",
):
    """Mirrors ``batch_eval_memmap.load_model_safe``.

    Imports are delayed so this module can be imported (``--help``, smoke
    tests against synthetic fixtures) without the Lightning stack.

    ATM-VFI checkpoints (``PixelAttentionVFI``) come from
    ``train_atm_vfi_12h_oddskip.py`` and use a different forward signature
    (``model.net(x0, xT, tau)`` over 24ch tensors). Detection mirrors
    ``tools/eval/region_season_12h_eval.py``: substring match on ckpt path,
    ``state_dict`` ``net.``-prefix fraction, or the ``in_channels`` hparam.
    """
    if ckpt_path.endswith(".ensemble.json"):
        from tools.eval.capmatched_loader import (
            load_capmatched_ensemble_manifest,
        )

        return load_capmatched_ensemble_manifest(
            ckpt_path,
            device,
            static_path=static_path,
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    state = ckpt.get("state_dict", ckpt)
    mt = hparams.get("model_type", "")

    # Capacity-matched checkpoints store the architecture under ``arch`` and
    # almost all weights below ``net.``. Detect them before the ATM-VFI
    # heuristic, whose legacy net-prefix test would otherwise misclassify Flow.
    if hparams.get("arch"):
        from tools.eval.capmatched_loader import load_capmatched_checkpoint

        return load_capmatched_checkpoint(
            ckpt_path,
            device,
            static_path=static_path,
        )

    # --- ATM-VFI detection + load --------------------------------------- #
    state_keys = list(state.keys())
    net_pref_frac = (
        sum(1 for k in state_keys if k.startswith("net.")) / max(1, len(state_keys))
    )
    is_atmvfi = (
        mt.startswith("atm_vfi")
        or "atm_vfi" in ckpt_path.lower()
        or "atmvfi" in ckpt_path.lower()
        or net_pref_frac > 0.95
        or "in_channels" in hparams  # PixelAttentionVFI uses this kwarg
    )
    if is_atmvfi:
        # Reuse the loader helper, including its static-augmented encoder path.
        from tools.eval.batch_eval_memmap import _load_atmvfi_model
        model = _load_atmvfi_model(ckpt_path, state, hparams, device)
        return model, "atm_vfi_pixel_attn"

    if mt == "dcae_adaln_residual_linear":
        from tools.eval.dcae_checkpoint_loader import load_dcae_checkpoint

        return load_dcae_checkpoint(ckpt_path, device)

    # Some evaluation mirrors intentionally omit the retired Hermite model
    # implementation, while the legacy Lightning wrapper still imports its
    # symbol unconditionally. Supply an explicit non-instantiable placeholder
    # so unrelated FuXi/ModAFNO/S-DYff checkpoints remain loadable.
    import weather_time_interp.model.WeatherInterpModel as model_exports

    if not hasattr(model_exports, "WeatherHermiteModel"):
        class _UnavailableLegacyHermite(torch.nn.Module):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__()
                raise RuntimeError(
                    "The retired WeatherHermiteModel implementation is not "
                    "present in this evaluation mirror."
                )

        model_exports.WeatherHermiteModel = _UnavailableLegacyHermite

    from trainer_weather_hermite import WeatherHermiteLightningModule  # type: ignore

    if "dcae" in mt:
        boc = tuple(hparams.get("block_out_channels", ()))
        hparams["block_out_channels"] = (
            boc if len(boc) >= 3 else (128, 256, 512)
        )
        lpb = tuple(hparams.get("layers_per_block", (3, 3, 3)))
        hparams["layers_per_block"] = lpb[:3] if len(lpb) > 3 else (lpb or (3, 3, 3))
        # Legacy compat (mirrors batch_eval_memmap.load_model_safe): some ckpts
        # were trained when the trainer ignored block_out_channels and built
        # models with class defaults. Sniff true widths from state_dict shapes.
        try:
            import re as _re
            idx_chan: dict[int, int] = {}
            for k, v in state.items():
                m = _re.match(r"model\.encoder\.down_blocks\.(\d+)\.conv1\.weight$", k)
                if m and v.dim() == 4:
                    idx_chan[int(m.group(1))] = int(v.shape[0])
                m2 = _re.match(
                    r"model\.encoder\.down_blocks\.(\d+)\.attn\.to_qkv_multiscale\.0\.proj_in\.weight$",
                    k,
                )
                if m2 and v.dim() == 4:
                    idx_chan[int(m2.group(1))] = int(v.shape[0]) // 3
            if idx_chan:
                widths = []
                for idx in sorted(idx_chan):
                    w = idx_chan[idx]
                    if not widths or widths[-1] != w:
                        widths.append(w)
                if (
                    len(widths) >= 3
                    and tuple(widths) != tuple(hparams["block_out_channels"])
                ):
                    print(
                        f"  [legacy-compat] overriding block_out_channels"
                        f" {hparams['block_out_channels']} → {tuple(widths)}"
                    )
                    hparams["block_out_channels"] = tuple(widths)
                    hparams["layers_per_block"] = (2,) * len(widths)
        except Exception as _e:
            print(f"  [legacy-compat] state_dict sniff failed: {_e}")
    # Filter dynamically to drop any ckpt hparams the current trainer signature
    # does not accept (e.g. `freeze_skip_gates`, `freq_cond_film`,
    # `pyramid_levels`, `lambda_pyramid` saved by legacy/experimental launchers).
    import inspect as _inspect
    _accepted = set(_inspect.signature(WeatherHermiteLightningModule.__init__).parameters.keys())
    for k in list(hparams.keys()):
        if k not in _accepted:
            hparams.pop(k, None)
    hparams["channel_groups"] = channel_groups
    model = WeatherHermiteLightningModule(**hparams)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"  state load: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    model.to(device).eval()
    return model, mt


# ---------------------------------------------------------------------------- #
# Climatology probing                                                          #
# ---------------------------------------------------------------------------- #


def _parse_year_window(value: object, *, field_name: str) -> Tuple[int, int]:
    """Parse a strict inclusive ``YYYY-YYYY`` provenance window."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a YYYY-YYYY string")
    parts = value.split("-")
    if (
        len(parts) != 2
        or any(len(part) != 4 or not part.isdigit() for part in parts)
    ):
        raise ValueError(f"{field_name} must be a YYYY-YYYY string")
    start_year, end_year = map(int, parts)
    if start_year > end_year:
        raise ValueError(f"{field_name} starts after it ends: {value}")
    return start_year, end_year


def validate_climatology_window(
    attrs: Mapping[str, object],
    *,
    evaluation_years: Sequence[int],
    expected_window: str,
) -> Dict[str, Any]:
    """Fail closed unless the declared climatology predates evaluation."""
    years = sorted({int(year) for year in evaluation_years})
    if not years:
        raise ValueError("evaluation_years must not be empty")
    expected_start, expected_end = _parse_year_window(
        expected_window,
        field_name="expected climatology window",
    )
    declared = attrs.get("climatology_window")
    declared_start, declared_end = _parse_year_window(
        declared,
        field_name="climatology_window attribute",
    )
    if (declared_start, declared_end) != (expected_start, expected_end):
        raise ValueError(
            "climatology window mismatch: "
            f"declared {declared!r}, expected {expected_window!r}"
        )
    if declared_end >= years[0]:
        raise ValueError(
            "climatology overlaps evaluation period: "
            f"{declared} vs {years}"
        )
    return {
        "declared_window": declared,
        "start_year": declared_start,
        "end_year": declared_end,
        "evaluation_years": years,
        "precedes_evaluation": True,
    }


def _climatology_semantic_provenance(
    climatology_path: Path,
    *,
    evaluation_years: Sequence[int],
    expected_window: str,
) -> Dict[str, Any]:
    probe = xr.open_zarr(str(climatology_path), consolidated=True)
    try:
        attrs = dict(probe.attrs)
        archive = validate_climatology_archive(probe, climatology_path)
    finally:
        probe.close()
    semantic = validate_climatology_window(
        attrs,
        evaluation_years=evaluation_years,
        expected_window=expected_window,
    )
    semantic["source"] = attrs.get("source")
    semantic["canonical_archive"] = archive
    return semantic


def _build_acc_indices(
    climatology_path: Path, channel_names: Sequence[str]
) -> List[int]:
    """Return the indices of ``channel_names`` for which the climatology has data."""
    probe = xr.open_zarr(str(climatology_path), consolidated=True)
    clim_vars = set(probe.data_vars)
    probe.close()
    skip = {"tisr"}  # always skip (analytic)
    for c in channel_names:
        is_pl = len(c) > 1 and c[0] in _CLIM_PL_PREFIXES and c[1:].isdigit()
        if is_pl:
            if _CLIM_PL_PREFIXES[c[0]] not in clim_vars:
                skip.add(c)
        else:
            if c not in clim_vars:
                skip.add(c)
    return [i for i, c in enumerate(channel_names) if c not in skip]


# ---------------------------------------------------------------------------- #
# CLI                                                                          #
# ---------------------------------------------------------------------------- #


def _parse_eval_hours(s: str) -> List[int]:
    return sorted({int(x) for x in s.split(",") if x.strip()})


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_code_provenance(
    sources: Dict[str, Path],
) -> Dict[str, str]:
    missing = [
        f"{name}={path}"
        for name, path in sources.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing evaluation source files: " + ", ".join(missing)
        )
    return {
        name: _sha256_file(path)
        for name, path in sources.items()
    }


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument(
        "--climatology",
        default=(
            "/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/"
            "climatology_1990-2019_0p5_canonical_v2.zarr"
        ),
    )
    ap.add_argument(
        "--expected-climatology-window",
        default="1990-2019",
        help=(
            "Required inclusive YYYY-YYYY window from the climatology zarr "
            "attribute. Evaluation fails if it differs or overlaps test data."
        ),
    )
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument(
        "--models",
        required=True,
        help="Comma-separated NAME:CKPT[:ENVS] entries (envs may use 'K=V K2=V2').",
    )
    ap.add_argument(
        "--out-dir",
        default="metrics/eval_12h_2020_pre_ep10",
        help="Output directory for per-model JSON files.",
    )
    ap.add_argument(
        "--paper-tag",
        default="12h_2020",
        help="Tag stored in the JSON for downstream table-builders.",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--samples-per-date",
        type=int,
        default=2,
        help=(
            "Number of start hours per day. For 12h windows the natural "
            "value is 2 (start 0h, 12h)."
        ),
    )
    ap.add_argument("--eval-days-per-month", type=int, default=4)
    ap.add_argument(
        "--days-of-month",
        default=None,
        help=(
            "Explicit comma-separated calendar days, for example "
            "1,8,15,22. This overrides the legacy economy-day mapping and "
            "is recorded in artifact provenance."
        ),
    )
    ap.add_argument(
        "--full-year",
        action="store_true",
        help="Disable the economy-day filter and evaluate every valid anchor window.",
    )
    ap.add_argument(
        "--proper-rmse",
        action="store_true",
        help=(
            "Report the direct spherical strip-area weighted spatial mean "
            "without post-processing."
        ),
    )
    ap.add_argument(
        "--save-window-metrics",
        action="store_true",
        help="Save paired per-window channel MSE and ACC arrays next to each JSON.",
    )
    ap.add_argument(
        "--save-physical-metrics",
        action="store_true",
        help=(
            "Add paired wind, kinetic-energy, hydrostatic, and moisture "
            "diagnostics; requires --save-window-metrics."
        ),
    )
    ap.add_argument(
        "--save-temporal-metrics",
        action="store_true",
        help=(
            "Save paired discrete temporal-curvature errors for complete "
            "integer-tau trajectories; requires --save-window-metrics."
        ),
    )
    ap.add_argument(
        "--max-tau-hours",
        type=int,
        default=12,
        help="Window width W in hours (default: 12 for this script).",
    )
    ap.add_argument(
        "--eval-hours",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11",
        help="Comma-separated τ values to evaluate (1 <= τ < W).",
    )
    ap.add_argument(
        "--seen-tau",
        type=str,
        default=",".join(str(t) for t in SEEN_TAU_12H),
        help="τ values seen during training (recorded in JSON; no filtering).",
    )
    ap.add_argument(
        "--unseen-tau",
        type=str,
        default=",".join(str(t) for t in UNSEEN_TAU_12H),
        help="τ values held out during training (recorded in JSON).",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Override torch device (e.g. 'cuda:0'). Defaults to auto-detect.",
    )
    ap.add_argument(
        "--no-acc",
        action="store_true",
        help=(
            "Skip ACC computation entirely (no climatology zarr needed). "
            "RMSE blocks (norm + phys) are still emitted for all τ."
        ),
    )
    ap.add_argument(
        "--lazy-climatology",
        action="store_true",
        help=(
            "Read only the requested climatology hour/day slices. This is "
            "slower but avoids materialising the multi-GiB zarr in RAM."
        ),
    )
    ap.add_argument(
        "--climatology-cache-dir",
        default=None,
        help=(
            "Optional directory for per-day lazy climatology .npy files. "
            "Useful when several evaluations share the same dates/channels."
        ),
    )
    ap.add_argument(
        "--keep-n-channels",
        type=int,
        default=None,
        help=(
            "Slice x0/xT/target to the first N channels before model forward. "
            "Use 24 for keep_24ch=true trained ckpts (drops sst/tcc/tcwv)."
        ),
    )
    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()

    if _DEFERRED_IMPORT_ERROR is not None:
        raise SystemExit(
            "batch_eval_12h_memmap requires xarray and the weather_time_interp "
            f"package; original ImportError: {_DEFERRED_IMPORT_ERROR}"
        )

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")

    eval_hours = _parse_eval_hours(args.eval_hours)
    seen_tau = _parse_eval_hours(args.seen_tau)
    unseen_tau = _parse_eval_hours(args.unseen_tau)
    eval_days_of_month = (
        None
        if args.days_of_month is None
        else _parse_eval_hours(args.days_of_month)
    )
    if args.full_year and eval_days_of_month is not None:
        raise SystemExit("--full-year and --days-of-month are mutually exclusive")
    if eval_days_of_month is not None and (
        not eval_days_of_month
        or any(day < 1 or day > 31 for day in eval_days_of_month)
    ):
        raise SystemExit("--days-of-month must contain unique days in [1, 31]")

    # Build a one-shot Config to extract channel names and ACC indices.
    cfg = Config(
        memmap_dir=args.memmap_dir,
        test_years=[args.test_year],
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
        static_path=args.static_path,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_per_date=args.samples_per_date,
        eval_days_per_month=(
            None
            if args.full_year or eval_days_of_month is not None
            else args.eval_days_per_month
        ),
        eval_days_of_month=eval_days_of_month,
        dt_hours=float(args.max_tau_hours),
        max_tau_hours=int(args.max_tau_hours),
        eval_hours=tuple(eval_hours),
        device=device,
        climatology_path=args.climatology,
    )

    # Build dataset once to extract channel groups + names; reused for every
    # model loaded below.
    ds_probe = ERA5MemmapDataset(
        memmap_dir=cfg.memmap_dir,
        years=list(cfg.test_years),
        max_tau_hours=cfg.max_tau_hours,
        samples_per_date=cfg.samples_per_date,
        train=False,
        eval_hours=list(cfg.eval_hours),
        static_path=cfg.static_path,
        stats_path=cfg.stats_path,
        surface_stats_path=cfg.surface_stats_path,
    )
    channel_groups = ds_probe.channel_groups
    channel_names = list(ds_probe.channel_names) + list(ds_probe.surface_variables)
    del ds_probe  # discard; the runner will re-open inside _setup_data
    print(f"  channels ({len(channel_names)}): {channel_names}")

    # ACC indices (climatology-backed channels only) — skipped in --no-acc mode.
    if args.no_acc:
        print("  --no-acc: ACC computation disabled, only RMSE will be reported")
        acc_indices: List[int] = []
        climatology = None  # type: ignore[assignment]
        climatology_provenance = None
    else:
        print(f"fingerprinting climatology {args.climatology}...")
        climatology_provenance = dict(
            zarr_store_provenance(args.climatology)
        )
        climatology_provenance["semantic"] = (
            _climatology_semantic_provenance(
                Path(args.climatology),
                evaluation_years=cfg.test_years,
                expected_window=args.expected_climatology_window,
            )
        )
        acc_indices = _build_acc_indices(Path(args.climatology), channel_names)
        # If keep_n_channels is set, drop ACC indices that would point past the
        # sliced model output (must match runner.acc_indices filtering).
        if args.keep_n_channels is not None:
            acc_indices = [i for i in acc_indices if i < args.keep_n_channels]
        print(f"  ACC channels: {[channel_names[i] for i in acc_indices]}")

        # Climatology (shared across all models).
        print(f"loading climatology {args.climatology}...")
        climatology = _ClimatologyLookup(
            Path(args.climatology),
            [channel_names[i] for i in acc_indices],
            device,
            lazy=args.lazy_climatology,
            cache_dir=(
                Path(args.climatology_cache_dir)
                if args.climatology_cache_dir
                else None
            ),
            store_identity=str(
                climatology_provenance["cache_identity_sha256"]
            ),
        )
        mode = "lazy" if climatology.lazy else "in-memory"
        print(f"  climatology shape: {climatology.storage_shape} ({mode})")

    # Parse models list.
    models: List[Tuple[str, str, str]] = []
    for entry in args.models.split(","):
        parts = entry.split(":")
        if len(parts) >= 2:
            name, ckpt = parts[0], parts[1]
            envs = ":".join(parts[2:]) if len(parts) > 2 else ""
            models.append((name, ckpt, envs))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from tools.train.training_protocol import memmap_dataset_provenance

    evaluation_dataset_provenance = memmap_dataset_provenance(
        cfg.memmap_dir,
        list(cfg.test_years),
    )
    evaluation_input_provenance = {
        "static_features": static_feature_provenance(cfg.static_path),
        "pressure_level_stats": file_provenance(cfg.stats_path),
        "surface_stats": file_provenance(cfg.surface_stats_path),
        "climatology": climatology_provenance,
    }

    for i, (name, ckpt, envs) in enumerate(models, 1):
        print(f"\n=== [{i}/{len(models)}] {name} ===")
        if not Path(ckpt).exists():
            print(f"  [MISS] {ckpt}")
            continue
        for kv in envs.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                os.environ[k] = v
                print(f"  env {k}={v}")
        t_m = time.time()
        model, mt = _load_model_safe(
            ckpt,
            device,
            channel_groups,
            static_path=cfg.static_path,
        )
        print(
            f"  model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
            f"params, type={mt}"
        )
        # Per-model override keeps checkpoint input metadata explicit while
        # paper-facing runs use the fixed 24-channel prognostic protocol.
        model_keep_n = args.keep_n_channels
        model_input_n = None
        if mt == "dcae_adaln_residual_linear":
            model_input_n = int(getattr(model, "in_channels", model_keep_n or 0))
            if model_keep_n is not None and model_input_n != model_keep_n:
                print(
                    f"  [WeatherDCAE] using {model_input_n} anchor-input "
                    f"channels; evaluating the first {model_keep_n} targets"
                )
        if mt == "atm_vfi_pixel_attn":
            try:
                eff = getattr(model.net, "_effective_in_channels", None)
                in_w = int(eff) if eff is not None else int(
                    model.net.frame_encoder[0][0].weight.shape[1]
                )
                if model_keep_n != in_w:
                    print(
                        f"  [atm_vfi] overriding keep_n_channels "
                        f"{model_keep_n} → {in_w} (effective prog input width)"
                    )
                    model_keep_n = in_w
            except (AttributeError, IndexError):
                pass
        runner = BatchModelRunner12h(
            cfg=cfg,
            model=model,
            model_type=mt,
            ckpt_path=ckpt,
            climatology=climatology,
            acc_indices=acc_indices,
            seen_tau=seen_tau,
            unseen_tau=unseen_tau,
            keep_n_channels=model_keep_n,
            input_n_channels=model_input_n,
            proper_rmse=args.proper_rmse,
            save_window_metrics=args.save_window_metrics,
            save_physical_metrics=args.save_physical_metrics,
            save_temporal_metrics=args.save_temporal_metrics,
        )
        payload = runner.run()
        payload["paper_tag"] = args.paper_tag
        checkpoint_path = Path(ckpt)
        payload["checkpoint_provenance"] = {
            "path": str(checkpoint_path.resolve()),
            "size_bytes": checkpoint_path.stat().st_size,
            "mtime_ns": checkpoint_path.stat().st_mtime_ns,
            "sha256": _sha256_file(checkpoint_path),
            "eval_git_commit": _git_commit(),
        }
        payload["evaluation_input_provenance"] = evaluation_input_provenance
        payload[
            "evaluation_dataset_provenance"
        ] = evaluation_dataset_provenance
        payload["static_feature_names"] = list(STATIC_FEATURES_3)
        from tools.eval.eval_artifact_status import (
            _evaluation_source_paths,
        )

        evaluation_sources = _evaluation_source_paths()
        payload["evaluation_code_provenance"] = (
            _evaluation_code_provenance(evaluation_sources)
        )
        # ensure canonical channel order; if keep_n_channels set, slice accordingly
        if model_keep_n is not None:
            payload["channel_names"] = channel_names[:model_keep_n]
        else:
            payload["channel_names"] = channel_names
        out_path = out_dir / f"{name}.json"
        if args.save_window_metrics:
            window_dir = out_dir / "window_metrics"
            window_dir.mkdir(parents=True, exist_ok=True)
            window_path = window_dir / f"{name}.npz"
            window_tmp = window_path.with_name(
                f".{window_path.name}.{os.getpid()}.tmp"
            )
            with window_tmp.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    **runner.window_metrics_payload(),
                )
                handle.flush()
                os.fsync(handle.fileno())
            window_tmp.replace(window_path)
            payload["window_metrics_file"] = str(
                window_path.relative_to(out_dir)
            )
            payload["window_metrics_provenance"] = {
                "size_bytes": window_path.stat().st_size,
                "sha256": _sha256_file(window_path),
                "index_sha256": payload["evaluation_protocol"][
                    "index_sha256"
                ],
            }
        output_tmp = out_path.with_name(
            f".{out_path.name}.{os.getpid()}.tmp"
        )
        with output_tmp.open("w") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        output_tmp.replace(out_path)
        print(f"  saved {out_path} in {(time.time() - t_m) / 60:.1f} min")
        del model, runner
        torch.cuda.empty_cache()

    print("\n=== ALL MODELS DONE ===")


if __name__ == "__main__":
    main()

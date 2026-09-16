"""EvaluationRunner: shared scaffolding for ``tools/eval/*`` baseline scripts.

Phase 0 + Phase 1 of the eval-script refactor: extract memmap+wrapper+DataLoader
setup, economy-days filter, latitude weights, per-batch dispatch and per-hour
RMSE accumulators into a single abstract class. Subclasses only implement
``method_names`` / ``predict`` / ``output_payload``.

Float reduction order matches the legacy scripts exactly:
nested loops are ``(batch, h_idx, sample_in_batch, channel, method)`` and each
contribution is ``float(err[i, ci].item())`` accumulated into ``float`` running
sums. ``sqrt`` is applied only once at the end. The default reduction divides
the longitude sum by grid width; ``normalize_longitude=False`` exists only for
auditing historical JSON snapshots that omitted this factor.
"""
from __future__ import annotations

import abc
import datetime as _dt
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from torch.utils.data import DataLoader

from weather_time_interp.eval_datasets import ERA5WeatherHermiteDataset
from weather_time_interp.grid import (
    WB2_BLOCK_GRID_NAME,
    latitude_strip_weights,
    wb2_block_average_latitudes,
)
from weather_time_interp.memmap_dataset import ERA5MemmapDataset


# Day-of-month picks used by the economy filter. Mirrors the dictionary baked
# into batch_eval_memmap.py / numerical_baseline_eval.py.
_DAY_PICKS = {
    3: [1, 11, 21],
    4: [1, 8, 15, 22],
    5: [1, 7, 14, 21, 28],
}


def _bilinear_time_interp(
    x0: torch.Tensor,
    x1: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """``(1-τ)·x0 + τ·xT`` mirroring ``interpolate_time_with_f_interpolate``.

    Inlined to avoid pulling heavy training-stack imports through
    ``evaluate_baselines.py``.
    """
    tau_ = tau.float().view(-1)
    B = x0.size(0)
    if tau_.numel() == 1 and B > 1:
        tau_ = tau_.expand(B)
    tau_b = tau_.view(B, 1, 1, 1)
    return (1.0 - tau_b) * x0 + tau_b * x1


@dataclass
class Config:
    """Shared configuration for any EvaluationRunner subclass."""

    memmap_dir: str
    test_years: List[int]
    stats_path: str = "data/json_stats_0p5.nc"
    surface_stats_path: str = "data/surface_stats_0p5.json"
    static_path: str = "data/static_features_0p5.pt"
    batch_size: int = 4
    num_workers: int = 2
    samples_per_date: int = 4
    eval_days_per_month: Optional[int] = 4
    eval_days_of_month: Optional[Sequence[int]] = None
    dt_hours: float = 6.0
    max_tau_hours: int = 6
    eval_hours: Sequence[int] = field(default_factory=lambda: tuple(range(7)))
    device: Optional[torch.device] = None
    climatology_path: Optional[str] = None
    skip_acc_channels: Set[str] = field(default_factory=lambda: {"tisr"})
    normalize_longitude: bool = True
    out_dir: str = "metrics"


@dataclass
class BatchContext:
    """Per-(batch, h_idx) container handed to ``predict`` and hooks."""

    B: int
    nH: int
    h_idx: int
    h: torch.Tensor          # (B,) hour index per sample (long tensor on CPU)
    hours_per_sample: List[int]
    target_h: torch.Tensor   # (B, C, H, W)
    target_h_phys: Optional[torch.Tensor]   # physical-units target (may be None)
    x0_phys: torch.Tensor    # (B, C, H, W) — denormalised x0
    xT_phys: torch.Tensor
    u0p: torch.Tensor        # (B, C, H, W) wind u at t=0 broadcast per channel
    v0p: torch.Tensor
    uTp: torch.Tensor
    vTp: torch.Tensor
    batch_idx: int
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerHourAccums:
    """Running sums for RMSE/ACC per (method, hour, channel)."""

    sum_sq: Dict[str, Dict[int, Dict[str, float]]] = field(default_factory=dict)
    n_per_hour: Dict[int, int] = field(default_factory=dict)
    acc_sxy: Optional[Dict[int, Dict[str, float]]] = None
    acc_sxx: Optional[Dict[int, Dict[str, float]]] = None
    acc_syy: Optional[Dict[int, Dict[str, float]]] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class EvaluationRunner(abc.ABC):
    """ABC: shared dataset + loader + per-hour accumulator scaffolding."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # Late-initialised by ``_setup_data``.
        self.device: torch.device
        self.ds_base: ERA5MemmapDataset
        self.test_wrapped: ERA5WeatherHermiteDataset
        self.loader: DataLoader
        self.channel_names: List[str] = []
        self.mu_all: torch.Tensor
        self.sigma_all: torch.Tensor
        self.w_lat: torch.Tensor
        self.H: int = 0
        self.W: int = 0

    # ------------------------------------------------------------------ setup

    def _setup_data(self) -> None:
        cfg = self.cfg
        self.device = cfg.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"device: {self.device}")

        self.ds_base = ERA5MemmapDataset(
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
        if cfg.eval_days_of_month is not None:
            self._calendar_day_filter(self.ds_base, cfg.eval_days_of_month)
        elif cfg.eval_days_per_month is not None:
            self._economy_filter(self.ds_base)

        self.channel_names = list(self.ds_base.channel_names) + list(
            self.ds_base.surface_variables
        )
        print(f"  channels ({len(self.channel_names)}): {self.channel_names}")

        self.mu_all = (
            torch.cat([self.ds_base.mu, self.ds_base.surface_mu])
            .to(self.device)
            .view(1, -1, 1, 1)
        )
        self.sigma_all = (
            torch.cat([self.ds_base.sigma, self.ds_base.surface_sigma])
            .to(self.device)
            .view(1, -1, 1, 1)
        )

        self.test_wrapped = ERA5WeatherHermiteDataset(
            self.ds_base, delta_t_hours=cfg.dt_hours
        )
        print(f"  test dataset: {len(self.test_wrapped)} windows")
        self.loader = DataLoader(
            self.test_wrapped,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )

        first_arr = list(self.ds_base.memmaps.values())[0]
        self.H = first_arr.shape[-2]
        self.W = first_arr.shape[-1]
        self.w_lat = self._lat_weights(self.H)

    # ----------------------------------------------------------- legacy parity

    def _economy_filter(self, ds_base: ERA5MemmapDataset) -> None:
        """Restrict ``ds_base.index`` to a few representative days per month.

        Bit-identical to the inline block in ``numerical_baseline_eval.py``
        (lines 83–101) — keep the loop shape, ``int`` casts and ``try/except``
        intact so the resulting ordering matches.
        """
        K = max(1, int(self.cfg.eval_days_per_month or 0))
        day_picks = _DAY_PICKS.get(
            K, sorted({1 + i * (30 // K) for i in range(K)})
        )
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
        print(
            f"  economy filter: {len(filt)}/{len(ds_base.index)} index entries"
        )
        ds_base.index = filt

    @staticmethod
    def _calendar_day_filter(
        ds_base: ERA5MemmapDataset,
        days_of_month: Sequence[int],
    ) -> None:
        """Restrict the evaluation index to an explicit calendar-day set."""
        allowed = {int(day) for day in days_of_month}
        if not allowed or any(day < 1 or day > 31 for day in allowed):
            raise ValueError("eval_days_of_month must contain days in [1, 31]")
        filtered = []
        for entry in ds_base.index:
            year, t0, _, _ = entry
            day_of_year = int(t0) // 24
            date = _dt.date(int(year), 1, 1) + _dt.timedelta(days=day_of_year)
            if date.day in allowed:
                filtered.append(entry)
        if not filtered:
            raise ValueError("explicit calendar-day filter selected no windows")
        print(
            "  calendar-day filter "
            f"{sorted(allowed)}: {len(filtered)}/{len(ds_base.index)} index entries"
        )
        ds_base.index = filtered

    def _lat_weights(self, H: int) -> torch.Tensor:
        """Spherical row-area weights, shaped ``(1, 1,H,1)``."""
        if H == 360:
            lat = wb2_block_average_latitudes(H)
            w = latitude_strip_weights(lat)
            self.latitude_grid_name = WB2_BLOCK_GRID_NAME
        elif H == 181:
            lat = np.linspace(90.0, -90.0, H, dtype=np.float32)
            w = np.cos(np.deg2rad(lat))
            self.latitude_grid_name = "legacy_181_endpoint_grid"
        else:
            lat = np.linspace(90.0, -90.0, H, dtype=np.float32)
            w = np.cos(np.deg2rad(lat))
            self.latitude_grid_name = "legacy_endpoint_grid"
        w = w / w.sum()
        return torch.from_numpy(w).to(device=self.device, dtype=torch.float32).view(
            1, 1, -1, 1
        )

    # ------------------------------------------------------------- main loop

    def _new_accums(self) -> PerHourAccums:
        methods = list(self.method_names())
        eval_hours = list(self.cfg.eval_hours)
        accums = PerHourAccums(
            sum_sq={m: {h: {} for h in eval_hours} for m in methods},
            n_per_hour={h: 0 for h in eval_hours},
        )
        return accums

    def _per_ch_sq(
        self, pred: torch.Tensor, target_h: torch.Tensor, w_lat: torch.Tensor
    ) -> torch.Tensor:
        squared_error = ((pred - target_h) ** 2 * w_lat).sum(dim=(-2, -1))
        if self.cfg.normalize_longitude:
            squared_error = squared_error / float(pred.shape[-1])
        return squared_error  # (B, C)

    def _compute_uv_for_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Denormalise ``x0`` / ``xT`` once per batch.

        Wind broadcasting (``u0p`` / ``v0p`` / ``uTp`` / ``vTp``) is delegated to
        subclasses via :meth:`expand_uv` so subclasses without UV-channels (e.g.
        Farneback) can return zero tensors.
        """
        device = self.device
        x0 = batch["x0"].to(device, non_blocking=True)
        xT = batch["xT"].to(device, non_blocking=True)
        tau_all = batch["tau"].to(device, non_blocking=True)
        tau_hour_all = batch["tau_hour"].long()
        target_all = batch["target"].to(device, non_blocking=True)

        x0_phys = x0 * self.sigma_all + self.mu_all
        xT_phys = xT * self.sigma_all + self.mu_all

        u0p, v0p = self.expand_uv(x0_phys)
        uTp, vTp = self.expand_uv(xT_phys)

        return {
            "x0": x0,
            "xT": xT,
            "tau_all": tau_all,
            "tau_hour_all": tau_hour_all,
            "target_all": target_all,
            "x0_phys": x0_phys,
            "xT_phys": xT_phys,
            "u0p": u0p,
            "v0p": v0p,
            "uTp": uTp,
            "vTp": vTp,
        }

    def expand_uv(self, x_phys: torch.Tensor) -> tuple:
        """Return ``(u_per_channel, v_per_channel)`` tensors.

        Default returns zero fields (no advection). Subclasses can override.
        """
        zeros = torch.zeros_like(x_phys)
        return zeros, zeros

    def _accumulate(
        self,
        method: str,
        h_idx: int,
        B: int,
        err_per_ch: torch.Tensor,
        hours_per_sample: List[int],
        accums: PerHourAccums,
    ) -> None:
        """Add one method's ``(B, C)`` SE tensor to running sums.

        ``n_per_hour`` is incremented once across all methods — see
        :meth:`_process_batch`. We only update ``sum_sq`` here.
        """
        sum_sq_method = accums.sum_sq[method]
        names = self.channel_names
        for i in range(B):
            h = int(hours_per_sample[i])
            if h not in sum_sq_method:
                continue  # hour outside eval_hours
            bucket = sum_sq_method[h]
            for ci, name in enumerate(names):
                bucket.setdefault(name, 0.0)
                bucket[name] += float(err_per_ch[i, ci].item())

    def _process_batch(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        accums: PerHourAccums,
    ) -> None:
        ctx_b = self._compute_uv_for_batch(batch)
        x0 = ctx_b["x0"]
        xT = ctx_b["xT"]
        tau_all = ctx_b["tau_all"]
        tau_hour_all = ctx_b["tau_hour_all"]
        target_all = ctx_b["target_all"]

        B = x0.size(0)
        nH = tau_all.size(1)
        methods = list(self.method_names())

        for h_idx in range(nH):
            tau_h = tau_all[:, h_idx, 0]
            target_h = target_all[:, h_idx]
            hours_per_sample = tau_hour_all[:, h_idx, 0].tolist()
            h_tensor = tau_hour_all[:, h_idx, 0]

            batch_ctx = BatchContext(
                B=B,
                nH=nH,
                h_idx=h_idx,
                h=h_tensor,
                hours_per_sample=hours_per_sample,
                target_h=target_h,
                target_h_phys=None,
                x0_phys=ctx_b["x0_phys"],
                xT_phys=ctx_b["xT_phys"],
                u0p=ctx_b["u0p"],
                v0p=ctx_b["v0p"],
                uTp=ctx_b["uTp"],
                vTp=ctx_b["vTp"],
                batch_idx=batch_idx,
            )

            # Predictions per method (order matches method_names())
            preds: Dict[str, torch.Tensor] = {}
            for m in methods:
                preds[m] = self.predict(m, x0, xT, tau_h, batch_ctx=batch_ctx)

            # Per-method squared error (B, C). Compute all before bumping n.
            err_map: Dict[str, torch.Tensor] = {
                m: self._per_ch_sq(preds[m], target_h, self.w_lat) for m in methods
            }

            # ----- canonical reduction loop (matches legacy exactly) -----
            for i in range(B):
                h = int(hours_per_sample[i])
                if h not in accums.n_per_hour:
                    continue
                accums.n_per_hour[h] += 1
                for ci, name in enumerate(self.channel_names):
                    for method, e_tensor in err_map.items():
                        accums.sum_sq[method][h].setdefault(name, 0.0)
                        accums.sum_sq[method][h][name] += float(
                            e_tensor[i, ci].item()
                        )

            # Optional per-sample hook (e.g. ACC).
            for i in range(B):
                self.collect_extra_per_sample(
                    method=None, h=int(hours_per_sample[i]),
                    sample_idx=i, batch_ctx=batch_ctx,
                )

        if batch_idx % 10 == 0:
            print(f"  batch {batch_idx}/{len(self.loader)}")

    # --------------------------------------------------------------- finalize

    def _finalize(self, accums: PerHourAccums) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for method in self.method_names():
            results[method] = self.output_payload(method, accums)
        return results

    def output_path(self, method: str) -> Path:
        return Path(self.cfg.out_dir) / f"{method}.json"

    def _dump(self, results: Dict[str, Any]) -> None:
        out_dir = Path(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for method, payload in results.items():
            p = self.output_path(method)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"saved {p}")

    # --------------------------------------------------------------- run

    def run(self) -> Dict[str, Any]:
        t_global = time.time()
        self._setup_data()
        self.on_setup_done()
        print(f"setup done in {time.time()-t_global:.1f}s")

        accums = self._new_accums()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.loader):
                self._process_batch(batch, batch_idx, accums)

        results = self._finalize(accums)
        self._dump(results)
        print(f"\n=== DONE in {(time.time()-t_global)/60:.1f} min ===")
        return results

    # --------------------------------------------------------------- hooks

    def on_setup_done(self) -> None:
        """Post-setup hook (e.g. compute uv assignment). No-op by default."""

    def collect_extra_per_sample(
        self,
        method: Optional[str],
        h: int,
        sample_idx: int,
        batch_ctx: BatchContext,
    ) -> None:
        """Per-sample hook for extras like ACC. Default no-op."""

    # --------------------------------------------------------------- abstract

    @abc.abstractmethod
    def method_names(self) -> List[str]:
        """Ordered list of methods this runner reports."""

    @abc.abstractmethod
    def predict(
        self,
        method: str,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau_h: torch.Tensor,
        *,
        batch_ctx: BatchContext,
    ) -> torch.Tensor:
        """Return normalised ``(B, C, H, W)`` prediction for ``method``."""

    @abc.abstractmethod
    def output_payload(self, method: str, accums: PerHourAccums) -> Dict[str, Any]:
        """Build the JSON payload for one method."""


# ---------------------------------------------------------------------------- #
# Phase 1: NumericalBaselineRunner                                              #
# ---------------------------------------------------------------------------- #


class NumericalBaselineRunner(EvaluationRunner):
    """Migrated ``tools/eval/numerical_baseline_eval.py`` body.

    Methods (fixed order): ``bilinear``, ``semi_lagrangian``, ``hermite_advection``,
    ``settls``, ``hermite_diffusion``. Output JSON schema mirrors the legacy
    script — each method's payload has a ``per_hour`` block with
    ``model`` / ``bilinear`` / ``bicubic`` sub-dicts (the latter two always
    reflect the legacy bilinear closed-form).
    """

    METHODS = (
        "bilinear",
        "semi_lagrangian",
        "hermite_advection",
        "settls",
        "hermite_diffusion",
    )

    def method_names(self) -> List[str]:
        return list(self.METHODS)

    def on_setup_done(self) -> None:
        # Lazy import: avoids pulling torch ops at module-import time so tests
        # that don't exercise these methods can still import the module.
        from tools.baselines.numerical_baselines import build_uv_assignment

        self.uv_map = build_uv_assignment(self.channel_names)
        adv = [
            self.channel_names[c]
            for c, (ui, _vi) in self.uv_map.items()
            if ui >= 0
        ]
        no_adv = [
            self.channel_names[c]
            for c, (ui, _vi) in self.uv_map.items()
            if ui < 0
        ]
        print(f"  advected channels ({len(adv)}): {adv}")
        print(f"  no-advection channels ({len(no_adv)}): {no_adv}")

    def expand_uv(self, x_phys: torch.Tensor) -> tuple:
        from tools.baselines.numerical_baselines import expand_uv_to_channels

        return expand_uv_to_channels(x_phys, self.uv_map)

    def predict(
        self,
        method: str,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau_h: torch.Tensor,
        *,
        batch_ctx: BatchContext,
    ) -> torch.Tensor:
        from tools.baselines.numerical_baselines import (
            hermite_advection_interp,
            hermite_diffusion_interp,
            semi_lagrangian_interp,
            settls_interp,
        )

        dt_hours = self.cfg.dt_hours
        if method == "bilinear":
            return _bilinear_time_interp(x0, xT, tau_h)
        if method == "semi_lagrangian":
            return semi_lagrangian_interp(
                x0, xT, batch_ctx.u0p, batch_ctx.v0p,
                batch_ctx.uTp, batch_ctx.vTp,
                tau_h, dt_hours=dt_hours, n_iter=2,
            )
        if method == "hermite_advection":
            return hermite_advection_interp(
                x0, xT, batch_ctx.u0p, batch_ctx.v0p,
                batch_ctx.uTp, batch_ctx.vTp,
                tau_h, dt_hours=dt_hours,
            )
        if method == "settls":
            return settls_interp(
                x0, xT, batch_ctx.u0p, batch_ctx.v0p,
                batch_ctx.uTp, batch_ctx.vTp,
                tau_h, dt_hours=dt_hours, n_iter=3,
            )
        if method == "hermite_diffusion":
            return hermite_diffusion_interp(
                x0, xT, batch_ctx.u0p, batch_ctx.v0p,
                batch_ctx.uTp, batch_ctx.vTp,
                tau_h, dt_hours=dt_hours,
            )
        raise ValueError(f"unknown method: {method}")

    def output_payload(self, method: str, accums: PerHourAccums) -> Dict[str, Any]:
        eval_hours = list(self.cfg.eval_hours)
        per_hour_out: Dict[str, Any] = {}
        for h in eval_hours:
            n = accums.n_per_hour[h]
            if n == 0:
                continue
            per_hour_out[str(h)] = {"model": {}, "bilinear": {}, "bicubic": {}}
            for name in self.channel_names:
                if name in accums.sum_sq[method][h]:
                    rmse = float(np.sqrt(accums.sum_sq[method][h][name] / n))
                    per_hour_out[str(h)]["model"][f"rmse_{name}"] = rmse
                    rmse_bil = float(
                        np.sqrt(accums.sum_sq["bilinear"][h][name] / n)
                    )
                    per_hour_out[str(h)]["bilinear"][f"rmse_{name}"] = rmse_bil
                    # 'bicubic' mirrors bilinear (2-anchor cubic collapses).
                    per_hour_out[str(h)]["bicubic"][f"rmse_{name}"] = rmse_bil
        return {
            "schema_version": 3,
            "method": method,
            "num_samples": sum(accums.n_per_hour.values()),
            "years": list(self.cfg.test_years),
            "channel_names": list(self.channel_names),
            "evaluation_protocol": {
                "latitude_weighting": "cell_area",
                "longitude_reduction": (
                    "mean" if self.cfg.normalize_longitude else "legacy_sum"
                ),
                "grid_width": self.W,
                "latitude_grid": self.latitude_grid_name,
                "transport_sampling": (
                    "canonical_cell_centres_periodic_longitude_border_latitude"
                ),
            },
            "per_hour": per_hour_out,
        }


__all__ = [
    "Config",
    "BatchContext",
    "PerHourAccums",
    "EvaluationRunner",
    "NumericalBaselineRunner",
]

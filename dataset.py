
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from weather_time_interp.config import (
    TAU_TRAIN_STEPS,
    TAU_VAL_STEPS,
    HOURS_PER_TAU_UNIT,
    DEFAULT_VARIABLES,
    DEFAULT_PRESSURE_LEVELS,
    VAR_SHORT_TO_NAME,
    VAR_TO_ERA5,
)

class ERA5ResNetODEDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        years: list,
        in_channels: int | None = None,
        variables: list[str] | None = None,
        pressure_levels: list[int] | None = None,
        max_tau_hours: int = 6,
        samples_per_date: int = 4,
        train: bool = True,
        static_path: str | None = None,
        stats_path: str = "data/json_stats.nc",
        train_hours: Optional[List[int]] = None,
        eval_hours: Optional[List[int]] = None,
        cache_in_ram: bool = False,
        surface_data_dir: str | None = None,
        surface_variables: List[str] | None = None,
        surface_stats_path: str | None = None,
        use_analytic_tisr: bool = False,
        eval_all_hours: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.years = years
        self.max_tau_hours = max_tau_hours
        self.samples_per_date = samples_per_date
        self.train = train
        self.train_hours = train_hours
        self.eval_hours = eval_hours
        self.cache_in_ram = cache_in_ram
        # Surface (single-level) variables, optional
        self.surface_data_dir = Path(surface_data_dir) if surface_data_dir else None
        self.surface_variables = list(surface_variables) if surface_variables else []
        self.surface_stats_path = surface_stats_path
        self.use_analytic_tisr = bool(use_analytic_tisr)
        self.eval_all_hours = bool(eval_all_hours)

        # Dynamic channel setup
        self.variables = [v.upper() for v in (variables or DEFAULT_VARIABLES)]
        self.pressure_levels = pressure_levels or DEFAULT_PRESSURE_LEVELS

        channel_specs = [(v, lvl) for v in self.variables for lvl in self.pressure_levels]
        if in_channels is None:
            in_channels = len(channel_specs)
        if in_channels < len(channel_specs):
            channel_specs = channel_specs[:in_channels]
        elif in_channels > len(channel_specs):
            print(
                f"Warning: Requested {in_channels} channels, but only {len(channel_specs)} available. "
                "Using all available channels."
            )
            in_channels = len(channel_specs)

        self.in_channels = in_channels
        self.channel_names = [f"{v}{lvl}" for v, lvl in channel_specs]
        self.selected_setup = [(VAR_TO_ERA5.get(v, v.lower()), v, lvl) for v, lvl in channel_specs]

        # Group channels by variable for metrics.
        # PL channels first (idx 0..n_pl-1), then surface (n_pl..n_pl+n_surf-1).
        self.channel_groups: Dict[str, List[int]] = {}
        for idx, (v, lvl) in enumerate(channel_specs):
            name = VAR_SHORT_TO_NAME.get(v, v)
            self.channel_groups.setdefault(name, []).append(idx)
            # Per-level alias (e.g. "T1000", "U925") for Aurora-style per-level weights.
            self.channel_groups.setdefault(f"{v.upper()}{int(lvl)}", []).append(idx)
        # Surface vars: extend channel_groups so trainer can per-channel weight (e.g. tisr=0).
        if surface_variables:
            n_pl = len(channel_specs)
            for s_idx, sv in enumerate(surface_variables):
                self.channel_groups.setdefault(sv, []).append(n_pl + s_idx)

        # 1. Загрузка Zarr
        self.datasets: Dict[int, xr.Dataset] = {}
        for y in years:
            zarr_path = self.data_dir / f"zarr_{y}.zarr"
            self.datasets[y] = xr.open_zarr(zarr_path, consolidated=True)

        # 1b. Optional in-RAM cache: pre-stack all selected channels into one
        # contiguous array per year shaped (T, C, H, W) for O(1) __getitem__.
        # Required for any sane training throughput on s3fs (chunk-aware reads
        # would otherwise decompress 100s of MB to slice 65 KB).
        self.cached: Dict[int, np.ndarray] = {}
        if cache_in_ram:
            import time as _t
            for y, ds in self.datasets.items():
                t0 = _t.time()
                lvl_index = {int(l): i for i, l in enumerate(ds.level.values.tolist())}
                T = ds.sizes["time"]
                # Load each variable fully (only selected pressure_levels), single decompress.
                # Shape per var after .values: (T, len(pressure_levels), H, W)
                var_arrays = {}
                for v_ds, _v_short, _lvl in self.selected_setup:
                    if v_ds in var_arrays:
                        continue
                    arr = ds[v_ds].sel(level=self.pressure_levels).values  # (T, L, H, W)
                    var_arrays[v_ds] = arr.astype(np.float32, copy=False)
                # Stack channels in selected_setup order: (T, C, H, W)
                C = len(self.selected_setup)
                H, W = next(iter(var_arrays.values())).shape[-2:]
                stacked = np.empty((T, C, H, W), dtype=np.float32)
                for c, (v_ds, _v_short, lvl) in enumerate(self.selected_setup):
                    li = lvl_index[int(lvl)]
                    li_in_subset = self.pressure_levels.index(int(lvl))
                    stacked[:, c] = var_arrays[v_ds][:, li_in_subset]
                self.cached[y] = stacked
                size_gib = stacked.nbytes / 1024**3
                print(f"[cache] year {y}: shape={stacked.shape} size={size_gib:.2f} GiB loaded in {_t.time()-t0:.1f}s",
                      flush=True)

        # 1c. Optional surface variables (single-level: t2m, mslp, u10, v10, tisr, tcwv)
        # Load from <surface_data_dir>/surface_<year>.zarr alongside pressure-level data.
        self.surface_datasets: Dict[int, xr.Dataset] = {}
        self.surface_cached: Dict[int, np.ndarray] = {}
        if self.surface_data_dir is not None and self.surface_variables:
            import time as _t
            for y in years:
                surf_path = self.surface_data_dir / f"surface_{y}.zarr"
                ds_s = xr.open_zarr(surf_path, consolidated=True)
                self.surface_datasets[y] = ds_s
                if cache_in_ram:
                    t0 = _t.time()
                    T_s = ds_s.sizes["time"]
                    H_s = ds_s.sizes["latitude"]
                    W_s = ds_s.sizes["longitude"]
                    Cs = len(self.surface_variables)
                    surf_stacked = np.empty((T_s, Cs, H_s, W_s), dtype=np.float32)
                    tisr_backend = "n/a"
                    for ci, vname in enumerate(self.surface_variables):
                        if vname == "tisr" and self.use_analytic_tisr:
                            tisr_backend = self._compute_analytic_tisr(
                                ds_s.time.values,
                                ds_s.latitude.values,
                                ds_s.longitude.values,
                                surf_stacked[:, ci],
                            )
                        else:
                            surf_stacked[:, ci] = ds_s[vname].values.astype(np.float32, copy=False)
                    self.surface_cached[y] = surf_stacked
                    src_tag = f" (tisr={tisr_backend})" if (self.use_analytic_tisr and "tisr" in self.surface_variables) else ""
                    print(f"[cache surface] year {y}: shape={surf_stacked.shape} size={surf_stacked.nbytes/1024**3:.2f} GiB in {_t.time()-t0:.1f}s{src_tag}",
                          flush=True)

        # 2. Загрузка статистики
        stats_file = Path(stats_path)
        if not stats_file.exists():
            raise FileNotFoundError(
                f"Normalization stats file not found: {stats_file}"
            )
        with xr.open_dataset(stats_file) as ds_stats:
            stats_subset = ds_stats["climate_statistics"].sel(params=self.channel_names)
            mu = stats_subset.isel(stats=0).values
            sigma = stats_subset.isel(stats=1).values
            
            # Проверка на NaN в статистике
            if np.isnan(mu).any() or np.isnan(sigma).any():
                print(f"Warning: Stats file contains NaN!")
                print(f"  mu: {mu}, has_nan: {np.isnan(mu).any()}")
                print(f"  sigma: {sigma}, has_nan: {np.isnan(sigma).any()}")
                print(f"  channel_names: {self.channel_names}")

        self.mu = torch.from_numpy(mu).float().view(-1, 1, 1)
        self.sigma = torch.from_numpy(sigma).float().view(-1, 1, 1)
        self.sigma = torch.clamp(self.sigma, min=1e-6)

        # 2b. Surface stats (optional)
        self.surface_mu: Optional[torch.Tensor] = None
        self.surface_sigma: Optional[torch.Tensor] = None
        if self.surface_variables and self.surface_stats_path:
            import json
            with open(self.surface_stats_path) as f:
                surf_stats = json.load(f)
            mus = []
            sigs = []
            for vname in self.surface_variables:
                if vname not in surf_stats:
                    raise KeyError(f"Surface var '{vname}' not in {self.surface_stats_path}")
                mus.append(float(surf_stats[vname]["mean"]))
                sigs.append(float(surf_stats[vname]["std"]))
            self.surface_mu = torch.tensor(mus, dtype=torch.float32).view(-1, 1, 1)
            self.surface_sigma = torch.tensor(sigs, dtype=torch.float32).view(-1, 1, 1).clamp_min(1e-6)
            print(f"[surface stats] {len(self.surface_variables)} vars: {self.surface_variables}", flush=True)

        # 3. Static data — supports both .pt (torch tensor) and .nc (xarray Dataset)
        self.use_static = static_path is not None
        self.static = None
        if self.use_static and static_path:
            sp = Path(static_path)
            if sp.suffix in (".pt", ".pth"):
                self.static = torch.load(str(sp), weights_only=False)
                if not isinstance(self.static, torch.Tensor):
                    raise ValueError(f"Expected torch.Tensor in {sp}, got {type(self.static)}")
            else:
                static_ds = xr.open_dataset(static_path)
                self.static = torch.from_numpy(static_ds.to_array().values.astype(np.float32))

        # 4. Indexing
        tau_steps_default = TAU_TRAIN_STEPS if train else TAU_VAL_STEPS
        default_hours = sorted({
            max(0, min(self.max_tau_hours, int(round(float(t) * HOURS_PER_TAU_UNIT))))
            for t in tau_steps_default
        })

        def _normalize_hours(hours: Optional[List[int]], default: List[int]) -> List[int]:
            if hours is None:
                values = default
            else:
                values = [int(h) for h in hours]
            uniq = sorted({max(0, min(self.max_tau_hours, h)) for h in values})
            if not uniq:
                raise ValueError("No valid tau hours after normalization.")
            return uniq

        if self.train:
            self.active_hours = _normalize_hours(self.train_hours, default_hours)
        else:
            eval_default = list(range(0, self.max_tau_hours + 1))
            self.active_hours = _normalize_hours(self.eval_hours, eval_default)

        self.active_taus = [
            float(h) / float(self.max_tau_hours)
            for h in self.active_hours
        ]

        self.index: List[Tuple[int, int, int, float]] = []
        for y, ds in self.datasets.items():
            T = ds.sizes["time"]
            step_per_day = 24
            for day_start in range(0, T - max_tau_hours - 1, step_per_day):
                for s in range(samples_per_date):
                    t0 = day_start + s * (step_per_day // max(1, samples_per_date))
                    if t0 + max_tau_hours >= T:
                        continue
                    for tau_hours, tau in zip(self.active_hours, self.active_taus):
                        if t0 + tau_hours < T:
                            self.index.append((y, t0, tau_hours, tau))

    def _read_fields(self, ds: xr.Dataset, t: int) -> np.ndarray:
        fields = []
        for v_ds, _, lvl in self.selected_setup:
            arr = ds[v_ds].sel(level=lvl).isel(time=t).values
            fields.append(arr)
        return np.stack(fields, axis=0)

    def _read_fields_cached(self, year: int, t: int) -> np.ndarray:
        # cached array shape (T, C, H, W) — direct slice
        return self.cached[year][t]

    def _compute_analytic_tisr(self, times, lat, lon, out_buffer: np.ndarray) -> str:
        """Fill `out_buffer` (shape (T,H,W)) with analytic TISR. Uses torch CUDA when
        available (≈65× faster than numpy), falls back to numpy on any error.

        Returns the backend name actually used: "torch:cuda" | "numpy".
        """
        from weather_time_interp.utils.solar_radiation import (
            hourly_tisr_accumulated,
            hourly_tisr_accumulated_torch,
        )
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                # Pick the GPU with the most free memory to avoid OOM with parallel training.
                best_dev, best_free = 0, 0
                for d in range(torch.cuda.device_count()):
                    try:
                        free, _ = torch.cuda.mem_get_info(d)
                    except Exception:
                        free = 0
                    if free > best_free:
                        best_free, best_dev = free, d
                # Need ~3 GB headroom (256 chunk × 64 × 181 × 360 × 4B ≈ 4 GB).
                if best_free > 4 * 1024**3:
                    out_buffer[:] = hourly_tisr_accumulated_torch(
                        times, lat, lon,
                        n_substeps=64, device=f"cuda:{best_dev}", chunk_T=256,
                    )
                    return f"torch:cuda:{best_dev}"
        except Exception as e:
            print(f"  [tisr] torch path failed ({e!r}), falling back to numpy", flush=True)
        out_buffer[:] = hourly_tisr_accumulated(times, lat, lon, n_substeps=64)
        return "numpy"

    def _read_surface_cached(self, year: int, t: int) -> Optional[np.ndarray]:
        if not self.surface_variables:
            return None
        if self.cache_in_ram and year in self.surface_cached:
            return self.surface_cached[year][t]
        # Fallback: read from open zarr (no-cache path)
        ds_s = self.surface_datasets[year]
        layers = []
        for v in self.surface_variables:
            if v == "tisr" and self.use_analytic_tisr:
                from weather_time_interp.utils.solar_radiation import hourly_tisr_accumulated
                layers.append(
                    hourly_tisr_accumulated(
                        np.array([ds_s.time.values[t]]),
                        ds_s.latitude.values,
                        ds_s.longitude.values,
                    )[0]
                )
            else:
                layers.append(ds_s[v].isel(time=t).values)
        return np.stack(layers, axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        year, t0, tau_hours, tau = self.index[idx]
        ds = self.datasets[year]

        if self.cache_in_ram:
            x0 = torch.from_numpy(self._read_fields_cached(year, t0))
            x1 = torch.from_numpy(self._read_fields_cached(year, t0 + self.max_tau_hours))
            y = torch.from_numpy(self._read_fields_cached(year, t0 + tau_hours))
        else:
            x0 = torch.from_numpy(self._read_fields(ds, t0).astype(np.float32))
            x1 = torch.from_numpy(self._read_fields(ds, t0 + self.max_tau_hours).astype(np.float32))
            y = torch.from_numpy(self._read_fields(ds, t0 + tau_hours).astype(np.float32))

        # Проверка на NaN в исходных данных
        if np.isnan(x0).any() or np.isnan(x1).any() or np.isnan(y).any():
            print(f"Warning: Sample {idx} from year {year} t={t0} contains NaN in raw data!")

        # Нормализация
        x0 = (x0 - self.mu) / self.sigma
        x1 = (x1 - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma

        # Surface variables: load + normalize + concat
        if self.surface_variables:
            s0 = torch.from_numpy(self._read_surface_cached(year, t0).astype(np.float32))
            s1 = torch.from_numpy(self._read_surface_cached(year, t0 + self.max_tau_hours).astype(np.float32))
            s_y = torch.from_numpy(self._read_surface_cached(year, t0 + tau_hours).astype(np.float32))
            if self.surface_mu is not None and self.surface_sigma is not None:
                s0 = (s0 - self.surface_mu) / self.surface_sigma
                s1 = (s1 - self.surface_mu) / self.surface_sigma
                s_y = (s_y - self.surface_mu) / self.surface_sigma
            # Concat surface AFTER pressure-level along channel dim.
            x0 = torch.cat([x0, s0], dim=0)
            x1 = torch.cat([x1, s1], dim=0)
            y = torch.cat([y, s_y], dim=0)

        # Замена NaN после нормализации (если sigma была NaN или Inf)
        if torch.isnan(x0).any():
            x0 = torch.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(x1).any():
            x1 = torch.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(y).any():
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        ts = datetime.utcfromtimestamp(ds.time.values[t0].astype("datetime64[s]").astype(int))
        day, hour = ts.timetuple().tm_yday / 365.0, ts.hour / 24.0
        time_emb = torch.tensor([np.sin(2*np.pi*day), np.cos(2*np.pi*day),
                                 np.sin(2*np.pi*hour), np.cos(2*np.pi*hour)], dtype=torch.float32)

        out = {
            "x0": x0,
            "x1": x1,
            "time_emb": time_emb,
            "tau": torch.tensor([tau], dtype=torch.float32),
            "tau_hour": torch.tensor([tau_hours], dtype=torch.long),
            "target": y,
        }
        # eval_all_hours: pack targets for every h ∈ {1..max_tau-1} so evaluate_per_hour_all
        # compares pred_h vs truth_h instead of pred_h vs (arbitrary batch_tau target).
        if getattr(self, "eval_all_hours", False):
            n_hours = self.max_tau_hours - 1   # exclude both endpoints (h=0 = x0, h=max = x1)
            all_targets = []
            for h in range(1, self.max_tau_hours):
                if self.cache_in_ram:
                    y_h = torch.from_numpy(self._read_fields_cached(year, t0 + h))
                else:
                    y_h = torch.from_numpy(self._read_fields(ds, t0 + h).astype(np.float32))
                y_h = (y_h - self.mu) / self.sigma
                if self.surface_variables:
                    s_h = torch.from_numpy(self._read_surface_cached(year, t0 + h).astype(np.float32))
                    if self.surface_mu is not None:
                        s_h = (s_h - self.surface_mu) / self.surface_sigma
                    y_h = torch.cat([y_h, s_h], dim=0)
                y_h = torch.nan_to_num(y_h, nan=0.0, posinf=0.0, neginf=0.0)
                all_targets.append(y_h)
            out["target_all_hours"] = torch.stack(all_targets, dim=0)  # (n_hours, C, H, W)
        if self.static is not None: out["static"] = self.static
        return out

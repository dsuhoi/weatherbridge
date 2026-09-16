"""Memmap-based Dataset for 0.5° pre-processed data.

Reads from /tmp/wb2_0p5_cache/wb2_YYYY.bin (np.memmap fp32 (T, 27, 360, 720))
+ wb2_YYYY.json metadata. Drop-in compatible with ERA5ResNetODEDataset interface.
"""
from __future__ import annotations

import json
import mmap
import numpy as np
import torch
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional


def release_memmap_time_slices(
    array: np.memmap,
    time_indices: tuple[int, ...],
) -> bool:
    """Drop copied time slices from this process's file-backed page mappings."""
    mapped = getattr(array, "_mmap", None)
    if mapped is None or not hasattr(mapped, "madvise"):
        return False
    stride = int(array.strides[0])
    mapped_size = int(mapped.size())
    for time_index in set(time_indices):
        start = int(array.offset) + time_index * stride
        aligned_start = start - start % mmap.PAGESIZE
        end = min(start + stride, mapped_size)
        aligned_length = (
            (end - aligned_start + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        ) * mmap.PAGESIZE
        aligned_length = min(aligned_length, mapped_size - aligned_start)
        try:
            mapped.madvise(
                mmap.MADV_DONTNEED,
                aligned_start,
                aligned_length,
            )
        except (OSError, ValueError):
            return False
    return True


class ERA5MemmapDataset(Dataset):
    """Drop-in replacement for ERA5ResNetODEDataset using fp32 memmap on local SSD."""

    PL_NAMES = ["T", "U", "V", "Q", "Z"]
    PL_LEVELS = [1000, 925, 850, 700]
    SURF_NAMES = ["t2m", "u10", "v10", "mslp", "sst", "tcc", "tcwv"]

    def __init__(
        self,
        memmap_dir: str,
        years: List[int],
        max_tau_hours: int = 6,
        samples_per_date: int = 4,
        train: bool = True,
        static_path: Optional[str] = None,
        stats_path: str = "data/json_stats_0p5.nc",
        surface_stats_path: str = "data/surface_stats_0p5.json",
        train_hours: Optional[List[int]] = None,
        eval_hours: Optional[List[int]] = None,
        use_analytic_tisr: bool = False,  # tisr replaced via analytic if needed
        release_memmap_pages: bool = False,
        **kwargs,  # absorb unused kwargs from old API
    ):
        self.memmap_dir = Path(memmap_dir)
        self.years = years
        self.max_tau_hours = max_tau_hours
        self.samples_per_date = samples_per_date
        self.train = train
        self.train_hours = train_hours
        self.eval_hours = eval_hours
        self.use_analytic_tisr = bool(use_analytic_tisr)
        self.release_memmap_pages = bool(release_memmap_pages)

        self.channel_names = [f"{v}{l}" for v in self.PL_NAMES for l in self.PL_LEVELS]
        self.surface_variables = list(self.SURF_NAMES)
        self.in_channels = len(self.channel_names)
        n_pl = self.in_channels
        n_surf = len(self.surface_variables)

        # Load memmaps per year
        self.memmaps: Dict[int, np.memmap] = {}
        self.years_T: Dict[int, int] = {}
        self.time_starts: Dict[int, datetime] = {}
        for y in years:
            meta_p = self.memmap_dir / f"wb2_{y}.json"
            bin_p = self.memmap_dir / f"wb2_{y}.bin"
            with open(meta_p) as f:
                meta = json.load(f)
            assert meta["n_channels"] == n_pl + n_surf, (
                f"Memmap n_channels={meta['n_channels']} != {n_pl+n_surf}"
            )
            T = meta["T"]; H = meta["H"]; W = meta["W"]
            arr = np.memmap(str(bin_p), dtype=np.float32, mode="r",
                            shape=(T, n_pl + n_surf, H, W))
            self.memmaps[y] = arr
            self.years_T[y] = T
            self.time_starts[y] = datetime(y, 1, 1)
            print(f"[memmap] year {y}: shape={arr.shape}, fp32 ({bin_p.stat().st_size/1024**3:.1f} GiB)",
                  flush=True)

        # Build index: (year, t0, tau_hours, tau_normalized)
        # Match ERA5ResNetODEDataset.index format
        step_per_day = 24
        self.index: List = []
        train_hour_set = set(train_hours) if train_hours else None
        eval_hour_set = set(eval_hours) if eval_hours else None
        for y in years:
            T = self.years_T[y]
            n_days = T // step_per_day
            for d in range(n_days):
                day_start = d * step_per_day
                for s in range(samples_per_date):
                    t0 = day_start + s * (step_per_day // max(1, samples_per_date))
                    if t0 + max_tau_hours >= T:
                        continue
                    for tau_h in range(1, max_tau_hours):
                        if train and train_hour_set and tau_h not in train_hour_set:
                            continue
                        if (not train) and eval_hour_set and tau_h not in eval_hour_set:
                            continue
                        tau_norm = tau_h / float(self.max_tau_hours)
                        self.index.append((y, t0, tau_h, tau_norm))

        # Load normalization stats
        with xr.open_dataset(stats_path) as ds_stats:
            stats_subset = ds_stats["climate_statistics"].sel(params=self.channel_names)
            self.mu = torch.from_numpy(stats_subset.isel(stats=0).values).float().view(-1, 1, 1)
            self.sigma = torch.from_numpy(stats_subset.isel(stats=1).values).float().view(-1, 1, 1)
            self.sigma = torch.clamp(self.sigma, min=1e-6)

        with open(surface_stats_path) as f:
            ss = json.load(f)
        smus, ssigs = [], []
        for v in self.surface_variables:
            smus.append(float(ss[v]["mean"]))
            ssigs.append(max(float(ss[v]["std"]), 1e-6))
        self.surface_mu = torch.tensor(smus).float().view(-1, 1, 1)
        self.surface_sigma = torch.tensor(ssigs).float().view(-1, 1, 1)

        # Static features
        self.static = None
        if static_path:
            self.static = torch.load(static_path, weights_only=False).float()

        # Channel groups for trainer (per-channel metrics)
        self.channel_groups: Dict[str, List[int]] = {}
        for i, v in enumerate(self.PL_NAMES):
            self.channel_groups[v] = [i * 4 + j for j in range(4)]
        for j, sv in enumerate(self.surface_variables):
            self.channel_groups[sv] = [n_pl + j]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        year, t0, tau_hours, tau = self.index[idx]
        arr = self.memmaps[year]
        n_pl = self.in_channels

        x0_array = np.array(arr[t0], dtype=np.float32, copy=True, order="C")
        x1_array = np.array(
            arr[t0 + self.max_tau_hours],
            dtype=np.float32,
            copy=True,
            order="C",
        )
        y_array = np.array(
            arr[t0 + tau_hours],
            dtype=np.float32,
            copy=True,
            order="C",
        )
        if self.release_memmap_pages:
            release_memmap_time_slices(
                arr,
                (t0, t0 + self.max_tau_hours, t0 + tau_hours),
            )
        x0 = torch.from_numpy(x0_array).float()
        x1 = torch.from_numpy(x1_array).float()
        y = torch.from_numpy(y_array).float()

        # Normalize (split into PL + surface)
        x0_pl = (x0[:n_pl] - self.mu) / self.sigma
        x1_pl = (x1[:n_pl] - self.mu) / self.sigma
        y_pl = (y[:n_pl] - self.mu) / self.sigma
        x0_s = (x0[n_pl:] - self.surface_mu) / self.surface_sigma
        x1_s = (x1[n_pl:] - self.surface_mu) / self.surface_sigma
        y_s = (y[n_pl:] - self.surface_mu) / self.surface_sigma

        x0 = torch.cat([x0_pl, x0_s], dim=0)
        x1 = torch.cat([x1_pl, x1_s], dim=0)
        y = torch.cat([y_pl, y_s], dim=0)

        x0 = torch.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)
        x1 = torch.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # time_emb (sin/cos for doy and hour)
        ts = self.time_starts[year] + timedelta(hours=int(t0))
        doy = ts.timetuple().tm_yday / 365.0
        hour = ts.hour / 24.0
        time_emb = torch.tensor([
            np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy),
            np.sin(2 * np.pi * hour), np.cos(2 * np.pi * hour),
        ], dtype=torch.float32)

        out = {
            "x0": x0, "x1": x1, "target": y,
            "time_emb": time_emb,
            "tau": torch.tensor([tau], dtype=torch.float32),
            "tau_hour": torch.tensor([tau_hours], dtype=torch.long),
        }
        if self.static is not None:
            out["static"] = self.static
        return out

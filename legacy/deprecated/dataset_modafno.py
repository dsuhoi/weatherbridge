# -*- coding: utf-8 -*-
"""
Специализированный датасет для ModAFNO.

Отличия от ERA5ResNetODEDataset:
- Вход: конкатенация x0 и x1 (а не отдельные тензоры)
- mod_input: tau + time_emb для модуляции
- Нет отдельного encoder для x0/x1 (всё обрабатывается совместно)
"""
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
)


class ERA5ModAFNODataset(Dataset):
    """
    Датасет для ModAFNO temporal interpolation.
    
    Возвращает:
        - input: конкатенация x0 и x1 [2C, H, W]
        - target: поле в момент tau [C, H, W]
        - mod_input: [tau, sin(day), cos(day), sin(hour), cos(hour)]
        - tau: скаляр [1]
        - time_emb: [4]
    """

    def __init__(
        self,
        data_dir: str,
        years: List[int],
        in_channels: Optional[int] = None,
        variables: Optional[List[str]] = None,
        pressure_levels: Optional[List[int]] = None,
        max_tau_hours: int = 6,
        samples_per_date: int = 3,
        train: bool = True,
        static_path: Optional[str] = None,
        mod_use_time_emb: bool = True,
        stats_path: str = "data/json_stats.nc",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.years = years
        self.max_tau_hours = max_tau_hours
        self.samples_per_date = samples_per_date
        self.train = train
        self.mod_use_time_emb = mod_use_time_emb

        # Dynamic channel setup
        self.variables = [v.upper() for v in (variables or DEFAULT_VARIABLES)]
        self.pressure_levels = pressure_levels or DEFAULT_PRESSURE_LEVELS

        channel_specs = [
            (v, lvl)
            for v in self.variables
            for lvl in self.pressure_levels
        ]
        if in_channels is None:
            in_channels = len(channel_specs)
        if in_channels < len(channel_specs):
            channel_specs = channel_specs[:in_channels]
        elif in_channels > len(channel_specs):
            print(
                f"Warning: Requested {in_channels} channels, but only "
                f"{len(channel_specs)} available. Using all available channels."
            )
            in_channels = len(channel_specs)

        self.in_channels = in_channels
        self.channel_names = [f"{v}{lvl}" for v, lvl in channel_specs]
        self.selected_setup = [
            (v.lower(), v, lvl)
            for v, lvl in channel_specs
        ]

        # Group channels by variable for metrics
        self.channel_groups: Dict[str, List[int]] = {}
        for idx, (v, _lvl) in enumerate(channel_specs):
            name = VAR_SHORT_TO_NAME.get(v, v)
            self.channel_groups.setdefault(name, []).append(idx)

        # 1. Загрузка Zarr файлов
        self.datasets: Dict[int, xr.Dataset] = {}
        for y in years:
            zarr_path = self.data_dir / f"{y}.zarr"
            self.datasets[y] = xr.open_zarr(zarr_path, consolidated=True)

        # 2. Загрузка статистики для нормализации
        stats_file = Path(stats_path)
        if not stats_file.exists():
            raise FileNotFoundError(
                f"Normalization stats file not found: {stats_file}"
            )
        with xr.open_dataset(stats_file) as ds_stats:
            stats_subset = ds_stats["climate_statistics"].sel(
                params=self.channel_names
            )
            mu = stats_subset.isel(stats=0).values
            sigma = stats_subset.isel(stats=1).values

        self.mu = torch.from_numpy(mu).float().view(-1, 1, 1)
        self.sigma = torch.from_numpy(sigma).float().view(-1, 1, 1)
        self.sigma = torch.clamp(self.sigma, min=1e-6)

        # 3. Static data (орография, маска моря и т.д.)
        self.use_static = static_path is not None
        self.static = None
        if self.use_static and static_path:
            static_ds = xr.open_dataset(static_path)
            self.static = torch.from_numpy(
                static_ds.to_array().values.astype(np.float32)
            )

        # 4. Индексация примеров
        tau_steps = TAU_TRAIN_STEPS if train else TAU_VAL_STEPS
        self.index: List[Tuple[int, int, int]] = []
        for y, ds in self.datasets.items():
            T = ds.sizes["time"]
            step_per_day = 24
            for day_start in range(0, T - max_tau_hours - 1, step_per_day):
                for s in range(samples_per_date):
                    t0 = day_start + s * (step_per_day // max(1, samples_per_date))
                    if t0 + max_tau_hours < T:
                        self.index.append((y, t0, s % len(tau_steps)))

    def _read_fields(self, ds: xr.Dataset, t: int) -> np.ndarray:
        """Прочитать все каналы для момента времени t."""
        fields = []
        for v_ds, _, lvl in self.selected_setup:
            arr = ds[v_ds].sel(pressure_level=lvl).isel(time=t).values
            fields.append(arr)
        return np.stack(fields, axis=0)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        year, t0, tau_idx = self.index[idx]
        ds = self.datasets[year]

        # Определяем tau и целевой час
        tau = (TAU_TRAIN_STEPS if self.train else TAU_VAL_STEPS)[tau_idx]
        tau_hours = int(round(tau * HOURS_PER_TAU_UNIT))
        tau_hours = max(0, min(tau_hours, self.max_tau_hours))

        # Читаем поля
        x0 = torch.from_numpy(
            self._read_fields(ds, t0).astype(np.float32)
        )
        x1 = torch.from_numpy(
            self._read_fields(ds, t0 + self.max_tau_hours).astype(np.float32)
        )
        target = torch.from_numpy(
            self._read_fields(ds, t0 + tau_hours).astype(np.float32)
        )

        # Нормализация
        x0 = (x0 - self.mu) / self.sigma
        x1 = (x1 - self.mu) / self.sigma
        target = (target - self.mu) / self.sigma

        # Time embedding для модуляции
        ts = datetime.utcfromtimestamp(
            ds.time.values[t0].astype("datetime64[s]").astype(int)
        )
        day = ts.timetuple().tm_yday / 365.0
        hour = ts.hour / 24.0
        time_emb = torch.tensor(
            [
                np.sin(2 * np.pi * day),
                np.cos(2 * np.pi * day),
                np.sin(2 * np.pi * hour),
                np.cos(2 * np.pi * hour),
            ],
            dtype=torch.float32,
        )

        # Формируем mod_input
        tau_tensor = torch.tensor([tau], dtype=torch.float32)
        if self.mod_use_time_emb:
            mod_input = torch.cat([tau_tensor, time_emb], dim=0)
        else:
            mod_input = tau_tensor

        # Вход модели: конкатенация x0 и x1
        x_input = torch.cat([x0, x1], dim=0)
        if self.static is not None:
            x_input = torch.cat([x_input, self.static], dim=0)

        out = {
            "input": x_input,          # [2C, H, W] или [2C+static, H, W]
            "target": target,          # [C, H, W]
            "x0": x0,                  # [C, H, W]
            "x1": x1,                  # [C, H, W]
            "tau": tau_tensor,         # [1]
            "time_emb": time_emb,      # [4]
            "mod_input": mod_input,    # [1] или [5]
        }
        if self.static is not None:
            out["static"] = self.static

        return out

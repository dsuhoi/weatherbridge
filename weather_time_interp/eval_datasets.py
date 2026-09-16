"""Lightweight Dataset wrappers used by EvaluationRunner without heavy deps.

This module mirrors the production training-side dataset adapter
but avoids pulling Lightning / model imports, so eval tooling can run in an
environment without the full training stack (e.g. tests with synthetic data).
The class behaviour MUST stay bit-identical to the trainer copy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class ERA5WeatherInterpDataset(Dataset):
    """Adapter over an ERA5 dataset for the WTI residual-linear backbones.

    Groups consecutive ``(year, t0)`` windows from the base dataset into one
    sample with stacked ``target`` / ``tau`` / ``tau_hour`` over ``nH`` hours.
    """

    def __init__(self, base_dataset: Dataset, delta_t_hours: float = 6.0) -> None:
        self.base_dataset = base_dataset
        self.delta_t_hours = float(delta_t_hours)
        self._grouped_indices: Optional[List[List[int]]] = None
        base_index = getattr(base_dataset, "index", None)
        if isinstance(base_index, list) and base_index:
            grouped: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
            for idx, item in enumerate(base_index):
                if not isinstance(item, tuple) or len(item) < 3:
                    grouped = {}
                    break
                key = (int(item[0]), int(item[1]))
                tau_hour = int(item[2])
                grouped.setdefault(key, []).append((tau_hour, idx))
            if grouped:
                self._grouped_indices = []
                for _key, pairs in grouped.items():
                    pairs.sort(key=lambda x: x[0])
                    self._grouped_indices.append([idx for _, idx in pairs])

    def __len__(self) -> int:
        if self._grouped_indices is not None:
            return len(self._grouped_indices)
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._grouped_indices is not None:
            samples = [self.base_dataset[i] for i in self._grouped_indices[idx]]
            first = samples[0]
            tau = torch.stack([s["tau"].float().view(1) for s in samples], dim=0)
            tau_hour = torch.stack([s["tau_hour"].long().view(1) for s in samples], dim=0)
            target = torch.stack([s["target"] for s in samples], dim=0)
            out = {
                "x0": first["x0"],
                "xT": first["x1"],
                "target": target,
                "tau": tau,
                "tau_hour": tau_hour,
                "cond": torch.tensor([self.delta_t_hours], dtype=torch.float32),
            }
            if "static" in first:
                out["static"] = first["static"]
            return out
        sample = self.base_dataset[idx]
        out = {
            "x0": sample["x0"],
            "xT": sample["x1"],
            "target": sample["target"],
            "tau": sample["tau"].float().view(1),
            "tau_hour": sample["tau_hour"].long().view(1),
            "cond": torch.tensor([self.delta_t_hours], dtype=torch.float32),
        }
        if "static" in sample:
            out["static"] = sample["static"]
        return out


# Backward-compat alias: many cluster-side scripts still import the legacy
# name. Keep both pointing at the same class so production keeps working
# while new code uses the cleaner name.
ERA5WeatherHermiteDataset = ERA5WeatherInterpDataset

__all__ = ["ERA5WeatherInterpDataset", "ERA5WeatherHermiteDataset"]

#!/usr/bin/env python3
"""Paired weekly-block tests for per-window physical diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.eval.paired_block_bootstrap import window_index_sha256
from weather_time_interp.metrics.physical_consistency import (
    GENERALIZATION_DIAGNOSTICS,
)


@dataclass(frozen=True)
class PhysicalWindows:
    year: np.ndarray
    t0: np.ndarray
    tau: np.ndarray
    values: dict[str, np.ndarray]

    @property
    def index_sha256(self) -> str:
        return window_index_sha256(self.year, self.t0, self.tau)


def load_physical_windows(
    path: Path,
    method: str = "model",
) -> PhysicalWindows:
    with np.load(path, allow_pickle=False) as data:
        year = np.asarray(data["year"])
        t0 = np.asarray(data["t0"])
        tau = np.asarray(data["tau"])
        if (
            year.ndim != 1
            or t0.ndim != 1
            or tau.ndim != 1
            or not (year.size == t0.size == tau.size)
            or year.size == 0
        ):
            raise ValueError(f"{path}: invalid physical window index")
        if not all(
            np.issubdtype(values.dtype, np.integer)
            for values in (year, t0, tau)
        ):
            raise ValueError(f"{path}: physical index must have integer dtype")
        values = {}
        for diagnostic in GENERALIZATION_DIAGNOSTICS:
            key = f"physical_{diagnostic}_{method}"
            if key not in data:
                raise ValueError(f"{path}: missing {key}")
            current = np.asarray(data[key], dtype=np.float64)
            if current.ndim != 2 or current.shape[0] != year.size:
                raise ValueError(f"{path}: invalid {key} shape {current.shape}")
            if not np.all(np.isfinite(current)) or np.any(current < 0.0):
                raise ValueError(
                    f"{path}: {key} must be finite and non-negative"
                )
            values[diagnostic] = current
        return PhysicalWindows(
            year=year,
            t0=t0,
            tau=tau,
            values=values,
        )


def compare_physical(
    left: PhysicalWindows,
    right: PhysicalWindows,
    *,
    taus: np.ndarray,
    block_days: int,
    draws: int,
    seed: int,
) -> dict:
    if block_days <= 0 or draws <= 0:
        raise ValueError("block_days and draws must be positive")
    for name in ("year", "t0", "tau"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            raise ValueError(f"window index mismatch in {name}")
    for diagnostic in GENERALIZATION_DIAGNOSTICS:
        left_values = left.values[diagnostic]
        right_values = right.values[diagnostic]
        if (
            left_values.ndim != 2
            or left_values.shape[0] != left.year.size
            or left_values.shape != right_values.shape
            or not np.all(np.isfinite(left_values))
            or not np.all(np.isfinite(right_values))
            or np.any(left_values < 0.0)
            or np.any(right_values < 0.0)
        ):
            raise ValueError(f"{diagnostic}: invalid physical values")
    requested_taus = sorted({int(value) for value in taus})
    if not requested_taus:
        raise ValueError("at least one tau must be requested")
    missing_taus = [
        tau for tau in requested_taus
        if not np.any(left.tau == tau)
    ]
    if missing_taus:
        raise ValueError(f"requested taus have no windows: {missing_taus}")
    selected = np.isin(left.tau, taus)
    year = left.year[selected].astype(np.int64)
    t0 = left.t0[selected].astype(np.int64)
    block_id = year * 100_000 + t0 // (24 * block_days)
    blocks = np.unique(block_id)
    if len(blocks) < 2:
        raise ValueError(
            "physical block test requires at least two blocks"
        )
    block_indices = [np.flatnonzero(block_id == block) for block in blocks]
    rng = np.random.default_rng(seed)
    sampled_blocks = rng.integers(
        0,
        len(blocks),
        size=(draws, len(blocks)),
    )

    def bootstrap(values: np.ndarray) -> np.ndarray:
        block_sums = np.asarray(
            [values[index].sum() for index in block_indices],
            dtype=np.float64,
        )
        block_counts = np.asarray(
            [len(index) for index in block_indices],
            dtype=np.float64,
        )
        return (
            block_sums[sampled_blocks].sum(axis=1)
            / block_counts[sampled_blocks].sum(axis=1)
        )

    diagnostics = {}
    for diagnostic in GENERALIZATION_DIAGNOSTICS:
        left_values = left.values[diagnostic][selected].mean(axis=1)
        right_values = right.values[diagnostic][selected].mean(axis=1)
        if left_values.shape != right_values.shape:
            raise ValueError(f"{diagnostic}: value shape mismatch")
        left_draws = bootstrap(left_values)
        right_draws = bootstrap(right_values)
        difference = left_values - right_values
        block_sums = np.asarray(
            [difference[index].sum() for index in block_indices],
            dtype=np.float64,
        )
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(draws, len(blocks)),
        )
        null_delta = signs @ block_sums / len(difference)
        observed = float(difference.mean())
        p_value = float(
            (
                np.count_nonzero(np.abs(null_delta) >= abs(observed)) + 1
            )
            / (draws + 1)
        )
        diagnostics[diagnostic] = {
            "better": "lower",
            "left": float(left_values.mean()),
            "right": float(right_values.mean()),
            "delta_left_minus_right": observed,
            "relative_delta_pct": (
                100.0 * observed / float(right_values.mean())
                if float(right_values.mean()) != 0.0
                else 0.0
            ),
            "delta_ci95": np.percentile(
                left_draws - right_draws,
                [2.5, 50.0, 97.5],
            ).tolist(),
            "p_paired_block_permutation": p_value,
        }
    return {
        "taus": sorted(int(value) for value in np.unique(left.tau[selected])),
        "n_windows": int(selected.sum()),
        "n_blocks": len(blocks),
        "block_days": block_days,
        "draws": draws,
        "diagnostics": diagnostics,
    }

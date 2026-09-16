#!/usr/bin/env python3
"""Paired block bootstrap for unified per-window interpolation metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def window_index_sha256(
    year: np.ndarray,
    t0: np.ndarray,
    tau: np.ndarray,
) -> str:
    """Hash the ordered evaluation index shared by paired metric arrays."""
    text = "\n".join(
        f"{current_year},{current_t0},{current_tau}"
        for current_year, current_t0, current_tau in zip(
            year,
            t0,
            tau,
            strict=True,
        )
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WindowMetrics:
    year: np.ndarray
    t0: np.ndarray
    tau: np.ndarray
    mse: np.ndarray
    channels: tuple[str, ...]

    @property
    def index_sha256(self) -> str:
        return window_index_sha256(self.year, self.t0, self.tau)


@dataclass(frozen=True)
class WindowScores:
    year: np.ndarray
    t0: np.ndarray
    tau: np.ndarray
    values: np.ndarray
    channels: tuple[str, ...]

    @property
    def index_sha256(self) -> str:
        return window_index_sha256(self.year, self.t0, self.tau)


def _validate_index(
    year: np.ndarray,
    t0: np.ndarray,
    tau: np.ndarray,
    *,
    n_windows: int,
    context: str,
) -> None:
    for name, values in (("year", year), ("t0", t0), ("tau", tau)):
        if values.ndim != 1 or values.size != n_windows:
            raise ValueError(
                f"{context}: {name} must be a length-{n_windows} vector"
            )
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{context}: {name} must have integer dtype")
    if n_windows == 0:
        raise ValueError(f"{context}: no windows")
    unique_index = set(
        zip(
            year.tolist(),
            t0.tolist(),
            tau.tolist(),
            strict=True,
        )
    )
    if len(unique_index) != n_windows:
        raise ValueError(f"{context}: duplicate (year, t0, tau) window index")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_resampling(
    *,
    tau_values: np.ndarray,
    taus: np.ndarray,
    block_days: int,
    draws: int,
) -> np.ndarray:
    if block_days <= 0:
        raise ValueError("block_days must be positive")
    if draws <= 0:
        raise ValueError("draws must be positive")
    tau_list = np.asarray(
        sorted({int(value) for value in taus}),
        dtype=np.int16,
    )
    if tau_list.size == 0:
        raise ValueError("at least one tau must be requested")
    missing = [
        int(tau)
        for tau in tau_list
        if not np.any(tau_values == tau)
    ]
    if missing:
        raise ValueError(f"requested taus have no windows: {missing}")
    return tau_list


def _draw_block_weights_covering_taus(
    counts: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Draw shared block weights conditional on representing every tau."""
    if counts.ndim != 2 or counts.shape[1] < 2:
        raise ValueError("block counts must have shape (n_tau, n_blocks>=2)")
    support = counts > 0
    if np.any(~support.any(axis=1)):
        raise ValueError("every requested tau must occur in at least one block")

    n_blocks = counts.shape[1]
    probabilities = np.full(n_blocks, 1.0 / n_blocks, dtype=np.float64)
    accepted: list[np.ndarray] = []
    accepted_count = 0
    rejected_count = 0
    attempted_count = 0
    max_attempts = max(100_000, draws * 1_000)
    while accepted_count < draws:
        needed = draws - accepted_count
        batch_size = min(8_192, max(256, needed * 2))
        weights = rng.multinomial(
            n_blocks,
            probabilities,
            size=batch_size,
        )
        represented = weights @ support.T
        valid = np.all(represented > 0, axis=1)
        valid_weights = weights[valid]
        take = min(needed, valid_weights.shape[0])
        if take:
            accepted.append(valid_weights[:take])
            accepted_count += take
        rejected_count += int(np.count_nonzero(~valid))
        attempted_count += batch_size
        if attempted_count >= max_attempts and accepted_count < draws:
            raise RuntimeError(
                "could not draw enough block samples containing every tau"
            )

    return np.concatenate(accepted, axis=0).astype(np.float64), rejected_count


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-Bonferroni adjusted p-values in the original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Holm correction requires a non-empty p-value vector")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate(
        (values.size - np.arange(values.size)) * ranked
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def load_metrics(path: Path, method: str = "model") -> WindowMetrics:
    with np.load(path, allow_pickle=False) as data:
        year = np.asarray(data["year"])
        t0 = np.asarray(data["t0"])
        tau = np.asarray(data["tau"])
        mse = np.asarray(data[f"mse_norm_{method}"], dtype=np.float64)
        channels = tuple(str(value) for value in data["channel_names"])
    if mse.ndim != 2 or mse.shape[1] != len(channels):
        raise ValueError(f"{path}: invalid MSE shape {mse.shape}")
    _validate_index(
        year,
        t0,
        tau,
        n_windows=mse.shape[0],
        context=str(path),
    )
    if not np.all(np.isfinite(mse)) or np.any(mse < 0.0):
        raise ValueError(f"{path}: MSE values must be finite and non-negative")
    return WindowMetrics(year, t0, tau, mse, channels)


def select_channels(
    metrics: WindowMetrics,
    channels: tuple[str, ...],
) -> WindowMetrics:
    """Select named channels in a deterministic caller-specified order."""
    if not channels:
        raise ValueError("at least one channel must be requested")
    if len(set(channels)) != len(channels):
        raise ValueError("requested channels must be unique")
    channel_to_index = {
        channel: index for index, channel in enumerate(metrics.channels)
    }
    missing = [channel for channel in channels if channel not in channel_to_index]
    if missing:
        raise ValueError(f"requested channels are missing: {missing}")
    indices = np.asarray(
        [channel_to_index[channel] for channel in channels],
        dtype=np.int64,
    )
    return WindowMetrics(
        year=metrics.year,
        t0=metrics.t0,
        tau=metrics.tau,
        mse=metrics.mse[:, indices],
        channels=channels,
    )


def load_acc_scores(path: Path, method: str = "model") -> WindowScores:
    with np.load(path, allow_pickle=False) as data:
        key = f"acc_{method}"
        if key not in data:
            raise ValueError(f"{path}: missing {key}")
        values = np.asarray(data[key], dtype=np.float64)
        channels = tuple(str(value) for value in data["acc_channel_names"])
        if values.ndim != 2 or values.shape[1] != len(channels):
            raise ValueError(f"{path}: invalid {key} shape {values.shape}")
        year = np.asarray(data["year"])
        t0 = np.asarray(data["t0"])
        tau = np.asarray(data["tau"])
        _validate_index(
            year,
            t0,
            tau,
            n_windows=values.shape[0],
            context=str(path),
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: non-finite {key} values")
        if np.any(values < -1.0001) or np.any(values > 1.0001):
            raise ValueError(f"{path}: {key} values fall outside [-1, 1]")
        return WindowScores(
            year=year,
            t0=t0,
            tau=tau,
            values=values,
            channels=channels,
        )


def load_temporal_metrics(
    path: Path,
    method: str = "model",
) -> WindowMetrics:
    """Load one curvature MSE per anchor window and prognostic channel."""
    with np.load(path, allow_pickle=False) as data:
        key = f"temporal_curvature_mse_{method}"
        if key not in data:
            raise ValueError(f"{path}: missing {key}")
        mse = np.asarray(data[key], dtype=np.float64)
        channels = tuple(str(value) for value in data["channel_names"])
        year = np.asarray(data["temporal_year"])
        t0 = np.asarray(data["temporal_t0"])
        center_count = np.asarray(data["temporal_center_count"])
    tau = np.zeros(mse.shape[0], dtype=np.int8)
    if mse.ndim != 2 or mse.shape[1] != len(channels):
        raise ValueError(f"{path}: invalid {key} shape {mse.shape}")
    _validate_index(
        year,
        t0,
        tau,
        n_windows=mse.shape[0],
        context=f"{path} temporal",
    )
    if center_count.shape != (mse.shape[0],):
        raise ValueError(f"{path}: invalid temporal centre-count shape")
    if np.any(center_count <= 0):
        raise ValueError(f"{path}: temporal centre count must be positive")
    if len(set(zip(year.tolist(), t0.tolist()))) != mse.shape[0]:
        raise ValueError(f"{path}: duplicate temporal anchor window")
    if not np.all(np.isfinite(mse)) or np.any(mse < 0.0):
        raise ValueError(
            f"{path}: temporal curvature MSE must be finite and non-negative"
        )
    return WindowMetrics(year, t0, tau, mse, channels)


def assert_paired(left: WindowMetrics, right: WindowMetrics) -> None:
    for name in ("year", "t0", "tau"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            raise ValueError(f"window index mismatch in {name}")
    if left.channels != right.channels:
        raise ValueError("channel order mismatch")
    if left.mse.shape != right.mse.shape:
        raise ValueError("MSE array shape mismatch")


def assert_scores_paired(left: WindowScores, right: WindowScores) -> None:
    for name in ("year", "t0", "tau"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            raise ValueError(f"window index mismatch in {name}")
    if left.channels != right.channels:
        raise ValueError("score channel order mismatch")
    if left.values.shape != right.values.shape:
        raise ValueError("score array shape mismatch")


def aggregate_rmse(metrics: WindowMetrics, taus: np.ndarray) -> float:
    values = []
    for tau in taus:
        mask = metrics.tau == tau
        if mask.any():
            values.append(np.sqrt(metrics.mse[mask].mean(axis=0)).mean())
    return float(np.mean(values))


def aggregate_score(metrics: WindowScores, taus: np.ndarray) -> float:
    values = []
    per_window = metrics.values.mean(axis=1)
    for tau in taus:
        mask = metrics.tau == tau
        if mask.any():
            values.append(float(per_window[mask].mean()))
    if not values:
        raise ValueError("no score windows selected")
    return float(np.mean(values))


def paired_block_score_test(
    left: WindowScores,
    right: WindowScores,
    *,
    taus: np.ndarray,
    block_days: int,
    draws: int,
    seed: int,
    better: str,
) -> dict:
    """Paired block bootstrap and permutation test for scalar window scores."""
    if better not in {"higher", "lower"}:
        raise ValueError("better must be 'higher' or 'lower'")
    assert_scores_paired(left, right)
    if left.values.ndim != 2 or left.values.shape[1] != len(left.channels):
        raise ValueError("score comparison has an invalid value shape")
    _validate_index(
        left.year,
        left.t0,
        left.tau,
        n_windows=left.values.shape[0],
        context="score comparison",
    )
    if not np.all(np.isfinite(left.values)) or not np.all(
        np.isfinite(right.values)
    ):
        raise ValueError("score comparison contains non-finite values")
    tau_list = _validate_resampling(
        tau_values=left.tau,
        taus=taus,
        block_days=block_days,
        draws=draws,
    )
    selected = np.isin(left.tau, taus)
    if not selected.any():
        raise ValueError("no score windows selected")
    year = left.year[selected].astype(np.int64)
    t0 = left.t0[selected].astype(np.int64)
    tau_values = left.tau[selected]
    left_values = left.values[selected].mean(axis=1)
    right_values = right.values[selected].mean(axis=1)

    block_id = year * 100_000 + t0 // (24 * block_days)
    blocks = np.unique(block_id)
    n_blocks = len(blocks)
    if n_blocks < 2:
        raise ValueError("paired block test requires at least two blocks")
    left_sums = np.zeros((len(tau_list), n_blocks), dtype=np.float64)
    right_sums = np.zeros_like(left_sums)
    counts = np.zeros_like(left_sums)
    for tau_index, tau in enumerate(tau_list):
        for block_index, block in enumerate(blocks):
            mask = (tau_values == tau) & (block_id == block)
            if mask.any():
                left_sums[tau_index, block_index] = left_values[mask].sum()
                right_sums[tau_index, block_index] = right_values[mask].sum()
                counts[tau_index, block_index] = int(mask.sum())

    rng = np.random.default_rng(seed)
    block_weights, rejected_draws = _draw_block_weights_covering_taus(
        counts,
        draws=draws,
        rng=rng,
    )
    left_draws = np.empty((draws, len(tau_list)), dtype=np.float64)
    right_draws = np.empty_like(left_draws)
    permutation_deltas = np.empty_like(left_draws)
    swap_signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(draws, n_blocks),
    )
    for tau_index in range(len(tau_list)):
        sample_count = block_weights @ counts[tau_index]
        if np.any(sample_count <= 0):
            raise ValueError("bootstrap draw contains no windows for a tau")
        left_draws[:, tau_index] = (
            block_weights @ left_sums[tau_index]
        ) / sample_count
        right_draws[:, tau_index] = (
            block_weights @ right_sums[tau_index]
        ) / sample_count
        total_count = counts[tau_index].sum()
        if total_count <= 0:
            raise ValueError(f"no score windows for tau={tau_list[tau_index]}")
        permutation_deltas[:, tau_index] = (
            swap_signs
            @ (left_sums[tau_index] - right_sums[tau_index])
            / total_count
        )

    left_point = aggregate_score(left, tau_list)
    right_point = aggregate_score(right, tau_list)
    observed_delta = left_point - right_point

    def permutation_p(null_values: np.ndarray, observed: float) -> float:
        exceedances = np.count_nonzero(np.abs(null_values) >= abs(observed))
        return float((exceedances + 1) / (draws + 1))

    delta_draws = left_draws.mean(axis=1) - right_draws.mean(axis=1)
    per_tau = {}
    for tau_index, tau in enumerate(tau_list):
        left_tau = aggregate_score(left, np.asarray([tau]))
        right_tau = aggregate_score(right, np.asarray([tau]))
        per_tau[str(int(tau))] = {
            "left": left_tau,
            "right": right_tau,
            "delta_left_minus_right": left_tau - right_tau,
            "delta_ci95": np.percentile(
                left_draws[:, tau_index] - right_draws[:, tau_index],
                [2.5, 50.0, 97.5],
            ).tolist(),
            "p_paired_block_permutation": permutation_p(
                permutation_deltas[:, tau_index],
                left_tau - right_tau,
            ),
        }
    return {
        "metric": "mean_window_acc" if better == "higher" else "window_score",
        "better": better,
        "taus": tau_list.tolist(),
        "block_days": block_days,
        "n_blocks": n_blocks,
        "n_windows": int(selected.sum()),
        "bootstrap_draws": draws,
        "bootstrap_rejected_draws_missing_tau": rejected_draws,
        "seed": seed,
        "left": left_point,
        "right": right_point,
        "delta_left_minus_right": observed_delta,
        "relative_delta_pct": (
            100.0 * observed_delta / abs(right_point)
            if right_point != 0.0
            else 0.0
        ),
        "delta_ci95": np.percentile(
            delta_draws,
            [2.5, 50.0, 97.5],
        ).tolist(),
        "p_paired_block_permutation": permutation_p(
            permutation_deltas.mean(axis=1),
            observed_delta,
        ),
        "per_tau": per_tau,
    }


def paired_cellwise_score_tests(
    left: WindowScores,
    right: WindowScores,
    *,
    taus: np.ndarray,
    block_days: int,
    draws: int,
    seed: int,
    better: str,
    family_alpha: float = 0.05,
) -> dict:
    """Attach Holm-corrected channel-by-tau tests to an aggregate score test."""
    if not 0.0 < family_alpha < 1.0:
        raise ValueError("family_alpha must lie strictly between 0 and 1")
    assert_scores_paired(left, right)
    result = paired_block_score_test(
        left,
        right,
        taus=taus,
        block_days=block_days,
        draws=draws,
        seed=seed,
        better=better,
    )
    tau_list = np.asarray(result["taus"], dtype=np.int16)
    records = []
    per_channel_tau: dict[str, dict[str, dict]] = {}
    for tau in tau_list:
        channel_rows = {}
        for channel_index, channel in enumerate(left.channels):
            left_channel = WindowScores(
                left.year,
                left.t0,
                left.tau,
                left.values[:, channel_index : channel_index + 1],
                (channel,),
            )
            right_channel = WindowScores(
                right.year,
                right.t0,
                right.tau,
                right.values[:, channel_index : channel_index + 1],
                (channel,),
            )
            row = paired_block_score_test(
                left_channel,
                right_channel,
                taus=np.asarray([tau]),
                block_days=block_days,
                draws=draws,
                seed=seed,
                better=better,
            )
            cell = {
                key: row[key]
                for key in (
                    "left",
                    "right",
                    "delta_left_minus_right",
                    "delta_ci95",
                    "p_paired_block_permutation",
                )
            }
            channel_rows[channel] = cell
            records.append(cell)
        per_channel_tau[str(int(tau))] = channel_rows

    adjusted = _holm_adjust(
        np.asarray(
            [row["p_paired_block_permutation"] for row in records],
            dtype=np.float64,
        )
    )
    pointwise_better = []
    pointwise_worse = []
    significant_better = []
    significant_worse = []
    for row, p_holm in zip(records, adjusted, strict=True):
        row["p_holm"] = float(p_holm)
        delta = float(row["delta_left_minus_right"])
        is_better = delta > 0.0 if better == "higher" else delta < 0.0
        is_worse = delta < 0.0 if better == "higher" else delta > 0.0
        pointwise_better.append(is_better)
        pointwise_worse.append(is_worse)
        significant_better.append(is_better and p_holm < family_alpha)
        significant_worse.append(is_worse and p_holm < family_alpha)
    result["per_channel_tau"] = per_channel_tau
    result["cellwise_family"] = {
        "correction": "Holm-Bonferroni",
        "alpha": family_alpha,
        "n_hypotheses": len(records),
        "n_pointwise_left_better": sum(pointwise_better),
        "n_pointwise_left_worse": sum(pointwise_worse),
        "n_significant_left_better_holm": sum(significant_better),
        "n_significant_left_worse_holm": sum(significant_worse),
        "all_pointwise_left_better": all(pointwise_better),
        "all_left_better_holm": all(significant_better),
        "any_left_worse_holm": any(significant_worse),
    }
    return result


def paired_block_bootstrap(
    left: WindowMetrics,
    right: WindowMetrics,
    *,
    taus: np.ndarray,
    block_days: int,
    draws: int,
    seed: int,
    include_cellwise: bool = False,
    family_alpha: float = 0.05,
) -> dict:
    assert_paired(left, right)
    if left.mse.ndim != 2 or left.mse.shape[1] != len(left.channels):
        raise ValueError("RMSE comparison has an invalid MSE shape")
    _validate_index(
        left.year,
        left.t0,
        left.tau,
        n_windows=left.mse.shape[0],
        context="RMSE comparison",
    )
    if (
        not np.all(np.isfinite(left.mse))
        or not np.all(np.isfinite(right.mse))
        or np.any(left.mse < 0.0)
        or np.any(right.mse < 0.0)
    ):
        raise ValueError(
            "RMSE comparison requires finite, non-negative MSE values"
        )
    tau_list = _validate_resampling(
        tau_values=left.tau,
        taus=taus,
        block_days=block_days,
        draws=draws,
    )
    if not 0.0 < family_alpha < 1.0:
        raise ValueError("family_alpha must lie strictly between 0 and 1")
    selected = np.isin(left.tau, taus)
    year = left.year[selected].astype(np.int64)
    t0 = left.t0[selected].astype(np.int64)
    tau_values = left.tau[selected]
    left_mse = left.mse[selected]
    right_mse = right.mse[selected]

    block_id = year * 100_000 + t0 // (24 * block_days)
    blocks = np.unique(block_id)
    n_blocks = len(blocks)
    if n_blocks < 2:
        raise ValueError("paired block test requires at least two blocks")
    n_channels = left_mse.shape[1]

    left_sums = np.zeros((len(tau_list), n_blocks, n_channels), dtype=np.float64)
    right_sums = np.zeros_like(left_sums)
    counts = np.zeros((len(tau_list), n_blocks), dtype=np.float64)
    for tau_index, tau in enumerate(tau_list):
        for block_index, block in enumerate(blocks):
            mask = (tau_values == tau) & (block_id == block)
            if mask.any():
                left_sums[tau_index, block_index] = left_mse[mask].sum(axis=0)
                right_sums[tau_index, block_index] = right_mse[mask].sum(axis=0)
                counts[tau_index, block_index] = int(mask.sum())

    rng = np.random.default_rng(seed)
    block_weights, rejected_draws = _draw_block_weights_covering_taus(
        counts,
        draws=draws,
        rng=rng,
    )
    left_scores = np.zeros((draws, len(tau_list)), dtype=np.float64)
    right_scores = np.zeros_like(left_scores)
    left_channel_scores = (
        np.zeros((draws, len(tau_list), n_channels), dtype=np.float64)
        if include_cellwise
        else None
    )
    right_channel_scores = (
        np.zeros((draws, len(tau_list), n_channels), dtype=np.float64)
        if include_cellwise
        else None
    )
    for tau_index in range(len(tau_list)):
        sample_count = block_weights @ counts[tau_index]
        if np.any(sample_count <= 0):
            raise ValueError("bootstrap draw contains no windows for a tau")
        left_mean = (
            block_weights @ left_sums[tau_index]
        ) / sample_count[:, None]
        right_mean = (
            block_weights @ right_sums[tau_index]
        ) / sample_count[:, None]
        left_rmse_draws = np.sqrt(left_mean)
        right_rmse_draws = np.sqrt(right_mean)
        left_scores[:, tau_index] = left_rmse_draws.mean(axis=1)
        right_scores[:, tau_index] = right_rmse_draws.mean(axis=1)
        if include_cellwise:
            assert left_channel_scores is not None
            assert right_channel_scores is not None
            left_channel_scores[:, tau_index] = left_rmse_draws
            right_channel_scores[:, tau_index] = right_rmse_draws

    delta_draws = left_scores.mean(axis=1) - right_scores.mean(axis=1)
    left_point = aggregate_rmse(left, tau_list)
    right_point = aggregate_rmse(right, tau_list)

    # A bootstrap distribution estimates uncertainty; it is not a null
    # distribution. Build the latter by swapping paired model errors within
    # complete time blocks and recomputing the nonlinear RMSE aggregate.
    swap_signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(draws, n_blocks),
    )
    permutation_deltas = np.zeros((draws, len(tau_list)), dtype=np.float64)
    permutation_channel_deltas = (
        np.zeros((draws, len(tau_list), n_channels), dtype=np.float64)
        if include_cellwise
        else None
    )
    for tau_index in range(len(tau_list)):
        total_count = counts[tau_index].sum()
        midpoint = 0.5 * (
            left_sums[tau_index].sum(axis=0)
            + right_sums[tau_index].sum(axis=0)
        )
        half_difference = 0.5 * (
            left_sums[tau_index] - right_sums[tau_index]
        )
        permuted_left = midpoint + swap_signs @ half_difference
        permuted_right = midpoint - swap_signs @ half_difference
        permuted_left_channel_rmse = np.sqrt(permuted_left / total_count)
        permuted_right_channel_rmse = np.sqrt(permuted_right / total_count)
        permuted_left_rmse = permuted_left_channel_rmse.mean(axis=1)
        permuted_right_rmse = permuted_right_channel_rmse.mean(axis=1)
        permutation_deltas[:, tau_index] = (
            permuted_left_rmse - permuted_right_rmse
        )
        if include_cellwise:
            assert permutation_channel_deltas is not None
            permutation_channel_deltas[:, tau_index] = (
                permuted_left_channel_rmse - permuted_right_channel_rmse
            )

    observed_per_tau = np.asarray(
        [
            aggregate_rmse(left, np.asarray([tau]))
            - aggregate_rmse(right, np.asarray([tau]))
            for tau in tau_list
        ]
    )

    def permutation_p(
        null_values: np.ndarray,
        observed: float,
    ) -> float:
        exceedances = np.count_nonzero(np.abs(null_values) >= abs(observed))
        return float((exceedances + 1) / (draws + 1))

    def permutation_p_one_sided(
        null_values: np.ndarray,
        observed: float,
        *,
        alternative: str,
    ) -> float:
        if alternative == "less":
            exceedances = np.count_nonzero(null_values <= observed)
        elif alternative == "greater":
            exceedances = np.count_nonzero(null_values >= observed)
        else:
            raise ValueError("alternative must be 'less' or 'greater'")
        return float((exceedances + 1) / (draws + 1))

    observed_delta = left_point - right_point
    p_two_sided = permutation_p(
        permutation_deltas.mean(axis=1),
        observed_delta,
    )
    per_tau = {}
    for tau_index, tau in enumerate(tau_list):
        left_tau = aggregate_rmse(left, np.asarray([tau]))
        right_tau = aggregate_rmse(right, np.asarray([tau]))
        delta_tau = left_scores[:, tau_index] - right_scores[:, tau_index]
        per_tau[str(int(tau))] = {
            "left_rmse": left_tau,
            "right_rmse": right_tau,
            "delta_left_minus_right": left_tau - right_tau,
            "relative_delta_pct": 100.0 * (left_tau / right_tau - 1.0),
            "delta_ci95": np.percentile(
                delta_tau,
                [2.5, 50.0, 97.5],
            ).tolist(),
            "p_paired_block_permutation": permutation_p(
                permutation_deltas[:, tau_index],
                observed_per_tau[tau_index],
            ),
            "p_left_better_one_sided": permutation_p_one_sided(
                permutation_deltas[:, tau_index],
                observed_per_tau[tau_index],
                alternative="less",
            ),
            "p_left_worse_one_sided": permutation_p_one_sided(
                permutation_deltas[:, tau_index],
                observed_per_tau[tau_index],
                alternative="greater",
            ),
        }
    result = {
        "taus": tau_list.tolist(),
        "block_days": block_days,
        "n_blocks": n_blocks,
        "n_windows": int(selected.sum()),
        "bootstrap_draws": draws,
        "bootstrap_rejected_draws_missing_tau": rejected_draws,
        "seed": seed,
        "left_rmse": left_point,
        "right_rmse": right_point,
        "left_rmse_ci95": np.percentile(
            left_scores.mean(axis=1),
            [2.5, 50.0, 97.5],
        ).tolist(),
        "right_rmse_ci95": np.percentile(
            right_scores.mean(axis=1),
            [2.5, 50.0, 97.5],
        ).tolist(),
        "delta_left_minus_right": left_point - right_point,
        "relative_delta_pct": 100.0 * (left_point / right_point - 1.0),
        "delta_ci95": np.percentile(delta_draws, [2.5, 50.0, 97.5]).tolist(),
        "p_paired_block_permutation": p_two_sided,
        "p_left_better_one_sided": permutation_p_one_sided(
            permutation_deltas.mean(axis=1),
            observed_delta,
            alternative="less",
        ),
        "p_left_worse_one_sided": permutation_p_one_sided(
            permutation_deltas.mean(axis=1),
            observed_delta,
            alternative="greater",
        ),
        "per_tau": per_tau,
    }
    if include_cellwise:
        assert left_channel_scores is not None
        assert right_channel_scores is not None
        assert permutation_channel_deltas is not None
        left_cell = np.empty((len(tau_list), n_channels), dtype=np.float64)
        right_cell = np.empty_like(left_cell)
        for tau_index in range(len(tau_list)):
            total_count = counts[tau_index].sum()
            left_cell[tau_index] = np.sqrt(
                left_sums[tau_index].sum(axis=0) / total_count
            )
            right_cell[tau_index] = np.sqrt(
                right_sums[tau_index].sum(axis=0) / total_count
            )

        observed_cell = left_cell - right_cell
        bootstrap_cell_delta = left_channel_scores - right_channel_scores
        p_better = np.empty_like(observed_cell)
        p_worse = np.empty_like(observed_cell)
        p_two_sided_cell = np.empty_like(observed_cell)
        for tau_index in range(len(tau_list)):
            for channel_index in range(n_channels):
                null = permutation_channel_deltas[:, tau_index, channel_index]
                observed = observed_cell[tau_index, channel_index]
                p_two_sided_cell[tau_index, channel_index] = permutation_p(
                    null,
                    observed,
                )
                p_better[tau_index, channel_index] = permutation_p_one_sided(
                    null,
                    observed,
                    alternative="less",
                )
                p_worse[tau_index, channel_index] = permutation_p_one_sided(
                    null,
                    observed,
                    alternative="greater",
                )

        p_better_holm = _holm_adjust(p_better.ravel()).reshape(p_better.shape)
        p_worse_holm = _holm_adjust(p_worse.ravel()).reshape(p_worse.shape)
        per_channel_tau: dict[str, dict[str, dict[str, float | list[float]]]] = {}
        for tau_index, tau in enumerate(tau_list):
            channel_rows = {}
            for channel_index, channel in enumerate(left.channels):
                delta = observed_cell[tau_index, channel_index]
                channel_rows[channel] = {
                    "left_rmse": float(left_cell[tau_index, channel_index]),
                    "right_rmse": float(right_cell[tau_index, channel_index]),
                    "delta_left_minus_right": float(delta),
                    "relative_delta_pct": float(
                        100.0
                        * (
                            left_cell[tau_index, channel_index]
                            / right_cell[tau_index, channel_index]
                            - 1.0
                        )
                    ),
                    "delta_ci95": np.percentile(
                        bootstrap_cell_delta[:, tau_index, channel_index],
                        [2.5, 50.0, 97.5],
                    ).tolist(),
                    "p_paired_block_permutation": float(
                        p_two_sided_cell[tau_index, channel_index]
                    ),
                    "p_left_better_one_sided": float(
                        p_better[tau_index, channel_index]
                    ),
                    "p_left_worse_one_sided": float(
                        p_worse[tau_index, channel_index]
                    ),
                    "p_left_better_holm": float(
                        p_better_holm[tau_index, channel_index]
                    ),
                    "p_left_worse_holm": float(
                        p_worse_holm[tau_index, channel_index]
                    ),
                }
            per_channel_tau[str(int(tau))] = channel_rows

        pointwise_better = observed_cell < 0.0
        significant_better = pointwise_better & (p_better_holm < family_alpha)
        significant_worse = (observed_cell > 0.0) & (
            p_worse_holm < family_alpha
        )
        result["per_channel_tau"] = per_channel_tau
        result["cellwise_family"] = {
            "correction": "Holm-Bonferroni",
            "alpha": family_alpha,
            "n_hypotheses": int(observed_cell.size),
            "n_pointwise_left_better": int(pointwise_better.sum()),
            "n_pointwise_left_worse": int((observed_cell > 0.0).sum()),
            "n_significant_left_better_holm": int(significant_better.sum()),
            "n_significant_left_worse_holm": int(significant_worse.sum()),
            "all_pointwise_left_better": bool(np.all(pointwise_better)),
            "all_left_better_holm": bool(np.all(significant_better)),
            "any_left_worse_holm": bool(np.any(significant_worse)),
        }
    return result


def parse_entry(value: str) -> tuple[str, Path, str]:
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("expected NAME:NPZ[:METHOD]")
    return parts[0], Path(parts[1]), parts[2] if len(parts) == 3 else "model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=parse_entry)
    parser.add_argument("--right", action="append", required=True, type=parse_entry)
    parser.add_argument("--taus", default="")
    parser.add_argument(
        "--channels",
        default="",
        help="comma-separated channel names; empty selects every channel",
    )
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--cellwise", action="store_true")
    parser.add_argument("--family-alpha", type=float, default=0.05)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    left_name, left_path, left_method = args.left
    left = load_metrics(left_path, left_method)
    requested_channels = tuple(
        value.strip() for value in args.channels.split(",") if value.strip()
    )
    if requested_channels:
        left = select_channels(left, requested_channels)
    taus = (
        np.asarray(
            [int(value) for value in args.taus.split(",") if value],
            dtype=np.int16,
        )
        if args.taus
        else np.unique(left.tau)
    )
    comparisons = {}
    right_sha256 = {}
    for right_name, right_path, right_method in args.right:
        right = load_metrics(right_path, right_method)
        if requested_channels:
            right = select_channels(right, requested_channels)
        comparisons[right_name] = paired_block_bootstrap(
            left,
            right,
            taus=taus,
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
            include_cellwise=args.cellwise,
            family_alpha=args.family_alpha,
        )
        right_sha256[right_name] = sha256_file(right_path)

    output = {
        "schema_version": 1,
        "left": left_name,
        "left_path": str(left_path),
        "left_sha256": sha256_file(left_path),
        "right_sha256": right_sha256,
        "index_sha256": left.index_sha256,
        "channels": list(left.channels),
        "comparisons": comparisons,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

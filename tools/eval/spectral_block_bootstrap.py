#!/usr/bin/env python3
"""Paired bootstrap over calendar-week bins for spectral diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SpectralWindows:
    year: np.ndarray
    t0: np.ndarray
    energy_ratio: np.ndarray
    shape_error: np.ndarray
    coherence: np.ndarray
    signed_cospectrum: np.ndarray
    channels: tuple[str, ...]
    tau: int


def load_windows(path: Path) -> SpectralWindows:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "window_year",
            "window_t0",
            "window_hf_energy_ratio",
            "window_hf_log_shape_error",
            "window_hf_coherence",
            "window_hf_signed_cospectrum",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"{path} lacks schema-v6 arrays: {sorted(missing)}"
            )
        result = SpectralWindows(
            year=np.asarray(data["window_year"]),
            t0=np.asarray(data["window_t0"]),
            energy_ratio=np.asarray(
                data["window_hf_energy_ratio"],
                dtype=np.float64,
            ),
            shape_error=np.asarray(
                data["window_hf_log_shape_error"],
                dtype=np.float64,
            ),
            coherence=np.asarray(
                data["window_hf_coherence"],
                dtype=np.float64,
            ),
            signed_cospectrum=np.asarray(
                data["window_hf_signed_cospectrum"],
                dtype=np.float64,
            ),
            channels=tuple(str(value) for value in data["channel_names"]),
            tau=int(data["tau"]),
        )
    _validate_windows(result, context=str(path))
    return result


def load_vector_windows(path: Path) -> SpectralWindows:
    """Load vector-SHT windows and flatten wind-pair/component axes."""
    with np.load(path, allow_pickle=False) as data:
        required = {
            "window_year",
            "window_t0",
            "window_vector_hf_energy_ratio",
            "window_vector_hf_log_shape_error",
            "window_vector_hf_coherence",
            "window_vector_hf_signed_cospectrum",
            "wind_pair_names",
            "vector_component_names",
            "tau",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"{path} lacks schema-v6 vector arrays: {sorted(missing)}"
            )
        pairs = [str(value) for value in data["wind_pair_names"]]
        components = [str(value) for value in data["vector_component_names"]]
        channels = tuple(
            f"{pair}:{component}" for pair in pairs for component in components
        )

        def flattened(name: str) -> np.ndarray:
            values = np.asarray(data[name], dtype=np.float64)
            return values.reshape(values.shape[0], -1)

        result = SpectralWindows(
            year=np.asarray(data["window_year"]),
            t0=np.asarray(data["window_t0"]),
            energy_ratio=flattened("window_vector_hf_energy_ratio"),
            shape_error=flattened("window_vector_hf_log_shape_error"),
            coherence=flattened("window_vector_hf_coherence"),
            signed_cospectrum=flattened(
                "window_vector_hf_signed_cospectrum"
            ),
            channels=channels,
            tau=int(data["tau"]),
        )
    _validate_windows(result, context=f"{path} vector diagnostics")
    return result


def _validate_windows(data: SpectralWindows, *, context: str) -> None:
    n_windows = data.year.size
    if (
        data.year.ndim != 1
        or data.t0.ndim != 1
        or data.t0.size != n_windows
        or n_windows == 0
    ):
        raise ValueError(f"{context}: invalid spectral window index")
    if not np.issubdtype(data.year.dtype, np.integer) or not np.issubdtype(
        data.t0.dtype,
        np.integer,
    ):
        raise ValueError(f"{context}: spectral index must have integer dtype")
    if len(set(zip(data.year.tolist(), data.t0.tolist(), strict=True))) != n_windows:
        raise ValueError(f"{context}: duplicate spectral window index")
    expected_shape = (n_windows, len(data.channels))
    if (
        data.energy_ratio.shape != expected_shape
        or data.shape_error.shape != expected_shape
        or data.coherence.shape != expected_shape
        or data.signed_cospectrum.shape != expected_shape
    ):
        raise ValueError(
            f"{context}: spectral arrays must have shape {expected_shape}"
        )
    if (
        not np.all(np.isfinite(data.energy_ratio))
        or not np.all(np.isfinite(data.shape_error))
        or not np.all(np.isfinite(data.coherence))
        or not np.all(np.isfinite(data.signed_cospectrum))
        or np.any(data.energy_ratio < 0.0)
        or np.any(data.shape_error < 0.0)
        or np.any(data.coherence < 0.0)
        or np.any(data.coherence > 1.0)
        or np.any(data.signed_cospectrum < -1.0)
        or np.any(data.signed_cospectrum > 1.0)
    ):
        raise ValueError(
            f"{context}: spectral values must be finite and errors non-negative"
        )


def compare(
    left: SpectralWindows,
    right: SpectralWindows,
    *,
    channel_indices: np.ndarray,
    block_days: int,
    draws: int,
    seed: int,
    cellwise: bool = False,
) -> dict:
    _validate_windows(left, context="left spectral input")
    _validate_windows(right, context="right spectral input")
    if block_days <= 0 or draws <= 0:
        raise ValueError("block_days and draws must be positive")
    if left.tau != right.tau:
        raise ValueError("tau mismatch")
    if left.channels != right.channels:
        raise ValueError("channel order mismatch")
    if not np.array_equal(left.year, right.year) or not np.array_equal(
        left.t0,
        right.t0,
    ):
        raise ValueError("window index mismatch")
    channel_indices = np.asarray(channel_indices)
    if (
        channel_indices.ndim != 1
        or channel_indices.size == 0
        or not np.issubdtype(channel_indices.dtype, np.integer)
        or np.any(channel_indices < 0)
        or np.any(channel_indices >= len(left.channels))
        or len(np.unique(channel_indices)) != channel_indices.size
    ):
        raise ValueError("channel_indices must be unique valid integers")

    def scores(
        data: SpectralWindows,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ratio = np.clip(data.energy_ratio[:, channel_indices], 1e-8, 1e8)
        energy_error = np.abs(np.log(ratio)).mean(axis=1)
        shape_error = data.shape_error[:, channel_indices].mean(axis=1)
        coherence = data.coherence[:, channel_indices].mean(axis=1)
        signed_cospectrum = data.signed_cospectrum[:, channel_indices].mean(axis=1)
        return energy_error, shape_error, coherence, signed_cospectrum

    (
        left_energy,
        left_shape,
        left_coherence,
        left_signed_cospectrum,
    ) = scores(left)
    (
        right_energy,
        right_shape,
        right_coherence,
        right_signed_cospectrum,
    ) = scores(right)
    block_id = (
        left.year.astype(np.int64) * 100_000
        + left.t0.astype(np.int64) // (24 * block_days)
    )
    blocks = np.unique(block_id)
    if len(blocks) < 2:
        raise ValueError("spectral block test requires at least two blocks")
    block_indices = [np.flatnonzero(block_id == block) for block in blocks]
    rng = np.random.default_rng(seed)
    sampled_blocks = rng.integers(0, len(blocks), size=(draws, len(blocks)))

    def bootstrap(values: np.ndarray) -> np.ndarray:
        block_sums = np.asarray([values[index].sum() for index in block_indices])
        block_counts = np.asarray([len(index) for index in block_indices])
        sums = block_sums[sampled_blocks].sum(axis=1)
        counts = block_counts[sampled_blocks].sum(axis=1)
        return sums / counts

    left_energy_draws = bootstrap(left_energy)
    right_energy_draws = bootstrap(right_energy)
    left_shape_draws = bootstrap(left_shape)
    right_shape_draws = bootstrap(right_shape)
    left_coherence_draws = bootstrap(left_coherence)
    right_coherence_draws = bootstrap(right_coherence)
    left_signed_cospectrum_draws = bootstrap(left_signed_cospectrum)
    right_signed_cospectrum_draws = bootstrap(right_signed_cospectrum)

    def paired_permutation_p(
        left_values: np.ndarray,
        right_values: np.ndarray,
    ) -> float:
        differences = left_values - right_values
        block_sums = np.asarray(
            [differences[index].sum() for index in block_indices],
            dtype=np.float64,
        )
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(draws, len(blocks)),
        )
        null_deltas = signs @ block_sums / len(differences)
        observed = float(differences.mean())
        exceedances = np.count_nonzero(np.abs(null_deltas) >= abs(observed))
        return float((exceedances + 1) / (draws + 1))

    def metric_payload(
        left_values: np.ndarray,
        right_values: np.ndarray,
        left_draws: np.ndarray,
        right_draws: np.ndarray,
        better: str,
    ) -> dict:
        delta_draws = left_draws - right_draws
        return {
            "better": better,
            "left": float(left_values.mean()),
            "right": float(right_values.mean()),
            "delta_left_minus_right": float(
                left_values.mean() - right_values.mean()
            ),
            "delta_ci95": np.percentile(
                delta_draws,
                [2.5, 50.0, 97.5],
            ).tolist(),
            "p_paired_block_permutation": paired_permutation_p(
                left_values,
                right_values,
            ),
        }

    result = {
        "tau": left.tau,
        "n_windows": len(left.year),
        "n_blocks": len(blocks),
        "block_days": block_days,
        "block_definition": "calendar_bins_over_dense_evaluation_windows",
        "n_unique_days": len(np.unique(left.year.astype(np.int64) * 1000 + left.t0 // 24)),
        "median_windows_per_block": float(
            np.median([len(index) for index in block_indices])
        ),
        "draws": draws,
        "energy_log_error": metric_payload(
            left_energy,
            right_energy,
            left_energy_draws,
            right_energy_draws,
            "lower",
        ),
        "shape_log_error": metric_payload(
            left_shape,
            right_shape,
            left_shape_draws,
            right_shape_draws,
            "lower",
        ),
        "coherence": metric_payload(
            left_coherence,
            right_coherence,
            left_coherence_draws,
            right_coherence_draws,
            "higher",
        ),
        "signed_cospectrum": metric_payload(
            left_signed_cospectrum,
            right_signed_cospectrum,
            left_signed_cospectrum_draws,
            right_signed_cospectrum_draws,
            "higher",
        ),
    }
    if not cellwise:
        return result

    per_channel: dict[str, dict] = {}
    for channel_index in channel_indices:
        channel_name = left.channels[int(channel_index)]
        left_ratio = np.clip(left.energy_ratio[:, channel_index], 1e-8, 1e8)
        right_ratio = np.clip(right.energy_ratio[:, channel_index], 1e-8, 1e8)
        channel_metrics = (
            (
                "energy_log_error",
                np.abs(np.log(left_ratio)),
                np.abs(np.log(right_ratio)),
                "lower",
            ),
            (
                "shape_log_error",
                left.shape_error[:, channel_index],
                right.shape_error[:, channel_index],
                "lower",
            ),
            (
                "coherence",
                left.coherence[:, channel_index],
                right.coherence[:, channel_index],
                "higher",
            ),
            (
                "signed_cospectrum",
                left.signed_cospectrum[:, channel_index],
                right.signed_cospectrum[:, channel_index],
                "higher",
            ),
        )
        per_channel[channel_name] = {
            metric_name: metric_payload(
                left_values,
                right_values,
                bootstrap(left_values),
                bootstrap(right_values),
                better,
            )
            for metric_name, left_values, right_values, better in channel_metrics
        }

    def holm_adjust(p_values: list[float]) -> list[float]:
        order = np.argsort(np.asarray(p_values, dtype=np.float64))
        adjusted = np.empty(len(p_values), dtype=np.float64)
        running = 0.0
        count = len(p_values)
        for rank, index in enumerate(order):
            running = max(running, (count - rank) * p_values[int(index)])
            adjusted[int(index)] = min(1.0, running)
        return adjusted.tolist()

    family: dict[str, dict] = {}
    for metric_name in (
        "energy_log_error",
        "shape_log_error",
        "coherence",
        "signed_cospectrum",
    ):
        names = list(per_channel)
        p_values = [
            per_channel[name][metric_name]["p_paired_block_permutation"]
            for name in names
        ]
        adjusted = holm_adjust(p_values)
        pointwise_better = []
        significant_better = []
        for name, p_holm in zip(names, adjusted):
            metric = per_channel[name][metric_name]
            metric["p_holm"] = p_holm
            delta = metric["delta_left_minus_right"]
            better = delta < 0.0 if metric["better"] == "lower" else delta > 0.0
            pointwise_better.append(better)
            significant_better.append(better and p_holm < 0.05)
        family[metric_name] = {
            "correction": "Holm-Bonferroni",
            "alpha": 0.05,
            "n_hypotheses": len(names),
            "n_pointwise_left_better": sum(pointwise_better),
            "n_significant_left_better_holm": sum(significant_better),
            "all_pointwise_left_better": all(pointwise_better),
            "all_left_better_holm": all(significant_better),
        }
    result["per_channel"] = per_channel
    result["cellwise_family"] = family
    return result


def parse_entry(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("expected NAME:NPZ")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=parse_entry)
    parser.add_argument("--right", action="append", required=True, type=parse_entry)
    parser.add_argument("--channels", default="all")
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--cellwise", action="store_true")
    parser.add_argument(
        "--vector",
        action="store_true",
        help="Compare vector-SHT wind-pair/component diagnostics.",
    )
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    left_name, left_path = args.left
    loader = load_vector_windows if args.vector else load_windows
    left = loader(left_path)
    if args.channels == "all":
        channel_indices = np.arange(len(left.channels))
        selected_channels = list(left.channels)
    else:
        selected_channels = [
            value.strip() for value in args.channels.split(",") if value.strip()
        ]
        channel_indices = np.asarray(
            [left.channels.index(value) for value in selected_channels],
            dtype=np.int16,
        )

    comparisons = {}
    right_sha256 = {}
    right_paths = {}
    for right_name, right_path in args.right:
        comparisons[right_name] = compare(
            left,
            loader(right_path),
            channel_indices=channel_indices,
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
            cellwise=args.cellwise,
        )
        right_sha256[right_name] = sha256_file(right_path)
        right_paths[right_name] = str(right_path)
    output = {
        "schema_version": 2,
        "left": left_name,
        "left_path": str(left_path),
        "left_sha256": sha256_file(left_path),
        "right_sha256": right_sha256,
        "right_paths": right_paths,
        "channels": selected_channels,
        "comparisons": comparisons,
        "diagnostic_space": "vector_sht" if args.vector else "scalar_sht",
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

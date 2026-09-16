#!/usr/bin/env python3
"""Summarise full-hour versus held-out-hour 6 h interpolation controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Pair:
    architecture: str
    heldout_json: str
    full_json: str
    comparability: str
    heldout_steps: int
    full_steps: int


PAIRS = (
    Pair(
        "WeatherBridge",
        "{new}/{year}/weatherbridge_heldout.json",
        "{new}/{year}/weatherbridge_full.json",
        "optimizer_budget_and_seed_matched",
        13136,
        13136,
    ),
    Pair(
        "WeatherDCAE-14M",
        "{new}/{year}/weatherdcae_heldout.json",
        "{new}/{year}/weatherdcae_full.json",
        "same_seed_full_has_more_optimizer_steps",
        13136,
        21888,
    ),
    Pair(
        "PixelAttn-VFI",
        "{corrected}/6h/{year}/full_year/pixelattn_vfi.json",
        "{new}/{year}/pixelattn_full.json",
        "eight_epochs_but_full_has_more_optimizer_steps_seed_unverified_for_heldout",
        13136,
        21888,
    ),
    Pair(
        "SwinV2",
        "{corrected}/6h/{year}/full_year/swinv2.json",
        "{new}/{year}/swinv2_full.json",
        "optimizer_steps_matched_seed_not_checkpoint_verified",
        17520,
        17520,
    ),
    Pair(
        "S-DYff",
        "{corrected}/6h/{year}/full_year/sdyff.json",
        "{new}/{year}/sdyff_full.json",
        "optimizer_steps_matched_seed_not_checkpoint_verified",
        35032,
        35032,
    ),
)

PHYSICAL_UNITS = {
    **{f"T{level}": "K" for level in (1000, 925, 850, 700)},
    **{f"U{level}": "m s-1" for level in (1000, 925, 850, 700)},
    **{f"V{level}": "m s-1" for level in (1000, 925, 850, 700)},
    **{f"Q{level}": "kg kg-1" for level in (1000, 925, 850, 700)},
    **{f"Z{level}": "m2 s-2" for level in (1000, 925, 850, 700)},
    "t2m": "K",
    "u10": "m s-1",
    "v10": "m s-1",
    "mslp": "Pa",
}


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if set(payload["per_tau"]) != {"1", "2", "3", "4", "5"}:
        raise ValueError(f"incomplete tau set: {path}")
    return payload


def metric(payload: dict[str, Any], hour: int, prefix: str, field: str) -> float:
    return float(payload["per_tau"][str(hour)]["model"][f"{prefix}_{field}"])


def window_path(payload: dict[str, Any], json_path: Path) -> Path:
    value = Path(payload["window_metrics_file"])
    if not value.is_absolute():
        value = json_path.parent / value
    return value


def paired_bootstrap(
    held_path: Path,
    full_path: Path,
    taus: tuple[int, ...],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    held = np.load(held_path, allow_pickle=False)
    full = np.load(full_path, allow_pickle=False)
    if not np.array_equal(held["channel_names"], full["channel_names"]):
        raise ValueError(f"channel mismatch: {held_path} vs {full_path}")

    held_keys = list(
        zip(held["year"].tolist(), held["t0"].tolist(), held["tau"].tolist(), strict=True)
    )
    full_keys = list(
        zip(full["year"].tolist(), full["t0"].tolist(), full["tau"].tolist(), strict=True)
    )
    if len(set(held_keys)) != len(held_keys) or len(set(full_keys)) != len(full_keys):
        raise ValueError("window metrics contain duplicate (year, t0, tau) keys")
    if set(held_keys) != set(full_keys):
        raise ValueError(f"paired window set mismatch: {held_path} vs {full_path}")
    full_lookup = {key: index for index, key in enumerate(full_keys)}
    full_order = np.asarray([full_lookup[key] for key in held_keys], dtype=np.int64)

    year = held["year"].astype(np.int64)
    t0 = held["t0"].astype(np.int64)
    tau = held["tau"].astype(np.int64)
    held_mse = held["mse_norm_model"].astype(np.float64)
    full_mse = full["mse_norm_model"][full_order].astype(np.float64)
    mask = np.isin(tau, np.asarray(taus))
    day = year * 1000 + t0 // 24
    blocks = np.unique(day[mask])
    rng = np.random.default_rng(seed)

    def score(mse: np.ndarray, selection: np.ndarray) -> float:
        values = []
        for current_tau in taus:
            current = selection & (tau == current_tau)
            values.append(np.sqrt(mse[current].mean(axis=0)).mean())
        return float(np.mean(values))

    held_point = score(held_mse, mask)
    full_point = score(full_mse, mask)
    block_lookup = {int(value): index for index, value in enumerate(blocks)}
    block_index = np.asarray([block_lookup[int(value)] for value in day], dtype=np.int64)
    shape = (len(blocks), len(taus), held_mse.shape[1])
    held_block_sum = np.zeros(shape, dtype=np.float64)
    full_block_sum = np.zeros(shape, dtype=np.float64)
    block_count = np.zeros((len(blocks), len(taus)), dtype=np.int64)
    for tau_index, current_tau in enumerate(taus):
        current = tau == current_tau
        np.add.at(
            held_block_sum[:, tau_index],
            block_index[current],
            held_mse[current],
        )
        np.add.at(
            full_block_sum[:, tau_index],
            block_index[current],
            full_mse[current],
        )
        np.add.at(block_count[:, tau_index], block_index[current], 1)

    draw_weights = rng.multinomial(
        len(blocks),
        np.full(len(blocks), 1.0 / len(blocks)),
        size=draws,
    ).astype(np.float64)
    held_draw_sum = np.einsum("db,btc->dtc", draw_weights, held_block_sum)
    full_draw_sum = np.einsum("db,btc->dtc", draw_weights, full_block_sum)
    draw_count = draw_weights @ block_count
    held_draws = np.sqrt(held_draw_sum / draw_count[:, :, None]).mean(axis=(1, 2))
    full_draws = np.sqrt(full_draw_sum / draw_count[:, :, None]).mean(axis=(1, 2))
    deltas = held_draws - full_draws
    held_field_draws = np.sqrt(held_draw_sum / draw_count[:, :, None]).mean(axis=1)
    full_field_draws = np.sqrt(full_draw_sum / draw_count[:, :, None]).mean(axis=1)
    field_deltas = held_field_draws - full_field_draws
    held_field_point = np.mean(
        [np.sqrt(held_mse[tau == current_tau].mean(axis=0)) for current_tau in taus],
        axis=0,
    )
    full_field_point = np.mean(
        [np.sqrt(full_mse[tau == current_tau].mean(axis=0)) for current_tau in taus],
        axis=0,
    )
    per_field: dict[str, Any] = {}
    for field_index, field in enumerate(held["channel_names"].astype(str)):
        non_better_field_draws = int(
            np.count_nonzero(field_deltas[:, field_index] <= 0.0)
        )
        per_field[field] = {
            "heldout_rmse": float(held_field_point[field_index]),
            "full_rmse": float(full_field_point[field_index]),
            "heldout_minus_full": float(
                held_field_point[field_index] - full_field_point[field_index]
            ),
            "full_gain_pct": float(
                100.0
                * (held_field_point[field_index] - full_field_point[field_index])
                / held_field_point[field_index]
            ),
            "delta_ci95": np.percentile(
                field_deltas[:, field_index], [2.5, 50.0, 97.5]
            ).tolist(),
            "bootstrap_draws_full_not_better": non_better_field_draws,
            "bootstrap_tail_probability_full_not_better": float(
                (non_better_field_draws + 1) / (draws + 1)
            ),
            "bootstrap_probability_at_floor": non_better_field_draws == 0,
        }
    non_better_draws = int(np.count_nonzero(deltas <= 0.0))
    return {
        "taus": list(taus),
        "n_days": len(blocks),
        "draws": draws,
        "window_set_sha256": hashlib.sha256(
            "\n".join(
                f"{current_year},{current_t0},{current_tau}"
                for current_year, current_t0, current_tau in sorted(held_keys)
            ).encode("utf-8")
        ).hexdigest(),
        "heldout_rmse": held_point,
        "full_rmse": full_point,
        "heldout_minus_full": held_point - full_point,
        "full_gain_pct": 100.0 * (held_point - full_point) / held_point,
        "delta_ci95": np.percentile(deltas, [2.5, 50.0, 97.5]).tolist(),
        "bootstrap_draws_full_not_better": non_better_draws,
        "bootstrap_tail_probability_full_not_better": float(
            (non_better_draws + 1) / (draws + 1)
        ),
        "bootstrap_probability_resolution": float(1.0 / (draws + 1)),
        "bootstrap_probability_at_floor": non_better_draws == 0,
        "per_field": per_field,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", required=True, type=Path)
    parser.add_argument("--corrected-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--years", default="2020,2021")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=202707)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    years = tuple(int(value) for value in args.years.split(",") if value)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    roots = {
        "new": str(args.new_root),
        "corrected": str(args.corrected_root),
    }

    rows: list[dict[str, Any]] = []
    bootstrap: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "years": list(years),
        "trained_hours": [1, 3, 5],
        "held_out_hours": [2, 4],
        "pairs": {},
    }
    for pair in PAIRS:
        bootstrap[pair.architecture] = {}
        manifest["pairs"][pair.architecture] = {
            "comparability": pair.comparability,
            "heldout_steps": pair.heldout_steps,
            "full_steps": pair.full_steps,
            "years": {},
        }
        for year in years:
            paths = {
                "heldout": Path(pair.heldout_json.format(year=year, **roots)),
                "full": Path(pair.full_json.format(year=year, **roots)),
            }
            payloads = {name: load_payload(path) for name, path in paths.items()}
            protocol_keys = (
                "rmse_reduction",
                "samples_per_date",
                "sample_strategy",
                "full_year",
                "eval_hours",
                "latitude_grid",
            )
            protocol = payloads["heldout"]["evaluation_protocol"]
            for key in protocol_keys:
                if protocol.get(key) != payloads["full"]["evaluation_protocol"].get(key):
                    raise ValueError(
                        f"evaluation protocol mismatch for {pair.architecture} "
                        f"{year}: {key}"
                    )
            code_keys = (
                "batch_eval_12h_memmap.py",
                "grid.py",
                "normalization.py",
                "physical_consistency.py",
                "capmatched_loader.py",
            )
            held_code = payloads["heldout"]["evaluation_code_provenance"]
            full_code = payloads["full"]["evaluation_code_provenance"]
            for key in code_keys:
                if held_code.get(key) != full_code.get(key):
                    raise ValueError(
                        f"evaluation code mismatch for {pair.architecture} "
                        f"{year}: {key}"
                    )
            manifest["pairs"][pair.architecture]["years"][str(year)] = {
                "heldout_json": str(paths["heldout"]),
                "full_json": str(paths["full"]),
                "heldout_checkpoint": payloads["heldout"]["checkpoint_provenance"],
                "full_checkpoint": payloads["full"]["checkpoint_provenance"],
                "evaluation_protocol": {
                    key: protocol.get(key) for key in protocol_keys
                },
                "evaluation_code_sha256": {
                    key: held_code.get(key) for key in code_keys
                },
            }
            channels = payloads["heldout"]["channel_names"]
            if channels != payloads["full"]["channel_names"]:
                raise ValueError(f"channel mismatch for {pair.architecture} {year}")
            for hour in range(1, 6):
                split = "held_out" if hour in (2, 4) else "trained"
                for field in channels:
                    held_norm = metric(payloads["heldout"], hour, "rmse_norm", field)
                    full_norm = metric(payloads["full"], hour, "rmse_norm", field)
                    held_phys = metric(payloads["heldout"], hour, "rmse_phys", field)
                    full_phys = metric(payloads["full"], hour, "rmse_phys", field)
                    held_acc = metric(payloads["heldout"], hour, "acc", field)
                    full_acc = metric(payloads["full"], hour, "acc", field)
                    rows.append(
                        {
                            "architecture": pair.architecture,
                            "year": year,
                            "field": field,
                            "hour": hour,
                            "hour_split": split,
                            "heldout_nrmse": held_norm,
                            "full_nrmse": full_norm,
                            "full_gain_pct": 100.0 * (held_norm - full_norm) / held_norm,
                            "heldout_physical_rmse": held_phys,
                            "full_physical_rmse": full_phys,
                            "physical_unit": PHYSICAL_UNITS[field],
                            "heldout_acc": held_acc,
                            "full_acc": full_acc,
                            "full_acc_delta": full_acc - held_acc,
                            "comparability": pair.comparability,
                            "heldout_steps": pair.heldout_steps,
                            "full_steps": pair.full_steps,
                        }
                    )
            held_window = window_path(payloads["heldout"], paths["heldout"])
            full_window = window_path(payloads["full"], paths["full"])
            bootstrap[pair.architecture][str(year)] = {
                "trained_hours": paired_bootstrap(
                    held_window,
                    full_window,
                    (1, 3, 5),
                    draws=args.bootstrap_draws,
                    seed=args.seed + year,
                ),
                "held_out_hours": paired_bootstrap(
                    held_window,
                    full_window,
                    (2, 4),
                    draws=args.bootstrap_draws,
                    seed=args.seed + year + 10000,
                ),
            }

    csv_path = args.out_dir / "full_vs_heldout_field_hour.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        for year in years:
            selected = [row for row in rows if row["architecture"] == pair.architecture and row["year"] == year]
            for split in ("trained", "held_out", "all"):
                subset = selected if split == "all" else [row for row in selected if row["hour_split"] == split]
                held = float(np.mean([row["heldout_nrmse"] for row in subset]))
                full = float(np.mean([row["full_nrmse"] for row in subset]))
                summary_rows.append(
                    {
                        "architecture": pair.architecture,
                        "year": year,
                        "hours": split,
                        "heldout_nrmse": held,
                        "full_nrmse": full,
                        "full_gain_pct": 100.0 * (held - full) / held,
                        "cells_full_better": sum(row["full_nrmse"] < row["heldout_nrmse"] for row in subset),
                        "cells_total": len(subset),
                        "comparability": pair.comparability,
                    }
                )
    summary_csv = args.out_dir / "full_vs_heldout_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    field_summary_rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        for year in years:
            for field in PHYSICAL_UNITS:
                selected = [
                    row for row in rows
                    if row["architecture"] == pair.architecture
                    and row["year"] == year
                    and row["field"] == field
                ]
                for split in ("trained", "held_out"):
                    subset = [row for row in selected if row["hour_split"] == split]
                    held = float(np.mean([row["heldout_nrmse"] for row in subset]))
                    full = float(np.mean([row["full_nrmse"] for row in subset]))
                    held_acc = float(np.mean([row["heldout_acc"] for row in subset]))
                    full_acc = float(np.mean([row["full_acc"] for row in subset]))
                    field_summary_rows.append(
                        {
                            "architecture": pair.architecture,
                            "year": year,
                            "field": field,
                            "hours": split,
                            "heldout_nrmse": held,
                            "full_nrmse": full,
                            "full_gain_pct": 100.0 * (held - full) / held,
                            "heldout_acc": held_acc,
                            "full_acc": full_acc,
                            "full_acc_delta": full_acc - held_acc,
                            "comparability": pair.comparability,
                        }
                    )
    field_summary_csv = args.out_dir / "full_vs_heldout_field_summary.csv"
    with field_summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(field_summary_rows[0]))
        writer.writeheader()
        writer.writerows(field_summary_rows)

    (args.out_dir / "full_vs_heldout_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2) + "\n"
    )
    field_bootstrap_rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        for year in years:
            for split in ("trained_hours", "held_out_hours"):
                split_payload = bootstrap[pair.architecture][str(year)][split]
                for field, field_payload in split_payload["per_field"].items():
                    ci_low, ci_median, ci_high = field_payload["delta_ci95"]
                    field_bootstrap_rows.append(
                        {
                            "architecture": pair.architecture,
                            "year": year,
                            "hours": split,
                            "field": field,
                            "heldout_nrmse": field_payload["heldout_rmse"],
                            "full_nrmse": field_payload["full_rmse"],
                            "heldout_minus_full": field_payload["heldout_minus_full"],
                            "full_gain_pct": field_payload["full_gain_pct"],
                            "delta_ci95_low": ci_low,
                            "delta_ci95_median": ci_median,
                            "delta_ci95_high": ci_high,
                            "bootstrap_draws_full_not_better": field_payload[
                                "bootstrap_draws_full_not_better"
                            ],
                            "bootstrap_tail_probability_full_not_better": (
                                field_payload[
                                    "bootstrap_tail_probability_full_not_better"
                                ]
                            ),
                            "bootstrap_probability_at_floor": field_payload[
                                "bootstrap_probability_at_floor"
                            ],
                            "comparability": pair.comparability,
                        }
                    )
    field_bootstrap_csv = args.out_dir / "full_vs_heldout_field_bootstrap.csv"
    with field_bootstrap_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(field_bootstrap_rows[0]),
        )
        writer.writeheader()
        writer.writerows(field_bootstrap_rows)
    (args.out_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    fields = [row["field"] for row in rows if row["architecture"] == PAIRS[0].architecture and row["year"] == years[0] and row["hour"] == 1]
    fig, axes = plt.subplots(len(PAIRS), len(years), figsize=(12, 21), constrained_layout=True)
    if len(years) == 1:
        axes = np.asarray(axes)[:, None]
    maximum_gain = float(np.nanmax(np.abs([row["full_gain_pct"] for row in rows])))
    heatmap_limit = max(5.0, 5.0 * np.ceil(maximum_gain / 5.0))
    image = None
    for row_index, pair in enumerate(PAIRS):
        evidence_label = (
            "matched"
            if pair.comparability == "optimizer_budget_and_seed_matched"
            else "diagnostic"
        )
        for column_index, year in enumerate(years):
            ax = axes[row_index, column_index]
            matrix = np.empty((len(fields), 5), dtype=np.float64)
            for field_index, field in enumerate(fields):
                for hour in range(1, 6):
                    item = next(
                        row for row in rows
                        if row["architecture"] == pair.architecture
                        and row["year"] == year
                        and row["field"] == field
                        and row["hour"] == hour
                    )
                    matrix[field_index, hour - 1] = item["full_gain_pct"]
            image = ax.imshow(
                matrix,
                cmap="RdBu_r",
                vmin=-heatmap_limit,
                vmax=heatmap_limit,
                aspect="auto",
            )
            ax.set_title(
                f"{pair.architecture}, {year} [{evidence_label}]",
                fontsize=11,
                weight="bold",
            )
            ax.set_xticks(range(5), ["1", "2*", "3", "4*", "5"])
            ax.set_xlabel("Query hour (* held out)")
            if column_index == 0:
                ax.set_yticks(range(len(fields)), fields, fontsize=7)
            else:
                ax.set_yticks(range(len(fields)), [])
            ax.axvline(0.5, color="black", linewidth=0.5)
            ax.axvline(1.5, color="black", linewidth=0.5)
            ax.axvline(2.5, color="black", linewidth=0.5)
            ax.axvline(3.5, color="black", linewidth=0.5)
            colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
            colorbar.set_label("Full-training gain (%)", fontsize=8)
    fig.savefig(args.out_dir / "full_vs_heldout_field_hour_heatmaps.png", dpi=220)
    fig.savefig(args.out_dir / "full_vs_heldout_field_hour_heatmaps.pdf")
    plt.close(fig)

    year_styles = ("-", "--", ":", "-.")
    for pair in PAIRS:
        evidence_label = (
            "matched"
            if pair.comparability == "optimizer_budget_and_seed_matched"
            else "diagnostic"
        )
        fig, axes = plt.subplots(
            6,
            4,
            figsize=(11, 13),
            sharex=True,
        )
        for field_index, field in enumerate(fields):
            ax = axes.flat[field_index]
            for year_index, year in enumerate(years):
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["architecture"] == pair.architecture
                        and row["year"] == year
                        and row["field"] == field
                    ),
                    key=lambda row: row["hour"],
                )
                hours = [row["hour"] for row in selected]
                style = year_styles[year_index % len(year_styles)]
                ax.plot(
                    hours,
                    [row["heldout_physical_rmse"] for row in selected],
                    color="#6b7280",
                    linestyle=style,
                    marker="o",
                    markersize=2.5,
                    linewidth=1.2,
                    label=f"Held-out, {year}",
                )
                ax.plot(
                    hours,
                    [row["full_physical_rmse"] for row in selected],
                    color="#d62728",
                    linestyle=style,
                    marker="o",
                    markersize=2.5,
                    linewidth=1.2,
                    label=f"Full, {year}",
                )
            ax.axvspan(1.75, 2.25, color="#f5dada", alpha=0.55, linewidth=0)
            ax.axvspan(3.75, 4.25, color="#f5dada", alpha=0.55, linewidth=0)
            ax.set_title(f"{field} [{PHYSICAL_UNITS[field]}]", fontsize=8, weight="bold")
            ax.grid(color="#d1d5db", linewidth=0.4, alpha=0.7)
            ax.tick_params(labelsize=7)
            ax.set_xticks(range(1, 6))
            if field_index % 4 == 0:
                ax.set_ylabel("RMSE", fontsize=7)
            if field_index >= 20:
                ax.set_xlabel("Query hour", fontsize=7)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.972),
            ncol=min(4, len(labels)),
            frameon=False,
            fontsize=8,
        )
        fig.suptitle(
            f"{pair.architecture}: full-hours versus held-out-hours training "
            f"[{evidence_label}]",
            fontsize=12,
            weight="bold",
            y=0.997,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=1.0, w_pad=0.8)
        file_stem = pair.architecture.lower().replace("-", "_").replace(" ", "_")
        fig.savefig(
            args.out_dir / f"full_vs_heldout_physical_rmse_{file_stem}.png",
            dpi=220,
        )
        fig.savefig(args.out_dir / f"full_vs_heldout_physical_rmse_{file_stem}.pdf")
        plt.close(fig)
    print(f"wrote {csv_path}")
    print(f"wrote {summary_csv}")
    print(f"wrote {field_summary_csv}")
    print(f"wrote {field_bootstrap_csv}")


if __name__ == "__main__":
    main()

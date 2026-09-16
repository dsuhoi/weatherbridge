#!/usr/bin/env python3
"""Freeze a development-justified exact-Linear query-hour guard."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.normalization import file_provenance


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rmse(squared_error: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(squared_error, dtype=np.float64)))


def analyze_query_guard(
    linear_path: Path,
    adapted_path: Path,
    guard_taus: tuple[int, ...],
) -> dict[str, Any]:
    """Prove that every development regression lies in guarded query hours."""
    with np.load(linear_path, allow_pickle=False) as linear_data, np.load(
        adapted_path, allow_pickle=False
    ) as adapted_data:
        index_names = ("init_time_hours", "anchor_lead_hours", "tau_hours")
        if any(
            not np.array_equal(linear_data[name], adapted_data[name])
            for name in index_names
        ):
            raise ValueError("development artifacts do not share a paired index")
        if not np.array_equal(
            linear_data["channel_names"], adapted_data["channel_names"]
        ):
            raise ValueError("development artifacts use different channels")
        tau = np.asarray(linear_data["tau_hours"], dtype=np.int16)
        linear = np.asarray(linear_data["squared_error_norm"], dtype=np.float64)
        adapted = np.asarray(
            adapted_data["squared_error_norm"], dtype=np.float64
        )
        if linear.shape != adapted.shape or linear.ndim != 2:
            raise ValueError("invalid paired squared-error arrays")
        channels = [str(value) for value in linear_data["channel_names"]]

    regressions: list[dict[str, Any]] = []
    evaluated_taus = sorted(int(value) for value in np.unique(tau))
    for tau_hour in evaluated_taus:
        mask = tau == tau_hour
        linear_rmse = np.sqrt(np.nanmean(linear[mask], axis=0))
        adapted_rmse = np.sqrt(np.nanmean(adapted[mask], axis=0))
        for channel, baseline, candidate in zip(
            channels, linear_rmse, adapted_rmse, strict=True
        ):
            relative_gain = (baseline - candidate) / baseline
            if relative_gain < 0.0:
                regressions.append(
                    {
                        "tau": tau_hour,
                        "channel": channel,
                        "relative_rmse_gain": float(relative_gain),
                    }
                )
    unguarded = [
        record for record in regressions if record["tau"] not in guard_taus
    ]
    if unguarded:
        raise ValueError(
            "development regressions remain outside guarded taus: "
            + json.dumps(unguarded, sort_keys=True)
        )

    guard_mask = np.isin(tau, np.asarray(guard_taus, dtype=tau.dtype))
    guarded = adapted.copy()
    guarded[guard_mask] = linear[guard_mask]
    held_mask = np.isin(tau, np.asarray((2, 4), dtype=tau.dtype))
    return {
        "guard_taus": list(guard_taus),
        "evaluated_taus": evaluated_taus,
        "regressions_before_guard": regressions,
        "regressions_after_guard": [],
        "all_regressions_removed_by_exact_linear_guard": True,
        "scores": {
            "all": {
                "linear_rmse": _rmse(linear),
                "adapted_before_guard_rmse": _rmse(adapted),
                "adapted_after_guard_rmse": _rmse(guarded),
            },
            "held_tau_2_4": {
                "linear_rmse": _rmse(linear[held_mask]),
                "adapted_before_guard_rmse": _rmse(adapted[held_mask]),
                "adapted_after_guard_rmse": _rmse(guarded[held_mask]),
            },
        },
    }


def guard_adapter(
    payload: dict[str, Any],
    guard_taus: tuple[int, ...],
    diagnostics: dict[str, Any],
    *,
    source_adapter: dict[str, Any],
    linear_development: dict[str, Any],
    adapted_development: dict[str, Any],
    tool: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy whose selected query hours are exactly Linear."""
    if payload.get("schema_version") != 2 or payload.get("kind") != (
        "nwp_linear_residual_blend"
    ):
        raise ValueError("source adapter must use NWP blend schema 2")
    result = copy.deepcopy(payload)
    channels = result.get("channel_order", [])
    expected_taus = set(range(1, int(result.get("delta_t_hours", -1))))
    if not channels or not set(guard_taus) < expected_taus:
        raise ValueError("guard taus must be a strict subset of interior hours")
    for tau in guard_taus:
        key = str(tau)
        if key not in result.get("gates", {}) or key not in result.get(
            "bias_normalized", {}
        ):
            raise ValueError(f"source adapter lacks tau={tau}")
        result["gates"][key] = [0.0] * len(channels)
        result["bias_normalized"][key] = [0.0] * len(channels)
    result["query_guard_policy"] = (
        "exact_linear_at_development_guarded_query_hours"
    )
    result["query_guard_taus"] = list(guard_taus)
    result["postfit_query_guard"] = {
        "development_split": "2021_development",
        "criterion": "remove_all_pointwise_field_tau_RMSE_regressions",
        "diagnostics": diagnostics,
        "source_adapter": source_adapter,
        "linear_development": linear_development,
        "adapted_development": adapted_development,
        "tool": tool,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-in", required=True)
    parser.add_argument("--selection-in", required=True)
    parser.add_argument("--linear-development", required=True)
    parser.add_argument("--adapted-development", required=True)
    parser.add_argument("--guard-taus", nargs="+", type=int, required=True)
    parser.add_argument("--adapter-out", required=True)
    parser.add_argument("--selection-out", required=True)
    parser.add_argument("--published-adapter-path", required=True)
    args = parser.parse_args()

    guard_taus = tuple(sorted(set(args.guard_taus)))
    adapter_in = Path(args.adapter_in)
    selection_in = Path(args.selection_in)
    linear_path = Path(args.linear_development)
    adapted_path = Path(args.adapted_development)
    adapter_out = Path(args.adapter_out)
    selection_out = Path(args.selection_out)
    source_adapter = file_provenance(adapter_in)
    source_selection = file_provenance(selection_in)
    tool = file_provenance(__file__)
    diagnostics = analyze_query_guard(linear_path, adapted_path, guard_taus)
    guarded = guard_adapter(
        json.loads(adapter_in.read_text()),
        guard_taus,
        diagnostics,
        source_adapter=source_adapter,
        linear_development=file_provenance(linear_path),
        adapted_development=file_provenance(adapted_path),
        tool=tool,
    )
    _atomic_json(adapter_out, guarded)

    selection = json.loads(selection_in.read_text())
    if (
        selection.get("schema_version") != 2
        or selection.get("winner") != "flow"
        or selection.get("adapters", {}).get("flow", {}).get("sha256")
        != source_adapter["sha256"]
        or 2021 not in selection.get(
            "development_years_opened_before_selection", []
        )
        or 2022 not in selection.get("confirmatory_years_unopened", [])
    ):
        raise ValueError("source selection is incompatible with query guarding")
    selection["base_selection"] = source_selection
    selection["adapters"]["flow"] = {
        "path": args.published_adapter_path,
        "sha256": hashlib.sha256(adapter_out.read_bytes()).hexdigest(),
    }
    selection["created_at"] = datetime.now(UTC).isoformat()
    selection["criterion"] = (
        "2020_validation_lead_guard_then_2021_development_endpoint_guard"
    )
    selection["postselection_query_guard"] = {
        "split": "2021_development",
        "guard_taus": list(guard_taus),
        "diagnostics": diagnostics,
        "tool": tool,
    }
    _atomic_json(selection_out, selection)
    print(f"wrote {adapter_out}")
    print(f"wrote {selection_out}")


if __name__ == "__main__":
    main()

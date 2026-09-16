#!/usr/bin/env python3
"""Assess a frozen winner's AVG3 checkpoint as a deployment auxiliary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.select_upr_lite_candidate import (
    load_field_metrics,
    load_spectral_metrics,
)
from tools.eval.validate_upr_lite_selection import (
    _annotate_holm,
    acc_comparisons,
    field_comparisons,
    physical_comparisons,
    spectral_comparisons,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard(
    label: str,
    result: dict[str, Any],
    *,
    better: str,
) -> dict[str, Any]:
    if better not in {"lower", "higher"}:
        raise ValueError(f"invalid metric direction: {better}")
    return {
        "label": label,
        "better": better,
        "result": result,
    }


def _annotate_guard_family(
    guards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = [guard["result"] for guard in guards]
    _annotate_holm(
        results,
        output_key="p_holm_deployment_safety_family",
    )
    regressions = []
    for guard in guards:
        result = guard["result"]
        delta = float(result["delta_left_minus_right"])
        right_is_worse = (
            delta < 0.0
            if guard["better"] == "lower"
            else delta > 0.0
        )
        if (
            right_is_worse
            and result["p_holm_deployment_safety_family"] < 0.05
        ):
            regressions.append(
                {
                    "metric": guard["label"],
                    "delta_raw_minus_avg3": delta,
                    "p_raw": result["p_paired_block_permutation"],
                    "p_holm": result[
                        "p_holm_deployment_safety_family"
                    ],
                }
            )
    return regressions


def summarize(
    raw_name: str,
    averaged_name: str,
    *,
    field: dict[str, Any],
    acc: dict[str, Any],
    physical: dict[str, Any],
    spectral: dict[str, Any],
    held_hours: tuple[int, ...] = (2, 4),
) -> dict[str, Any]:
    if not held_hours:
        raise ValueError("held_hours must not be empty")
    primary = field["2020"]["unseen"][averaged_name]
    per_hour = primary["per_tau"]
    improves_all_held_hours = all(
        float(per_hour[str(hour)]["delta_left_minus_right"]) >= 0.0
        for hour in held_hours
    )

    guards_2020 = [
        _guard(
            "field_seen_rmse",
            field["2020"]["seen"][averaged_name],
            better="lower",
        ),
        _guard(
            "acc_unseen",
            acc["2020"]["unseen"][averaged_name],
            better="higher",
        ),
    ]
    for diagnostic, result in physical["2020"]["unseen"][
        averaged_name
    ]["diagnostics"].items():
        guards_2020.append(
            _guard(
                f"physical_unseen/{diagnostic}",
                result,
                better="lower",
            )
        )
    for tau, comparison in spectral.items():
        result = comparison[averaged_name]
        for metric in ("energy_log_error", "shape_log_error"):
            guards_2020.append(
                _guard(
                    f"spectral_h{tau}/{metric}",
                    result[metric],
                    better="lower",
                )
            )
        guards_2020.append(
            _guard(
                f"spectral_h{tau}/coherence",
                result["coherence"],
                better="higher",
            )
        )
    regressions_2020 = _annotate_guard_family(guards_2020)

    guards_2021 = [
        _guard(
            "field_unseen_rmse",
            field["2021"]["unseen"][averaged_name],
            better="lower",
        ),
        _guard(
            "acc_unseen",
            acc["2021"]["unseen"][averaged_name],
            better="higher",
        ),
    ]
    for hour in held_hours:
        guards_2021.append(
            _guard(
                f"field_h{hour}_rmse",
                field["2021"]["unseen"][averaged_name]["per_tau"][
                    str(hour)
                ],
                better="lower",
            )
        )
    for diagnostic, result in physical["2021"]["unseen"][
        averaged_name
    ]["diagnostics"].items():
        guards_2021.append(
            _guard(
                f"physical_unseen/{diagnostic}",
                result,
                better="lower",
            )
        )
    regressions_2021 = _annotate_guard_family(guards_2021)

    selected_on_2020 = (
        float(primary["delta_left_minus_right"]) > 0.0
        and improves_all_held_hours
        and not regressions_2020
    )
    confirmed_on_2021 = not regressions_2021
    recommend_average = selected_on_2020 and confirmed_on_2021
    return {
        "available": True,
        "raw_name": raw_name,
        "averaged_name": averaged_name,
        "selected_on_2020": selected_on_2020,
        "confirmed_on_frozen_2021": confirmed_on_2021,
        "recommended_name": (
            averaged_name if recommend_average else raw_name
        ),
        "primary_2020_unseen_rmse": primary,
        "held_hours": list(held_hours),
        "improves_all_2020_held_hours": improves_all_held_hours,
        "significant_2020_safety_regressions": regressions_2020,
        "significant_2021_confirmation_regressions": regressions_2021,
        "multiplicity_correction": (
            "Holm within the 2020 deployment-safety family and separately "
            "within the frozen-2021 confirmation family"
        ),
        "note": (
            "AVG3 is a deployment-only auxiliary. The frozen raw checkpoint "
            "remains the architecture-selection result. Frozen 2021 can veto "
            "deployment but cannot promote AVG3 after a failed 2020 gate."
        ),
    }


def _validate_metric_provenance(
    selection: dict[str, Any],
    raw_name: str,
    averaged_name: str,
    root_2020: Path,
    root_2021: Path,
    spectra_root: Path,
    *,
    spectral_taus: tuple[int, ...],
) -> dict[str, Any]:
    field_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for year, root in (("2020", root_2020), ("2021", root_2021)):
        field_metrics[year] = {
            name: load_field_metrics(root / f"{name}.json")
            for name in (raw_name, averaged_name)
        }
        raw = field_metrics[year][raw_name]
        averaged = field_metrics[year][averaged_name]
        for key in (
            "window_index_sha256",
            "evaluation_input_provenance",
            "evaluation_dataset_provenance",
        ):
            if raw[key] != averaged[key]:
                raise ValueError(
                    f"{year}: raw/AVG3 field provenance mismatch: {key}"
                )

    raw_expected_hash = selection["models"][raw_name]["checkpoint_sha256"]
    raw_hashes = {
        field_metrics[year][raw_name]["checkpoint_sha256"]
        for year in ("2020", "2021")
    }
    if raw_hashes != {raw_expected_hash}:
        raise ValueError("raw metrics do not match the frozen selection")
    averaged_hashes = {
        field_metrics[year][averaged_name]["checkpoint_sha256"]
        for year in ("2020", "2021")
    }
    if len(averaged_hashes) != 1:
        raise ValueError("AVG3 checkpoint changed between evaluation years")

    spectral_metrics = {
        name: load_spectral_metrics(
            spectra_root,
            name,
            180,
            taus=set(spectral_taus),
            required_lmax=359,
            expected_channels=24,
        )
        for name in (raw_name, averaged_name)
    }
    for key in (
        "window_index_sha256",
        "evaluation_input_provenance",
        "evaluation_dataset_provenance",
    ):
        if spectral_metrics[raw_name][key] != spectral_metrics[
            averaged_name
        ][key]:
            raise ValueError(
                f"raw/AVG3 spectral provenance mismatch: {key}"
            )
    if spectral_metrics[raw_name]["checkpoint_sha256"] != raw_expected_hash:
        raise ValueError("raw spectrum does not match the frozen selection")
    averaged_hash = next(iter(averaged_hashes))
    if (
        spectral_metrics[averaged_name]["checkpoint_sha256"]
        != averaged_hash
    ):
        raise ValueError("AVG3 field/spectral checkpoint mismatch")
    return {
        "raw_checkpoint_sha256": raw_expected_hash,
        "averaged_checkpoint_sha256": averaged_hash,
    }


def assess(
    selection_path: Path,
    root_2020: Path,
    root_2021: Path,
    spectra_root: Path,
    *,
    block_days: int,
    draws: int,
    seed: int,
    all_taus: tuple[int, ...] = (1, 2, 3, 4, 5),
    seen_taus: tuple[int, ...] = (1, 3, 5),
    unseen_taus: tuple[int, ...] = (2, 4),
    spectral_taus: tuple[int, ...] = (2, 4),
) -> dict[str, Any]:
    tau_sets = {
        "all": all_taus,
        "seen": seen_taus,
        "unseen": unseen_taus,
        "spectral": spectral_taus,
    }
    for label, values in tau_sets.items():
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{label}_taus must be non-empty and unique")
    if set(seen_taus) | set(unseen_taus) != set(all_taus):
        raise ValueError("seen_taus and unseen_taus must partition all_taus")
    if set(seen_taus) & set(unseen_taus):
        raise ValueError("seen_taus and unseen_taus must be disjoint")
    if not set(spectral_taus).issubset(all_taus):
        raise ValueError("spectral_taus must be a subset of all_taus")

    selection = json.loads(selection_path.read_text())
    raw_name = str(selection["winner"])
    averaged_name = f"{raw_name}_avg3"
    required = [
        root_2020 / f"{averaged_name}.json",
        root_2021 / f"{averaged_name}.json",
        *[
            spectra_root / f"{averaged_name}_tau{tau}.npz"
            for tau in spectral_taus
        ],
    ]
    if not all(path.is_file() for path in required):
        return {
            "schema_version": 1,
            "available": False,
            "raw_name": raw_name,
            "averaged_name": averaged_name,
            "recommended_name": raw_name,
            "missing_artifacts": [
                str(path) for path in required if not path.is_file()
            ],
            "selection_sha256": _sha256(selection_path),
            "note": (
                "The frozen winner has no evaluated AVG3 auxiliary; retain "
                "the raw architecture-selection checkpoint."
            ),
        }

    provenance = _validate_metric_provenance(
        selection,
        raw_name,
        averaged_name,
        root_2020,
        root_2021,
        spectra_root,
        spectral_taus=spectral_taus,
    )
    models = (raw_name, averaged_name)
    tau_groups = {
        "all": np.asarray(all_taus, dtype=np.int16),
        "seen": np.asarray(seen_taus, dtype=np.int16),
        "unseen": np.asarray(unseen_taus, dtype=np.int16),
    }
    field = {
        year: field_comparisons(
            raw_name,
            models,
            root,
            block_days=block_days,
            draws=draws,
            seed=seed,
            tau_groups=tau_groups,
        )
        for year, root in (("2020", root_2020), ("2021", root_2021))
    }
    acc = {
        year: acc_comparisons(
            raw_name,
            models,
            root,
            block_days=block_days,
            draws=draws,
            seed=seed,
            tau_groups=tau_groups,
        )
        for year, root in (("2020", root_2020), ("2021", root_2021))
    }
    physical = {
        year: physical_comparisons(
            raw_name,
            models,
            root,
            block_days=block_days,
            draws=draws,
            seed=seed,
            tau_groups=tau_groups,
        )
        for year, root in (("2020", root_2020), ("2021", root_2021))
    }
    spectral = spectral_comparisons(
        raw_name,
        models,
        spectra_root,
        block_days=block_days,
        draws=draws,
        seed=seed,
        taus=np.asarray(spectral_taus, dtype=np.int16),
    )
    report = summarize(
        raw_name,
        averaged_name,
        field=field,
        acc=acc,
        physical=physical,
        spectral=spectral,
        held_hours=unseen_taus,
    )
    return {
        "schema_version": 1,
        **report,
        **provenance,
        "selection_sha256": _sha256(selection_path),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "evaluation_protocol": {
            "all_taus": list(all_taus),
            "seen_taus": list(seen_taus),
            "unseen_taus": list(unseen_taus),
            "spectral_taus": list(spectral_taus),
        },
        "comparisons": {
            "field": field,
            "acc": acc,
            "physical": physical,
            "spectral": spectral,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--root-2020", type=Path, required=True)
    parser.add_argument("--root-2021", type=Path, required=True)
    parser.add_argument("--spectra-root", type=Path, required=True)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--all-taus", default="1,2,3,4,5")
    parser.add_argument("--seen-taus", default="1,3,5")
    parser.add_argument("--unseen-taus", default="2,4")
    parser.add_argument("--spectral-taus", default="2,4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def parse_taus(value: str) -> tuple[int, ...]:
        return tuple(int(item) for item in value.split(",") if item)

    report = assess(
        args.selection,
        args.root_2020,
        args.root_2021,
        args.spectra_root,
        block_days=args.block_days,
        draws=args.draws,
        seed=args.seed,
        all_taus=parse_taus(args.all_taus),
        seen_taus=parse_taus(args.seen_taus),
        unseen_taus=parse_taus(args.unseen_taus),
        spectral_taus=parse_taus(args.spectral_taus),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "available": report["available"],
                "recommended_name": report["recommended_name"],
            }
        )
    )


if __name__ == "__main__":
    main()

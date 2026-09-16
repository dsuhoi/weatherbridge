#!/usr/bin/env python3
"""Summarize a validation-frozen NWP blend on a declared test year."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.eval.summarize_forecast_anchor import (
    load_artifact,
    summarize_artifacts,
)
from weather_time_interp.normalization import file_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--test-year", type=int, default=2021)
    parser.add_argument(
        "--test-role",
        choices=("independent", "development"),
        default="independent",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("schema_version") not in (1, 2)
        or selection.get("selection_split") != "2020_validation"
    ):
        raise SystemExit("invalid validation-frozen NWP selection")
    if selection["schema_version"] == 1:
        valid_role = (
            args.test_year == 2021
            and args.test_role == "independent"
            and selection.get("test_year_opened_at_selection_time") is False
        )
    elif args.test_role == "development":
        valid_role = args.test_year in selection.get(
            "development_years_opened_before_selection",
            [],
        )
    else:
        valid_role = args.test_year in selection.get(
            "confirmatory_years_unopened",
            [],
        )
    if not valid_role:
        raise SystemExit("test year/role conflicts with frozen selection")
    winner = f"{selection['winner']}_adapted"
    paths: dict[str, Path] = {}
    for spec in args.artifact:
        if "=" not in spec:
            raise SystemExit(f"invalid artifact specification: {spec}")
        name, raw_path = spec.split("=", 1)
        if not name or name in paths:
            raise SystemExit(f"duplicate artifact name: {name}")
        paths[name] = Path(raw_path)
    artifacts = {
        name: load_artifact(
            name,
            path,
            expected_target_year=args.test_year,
        )
        for name, path in paths.items()
    }
    if winner not in artifacts or "linear" not in artifacts:
        raise SystemExit("selection winner and Linear artifacts are required")
    selected_adapter = selection["adapters"][selection["winner"]]
    winner_model = json.loads(artifacts[winner].json_path.read_text())[
        "provenance"
    ]["model"]
    if winner_model["blend_adapter"]["sha256"] != selected_adapter["sha256"]:
        raise SystemExit("test winner does not use the frozen adapter")
    result = summarize_artifacts(
        artifacts,
        winner,
        draws=args.draws,
        seed=args.seed,
    )
    result["diagnostic_only"] = False
    result["selection_independent"] = args.test_role == "independent"
    result["selection_split"] = "2020_validation"
    result["test_split"] = f"{args.test_year}_{args.test_role}"
    result["provenance"] = {
        "summarizer": file_provenance(__file__),
        "selection": file_provenance(selection_path),
        "artifacts": {
            name: {
                "json": file_provenance(artifact.json_path),
                "paired_npz": file_provenance(artifact.npz_path),
            }
            for name, artifact in artifacts.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

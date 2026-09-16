#!/usr/bin/env python3
"""Verify declared inputs and outputs without running an experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.repro.manifest import inventory_paths, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="repro/paper_experiments.json")
    parser.add_argument("--experiment")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / args.manifest)
    selected = (
        [manifest.experiments[args.experiment]]
        if args.experiment
        else list(manifest.experiments.values())
    )
    failed = False
    report = []
    for experiment in selected:
        inputs = inventory_paths(root, experiment.inputs)
        outputs = inventory_paths(root, experiment.outputs)
        complete = all(item["exists"] for item in (*inputs, *outputs))
        failed |= not complete
        report.append(
            {
                "experiment": experiment.name,
                "complete": complete,
                "inputs": inputs,
                "outputs": outputs,
            }
        )
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

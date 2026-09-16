#!/usr/bin/env python3
"""Run one declared experiment and write an immutable execution record."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from tools.repro.manifest import inventory_paths, load_manifest


def _git(root: Path, *args: str) -> str | None:
    # An extracted release must not inherit the enclosing checkout's identity.
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?")
    parser.add_argument("--manifest", default="repro/paper_experiments.json")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-inputs", action="store_true")
    parser.add_argument("--records-dir", default="artifacts/repro_runs")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / args.manifest)
    if args.list:
        for experiment in manifest.experiments.values():
            print(f"{experiment.name:28s} {experiment.description}")
        return 0
    if not args.experiment or args.experiment not in manifest.experiments:
        parser.error("choose an experiment from --list")
    experiment = manifest.experiments[args.experiment]
    input_inventory = inventory_paths(root, experiment.inputs)
    missing = [item["path"] for item in input_inventory if not item["exists"]]
    if missing and not args.allow_missing_inputs:
        raise FileNotFoundError(f"missing declared inputs: {', '.join(missing)}")

    command = list(experiment.command)
    if command[0] == "python":
        command[0] = sys.executable
    if args.dry_run:
        print(json.dumps({"cwd": str(root), "command": command}, indent=2))
        return 0

    started = dt.datetime.now(dt.timezone.utc)
    environment = os.environ.copy()
    environment.update(experiment.environment)
    environment.setdefault("PYTHON", sys.executable)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root), str(root / "scripts"), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(command, cwd=root, env=environment, check=False)
    finished = dt.datetime.now(dt.timezone.utc)
    output_inventory = inventory_paths(root, experiment.outputs)
    missing_outputs = [item["path"] for item in output_inventory if not item["exists"]]

    git_status = _git(root, "status", "--porcelain")
    release_provenance = root / "RELEASE_PROVENANCE.txt"
    record = {
        "schema_version": 1,
        "experiment": experiment.name,
        "manifest": args.manifest,
        "command": command,
        "environment_overrides": experiment.environment,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "return_code": completed.returncode,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": None if git_status is None else bool(git_status),
        "release_provenance": (
            release_provenance.read_text() if release_provenance.is_file() else None
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": input_inventory,
        "outputs": output_inventory,
    }
    records_dir = root / args.records_dir
    records_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    record_path = records_dir / f"{stamp}_{experiment.name}.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"execution record: {record_path}")
    if completed.returncode:
        return completed.returncode
    if missing_outputs:
        print(
            f"missing declared outputs: {', '.join(missing_outputs)}", file=sys.stderr
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

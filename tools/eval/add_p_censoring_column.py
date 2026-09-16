#!/usr/bin/env python3
"""Add the Monte Carlo resolution-floor flag to an exported statistics CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from tools.eval.export_journal_statistics import _annotate_p_censoring


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate(csv_path: Path, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("csv_sha256") != _sha256(csv_path):
        raise ValueError("statistics CSV does not match its manifest")

    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    if "p_raw_two_sided" not in fields:
        raise ValueError("statistics CSV has no raw two-sided p-value column")
    if "p_censored" not in fields:
        index = fields.index("p_raw_two_sided") + 1
        fields.insert(index, "p_censored")
    for row in rows:
        _annotate_p_censoring(row)

    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)

    exporter = Path(__file__).with_name("export_journal_statistics.py")
    manifest.update(
        {
            "csv_sha256": _sha256(csv_path),
            "row_count": len(rows),
            "exporter_sha256": _sha256(exporter),
            "p_censoring": {
                "definition": "raw two-sided p equals 1/(draws+1)",
                "censored_rows": sum(
                    row["p_censored"] == "true" for row in rows
                ),
            },
            "p_censoring_migration_sha256": _sha256(Path(__file__)),
        }
    )
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = migrate(args.csv, args.manifest)
    print(json.dumps(result["p_censoring"], sort_keys=True))


if __name__ == "__main__":
    main()

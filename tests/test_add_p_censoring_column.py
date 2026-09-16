from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from tools.eval.add_p_censoring_column import migrate


def test_migrate_marks_only_resolution_floor(tmp_path: Path) -> None:
    csv_path = tmp_path / "statistics.csv"
    fields = ["draws", "p_raw_two_sided"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {"draws": 5000, "p_raw_two_sided": 1.0 / 5001.0},
                {"draws": 5000, "p_raw_two_sided": 0.01},
            ]
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "source_sha256": {"source.json": "a" * 64},
            }
        )
    )

    result = migrate(csv_path, manifest_path)

    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["p_censored"] for row in rows] == ["true", "false"]
    assert result["p_censoring"]["censored_rows"] == 1

#!/usr/bin/env python3
"""Write a reproducible fingerprint of matched-training input files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from tools.train.training_protocol import memmap_dataset_provenance
from weather_time_interp.normalization import file_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memmap-dir",
        type=Path,
        default=Path("/tmp/wb2_0p5_cache"),
    )
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument(
        "--static-path",
        type=Path,
        default=Path("data/static_features_0p5.pt"),
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("data/json_stats_0p5.nc"),
    )
    parser.add_argument(
        "--surface-stats-path",
        type=Path,
        default=Path("data/surface_stats_0p5.json"),
    )
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "inputs": {
            "memmap": memmap_dataset_provenance(
                args.memmap_dir,
                args.years,
            ),
            "static_features": file_provenance(args.static_path),
            "pressure_level_stats": file_provenance(args.stats_path),
            "surface_stats": file_provenance(
                args.surface_stats_path
            ),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_json.with_name(
        f".{args.out_json.name}.{os.getpid()}.tmp"
    )
    with temporary.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.out_json)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "memmap_identity_sha256": report["inputs"]["memmap"][
                    "identity_sha256"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()

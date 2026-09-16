#!/usr/bin/env python3
"""Promote validated corrected-baseline JSONs to paper-facing aliases."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_ALIASES = {
    6: {
        "swinv2": "fuxi_24ch_6yr_ep8.json",
        "modafno": "modafno_24ch_6yr_ep8.json",
        "sdyff": "sdyff_24ch_6yr_ep8.json",
        "pixelattn_vfi": "atm_vfi_6yr_ep8_matched.json",
    },
    12: {
        "swinv2": "fuxi_3yr_ep10.json",
        "modafno": "modafno_3yr_ep10.json",
        "sdyff": "sdyff_3yr_ep10.json",
        "pixelattn_vfi": "atm_vfi_3yr_ep10_matched.json",
    },
}
COHORTS = ((6, 2020), (6, 2021), (12, 2020))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_payload(path: Path, horizon: int, year: int) -> dict:
    payload = json.loads(path.read_text())
    channels = payload.get("channel_names")
    if not isinstance(channels, list) or len(channels) != 24:
        raise ValueError(f"{path}: expected 24 scored channels")
    if payload.get("years") != [year]:
        raise ValueError(f"{path}: expected years=[{year}]")
    required_taus = set(map(str, range(1, horizon)))
    if set(payload.get("per_tau", {})) != required_taus:
        raise ValueError(f"{path}: incomplete query-hour set")
    for tau in required_taus:
        entry = payload["per_tau"][tau].get("model", {})
        for channel in channels:
            for prefix in ("rmse_norm_", "acc_"):
                value = entry.get(f"{prefix}{channel}")
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(
                        f"{path}: invalid {prefix}{channel} at tau={tau}"
                    )
    protocol = payload.get("evaluation_protocol", {})
    if (
        protocol.get("full_year") is not True
        or protocol.get("sample_strategy") != "all_valid_anchor_windows"
    ):
        raise ValueError(f"{path}: expected full-year evaluation")
    return payload


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(source.read_bytes())
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "metrics" / "corrected_baselines_v1",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=ROOT / "metrics" / "journal_unified",
    )
    args = parser.parse_args()

    completion = args.source_root / ".complete"
    marker = json.loads(completion.read_text())
    if marker.get("status") != "complete":
        raise ValueError(f"incomplete corrected-baseline source: {completion}")

    records = []
    for horizon, year in COHORTS:
        source_dir = args.source_root / f"{horizon}h" / str(year) / "full_year"
        target_dir = args.target_root / f"{horizon}h_{year}"
        for model, alias in MODEL_ALIASES[horizon].items():
            source = source_dir / f"{model}.json"
            target = target_dir / alias
            payload = validate_payload(source, horizon, year)
            atomic_copy(source, target)
            records.append(
                {
                    "horizon_hours": horizon,
                    "year": year,
                    "model": model,
                    "source": str(source.relative_to(ROOT)),
                    "source_sha256": sha256_file(source),
                    "paper_alias": str(target.relative_to(ROOT)),
                    "paper_alias_sha256": sha256_file(target),
                    "checkpoint_sha256": payload["checkpoint_provenance"]["sha256"],
                }
            )

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "source_completion_marker": str(completion.relative_to(ROOT)),
        "source_completion_sha256": sha256_file(completion),
        "records": records,
    }
    output = args.target_root / "corrected_baselines_v1.manifest.json"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(f"promoted {len(records)} artifacts; wrote {output}")


if __name__ == "__main__":
    main()

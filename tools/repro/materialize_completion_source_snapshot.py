#!/usr/bin/env python3
"""Materialize source files bound by a completion marker's SHA-256 map."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(marker: dict[str, Any]) -> dict[str, str]:
    hashes = marker.get("source_sha256")
    if not isinstance(hashes, dict):
        hashes = marker.get("source_files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("completion marker has no source-file hashes")
    return {str(name): str(value) for name, value in hashes.items()}


def _candidate_paths(
    source_name: str,
    marker_path: Path,
    search_root: Path,
) -> list[Path]:
    source = Path(source_name)
    if source.is_absolute():
        return [source]
    candidates = [marker_path.parent / "source_snapshot" / source]
    candidates.extend(parent / source for parent in marker_path.parents)
    candidates.append(search_root / source)
    candidates.extend(
        child / source
        for child in sorted(search_root.iterdir())
        if child.is_dir()
    )
    return candidates


def materialize_snapshot(
    marker_path: Path,
    search_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    marker_path = marker_path.resolve()
    search_root = search_root.resolve()
    marker = json.loads(marker_path.read_text())
    source_hashes = _source_hashes(marker)
    target_root = (
        output_dir.resolve()
        if output_dir is not None
        else marker_path.parent / "source_snapshot"
    )
    records: dict[str, dict[str, str]] = {}
    missing: list[str] = []

    for source_name, expected in sorted(source_hashes.items()):
        match = next(
            (
                candidate
                for candidate in _candidate_paths(
                    source_name,
                    marker_path,
                    search_root,
                )
                if candidate.is_file() and _sha256(candidate) == expected
            ),
            None,
        )
        if match is None:
            missing.append(source_name)
            continue
        relative = Path(source_name)
        if relative.is_absolute():
            relative = Path("absolute") / relative.relative_to(relative.anchor)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(match, temporary)
        if _sha256(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"snapshot copy hash mismatch for {source_name}")
        os.replace(temporary, target)
        records[source_name] = {
            "sha256": expected,
            "snapshot_path": str(target),
            "discovered_from": str(match.resolve()),
        }

    if missing:
        raise FileNotFoundError(
            "no matching source content for: " + ", ".join(missing)
        )

    manifest = {
        "schema_version": 1,
        "completion_marker": str(marker_path),
        "completion_marker_sha256": _sha256(marker_path),
        "source_composite_sha256": marker.get("source_composite_sha256"),
        "source_files": records,
    }
    target_root.mkdir(parents=True, exist_ok=True)
    manifest_path = target_root / "manifest.json"
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = materialize_snapshot(
        args.marker,
        args.search_root,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "completion_marker": manifest["completion_marker"],
                "n_source_files": len(manifest["source_files"]),
                "status": "complete",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

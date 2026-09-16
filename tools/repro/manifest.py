"""Validated, dependency-free experiment manifest loader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    command: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    environment: dict[str, str]


@dataclass(frozen=True)
class Manifest:
    path: Path
    schema_version: int
    experiments: dict[str, Experiment]


def _relative_path(value: Any, *, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key}: expected a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{key}: path must stay inside the repository: {value}")
    return path.as_posix()


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, dict) or not raw_experiments:
        raise ValueError("manifest must define at least one experiment")

    experiments: dict[str, Experiment] = {}
    for name, raw in raw_experiments.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise ValueError("experiment entries must be named objects")
        command = raw.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise ValueError(f"{name}.command must be a non-empty string list")
        environment = raw.get("environment", {})
        if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ValueError(f"{name}.environment must map strings to strings")
        experiments[name] = Experiment(
            name=name,
            description=str(raw.get("description", "")),
            command=tuple(command),
            inputs=tuple(
                _relative_path(item, key=f"{name}.inputs")
                for item in raw.get("inputs", [])
            ),
            outputs=tuple(
                _relative_path(item, key=f"{name}.outputs")
                for item in raw.get("outputs", [])
            ),
            environment=dict(environment),
        )
    return Manifest(manifest_path, 1, experiments)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_paths(root: Path, paths: tuple[str, ...]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for relative in paths:
        path = root / relative
        record: dict[str, Any] = {"path": relative, "exists": path.exists()}
        if path.is_file():
            record.update(size_bytes=path.stat().st_size, sha256=sha256_path(path))
        elif path.is_dir():
            files = sorted(
                candidate for candidate in path.rglob("*") if candidate.is_file()
            )
            record.update(
                file_count=len(files),
                size_bytes=sum(item.stat().st_size for item in files),
            )
        inventory.append(record)
    return inventory

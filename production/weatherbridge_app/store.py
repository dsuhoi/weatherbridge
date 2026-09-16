"""Immutable forecast bundle store with atomic latest-pointer updates."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")


class BundleStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.runs = self.root / "runs"

    def initialize(self) -> None:
        self.runs.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate(value: str, pattern: re.Pattern[str], kind: str) -> str:
        if not pattern.fullmatch(value):
            raise ValueError(f"invalid {kind}: {value!r}")
        return value

    def latest_id(self) -> str | None:
        path = self.root / "latest.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self._validate(str(payload["run_id"]), RUN_ID, "run id")

    def list_runs(self, limit: int = 24) -> list[dict[str, Any]]:
        if not self.runs.is_dir():
            return []
        manifests = sorted(
            self.runs.glob("*/manifest.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return [self.load_manifest(path.parent.name) for path in manifests[:limit]]

    def load_manifest(self, run_id: str) -> dict[str, Any]:
        run_id = self._validate(run_id, RUN_ID, "run id")
        path = self.runs / run_id / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_id") != run_id:
            raise ValueError(f"bundle identity mismatch for {run_id}")
        return payload

    def image_path(self, run_id: str, field: str, tau: int) -> Path:
        run_id = self._validate(run_id, RUN_ID, "run id")
        field = self._validate(field, FIELD, "field")
        if tau < 1 or tau > 11:
            raise ValueError(f"invalid interpolation hour: {tau}")
        manifest = self.load_manifest(run_id)
        try:
            relative = manifest["fields"][field]["images"][str(tau)]
        except KeyError as exc:
            raise FileNotFoundError(f"{field} at tau={tau} is unavailable") from exc
        path = (self.runs / run_id / relative).resolve()
        run_root = (self.runs / run_id).resolve()
        if run_root not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        return path

    def publish(self, staged_run: Path, run_id: str) -> Path:
        run_id = self._validate(run_id, RUN_ID, "run id")
        self.initialize()
        target = self.runs / run_id
        if target.exists():
            raise FileExistsError(target)
        os.replace(staged_run, target)
        self._atomic_json(self.root / "latest.json", {"run_id": run_id})
        return target

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)

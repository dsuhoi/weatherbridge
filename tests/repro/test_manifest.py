from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.repro.manifest import inventory_paths, load_manifest, sha256_path
from tools.repro.run import _git


def test_archive_does_not_inherit_enclosing_git_identity(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    assert _git(extracted, "rev-parse", "--show-toplevel") is None
    assert _git(tmp_path, "rev-parse", "--show-toplevel") == str(tmp_path.resolve())


def test_repository_manifest_is_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / "repro" / "paper_experiments.json")
    assert "weatherbridge_spectral_figure" in manifest.experiments
    assert "weatherbridge_case_figures" in manifest.experiments
    assert "npj_manuscript" in manifest.experiments
    assert "npj_submission_package" in manifest.experiments
    assert "cloud_journal_spectra" in manifest.experiments
    assert "cloud_champion_selection" in manifest.experiments
    assert "cloud_postselection_2022" in manifest.experiments
    serialized = (root / "repro" / "paper_experiments.json").read_text()
    assert "detailed_benchmark_interim" not in serialized
    assert "detailed_benchmark_v1" not in serialized
    assert "FINAL_PUBLICATION=1" in serialized


def test_runner_preserves_python_and_import_paths(tmp_path, monkeypatch):
    import os
    import sys
    from types import SimpleNamespace
    from tools.repro import run

    manifest = tmp_path / "experiment.json"
    manifest.write_text(json.dumps({"schema_version": 1, "experiments": {
        "probe": {"description": "Environment check", "command": ["python", "-V"],
                  "inputs": [], "outputs": []}}}))
    monkeypatch.delenv("PYTHON", raising=False)
    monkeypatch.setattr(sys, "argv", ["run", "probe", "--manifest", str(manifest),
                                     "--records-dir", str(tmp_path / "records")])
    monkeypatch.setattr(run, "_git", lambda *args: None)
    monkeypatch.setattr(run.platform, "platform", lambda: "test-platform")

    def execute(command, *, cwd, env, check):
        assert command[0] == sys.executable
        assert env["PYTHON"] == sys.executable
        assert str(cwd) in env["PYTHONPATH"].split(os.pathsep)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run.subprocess, "run", execute)
    assert run.main() == 0


def test_manifest_rejects_paths_outside_repository(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiments": {"bad": {"command": ["true"], "inputs": ["../secret"]}},
            }
        )
    )
    with pytest.raises(ValueError, match="inside the repository"):
        load_manifest(path)


def test_inventory_fingerprints_files(tmp_path: Path) -> None:
    payload = tmp_path / "result.txt"
    payload.write_text("weatherbridge\n")
    inventory = inventory_paths(tmp_path, ("result.txt", "missing.txt"))
    assert inventory[0]["sha256"] == sha256_path(payload)
    assert inventory[1] == {"path": "missing.txt", "exists": False}

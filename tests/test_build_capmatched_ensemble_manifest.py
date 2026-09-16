from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


SCRIPT = Path("tools/eval/build_capmatched_ensemble_manifest.py")


def _checkpoint(path: Path, seed: int) -> None:
    torch.save(
        {
            "global_step": 410,
            "hyper_parameters": {
                "arch": "dcae_fm_14m",
                "training_seed": seed,
                "training_code_sha256": {"trainer": "abc"},
                "training_input_provenance": {"dataset": "def"},
            },
            "state_dict": {"net.weight": torch.ones(2, 3)},
        },
        path,
    )


def test_builder_writes_hash_locked_independent_members(tmp_path: Path) -> None:
    first = tmp_path / "first.ckpt"
    second = tmp_path / "second.ckpt"
    output = tmp_path / "fm.ensemble.json"
    _checkpoint(first, 1)
    _checkpoint(second, 2)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoints",
            str(first),
            str(second),
            "--expected-arch",
            "dcae_fm_14m",
            "--name",
            "test-ensemble",
            "--output",
            str(output),
        ],
        check=False,
    )

    assert result.returncode == 0
    manifest = json.loads(output.read_text())
    assert manifest["reduction"] == "mean"
    assert [member["seed"] for member in manifest["members"]] == [1, 2]
    assert all(len(member["sha256"]) == 64 for member in manifest["members"])

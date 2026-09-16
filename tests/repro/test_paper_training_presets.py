import argparse
import ast
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "tools/train/train_capacity_matched_6h.py"


@pytest.mark.parametrize("model", ["weatherbridge", "weatherdcae", "pixelattn_vfi"])
@pytest.mark.parametrize("horizon", [6, 12])
def test_paper_training_command(model, horizon):
    result = subprocess.run(
        ["bash", "repro/scripts/train_paper_matched.sh", "--dry-run", model, str(horizon)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    command = shlex.split(result.stdout)
    def values(flag):
        start = command.index(flag) + 1
        end = next((i for i in range(start, len(command)) if command[i].startswith("--")), len(command))
        return command[start:end]

    assert values("--arch") == [{"pixelattn_vfi": "atmvfi"}.get(model, model)]
    assert values("--train_tau_subset") == (
        ["1", "3", "5"] if horizon == 6 else ["1", "2", "3", "5", "7", "9", "10", "11"]
    )
    assert values("--eval_tau") == list(map(str, range(1, horizon)))
    assert values("--train_batches_per_epoch") == ["6568" if horizon == 6 else "4372"]
    assert values("--lambda_spec_override") == ["0.02" if model == "weatherbridge" else "0"]
    assert values("--lambda_hf_override") == ["0.05" if model == "weatherbridge" else "0"]
    assert values("--spectral_mask_profile") == ["advected" if model == "weatherbridge" else "all"]
    assert "--lr" not in command


def test_public_defaults_preserve_legacy_flags_and_explicit_overrides():
    # Test the CLI policy without importing the GPU training dependencies.
    tree = ast.parse(TRAINER.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_public_arch_defaults")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TRAINER), "exec"), namespace)
    apply = namespace[function.name]
    args = argparse.Namespace(arch="weatherbridge", lambda_hf_override=None,
                              lambda_spec_override=None, lambda_band_override=None,
                              spectral_mask_profile="auto", lr=2e-4)
    apply(args)
    assert (args.lambda_hf_override, args.lambda_spec_override, args.lambda_band_override) == (0.05, 0.02, 0.0)
    assert args.spectral_mask_profile == "advected"
    assert args.lr == 2e-4
    for arch in ("flow_pp3", "weatherbridge"):
        args = argparse.Namespace(arch=arch, lambda_hf_override=0,
                                  lambda_spec_override=0, lambda_band_override=0,
                                  spectral_mask_profile="all", lr=1e-4)
        before = vars(args).copy()
        apply(args)
        assert vars(args) == before
    args.arch = "flow_pp3"
    args.lambda_spec_override = None
    args.spectral_mask_profile = "auto"
    apply(args)
    assert args.lambda_spec_override is None and args.spectral_mask_profile == "auto"


def test_manuscript_describes_residual_supervision_and_conditioning():
    main = " ".join((ROOT / "paper/main.tex").read_text().split())
    table = (ROOT / "paper/tab_architecture_settings.tex").read_text()
    assert "with unit residual gain" in main
    assert "clamps its magnitude to at least 0.05" in main
    assert "penalises residual gain values" not in main
    assert "The encoder is conditioned" not in main
    assert "decoder skip gates" in main
    assert "AdaLN decoder" not in table
    assert "bottleneck self-attention and AdaLN-Zero" in table

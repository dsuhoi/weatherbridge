"""Release acceptance tests.

Three things are checked, in increasing strength:

1. the archive is intact — every file matches its recorded SHA-256;
2. every model loads and runs, deterministically, on synthetic input;
3. every model reproduces its recorded error on the shipped ERA5 window, and
   beats linear interpolation there.

Only (3) can tell a correct checkpoint from a corrupted or swapped one, which
is why the archive carries a real sample rather than synthetic input alone.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import weatherbridge as wb
from weatherbridge._loader import DATA_DIR, WEIGHTS_DIR

REPO = Path(__file__).resolve().parents[1]
SAMPLE = DATA_DIR / "sample_era5_2020070100.npz"
REFERENCE = DATA_DIR / "reference_scores.json"
MANIFEST = REPO / "MANIFEST.sha256"

# Absolute tolerance on a reproduced RMSE. Comfortably above float ordering
# noise across BLAS builds, far below the gap between any two of these models.
RMSE_TOLERANCE = 2e-3

# S-DYff's refiner samples noise inside its forward. Seeding reproduces the
# recorded scores exactly, so the tolerance above still applies to it.
REFERENCE_SEED = 0


def _require_weights(name: str) -> None:
    """Skip rather than fail in a git clone, where the binaries are absent."""
    path = WEIGHTS_DIR / wb.model_info(name)["weights"]
    if not path.is_file():
        pytest.skip(f"{path.name} is not unpacked; run scripts/make_archive.sh output")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def sample() -> dict:
    if not SAMPLE.is_file():
        pytest.skip(f"{SAMPLE.name} is not unpacked")
    data = np.load(str(SAMPLE))
    return {
        "x0": torch.from_numpy(data["x0"].astype(np.float32)).unsqueeze(0),
        "xT": torch.from_numpy(data["xT"].astype(np.float32)).unsqueeze(0),
        "targets": {
            int(t): torch.from_numpy(
                data[f"target_tau{int(t)}"].astype(np.float32)
            ).unsqueeze(0)
            for t in data["taus"]
        },
        "delta_t": int(data["delta_t_hours"]),
    }


@pytest.fixture(scope="session")
def reference() -> dict:
    if not REFERENCE.is_file():
        pytest.skip("reference_scores.json is not unpacked")
    return json.loads(REFERENCE.read_text())


def _latitude_weights(height: int) -> torch.Tensor:
    edges = torch.linspace(90.0, -90.0, height + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    weights = torch.cos(torch.deg2rad(centres)).clamp_min(0.0)
    return (weights / weights.mean()).view(1, 1, -1, 1)


def _weighted_rmse(pred, target, weights) -> float:
    return float(torch.sqrt((weights * (pred - target) ** 2).mean()))


def test_manifest_matches_files():
    """Every file recorded in the manifest is present and unmodified."""
    if not MANIFEST.is_file():
        pytest.skip("MANIFEST.sha256 is not present")
    mismatched, missing = [], []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected, name = line.split(maxsplit=1)
        path = REPO / name.strip()
        if not path.is_file():
            missing.append(name.strip())
        elif _sha256(path) != expected:
            mismatched.append(name.strip())
    assert not missing, f"missing from the archive: {missing}"
    assert not mismatched, f"checksum mismatch: {mismatched}"


def test_catalogue_lists_seven_models():
    names = wb.list_models()
    assert len(names) == 7, names
    # WeatherBridge is exactly the two released 6 h and 12 h checkpoints.
    bridges = [n for n in names if wb.model_info(n)["family"] == "weatherbridge"]
    assert sorted(bridges) == ["weatherbridge-12h", "weatherbridge-6h"]


def test_weatherdcae_uses_the_paper_channel_contract():
    """Reject the incompatible 27-channel development checkpoint."""
    _require_weights("weatherdcae-14m-6h")
    path = WEIGHTS_DIR / wb.model_info("weatherdcae-14m-6h")["weights"]
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    assert blob["kwargs"]["in_channels"] == 24
    assert blob["kwargs"]["out_channels"] == 24
    assert blob["kwargs"]["n_static_features"] == 3


@pytest.mark.parametrize("name", wb.list_models())
def test_loads_and_runs(name):
    """The model builds from its own recorded kwargs and produces 24 channels."""
    _require_weights(name)
    model = wb.load_model(name)
    delta_t = model._wb_delta_t
    x0 = torch.zeros(1, 24, 360, 720)
    xT = torch.zeros(1, 24, 360, 720)
    out = wb.predict(model, x0, xT, tau_hours=delta_t / 2.0)
    assert out.shape == (1, 24, 360, 720)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["weatherbridge-6h", "weatherdcae-14m-6h"])
def test_deterministic(name):
    """Two runs of a deterministic model on identical input agree exactly."""
    _require_weights(name)
    model = wb.load_model(name)
    torch.manual_seed(0)
    x0 = torch.randn(1, 24, 360, 720)
    xT = torch.randn(1, 24, 360, 720)
    first = wb.predict(model, x0, xT, 3.0)
    second = wb.predict(model, x0, xT, 3.0)
    assert torch.equal(first, second)


def test_linear_interpolation_endpoints(sample):
    """The baseline returns the anchors exactly at the ends of the window."""
    x0, xT, delta_t = sample["x0"], sample["xT"], sample["delta_t"]
    assert torch.allclose(wb.linear_interpolation(x0, xT, 0, delta_t), x0)
    assert torch.allclose(wb.linear_interpolation(x0, xT, delta_t, delta_t), xT)


@pytest.mark.parametrize("name", [n for n in wb.list_models() if n.endswith("-6h")])
def test_reference_scores_reproduce(name, sample, reference):
    """Each model reproduces its recorded error on the shipped ERA5 window."""
    expected = reference["scores"].get(name)
    if expected is None:
        pytest.skip(f"no reference recorded for {name}")
    _require_weights(name)
    model = wb.load_model(name)
    weights = _latitude_weights(sample["x0"].shape[-2])
    for tau_text, want in expected.items():
        tau = int(tau_text)
        torch.manual_seed(REFERENCE_SEED)
        pred = wb.predict(model, sample["x0"], sample["xT"], tau)
        got = _weighted_rmse(pred, sample["targets"][tau], weights)
        assert abs(got - want) < RMSE_TOLERANCE, (
            f"{name} tau={tau}: recorded {want:.6f}, reproduced {got:.6f}"
        )


@pytest.mark.parametrize("name", [n for n in wb.list_models() if n.endswith("-6h")])
def test_beats_linear_interpolation(name, sample, reference):
    """Every released model improves on the baseline it is measured against."""
    scores = reference["scores"]
    if name not in scores:
        pytest.skip(f"no reference recorded for {name}")
    model_mean = float(np.mean(list(scores[name].values())))
    linear_mean = float(np.mean(list(scores["linear"].values())))
    assert model_mean < linear_mean, (
        f"{name} scores {model_mean:.4f} against linear {linear_mean:.4f}"
    )


def test_weights_are_bare_blobs():
    """No checkpoint smuggles in optimiser state or a training framework."""
    for name in wb.list_models():
        _require_weights(name)
        path = WEIGHTS_DIR / wb.model_info(name)["weights"]
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        assert set(blob) == {"arch", "kwargs", "state_dict"}, (
            f"{path.name} carries unexpected keys: {sorted(set(blob))}"
        )

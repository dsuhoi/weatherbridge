from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_fig3_spectra.py"


def _load_generator():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("make_fig3_spectra", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_metrics(module, root: Path, *, schema_version: int = 6) -> None:
    ell = np.arange(181, dtype=np.int32)
    truth = np.stack(
        [
            1.2e4 / (ell + 3.0) ** 1.7,
            9.0e3 / (ell + 4.0) ** 1.6,
        ]
    )
    metadata = {
        "schema_version": schema_version,
        "sht_grid": module.WB2_BLOCK_GRID_NAME
        + "_latitude_strip_area_sht",
        "sample_strategy": "all_valid_anchor_windows",
    }
    for model_index, stem in enumerate(module.MODELS):
        for tau in module.TAUS:
            deficit = 0.04 + 0.01 * model_index + 0.01 * (tau == 8)
            ratio = 1.0 - deficit * (ell / 180.0) ** 1.2
            np.savez(
                root / f"{stem}_tau{tau}.npz",
                ell=ell,
                pred_El=truth * ratio[None, :],
                gt_El=truth,
                channel_names=np.array(["Q850", "V850"]),
                metadata_json=np.array(json.dumps(metadata)),
            )


def test_generates_compact_four_panel_figure(tmp_path: Path) -> None:
    module = _load_generator()
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    _write_metrics(module, metrics)
    module.NPZ_DIR = metrics
    module.OUT_PDF = tmp_path / "spectra.pdf"

    module.main()

    assert module.OUT_PDF.read_bytes().startswith(b"%PDF")
    assert module.PANEL_CONFIGS == [
        ("Q850", "Q850", 5),
        ("Q850", "Q850", 6),
        ("V850", "V850", 5),
        ("V850", "V850", 6),
    ]


def test_rejects_pre_v6_artifacts(tmp_path: Path) -> None:
    module = _load_generator()
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    _write_metrics(module, metrics, schema_version=5)
    module.NPZ_DIR = metrics
    module.OUT_PDF = tmp_path / "spectra.pdf"

    with pytest.raises(ValueError, match="dense corrected spectral artifact"):
        module.main()

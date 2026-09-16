import json
from pathlib import Path

import numpy as np

from tools.eval.export_anchor_exchange_tex import export_tex


def _artifact(root: Path, year: int) -> Path:
    paired = root / f"exchange_{year}.npz"
    np.savez(paired, value=np.asarray([year]))
    import hashlib

    models = {}
    for key, exchange, forward in (
        ("flow_spectral", 0.3, 0.1),
        ("weatherdcae_14m_6yr", 0.2, 0.1),
    ):
        models[key] = {
            "exchange": {"pooled_rmse": exchange},
            "forward_error": {"pooled_rmse": forward},
            "exchange_to_forward_rmse_ratio": exchange / forward,
        }
    payload = {
        "schema_version": 2,
        "diagnostic_only": True,
        "test_year": year,
        "delta_t_hours": 6,
        "eval_hours": [1, 2, 3, 4, 5],
        "channel_names": [f"c{i}" for i in range(24)],
        "evaluation_protocol": {"n_windows": 240, "n_endpoints": 48},
        "paired_windows_file": paired.name,
        "paired_windows_size_bytes": paired.stat().st_size,
        "paired_windows_sha256": hashlib.sha256(paired.read_bytes()).hexdigest(),
        "models": models,
    }
    path = root / f"exchange_{year}.json"
    path.write_text(json.dumps(payload))
    return path


def test_export_anchor_exchange_tex_reports_both_models_and_years(
    tmp_path: Path,
) -> None:
    output = tmp_path / "table.tex"
    export_tex(_artifact(tmp_path, 2020), _artifact(tmp_path, 2021), output)
    text = output.read_text()
    assert "WeatherBridge & 2020 & 0.300 & 0.100 & 3.00" in text
    assert "WeatherDCAE-14M & 2021 & 0.200 & 0.100 & 2.00" in text

import hashlib
import json
from pathlib import Path

import pytest

from tools.eval.export_downstream_metrics_tex import (
    MODELS,
    _expected_weight_name,
    _internal_name,
    export_tables,
)
from tools.train.training_protocol import memmap_dataset_provenance


def _record(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_inputs(root: Path) -> dict[tuple[int, str], Path]:
    support = root / "support"
    support.mkdir(parents=True)
    normalization_paths = {}
    for name in ("stats", "surface_stats", "static_features"):
        path = support / name
        path.write_text(name)
        normalization_paths[name] = path
    normalization = {
        "scheme": "(x - mean) / std",
        **{name: _record(path) for name, path in normalization_paths.items()},
    }
    memmap = support / "memmap"
    memmap.mkdir()
    (memmap / "wb2_2020.json").write_text(
        json.dumps(
            {
                "T": 2,
                "n_channels": 1,
                "H": 1,
                "W": 1,
                "shape": [2, 1, 1, 1],
            }
        )
    )
    (memmap / "wb2_2020.bin").write_bytes(b"12345678")
    dataset_provenance = memmap_dataset_provenance(memmap, [2020])
    weights: dict[tuple[int, str], Path] = {}
    months = {f"2020-{month:02d}" for month in range(1, 13)}
    regions = {name: {} for name in ("sahel", "amazon", "congo", "se_aus")}

    for horizon in (6, 12):
        output = root / f"{horizon}h" / "2020"
        output.mkdir(parents=True)
        start_hours = [48 * index for index in range(180)]
        index_sha = hashlib.sha256(
            "\n".join(map(str, start_hours)).encode()
        ).hexdigest()
        coverage = {
            "anchor_hours_utc": [0, 6, 12, 18] if horizon == 6 else [0, 12],
            "months": 12,
            "last_anchor_interval_included": True,
            "time_basis": "approximate local solar time",
            "month_windows": {
                month: {
                    "n_hours": 24 * 28,
                    "n_complete_days": 28,
                    "dropped_tail_hours_without_right_anchor": 0,
                }
                for month in months
            },
        }
        for model_index, (stem, _) in enumerate(MODELS):
            internal_name = _internal_name(stem, horizon)
            weight_name = _expected_weight_name(stem, horizon)
            if weight_name is None:
                model_provenance = {"kind": "linear"}
            else:
                weight = support / weight_name
                weight.write_text(f"{stem}-{horizon}")
                weights[horizon, stem] = weight
                model_provenance = _record(weight)
            value = 0.8 + 0.02 * model_index
            physics = {
                "model_name": internal_name,
                "year": 2020,
                "delta_t_hours": horizon,
                "taus": list(range(1, horizon)),
                "n_pairs_used": 180,
                "sample_start_hours": start_hours,
                "sampling": {
                    "strategy": "uniform_over_year",
                    "candidate_stride_hours": 48,
                    "max_pairs": 180,
                    "index_sha256": index_sha,
                },
                "normalization": normalization,
                "evaluation_dataset_provenance": dataset_provenance,
                "model_provenance": model_provenance,
                "per_tau": {
                    str(tau): {
                        internal_name: {
                            "ageo_ratio_850": value,
                            "ageo_ratio_700": value + 0.01,
                            "hydrostatic_ratio": value + 0.02,
                            "n_pairs": 180,
                        }
                    }
                    for tau in range(1, horizon)
                },
            }
            monthly = {
                region: {
                    month: {
                        field: {
                            "amp_ratio": 1.0 + 0.02 * (model_index + 1),
                            "peak_hour_error": float(model_index + 1),
                        }
                        for field in ("t2m", "u10", "v10")
                    }
                    for month in months
                }
                for region in regions
            }
            diurnal = {
                "model_name": internal_name,
                "year": 2020,
                "delta_t_hours": horizon,
                "taus": list(range(1, horizon)),
                "surface_channels": ["t2m", "u10", "v10"],
                "regions": regions,
                "coverage": coverage,
                "normalization": normalization,
                "evaluation_dataset_provenance": dataset_provenance,
                "model_provenance": model_provenance,
                "monthly": monthly,
            }
            (output / f"physics_{internal_name}.json").write_text(
                json.dumps(physics)
            )
            (output / f"diurnal_{internal_name}.json").write_text(
                json.dumps(diurnal)
            )
    return weights


def test_export_downstream_tables_bind_all_models_and_protocol(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    physics = tmp_path / "physics.tex"
    diurnal = tmp_path / "diurnal.tex"

    exported = export_tables(tmp_path, physics, diurnal)

    assert set(exported) == {str(physics), str(diurnal)}
    for text in (physics.read_text(), diurnal.read_text()):
        assert "WeatherBridge" in text
        assert "WeatherDCAE-14M" in text
        assert "Linear Interp." in text
    assert "180 uniformly spaced 2020 anchor windows" in physics.read_text()


def test_export_rejects_stale_model_artifact(tmp_path: Path) -> None:
    weights = _write_inputs(tmp_path)
    weights[6, "flow_spectral"].write_text("changed")

    with pytest.raises(ValueError, match="stale provenance"):
        export_tables(tmp_path, tmp_path / "physics.tex", tmp_path / "diurnal.tex")


def test_export_rejects_mismatched_sample_index(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    path = tmp_path / "6h/2020/physics_flow_spectral.json"
    payload = json.loads(path.read_text())
    hours = [hour + 48 for hour in payload["sample_start_hours"]]
    payload["sample_start_hours"] = hours
    payload["sampling"]["index_sha256"] = hashlib.sha256(
        "\n".join(map(str, hours)).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="different indices or normalization"):
        export_tables(tmp_path, tmp_path / "physics.tex", tmp_path / "diurnal.tex")


def test_export_rejects_mutated_evaluation_dataset(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    (tmp_path / "support/memmap/wb2_2020.bin").write_bytes(b"87654321")

    with pytest.raises(ValueError, match="stale evaluation dataset provenance"):
        export_tables(tmp_path, tmp_path / "physics.tex", tmp_path / "diurnal.tex")

import hashlib
import json
from pathlib import Path

import pytest

from tools.eval.export_case_metrics_tex import (
    HAISHEN_METHODS,
    HAISHEN_REFERENCES,
    HAISHEN_SOURCES,
    HAISHEN_SUMMARY,
    IDA_METHODS,
    IDA_SOURCES,
    IDA_SUMMARY,
    export_case_metrics,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _provenance(label: str, arch: str, checkpoint: Path, support: Path) -> dict:
    return {
        "schema_version": 3,
        "model_label": label,
        "model_arch": arch,
        "grid_convention": "wb2_0p25_pair_average_cell_centres_v1",
        "checkpoint": _record(checkpoint),
        "exporter": _record(support / "exporter.py"),
        "static": _record(support / "static.pt"),
        "stats": _record(support / "stats.nc"),
        "surface_stats": _record(support / "surface.json"),
        "memmap": {
            "path": str(support / "year.bin"),
            "size_bytes": (support / "year.bin").stat().st_size,
        },
        "target_crop_sha256": "a" * 64,
        "x0_24_sha256": "b" * 64,
        "xT_24_sha256": "c" * 64,
        "latitude_sha256": "d" * 64,
        "longitude_sha256": "e" * 64,
    }


def _metrics(unit: str, labels: tuple[str, ...] | None = None) -> dict:
    per_tau_key = "per_tau_hpa" if unit == "hPa" else "per_tau_m_s"
    mean_key = "mean_all_tau_hpa" if unit == "hPa" else "mean_all_tau_m_s"
    result = {}
    if labels is None:
        labels = HAISHEN_METHODS
    for index, label in enumerate(labels, start=1):
        result[label] = {
            per_tau_key: {str(tau): index / 10 + tau / 100 for tau in range(1, 6)},
            mean_key: index / 10,
        }
        if unit != "hPa":
            result[label]["tau3_error_at_truth_max_m_s"] = -float(index)
    return result


def _write_case_inputs(root: Path) -> Path:
    support = root / "support"
    support.mkdir(parents=True)
    for name in ("exporter.py", "static.pt", "stats.nc", "surface.json", "year.bin"):
        (support / name).write_text(name)
    weatherbridge_checkpoint = support / "weatherbridge.ckpt"
    weatherbridge_checkpoint.write_text("weatherbridge")
    for relative in HAISHEN_SOURCES | IDA_SOURCES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    def source_hashes(values: set[str]) -> dict[str, str]:
        return {value: _sha256(root / value) for value in values}
    weatherbridge_provenance = _provenance(
        "WeatherBridge", "flow_pp3", weatherbridge_checkpoint, support
    )
    comparator_provenance = {}
    for label, (arch, reference) in HAISHEN_REFERENCES.items():
        checkpoint = support / f"{arch}.ckpt"
        checkpoint.write_text(arch)
        comparator_provenance[label] = _provenance(
            label, arch, checkpoint, support
        )
        reference_path = root / reference
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(
            json.dumps({"checkpoint_provenance": _record(checkpoint)})
        )
    haishen = {
        "field": "mslp",
        "units": "hPa",
        "event": "Typhoon Haishen",
        "date": "2020-09-07",
        "taus_hours": [1, 2, 3, 4, 5],
        "evaluation_bbox": {
            "lat_min": 25.0,
            "lat_max": 48.0,
            "lon_min": 115.0,
            "lon_max": 145.0,
        },
        "metrics": _metrics("hPa"),
        "source_provenance": source_hashes(HAISHEN_SOURCES),
        "model_provenance": {
            **comparator_provenance,
            "WeatherBridge": weatherbridge_provenance,
        },
    }
    ida = {
        "field": "10m_wind_speed",
        "units": "m s-1",
        "event": "Hurricane Ida",
        "date": "2021-08-29",
        "anchor_hours_utc": [12, 18],
        "taus_hours": [1, 2, 3, 4, 5],
        "evaluation_bbox": {
            "lat_min": 20.0,
            "lat_max": 32.0,
            "lon_min": 260.0,
            "lon_max": 285.0,
        },
        "truth_max_tau3": {
            "latitude": 29.25,
            "longitude": 270.5,
            "speed_m_s": 30.2,
        },
        "metrics": _metrics("m s-1", IDA_METHODS),
        "source_provenance": source_hashes(IDA_SOURCES),
        "model_provenance": {
            "WeatherBridge": {
                "u10": weatherbridge_provenance,
                "v10": weatherbridge_provenance,
            },
        },
    }
    for relative, payload in ((HAISHEN_SUMMARY, haishen), (IDA_SUMMARY, ida)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return weatherbridge_checkpoint


def test_export_case_metrics_binds_sources_and_checkpoints(tmp_path: Path) -> None:
    weatherbridge = _write_case_inputs(tmp_path)
    output = tmp_path / "case_metrics.tex"

    checksum = export_case_metrics(tmp_path, weatherbridge, output)

    assert checksum == _sha256(output)
    text = output.read_text()
    assert "\\WBHaishenWeatherBridgeTauThree" in text
    assert "\\WBIdaDCAETauThree" in text
    assert "\\WBIdaWeatherBridgePeakError" in text


def test_export_case_metrics_rejects_stale_source(tmp_path: Path) -> None:
    weatherbridge = _write_case_inputs(tmp_path)
    (tmp_path / next(iter(HAISHEN_SOURCES))).write_text("changed")

    with pytest.raises(ValueError, match="source hash mismatch"):
        export_case_metrics(tmp_path, weatherbridge, tmp_path / "case_metrics.tex")


def test_export_case_metrics_rejects_wrong_checkpoint(tmp_path: Path) -> None:
    _write_case_inputs(tmp_path)
    replacement = tmp_path / "replacement.ckpt"
    replacement.write_text("replacement")

    with pytest.raises(ValueError, match="stale file"):
        export_case_metrics(tmp_path, replacement, tmp_path / "case_metrics.tex")

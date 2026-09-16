#!/usr/bin/env python3
"""Validate case-study provenance and export manuscript metric macros."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

HAISHEN_SUMMARY = Path(
    "metrics/case_studies_weatherbridge/typhoon_haishen/mslp_summary.json"
)
IDA_SUMMARY = Path(
    "metrics/case_studies_weatherbridge/"
    "hurricane_ida_2021/wind_speed_summary.json"
)
HAISHEN_SOURCES = {
    "demo/precomputed/typhoon_haishen/East_Asia__mslp.npz",
    "metrics/case_studies_pixelattn_vfi/typhoon_haishen/mslp.npz",
    "metrics/case_studies_weatherdcae_14m/typhoon_haishen/mslp.npz",
    "metrics/case_studies_weatherbridge/typhoon_haishen/mslp.npz",
}
HAISHEN_METHODS = (
    "Linear Interp.",
    "WeatherDCAE-14M",
    "PixelAttn-VFI",
    "WeatherBridge",
)
HAISHEN_REFERENCES = {
    "WeatherDCAE-14M": (
        "dcae_14m",
        Path("metrics/journal_unified/6h_2020/weatherdcae_14m_6yr_ep8_matched.json"),
    ),
    "PixelAttn-VFI": (
        "atmvfi",
        Path("metrics/journal_unified/6h_2020/atm_vfi_6yr_ep8_matched.json"),
    ),
}
IDA_SOURCES = {
    "demo/precomputed_2021/hurricane_ida_2021/North_America__u10.npz",
    "demo/precomputed_2021/hurricane_ida_2021/North_America__v10.npz",
    "metrics/case_studies_weatherbridge/hurricane_ida_2021/u10.npz",
    "metrics/case_studies_weatherbridge/hurricane_ida_2021/v10.npz",
}
IDA_METHODS = (
    "Linear Interp.",
    "SwinV2",
    "S-DYff",
    "WeatherDCAE-14M",
    "WeatherBridge",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite value")
    return result


def _validate_record(
    record: dict[str, Any],
    context: str,
    expected_path: Path | None = None,
) -> None:
    recorded_path = Path(str(record.get("path", "")))
    path = expected_path if expected_path is not None else recorded_path
    if not str(record.get("path", "")):
        raise ValueError(f"{context}: missing recorded path")
    if not path.is_file():
        raise ValueError(f"{context}: missing file {path}")
    if record.get("sha256") != _sha256(path):
        raise ValueError(f"{context}: stale file {path}")
    if "size_bytes" in record and int(record["size_bytes"]) != path.stat().st_size:
        raise ValueError(f"{context}: file size mismatch {path}")


def _validate_source_files(
    root: Path,
    payload: dict[str, Any],
    expected: set[str],
) -> None:
    records = payload.get("source_provenance", {})
    if set(records) != expected:
        raise ValueError("case summary has an unexpected source set")
    for relative, expected_sha in records.items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected_sha:
            raise ValueError(f"case source hash mismatch: {path}")


def _validate_model_provenance(
    provenance: dict[str, Any],
    *,
    arch: str,
    checkpoint: Path,
) -> None:
    if (
        provenance.get("schema_version") != 3
        or provenance.get("model_arch") != arch
        or provenance.get("grid_convention")
        != "wb2_0p25_pair_average_cell_centres_v1"
    ):
        raise ValueError(f"invalid {arch} model provenance")
    _validate_record(
        provenance.get("checkpoint", {}),
        f"{arch} checkpoint",
        checkpoint,
    )
    for key in ("exporter", "static", "stats", "surface_stats"):
        _validate_record(provenance.get(key, {}), f"{arch} {key}")
    memmap = provenance.get("memmap", {})
    memmap_path = Path(str(memmap.get("path", "")))
    if (
        not memmap_path.is_file()
        or int(memmap.get("size_bytes", -1)) != memmap_path.stat().st_size
    ):
        raise ValueError(f"{arch}: invalid memmap provenance")
    for key in (
        "target_crop_sha256",
        "x0_24_sha256",
        "xT_24_sha256",
        "latitude_sha256",
        "longitude_sha256",
    ):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{arch}: missing {key}")


def _validate_case_model_reference(
    root: Path,
    provenance: dict[str, Any],
    *,
    arch: str,
    reference: Path,
) -> None:
    if (
        provenance.get("schema_version") != 3
        or provenance.get("model_arch") != arch
        or provenance.get("grid_convention")
        != "wb2_0p25_pair_average_cell_centres_v1"
    ):
        raise ValueError(f"invalid {arch} case provenance")
    payload = json.loads((root / reference).read_text())
    expected = payload.get("checkpoint_provenance", {})
    recorded = provenance.get("checkpoint", {})
    for key in ("path", "sha256", "size_bytes"):
        if recorded.get(key) != expected.get(key):
            raise ValueError(f"{arch}: case checkpoint does not match {reference}")


def _validate_metrics(
    payload: dict[str, Any],
    labels: tuple[str, ...],
    per_tau_key: str,
    mean_key: str,
) -> None:
    metrics = payload.get("metrics", {})
    if set(metrics) != set(labels):
        raise ValueError("case summary has an unexpected method set")
    for label in labels:
        row = metrics[label]
        per_tau = row.get(per_tau_key, {})
        if set(per_tau) != {"1", "2", "3", "4", "5"}:
            raise ValueError(f"{label}: incomplete case tau grid")
        for tau, value in per_tau.items():
            _finite(value, f"{label} tau={tau}")
        _finite(row.get(mean_key), f"{label} mean")


def _format(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def export_case_metrics(
    root: Path,
    weatherbridge_checkpoint: Path,
    out_tex: Path,
) -> str:
    haishen = json.loads((root / HAISHEN_SUMMARY).read_text())
    ida = json.loads((root / IDA_SUMMARY).read_text())
    if (
        haishen.get("field") != "mslp"
        or haishen.get("units") != "hPa"
        or haishen.get("event") != "Typhoon Haishen"
        or haishen.get("date") != "2020-09-07"
        or haishen.get("taus_hours") != [1, 2, 3, 4, 5]
        or haishen.get("evaluation_bbox")
        != {"lat_min": 25.0, "lat_max": 48.0, "lon_min": 115.0, "lon_max": 145.0}
    ):
        raise ValueError("invalid Haishen case protocol")
    if (
        ida.get("field") != "10m_wind_speed"
        or ida.get("units") != "m s-1"
        or ida.get("event") != "Hurricane Ida"
        or ida.get("date") != "2021-08-29"
        or ida.get("anchor_hours_utc") != [12, 18]
        or ida.get("taus_hours") != [1, 2, 3, 4, 5]
        or ida.get("evaluation_bbox")
        != {"lat_min": 20.0, "lat_max": 32.0, "lon_min": 260.0, "lon_max": 285.0}
    ):
        raise ValueError("invalid Ida case protocol")
    _validate_source_files(root, haishen, HAISHEN_SOURCES)
    _validate_source_files(root, ida, IDA_SOURCES)
    _validate_metrics(
        haishen,
        HAISHEN_METHODS,
        "per_tau_hpa",
        "mean_all_tau_hpa",
    )
    _validate_metrics(
        ida,
        IDA_METHODS,
        "per_tau_m_s",
        "mean_all_tau_m_s",
    )
    for label in IDA_METHODS:
        _finite(
            ida["metrics"][label].get("tau3_error_at_truth_max_m_s"),
            f"{label} Ida peak error",
        )
    for label, arch, checkpoint in (
        ("WeatherBridge", "flow_pp3", weatherbridge_checkpoint),
    ):
        _validate_model_provenance(
            haishen.get("model_provenance", {}).get(label, {}),
            arch=arch,
            checkpoint=checkpoint,
        )
        components = ida.get("model_provenance", {}).get(label, {})
        if set(components) != {"u10", "v10"}:
            raise ValueError(f"{label}: incomplete Ida component provenance")
        for component in ("u10", "v10"):
            _validate_model_provenance(
                components[component],
                arch=arch,
                checkpoint=checkpoint,
            )
    for label, (arch, reference) in HAISHEN_REFERENCES.items():
        _validate_case_model_reference(
            root,
            haishen.get("model_provenance", {}).get(label, {}),
            arch=arch,
            reference=reference,
        )

    hm = haishen["metrics"]
    im = ida["metrics"]
    truth = ida.get("truth_max_tau3", {})
    commands = {
        "WBHaishenLinearTauThree": _format(hm["Linear Interp."]["per_tau_hpa"]["3"]),
        "WBHaishenWeatherBridgeTauThree": _format(hm["WeatherBridge"]["per_tau_hpa"]["3"]),
        "WBHaishenLinearMean": _format(hm["Linear Interp."]["mean_all_tau_hpa"]),
        "WBHaishenWeatherBridgeMean": _format(hm["WeatherBridge"]["mean_all_tau_hpa"]),
        "WBIdaLinearTauThree": _format(im["Linear Interp."]["per_tau_m_s"]["3"]),
        "WBIdaDCAETauThree": _format(im["WeatherDCAE-14M"]["per_tau_m_s"]["3"]),
        "WBIdaWeatherBridgeTauThree": _format(im["WeatherBridge"]["per_tau_m_s"]["3"]),
        "WBIdaLinearMean": _format(im["Linear Interp."]["mean_all_tau_m_s"]),
        "WBIdaDCAEMean": _format(im["WeatherDCAE-14M"]["mean_all_tau_m_s"]),
        "WBIdaWeatherBridgeMean": _format(im["WeatherBridge"]["mean_all_tau_m_s"]),
        "WBIdaTruthMax": _format(_finite(truth.get("speed_m_s"), "Ida truth max"), 1),
        "WBIdaTruthMaxLat": _format(_finite(truth.get("latitude"), "Ida latitude"), 2),
        "WBIdaTruthMaxLon": _format(_finite(truth.get("longitude"), "Ida longitude"), 1),
        "WBIdaLinearPeakError": _format(im["Linear Interp."]["tau3_error_at_truth_max_m_s"], 1),
        "WBIdaDCAEPeakError": _format(im["WeatherDCAE-14M"]["tau3_error_at_truth_max_m_s"], 1),
        "WBIdaWeatherBridgePeakError": _format(im["WeatherBridge"]["tau3_error_at_truth_max_m_s"], 1),
    }
    lines = [
        "% Auto-generated by tools/eval/export_case_metrics_tex.py.",
        *[
            f"\\newcommand{{\\{name}}}{{{value}}}"
            for name, value in commands.items()
        ],
    ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_tex.with_suffix(out_tex.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, out_tex)
    return _sha256(out_tex)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--weatherbridge-checkpoint", type=Path, required=True)
    parser.add_argument("--out-tex", type=Path, required=True)
    args = parser.parse_args()
    checksum = export_case_metrics(
        args.root.resolve(),
        args.weatherbridge_checkpoint.resolve(),
        args.out_tex,
    )
    print(json.dumps({str(args.out_tex): checksum}, sort_keys=True))


if __name__ == "__main__":
    main()

import json
from pathlib import Path

from tools.eval.export_main_metrics_tex import (
    FILES,
    TABLE_12_MODELS,
    _commands,
    _write_query_generalization_table,
    _write_12h_table,
)


def _payload(horizon: int, year: int, value: float, index: str) -> dict:
    channels = [f"c{i}" for i in range(24)]
    per_tau = {}
    for tau in range(1, horizon):
        model = {f"rmse_norm_{channel}": value for channel in channels}
        model.update({f"acc_{channel}": 1.0 - value for channel in channels})
        bilinear = {f"rmse_norm_{channel}": 2.0 * value for channel in channels}
        per_tau[str(tau)] = {"model": model, "bilinear": bilinear}
    return {
        "years": [year],
        "delta_t_hours": float(horizon),
        "channel_names": channels,
        "evaluation_protocol": {
            "full_year": True,
            "latitude_grid": "wb2_0p25_2x2_block_average_v1",
            "eval_hours": list(range(1, horizon)),
            "index_sha256": index,
        },
        "per_tau": per_tau,
    }


def test_exported_commands_use_matched_full_year_metrics(tmp_path: Path) -> None:
    for key, (horizon, year, filename) in FILES.items():
        value = 0.1 if key.startswith("bridge") else 0.2
        if key.startswith("dcae"):
            value = 0.15
        path = tmp_path / f"{horizon}h_{year}" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_payload(horizon, year, value, f"idx-{horizon}")))

    commands = _commands(tmp_path)

    assert commands["WBSixRMSE"] == "0.10000"
    assert commands["WBSixDCAERMSE"] == "0.15000"
    assert commands["WBSixLinearRMSE"] == "0.20000"
    assert commands["WBSixVsLinearGainPct"] == "50.0"
    assert commands["WBSixHeldRMSE"] == "0.10000"
    assert commands["WBSixHeldVsDCAEGainPct"] == "33.3"
    assert commands["WBSixHeldVsLinearGainPct"] == "50.0"
    assert commands["WBTwelveHeldACC"] == "0.90000"
    assert commands["WBTwelveHeldVsDCAEGainPct"] == "33.3"
    assert commands["WBTwelveVsLinearHeldGainPct"] == "50.0"


def test_export_rejects_mismatched_window_indices(tmp_path: Path) -> None:
    for key, (horizon, year, filename) in FILES.items():
        index = f"idx-{horizon}"
        if key == "bridge6":
            index = "wrong"
        path = tmp_path / f"{horizon}h_{year}" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_payload(horizon, year, 0.1, index)))

    try:
        _commands(tmp_path)
    except ValueError as error:
        assert "different window indices" in str(error)
    else:
        raise AssertionError("mismatched indices were accepted")


def test_export_rejects_unverified_latitude_grid(tmp_path: Path) -> None:
    for key, (horizon, year, filename) in FILES.items():
        payload = _payload(horizon, year, 0.1, f"idx-{horizon}")
        if key == "bridge6":
            payload["evaluation_protocol"]["latitude_grid"] = "legacy"
        path = tmp_path / f"{horizon}h_{year}" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    try:
        _commands(tmp_path)
    except ValueError as error:
        assert "latitude grid" in str(error)
    else:
        raise AssertionError("unverified latitude geometry was accepted")


def test_generated_12h_table_contains_central_model(tmp_path: Path) -> None:
    for index, (_, _, filename) in enumerate(TABLE_12_MODELS):
        path = tmp_path / "12h_2020" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_payload(12, 2020, 0.1 + index / 100, "same")))
    output = tmp_path / "table.tex"

    _write_12h_table(tmp_path, output)

    text = output.read_text()
    assert "WeatherBridge" in text
    assert "WeatherBridge" in text
    assert "Refine" not in text
    assert "WeatherDCAE-14M" in text
    assert "SwinV2" in text
    assert "\\label{tab:main-12h}" in text
    assert "standard deviation" in text
    assert "\\pm" in text
    assert "oddskip" not in text


def test_generated_query_generalization_table_separates_seen_and_held(
    tmp_path: Path,
) -> None:
    for key, (horizon, year, filename) in FILES.items():
        value = 0.1 if key.startswith("bridge") else 0.15
        path = tmp_path / f"{horizon}h_{year}" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_payload(horizon, year, value, f"idx-{horizon}")))
    output = tmp_path / "query_table.tex"

    _write_query_generalization_table(tmp_path, output)

    text = output.read_text()
    assert "6 h seen" in text
    assert "6 h held out" in text
    assert "12 h seen" in text
    assert "12 h held out" in text
    assert "WeatherBridge" in text
    assert "WeatherDCAE-14M" in text
    assert "Linear Interp." in text
    assert "\\label{tab:query-generalization}" in text
    assert "\\setlength{\\tabcolsep}{3.5pt}" in text

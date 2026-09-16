"""Export validated full-year headline metrics as LaTeX commands."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

CHANNEL_COUNT = 24
LATITUDE_GRID = "wb2_0p25_2x2_block_average_v1"
FILES = {
    "bridge6": (6, 2020, "weatherbridge_pp3_14m_6yr_ep8.json"),
    "dcae6": (6, 2020, "weatherdcae_14m_6yr_ep8_matched.json"),
    "bridge12": (12, 2020, "weatherbridge_14m_3yr_ep10.json"),
    "dcae12": (12, 2020, "weatherdcae_14m_3yr_ep10_matched.json"),
}
TABLE_12_MODELS = (
    ("WeatherBridge", 14.3, "weatherbridge_14m_3yr_ep10.json"),
    ("WeatherDCAE-14M", 14.4, "weatherdcae_14m_3yr_ep10_matched.json"),
    ("PixelAttn-VFI", 14.7, "atm_vfi_3yr_ep10_matched.json"),
    ("SwinV2", 7.9, "fuxi_3yr_ep10.json"),
    ("S-DYff", 85.5, "sdyff_3yr_ep10.json"),
    ("ModAFNO", 151.7, "modafno_3yr_ep10.json"),
)


def _load(root: Path, key: str) -> dict:
    horizon, year, filename = FILES[key]
    path = root / f"{horizon}h_{year}" / filename
    payload = json.loads(path.read_text())
    protocol = payload.get("evaluation_protocol", {})
    if payload.get("years") != [year]:
        raise ValueError(f"{path}: wrong test year")
    if float(payload.get("delta_t_hours", -1)) != float(horizon):
        raise ValueError(f"{path}: wrong horizon")
    if protocol.get("full_year") is not True:
        raise ValueError(f"{path}: refusing a non-full-year result")
    if protocol.get("latitude_grid") != LATITUDE_GRID:
        raise ValueError(f"{path}: wrong or unverified latitude grid")
    if protocol.get("eval_hours") != list(range(1, horizon)):
        raise ValueError(f"{path}: incomplete query-hour grid")
    if len(payload.get("channel_names", [])) != CHANNEL_COUNT:
        raise ValueError(f"{path}: expected {CHANNEL_COUNT} channels")
    if not protocol.get("index_sha256"):
        raise ValueError(f"{path}: missing window-index provenance")
    return payload


def _load_12h_table_inputs(root: Path) -> list[tuple[str, float, dict]]:
    loaded = []
    indices = set()
    for name, parameters, filename in TABLE_12_MODELS:
        path = root / "12h_2020" / filename
        payload = json.loads(path.read_text())
        protocol = payload.get("evaluation_protocol", {})
        if (
            payload.get("years") != [2020]
            or float(payload.get("delta_t_hours", -1)) != 12.0
            or protocol.get("full_year") is not True
            or protocol.get("latitude_grid") != LATITUDE_GRID
            or protocol.get("eval_hours") != list(range(1, 12))
            or len(payload.get("channel_names", [])) != CHANNEL_COUNT
        ):
            raise ValueError(f"{path}: invalid 12h full-year protocol")
        indices.add(protocol.get("index_sha256"))
        loaded.append((name, parameters, payload))
    if None in indices or len(indices) != 1:
        raise ValueError("12h table models use different window indices")
    return loaded


def _mean(payload: dict, method: str, taus: list[int], prefix: str) -> float:
    channels = payload["channel_names"]
    values = [
        float(payload["per_tau"][str(tau)][method][f"{prefix}_{channel}"])
        for tau in taus
        for channel in channels
    ]
    if len(values) != len(taus) * CHANNEL_COUNT or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid {prefix} values")
    return float(np.mean(values))


def _query_hour_mean_sd(
    payload: dict, method: str, taus: list[int], prefix: str
) -> tuple[float, float]:
    channels = payload["channel_names"]
    per_tau = np.asarray(
        [
            np.mean(
                [
                    float(payload["per_tau"][str(tau)][method][f"{prefix}_{channel}"])
                    for channel in channels
                ]
            )
            for tau in taus
        ],
        dtype=np.float64,
    )
    if per_tau.size != len(taus) or not np.all(np.isfinite(per_tau)):
        raise ValueError(f"invalid query-hour {prefix} values")
    return float(per_tau.mean()), float(per_tau.std(ddof=1 if per_tau.size > 1 else 0))


def _commands(root: Path) -> dict[str, str]:
    data = {key: _load(root, key) for key in FILES}
    for horizon, keys in (
        (6, ("bridge6", "dcae6")),
        (12, ("bridge12", "dcae12")),
    ):
        hashes = {
            data[key]["evaluation_protocol"]["index_sha256"] for key in keys
        }
        if len(hashes) != 1:
            raise ValueError(f"{horizon}h core models use different window indices")

    six_taus = list(range(1, 6))
    six_seen_taus = [1, 3, 5]
    six_held_taus = [2, 4]
    twelve_taus = list(range(1, 12))
    twelve_seen_taus = [1, 2, 3, 5, 7, 9, 10, 11]
    twelve_held_taus = [4, 6, 8]
    bridge6 = _mean(data["bridge6"], "model", six_taus, "rmse_norm")
    dcae6 = _mean(data["dcae6"], "model", six_taus, "rmse_norm")
    linear6 = _mean(data["bridge6"], "bilinear", six_taus, "rmse_norm")
    bridge6_seen = _mean(data["bridge6"], "model", six_seen_taus, "rmse_norm")
    bridge6_held = _mean(data["bridge6"], "model", six_held_taus, "rmse_norm")
    dcae6_held = _mean(data["dcae6"], "model", six_held_taus, "rmse_norm")
    linear6_held = _mean(
        data["bridge6"], "bilinear", six_held_taus, "rmse_norm"
    )
    bridge12 = _mean(data["bridge12"], "model", twelve_taus, "rmse_norm")
    dcae12 = _mean(data["dcae12"], "model", twelve_taus, "rmse_norm")
    linear12 = _mean(data["bridge12"], "bilinear", twelve_taus, "rmse_norm")
    bridge12_seen = _mean(
        data["bridge12"], "model", twelve_seen_taus, "rmse_norm"
    )
    bridge12_held = _mean(
        data["bridge12"], "model", twelve_held_taus, "rmse_norm"
    )
    dcae12_held = _mean(
        data["dcae12"], "model", twelve_held_taus, "rmse_norm"
    )
    linear12_held = _mean(
        data["bridge12"], "bilinear", twelve_held_taus, "rmse_norm"
    )
    bridge12_acc = _mean(data["bridge12"], "model", twelve_taus, "acc")
    bridge12_held_acc = _mean(
        data["bridge12"], "model", twelve_held_taus, "acc"
    )

    def number(value: float) -> str:
        return f"{value:.5f}"

    def gain(reference: float, value: float) -> str:
        return f"{100.0 * (reference - value) / reference:.1f}"

    return {
        "WBSixRMSE": number(bridge6),
        "WBSixDCAERMSE": number(dcae6),
        "WBSixLinearRMSE": number(linear6),
        "WBSixVsLinearGainPct": gain(linear6, bridge6),
        "WBSixSeenRMSE": number(bridge6_seen),
        "WBSixHeldRMSE": number(bridge6_held),
        "WBSixDCAEHeldRMSE": number(dcae6_held),
        "WBSixLinearHeldRMSE": number(linear6_held),
        "WBSixHeldVsDCAEGainPct": gain(dcae6_held, bridge6_held),
        "WBSixHeldVsLinearGainPct": gain(linear6_held, bridge6_held),
        "WBTwelveRMSE": number(bridge12),
        "WBTwelveDCAERMSE": number(dcae12),
        "WBTwelveLinearRMSE": number(linear12),
        "WBTwelveSeenRMSE": number(bridge12_seen),
        "WBTwelveHeldRMSE": number(bridge12_held),
        "WBTwelveDCAEHeldRMSE": number(dcae12_held),
        "WBTwelveLinearHeldRMSE": number(linear12_held),
        "WBTwelveVsLinearHeldGainPct": gain(linear12_held, bridge12_held),
        "WBTwelveHeldVsDCAEGainPct": gain(dcae12_held, bridge12_held),
        "WBTwelveVsLinearGainPct": gain(linear12, bridge12),
        "WBTwelveACC": number(bridge12_acc),
        "WBTwelveHeldACC": number(bridge12_held_acc),
    }


def _write_query_generalization_table(root: Path, path: Path) -> None:
    data = {key: _load(root, key) for key in FILES}
    groups = (
        ("6 h seen", "bridge6", [1, 3, 5]),
        ("6 h held out", "bridge6", [2, 4]),
        ("12 h seen", "bridge12", [1, 2, 3, 5, 7, 9, 10, 11]),
        ("12 h held out", "bridge12", [4, 6, 8]),
    )
    rows = (
        ("WeatherBridge", "bridge", "model"),
        ("WeatherDCAE-14M", "dcae", "model"),
        ("\\textit{Linear Interp.}", "bridge", "bilinear"),
    )

    rendered_rows: list[tuple[str, list[tuple[float, float]]]] = []
    for name, key_prefix, method in rows:
        values = []
        for _, bridge_key, taus in groups:
            horizon = "6" if bridge_key.endswith("6") else "12"
            payload = data[f"{key_prefix}{horizon}"]
            values.append(_query_hour_mean_sd(payload, method, taus, "rmse_norm"))
        rendered_rows.append((name, values))

    columns = np.asarray(
        [[mean for mean, _ in values] for _, values in rendered_rows],
        dtype=np.float64,
    )
    orders = [np.argsort(columns[:, index]).tolist() for index in range(4)]

    def formatted(
        row_index: int, column_index: int, value: tuple[float, float]
    ) -> str:
        mean, sd = value
        text = f"{mean:.3f}\\pm{sd:.3f}"
        if row_index == orders[column_index][0]:
            return f"$\\mathbf{{{text}}}$"
        if row_index == orders[column_index][1]:
            return f"\\underline{{${text}$}}"
        return f"${text}$"

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Generalisation from trained to held-out query hours on the 2020 ERA5 benchmark. Six-hour models train at $\\tau\\in\\{1,3,5\\}$ and hold out $\\{2,4\\}$; twelve-hour models hold out $\\{4,6,8\\}$. Entries are mean $\\pm$ standard deviation of the 24-channel macro RMSE across query hours. The standard deviation describes variation between hours, not sampling uncertainty over dates.}",
        "\\label{tab:query-generalization}",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & 6 h seen & 6 h held out & 12 h seen & 12 h held out \\\\",
        "\\midrule",
    ]
    for row_index, (name, values) in enumerate(rendered_rows):
        if name.startswith("\\textit"):
            lines.append("\\midrule")
        cells = [
            formatted(row_index, column_index, value)
            for column_index, value in enumerate(values)
        ]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table*}"))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def _write_12h_table(root: Path, path: Path) -> None:
    loaded = _load_12h_table_inputs(root)
    seen = [1, 2, 3, 5, 7, 9, 10, 11]
    held = [4, 6, 8]
    all_taus = list(range(1, 12))
    rows: list[tuple[str, str, list[tuple[float, float]]]] = []
    for name, parameters, payload in loaded:
        rows.append(
            (
                name,
                f"{parameters:.1f}",
                [
                    _query_hour_mean_sd(payload, "model", seen, "rmse_norm"),
                    _query_hour_mean_sd(payload, "model", held, "rmse_norm"),
                    _query_hour_mean_sd(payload, "model", all_taus, "rmse_norm"),
                ],
            )
        )
    reference = loaded[0][2]
    rows.append(
        (
            "\\textit{Linear Interp.}",
            "--",
            [
                _query_hour_mean_sd(reference, "bilinear", seen, "rmse_norm"),
                _query_hour_mean_sd(reference, "bilinear", held, "rmse_norm"),
                _query_hour_mean_sd(reference, "bilinear", all_taus, "rmse_norm"),
            ],
        )
    )
    columns = np.asarray(
        [[mean for mean, _ in values] for _, _, values in rows], dtype=np.float64
    )
    orders = [np.argsort(columns[:, index]).tolist() for index in range(3)]

    def formatted(
        row_index: int, column_index: int, value: tuple[float, float]
    ) -> str:
        mean, sd = value
        text = f"{mean:.3f}\\pm{sd:.3f}"
        if row_index == orders[column_index][0]:
            return f"$\\mathbf{{{text}}}$"
        if row_index == orders[column_index][1]:
            return f"\\underline{{${text}$}}"
        return f"${text}$"

    lines = [
        "\\begin{table}[b]",
        "\\centering",
        "\\caption{Matched-v2 12\\,h interpolation on 2020 ERA5. Entries are mean $\\pm$ standard deviation of the 24-channel macro RMSE across the indicated query hours. The standard deviation describes hour-to-hour variation, not sampling uncertainty over dates.}",
        "\\label{tab:main-12h}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & Params (M) & Seen & Held out & All hours \\\\",
        "\\midrule",
    ]
    for row_index, (name, parameters, values) in enumerate(rows):
        rendered = [formatted(row_index, index, value) for index, value in enumerate(values)]
        if name.startswith("\\textit"):
            lines.append("\\midrule")
        lines.append(f"{name} & {parameters} & " + " & ".join(rendered) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table}"))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=Path("metrics/journal_unified"))
    parser.add_argument("--out-tex", type=Path, default=Path("paper/main_metrics_v2.tex"))
    parser.add_argument("--out-table", type=Path, default=Path("paper/tab_main_12h_v2.tex"))
    parser.add_argument(
        "--out-query-table",
        type=Path,
        default=Path("paper/tab_query_generalization.tex"),
    )
    args = parser.parse_args()
    commands = _commands(args.metrics_root)
    lines = ["% Generated from validated matched-v2 full-year metrics."]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in commands.items())
    temporary = args.out_tex.with_suffix(args.out_tex.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, args.out_tex)
    _write_12h_table(args.metrics_root, args.out_table)
    _write_query_generalization_table(args.metrics_root, args.out_query_table)
    print(f"wrote {args.out_tex}")
    print(f"wrote {args.out_table}")
    print(f"wrote {args.out_query_table}")


if __name__ == "__main__":
    main()

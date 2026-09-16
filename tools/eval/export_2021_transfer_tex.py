#!/usr/bin/env python3
"""Export the matched-index 2021 calendar-transfer table."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

CHANNEL_COUNT = 24
EXPECTED_INDEX = "eeaeed65b31e15f60355eb0fa5bf9f172d9619147447ad67d728c8eceab31f55"
LATITUDE_GRID = "wb2_0p25_2x2_block_average_v1"
MODELS = (
    ("WeatherBridge", "weatherbridge_pp3_14m_6yr_ep8.json", False),
    ("WeatherDCAE-14M", "weatherdcae_14m_6yr_ep8_matched.json", False),
    ("PixelAttn-VFI", "atm_vfi_6yr_ep8_matched.json", True),
    ("SwinV2", "fuxi_24ch_6yr_ep8.json", True),
    ("S-DYff", "sdyff_24ch_6yr_ep8.json", True),
    ("ModAFNO", "modafno_24ch_6yr_ep8.json", True),
)


def load(path: Path, require_dataset_provenance: bool) -> dict:
    payload = json.loads(path.read_text())
    protocol = payload.get("evaluation_protocol", {})
    if (
        payload.get("years") != [2021]
        or float(payload.get("delta_t_hours", -1)) != 6.0
        or int(payload.get("num_samples", -1)) != 7295
        or len(payload.get("channel_names", [])) != CHANNEL_COUNT
        or protocol.get("full_year") is not True
        or protocol.get("eval_hours") != [1, 2, 3, 4, 5]
        or protocol.get("index_sha256") != EXPECTED_INDEX
    ):
        raise ValueError(f"{path}: invalid 2021 matched-index protocol")
    if require_dataset_provenance:
        if protocol.get("latitude_grid") != LATITUDE_GRID:
            raise ValueError(f"{path}: missing corrected latitude-grid protocol")
        provenance = payload.get("evaluation_dataset_provenance", {})
        if not provenance.get("identity_sha256"):
            raise ValueError(f"{path}: missing dataset identity provenance")
    return payload


def mean(payload: dict, method: str, taus: list[int], prefix: str) -> float:
    values = [
        float(payload["per_tau"][str(tau)][method][f"{prefix}_{channel}"])
        for tau in taus
        for channel in payload["channel_names"]
    ]
    if len(values) != len(taus) * CHANNEL_COUNT or not all(
        math.isfinite(value) for value in values
    ):
        raise ValueError(f"invalid {prefix} values")
    return float(np.mean(values))


def query_hour_mean_sd(
    payload: dict, method: str, taus: list[int], prefix: str
) -> tuple[float, float]:
    per_tau = np.asarray(
        [
            np.mean(
                [
                    float(payload["per_tau"][str(tau)][method][f"{prefix}_{channel}"])
                    for channel in payload["channel_names"]
                ]
            )
            for tau in taus
        ],
        dtype=np.float64,
    )
    if per_tau.size != len(taus) or not np.all(np.isfinite(per_tau)):
        raise ValueError(f"invalid query-hour {prefix} values")
    return float(per_tau.mean()), float(per_tau.std(ddof=1 if per_tau.size > 1 else 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir", type=Path, default=Path("metrics/journal_unified/6h_2021")
    )
    parser.add_argument(
        "--out-tex", type=Path, default=Path("paper/tab_6h_2021_transfer.tex")
    )
    args = parser.parse_args()

    rows = []
    loaded = {}
    for label, filename, legacy in MODELS:
        payload = load(
            args.metrics_dir / filename,
            require_dataset_provenance=not legacy,
        )
        loaded[label] = payload
        rows.append(
            (
                label + (r"$^{\dagger}$" if legacy else ""),
                [
                    query_hour_mean_sd(payload, "model", [1, 3, 5], "rmse_norm"),
                    query_hour_mean_sd(payload, "model", [2, 4], "rmse_norm"),
                    query_hour_mean_sd(payload, "model", [1, 2, 3, 4, 5], "rmse_norm"),
                    query_hour_mean_sd(payload, "model", [1, 2, 3, 4, 5], "acc"),
                ],
            )
        )
    reference = loaded["WeatherBridge"]
    rows.append(
        (
            "Linear Interp.",
            [
                query_hour_mean_sd(reference, "bilinear", [1, 3, 5], "rmse_norm"),
                query_hour_mean_sd(reference, "bilinear", [2, 4], "rmse_norm"),
                query_hour_mean_sd(reference, "bilinear", [1, 2, 3, 4, 5], "rmse_norm"),
                query_hour_mean_sd(reference, "bilinear", [1, 2, 3, 4, 5], "acc"),
            ],
        )
    )

    metric_values = np.asarray([[mean for mean, _ in values] for _, values in rows])
    metric_sds = np.asarray([[sd for _, sd in values] for _, values in rows])
    rmse_orders = [np.argsort(metric_values[:, column]) for column in range(3)]
    acc_order = np.argsort(-metric_values[:, 3])

    def render(row: int, column: int) -> str:
        text = f"{metric_values[row, column]:.3f}\\pm{metric_sds[row, column]:.3f}"
        order = rmse_orders[column] if column < 3 else acc_order
        if row == order[0]:
            return rf"$\mathbf{{{text}}}$"
        if row == order[1]:
            return rf"\underline{{${text}$}}"
        return rf"${text}$"

    lines = [
        "% Auto-generated by tools/eval/export_2021_transfer_tex.py",
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Six-hour calendar transfer on the matched full-year 2021 "
            r"index (7,295 window--hour samples). Seen hours are $\tau\in\{1,3,5\}$ "
            r"and held-out hours are $\tau\in\{2,4\}$. Entries are mean $\pm$ "
            r"standard deviation of 24-channel macro scores across the indicated "
            r"query hours; this is descriptive hour-to-hour variation, not "
            r"sampling uncertainty over dates. Lower RMSE and higher ACC "
            r"are better. $\dagger$ marks legacy evaluator records that share the "
            r"frozen index but predate corrected-grid and dataset-identity provenance. "
            r"They are kept for context and do not meet the current provenance standard.}"
        ),
        r"\label{tab:transfer-2021}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Seen RMSE & Held-out RMSE & Avg. RMSE & ACC \\",
        r"\midrule",
    ]
    for row, (label, _) in enumerate(rows):
        if label == "Linear Interp.":
            lines.append(r"\midrule")
        lines.append(
            f"{label} & "
            + " & ".join(render(row, column) for column in range(4))
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}"))
    temporary = args.out_tex.with_suffix(args.out_tex.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, args.out_tex)
    print(f"wrote {args.out_tex}")


if __name__ == "__main__":
    main()

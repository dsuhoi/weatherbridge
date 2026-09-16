#!/usr/bin/env python3
"""Render the frozen 2022 holdout assessment as manuscript LaTeX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_NAMES = {
    "weatherbridge_detail": "Detail-bypass ablation",
    "refine": "WeatherBridge-Refine",
    "flow_spectral": "WeatherBridge",
    "weatherdcae_14m": "WeatherDCAE-14M",
}


def render(payload: dict[str, Any]) -> str:
    winner = str(payload["frozen_winner"])
    evaluated = str(payload.get("evaluated_candidate", winner))
    display = MODEL_NAMES[evaluated]
    selector_confirmation = bool(payload.get("selector_confirmation", evaluated == winner))
    lines = []
    lines.append(
        "The post-selection protocol fixed the 48-window 2022 index and aggregate "
        "endpoint before the "
        "final selector was executed. Predictions and target-derived metrics "
        "from this sample were not selector inputs; inference began only after "
        "selection completed. The sample contains 48 disjoint 00 UTC windows on "
        "days 1, 8, 15 and 22 of each month."
    )
    if not payload.get("primary_endpoint_evaluated", bool(payload.get("horizons"))):
        if payload["status"] not in {
            "reference_retained",
            "reference_retained_holdout_unopened",
        }:
            raise ValueError(
                "assessment has neither an evaluated endpoint nor reference "
                "retention"
            )
        lines.append(
            f"{MODEL_NAMES[winner]} remained the selected reference after the "
            "retrospective "
            "gates. Because there was no selected challenger, the declared "
            "candidate-versus-reference endpoint was not evaluated and supplies "
            "no numerical confirmation result."
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Pre-specified 2022 comparison against WeatherDCAE-14M. "
            r"Values are paired differences in mean normalised RMSE; negative "
            r"values favour the evaluated candidate. Confidence intervals use "
            r"10,000 "
            r"seven-day block-bootstrap draws.}",
            r"\label{tab:postselection-2022}",
            r"\small",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Horizon & $\Delta$RMSE & Relative $\Delta$ (\%) & 95\% CI \\",
            r"\midrule",
        ]
    )
    for horizon in ("6h", "12h"):
        row = payload["horizons"][horizon]
        ci = row["delta_ci95"]
        lines.append(
            f"{horizon[:-1]} h & {row['delta_normalized_rmse']:.5f} & "
            f"{row['relative_delta_pct']:.2f} & "
            f"[{ci[0]:.5f}, {ci[2]:.5f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    if payload["aggregate_confirmation_pass"]:
        if selector_confirmation:
            lines.append(
                f"{display} has an upper confidence bound below zero at both "
                "horizons, confirming the frozen aggregate endpoint."
            )
        else:
            lines.append(
                f"{display} has an upper confidence bound below zero at both "
                "horizons on the frozen sample. This confirms its aggregate "
                "accuracy advantage without reopening the earlier multi-metric "
                f"selector, which retained {MODEL_NAMES[winner]}."
            )
    else:
        lines.append(
            f"{display} does not pass the frozen aggregate endpoint at "
            "both horizons; the retrospective advantage therefore does not "
            "generalise as a confirmatory claim."
        )
    if payload["universal_dominance_claim_allowed"]:
        lines.append(
            "Every RMSE, hard-window, ACC, spectral, temporal-curvature and "
            "physical diagnostic cell also improves pointwise. This is a "
            "descriptive strict-dominance result, not a simultaneous "
            "population-level proof."
        )
    else:
        lines.append(
            "At least one declared diagnostic cell does not improve "
            "pointwise; we therefore make no claim of universal diagnostic "
            "dominance."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--out-tex", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.assessment.read_text())
    output = render(payload)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(output)
    print(f"wrote {args.out_tex}")


if __name__ == "__main__":
    main()

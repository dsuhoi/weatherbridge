"""Export every S-DYff-ENS field/hour and paired aggregate uncertainty."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.eval.paired_block_bootstrap import load_metrics, paired_block_bootstrap, sha256_file


def summarize(path: Path, reference_path: Path, draws: int = 5000):
    data = json.loads(path.read_text())
    reference = json.loads(reference_path.read_text())
    for key in ("channel_names", "years", "n_per_tau", "seen_tau", "unseen_tau", "delta_t_hours"):
        if data[key] != reference[key]:
            raise ValueError(f"unmatched reference {key}")
    index = data.get("evaluation_protocol", {}).get("index_sha256")
    if not index or index != reference.get("evaluation_protocol", {}).get("index_sha256"):
        raise ValueError("unmatched reference window index")
    if data["checkpoint_provenance"]["sha256"] != reference["checkpoint_provenance"]["sha256"]:
        raise ValueError("checkpoint differs from the manuscript")
    for tau, values in data["per_tau"].items():
        for key, value in values["bilinear"].items():
            if not np.isclose(value, reference["per_tau"][tau]["bilinear"][key], rtol=2e-5, atol=2e-7):
                raise ValueError(f"linear reference mismatch: {tau} {key}")
    window = path.parent / data["window_metrics_file"]
    if sha256_file(window) != data["window_metrics_provenance"]["sha256"]:
        raise ValueError("window artifact checksum mismatch")
    single = load_metrics(window, "ens1")
    if single.index_sha256 != data["evaluation_protocol"]["index_sha256"]:
        raise ValueError("window index checksum mismatch")
    rows = []
    tests = {}
    methods = data["ensemble_protocol"]["method_sizes"]
    for method, size in sorted(methods.items(), key=lambda item: item[1]):
        for tau, values in data["per_tau"].items():
            for field in data["channel_names"]:
                score = values[method]
                rmse = score[f"rmse_norm_{field}"]
                control = values["ens1"][f"rmse_norm_{field}"]
                linear = values["bilinear"][f"rmse_norm_{field}"]
                rows.append({
                    "year": data["years"][0], "horizon": data["delta_t_hours"],
                    "hour": int(tau), "field": field, "members": size,
                    "held_out": int(tau) in data["unseen_tau"],
                    "rmse_norm": rmse, "rmse_phys": score[f"rmse_phys_{field}"],
                    "acc": score[f"acc_{field}"],
                    "gain_vs_single_pct": 100 * (1 - rmse / control),
                    "gain_vs_linear_pct": 100 * (1 - rmse / linear),
                    "acc_delta_vs_single": score[f"acc_{field}"] - values["ens1"][f"acc_{field}"],
                })
        if size == 1:
            continue
        current = load_metrics(window, method)
        tests[str(size)] = {}
        for group, taus in {"all": list(map(int, data["per_tau"])),
                            "trained": data["seen_tau"], "held_out": data["unseen_tau"]}.items():
            tests[str(size)][group] = paired_block_bootstrap(
                current, single, taus=np.asarray(taus), block_days=7,
                draws=draws, seed=20260914,
                include_cellwise=(size == max(methods.values()) and group == "all"),
            )
    with path.with_name("field_hour_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_sha256": sha256_file(path), "reference_sha256": sha256_file(reference_path),
        "ensemble_protocol": data["ensemble_protocol"], "paired_vs_single": tests,
        "interval_scope": "evaluation-date variability conditional on one sampling seed",
        "cellwise_family": "largest ensemble versus single draw; all fields and hours within this cohort",
    }
    path.with_name("ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    return rows, summary


def plot_fields(path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(path.read_text())
    hours = sorted(map(int, data["per_tau"]))
    fields = data["channel_names"]
    size = max(data["ensemble_protocol"]["sizes"])
    curves = [("bilinear", "Linear", "#6B7280", "--"),
              ("ens1", "S-DYff, paired N=1", "#9467BD", "--"),
              ("model", f"S-DYff-ENS, N={size}", "#145A52", "-")]
    for metric, label in [("rmse_norm", "Normalised RMSE"), ("acc", "ACC")]:
        fig, axes = plt.subplots(6, 4, figsize=(7.2, 9.0), layout="constrained")
        for ax, field in zip(axes.flat, fields, strict=True):
            for hour in set(data["unseen_tau"]) & set(hours):
                ax.axvspan(hour - 0.15, hour + 0.15, color="#F6DFE8", zorder=0)
            for method, name, color, style in curves:
                values = [data["per_tau"][str(h)][method][f"{metric}_{field}"] for h in hours]
                ax.plot(hours, values, label=name, color=color, linestyle=style,
                        linewidth=1.1, marker="o", markersize=2)
            ax.set_title(field, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_xticks(hours[::2] if len(hours) > 5 else hours)
            ax.grid(alpha=0.2)
        for ax in axes[-1]:
            ax.set_xlabel("Query hour", fontsize=8)
        fig.supylabel(label, fontsize=9)
        fig.suptitle(f"{data['years'][0]} | {int(data['delta_t_hours'])} h interpolation", fontsize=11)
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center",
                   ncol=3, frameon=False, fontsize=8)
        stem = path.with_name(f"sdyff_ens_{metric}_fields")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input, args.reference)
    plot_fields(args.input)

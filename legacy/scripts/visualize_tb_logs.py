
import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator


sns.set_style("whitegrid")
sns.set_context("talk")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11


def find_event_files(log_dir: Path) -> List[Path]:
    event_files = sorted(log_dir.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {log_dir}")
    return event_files


def load_scalars_from_event_file(event_file: Path) -> pd.DataFrame:
    accumulator = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()

    rows = []
    scalar_tags = accumulator.Tags().get("scalars", [])
    for tag in scalar_tags:
        for event in accumulator.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": event.step,
                    "value": event.value,
                    "wall_time": event.wall_time,
                    "source_file": str(event_file),
                }
            )
    return pd.DataFrame(rows)


def parse_tb_logs(log_dir: str) -> pd.DataFrame:
    log_path = Path(log_dir)
    event_files = find_event_files(log_path)
    print(f"Found {len(event_files)} event file(s)")

    frames = []
    for event_file in event_files:
        df = load_scalars_from_event_file(event_file)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["tag", "step", "value", "wall_time", "source_file"])

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["tag", "step", "wall_time"]).drop_duplicates(
        subset=["tag", "step"], keep="last"
    )

    data["phase"] = data["tag"].apply(infer_phase)
    data["metric"] = data["tag"].apply(metric_name_from_tag)
    return data.reset_index(drop=True)


def infer_phase(tag: str) -> str:
    if tag.startswith("train/"):
        return "train"
    if tag.startswith("val/"):
        return "val"
    if tag.startswith("test/"):
        return "test"
    return "other"


def metric_name_from_tag(tag: str) -> str:
    if "/" in tag:
        return tag.split("/", 1)[1]
    return tag


def safe_slug(tag: str) -> str:
    return (
        tag.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def plot_overview(data: pd.DataFrame, output_dir: Path) -> None:
    priority_tags = [
        "train/loss",
        "val/loss",
        "train/loss_recon",
        "val/loss_recon",
        "train/loss_latent",
        "val/loss_latent",
        "train/rmse",
        "val/rmse",
        "train/mae",
        "val/mae",
        "train/bias",
        "val/bias",
        "train/psnr",
        "val/psnr",
        "train/silhouette",
        "val/silhouette",
        "lr-AdamW",
    ]
    available = [tag for tag in priority_tags if tag in set(data["tag"])]
    if not available:
        return

    ncols = 2
    nrows = math.ceil(len(available) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(6, 4 * nrows)))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, tag in zip(axes, available):
        subset = data[data["tag"] == tag]
        ax.plot(subset["step"], subset["value"], linewidth=2)
        ax.set_title(tag)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        if tag.lower().startswith("lr"):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(available):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "overview.png", dpi=250, bbox_inches="tight")
    plt.close()


def plot_phase_comparisons(data: pd.DataFrame, output_dir: Path) -> None:
    metrics = sorted(
        metric
        for metric in data["metric"].unique()
        if any(data["tag"] == f"train/{metric}") or any(data["tag"] == f"val/{metric}")
    )

    for metric in metrics:
        subset = data[
            data["tag"].isin([f"train/{metric}", f"val/{metric}", f"test/{metric}"])
        ]
        if subset.empty:
            continue

        plt.figure(figsize=(12, 7))
        for phase in ["train", "val", "test"]:
            phase_df = subset[subset["phase"] == phase]
            if phase_df.empty:
                continue
            plt.plot(phase_df["step"], phase_df["value"], label=phase, linewidth=2)
        plt.title(metric)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"metric_{safe_slug(metric)}.png", dpi=250, bbox_inches="tight")
        plt.close()


def plot_per_variable_groups(data: pd.DataFrame, output_dir: Path) -> None:
    grouped_metrics = [prefix for prefix in ["rmse_", "mae_", "bias_", "silhouette_"]]
    for prefix in grouped_metrics:
        metrics = sorted(m for m in data["metric"].unique() if m.startswith(prefix))
        if not metrics:
            continue

        ncols = 2
        nrows = math.ceil(len(metrics) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(6, 4 * nrows)))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax, metric in zip(axes, metrics):
            subset = data[data["metric"] == metric]
            for phase in ["train", "val", "test"]:
                phase_df = subset[subset["phase"] == phase]
                if phase_df.empty:
                    continue
                ax.plot(phase_df["step"], phase_df["value"], label=phase, linewidth=2)
            ax.set_title(metric)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend()

        for ax in axes[len(metrics):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / f"group_{safe_slug(prefix)}.png", dpi=250, bbox_inches="tight")
        plt.close()


def save_metric_table(data: pd.DataFrame, output_dir: Path) -> None:
    summary_rows = []
    for tag, subset in data.groupby("tag"):
        subset = subset.sort_values("step")
        summary_rows.append(
            {
                "tag": tag,
                "phase": subset["phase"].iloc[-1],
                "last_step": int(subset["step"].iloc[-1]),
                "last_value": float(subset["value"].iloc[-1]),
                "min_value": float(subset["value"].min()),
                "max_value": float(subset["value"].max()),
                "num_points": int(len(subset)),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["phase", "tag"])
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize TensorBoard scalar logs")
    parser.add_argument("--log-dir", type=str, required=True, help="Training run directory")
    parser.add_argument(
        "--output",
        type=str,
        default="plots/training_plots",
        help="Directory where plots and CSV summary will be saved",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing TensorBoard logs from: {args.log_dir}")
    data = parse_tb_logs(args.log_dir)

    if data.empty:
        print("No scalar data found in TensorBoard events.")
        return

    data.to_csv(output_dir / "all_scalars.csv", index=False)
    print(f"Parsed {len(data)} scalar points across {data['tag'].nunique()} tags")

    plot_overview(data, output_dir)
    plot_phase_comparisons(data, output_dir)
    plot_per_variable_groups(data, output_dir)
    save_metric_table(data, output_dir)

    print(f"Saved plots and tables to: {output_dir}")


if __name__ == "__main__":
    main()

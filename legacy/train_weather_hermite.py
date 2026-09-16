"""
CLI training entrypoint for WeatherHermiteModel.

Examples:
python legacy/train_weather_hermite.py --preset stable --years 2016 2017 2018 2019
python legacy/train_weather_hermite.py --preset quality --run-name weather_hermite_quality_run1
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from weather_time_interp.config import DEFAULT_PRESSURE_LEVELS, DEFAULT_VARIABLES
from trainer_weather_hermite import (
    WeatherHermiteDataModule,
    WeatherHermiteLightningModule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train WeatherHermiteModel with dedicated datamodule and CLI"
    )

    parser.add_argument("--preset", type=str, default="stable", choices=["fast", "stable", "quality"])
    parser.add_argument(
        "--model-type",
        type=str,
        default="hermite",
        choices=["hermite", "unet_residual_linear", "unet_direct", "koopman_linear",
                 "dcae_residual_linear", "dcae_adaln_residual_linear",
                 "dcae_adaln_skip_residual_linear",
                 "dcae_swin_skip_residual_linear",
                 "dcae_bilinear_xattn_residual_linear",
                 "sfno_residual_linear", "sdyff_residual_linear",
                 "sdyff_dyffusion_residual_linear",
                 "fuxi_swinv2_residual_linear",
                 "true_unet_residual_linear", "modafno_residual_linear",
                 "modafno_official_residual_linear",
                 "afno_official_residual_linear",
                 "fm_residual_linear",
                 "channels_residual_linear"],
    )

    parser.add_argument("--data-dir", type=str, default="~/data/weather_data/time_interpolation/")
    parser.add_argument("--years", type=int, nargs="+", default=[2016, 2017, 2018, 2019, 2020])
    parser.add_argument("--train-years", type=int, nargs="+", default=None)
    parser.add_argument("--test-years", type=int, nargs="+", default=None)
    parser.add_argument("--variables", type=str, nargs="+", default=DEFAULT_VARIABLES)
    parser.add_argument(
        "--pressure-levels",
        type=int,
        nargs="+",
        default=DEFAULT_PRESSURE_LEVELS,
        help="Only first level will be used to keep exactly 5 channels.",
    )
    parser.add_argument("--stats-path", type=str, default="data/json_stats.nc")
    parser.add_argument("--static-path", type=str, default=None,
                        help="Path to static features (.pt or .nc). E.g. data/static_features.pt")
    parser.add_argument("--n-static-features", type=int, default=0,
                        help="Number of static feature channels (must match static_path tensor channel-0 size).")
    parser.add_argument("--modafno-depth", type=int, default=8,
                        help="ModAFNO num blocks (paper: 12). Affects modafno_*_residual_linear only.")
    parser.add_argument("--modafno-num-blocks", type=int, default=8,
                        help="ModAFNO num_blocks for block-diagonal spectral weights (paper: 12, embed must be divisible).")
    parser.add_argument("--modafno-drop-rate", type=float, default=0.0,
                        help="Dropout rate inside ModAFNO MLP/pos_drop. Set >0 (e.g. 0.1) for MC-dropout ensemble support.")
    parser.add_argument("--sdyff-num-layers", type=int, default=4,
                        help="S-DYff: number of SDyffBlock layers (paper Cachay 2024 uses 8).")
    parser.add_argument("--sdyff-n-modes-lat", type=int, default=16,
                        help="S-DYff: number of spectral modes in latitude.")
    parser.add_argument("--sdyff-n-modes-lon", type=int, default=32,
                        help="S-DYff: number of spectral modes in longitude.")
    parser.add_argument("--sdyff-dropout", type=float, default=0.1,
                        help="S-DYff: dropout rate inside MLP/SDyffBlock (active during eval for MC-dropout).")
    parser.add_argument("--sdyff-drop-path", type=float, default=0.1,
                        help="S-DYff: stochastic depth (drop path) rate, applied in train and eval.")
    parser.add_argument("--sdyff-inference-steps", type=int, default=5,
                        help="S-DYff DYffusion: K refiner iterations at inference.")
    parser.add_argument("--sdyff-train-refine-steps", type=int, default=1,
                        help="S-DYff DYffusion: refiner iterations during training (usually 1).")
    parser.add_argument("--fuxi-depth", type=int, default=8,
                        help="FuXi SwinV2: number of transformer blocks.")
    parser.add_argument("--fuxi-num-heads", type=int, default=8,
                        help="FuXi SwinV2: number of attention heads per block.")
    parser.add_argument("--fuxi-window-size-h", type=int, default=5,
                        help="FuXi SwinV2: window size in latitude (45 / 5 = 9 windows).")
    parser.add_argument("--fuxi-window-size-w", type=int, default=9,
                        help="FuXi SwinV2: window size in longitude (90 / 9 = 10 windows).")
    parser.add_argument("--fuxi-patch-size", type=int, default=4,
                        help="FuXi SwinV2: patch_embed kernel/stride (180 / 4 = 45 lat tokens).")
    parser.add_argument("--fuxi-drop-path", type=float, default=0.1,
                        help="FuXi SwinV2: stochastic depth per block (linear schedule).")
    parser.add_argument("--fm-ode-steps", type=int, default=1,
                        help="FM: number of ODE refinement steps (1 = single-step closed-form, 4-8 = multi-step Euler).")
    parser.add_argument("--surface-data-dir", type=str, default=None,
                        help="Directory with surface_<year>.zarr files. Adds time-varying surface vars.")
    parser.add_argument("--surface-variables", type=str, nargs="*", default=None,
                        help="Surface variable short names (t2m, u10, v10, mslp, tisr, tcwv).")
    parser.add_argument("--surface-stats-path", type=str, default=None,
                        help="JSON with mean/std for surface variables.")
    parser.add_argument("--analytic-tisr", action="store_true", default=False,
                        help="Compute TISR analytically (Spencer 1971) instead of reading from zarr surface store. "
                             "Saves ~10 GiB/year RAM cache and gives exact astronomical values.")
    parser.add_argument("--samples-per-date", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--delta-t-hours", type=float, default=6.0)
    parser.add_argument("--train-hours", type=int, nargs="+", default=None)
    parser.add_argument("--eval-hours", type=int, nargs="+", default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=None,
        help="Gradient clipping value.",
    )
    parser.add_argument(
        "--gradient-clip-algorithm",
        type=str,
        default="norm",
        choices=["norm", "value"],
        help="Gradient clipping algorithm.",
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)

    parser.add_argument(
        "--latent-channels",
        type=int,
        default=None,
        help="Number of channels in latent space.",
    )
    parser.add_argument("--cond-dim", type=int, default=1)

    parser.add_argument(
        "--block-out-channels",
        type=int,
        nargs="+",
        default=None,
        help="Encoder/decoder channels for each resolution stage.",
    )
    parser.add_argument(
        "--layers-per-block",
        type=int,
        nargs="+",
        default=None,
        help="Number of ResBlocks at each stage.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=None,
        help="Hidden channels in the Hermite parameter network.",
    )
    parser.add_argument(
        "--cond-mlp-dim",
        type=int,
        default=None,
        help="Hidden size of the conditioning MLP.",
    )

    parser.add_argument("--lat-crop", type=int, default=1)
    parser.add_argument("--residual-scale-init", type=float, default=0.10)
    parser.add_argument("--residual-clip", type=float, default=3.0)
    parser.add_argument("--residual-scale-learnable", action="store_true", default=True)
    parser.add_argument("--fixed-residual-scale", action="store_false", dest="residual_scale_learnable")

    parser.add_argument("--recon-loss", type=str, default="smooth_l1", choices=["smooth_l1", "mse"])
    parser.add_argument("--smooth-l1-beta", type=float, default=0.02)
    parser.add_argument(
        "--use-physical-scales-loss",
        action="store_true",
        default=True,
        help="Scale per-channel reconstruction loss by physical units (T,U,V,Q,Z).",
    )
    parser.add_argument(
        "--no-physical-scales-loss",
        action="store_false",
        dest="use_physical_scales_loss",
        help="Disable physical-unit scaling in reconstruction loss.",
    )
    parser.add_argument(
        "--use-aurora-weights",
        action="store_true",
        default=False,
        help="Apply Aurora-style per-channel/per-level loss weights (on normalized space). "
             "Recommended together with --no-physical-scales-loss.",
    )
    parser.add_argument(
        "--physical-loss-scales",
        type=float,
        nargs=5,
        metavar=("T", "U", "V", "Q", "Z"),
        default=[1.0, 2.0, 2.0, 1e-3, 98.1],
        help="Physical scales for (T[K], U[m/s], V[m/s], Q[kg/kg], Z[m^2/s^2]).",
    )
    parser.add_argument(
        "--hour-loss-weights",
        type=str,
        nargs="*",
        default=None,
        metavar="HOUR:W",
        help=(
            "Optional per-hour train loss weights (e.g. 1:2.0 3:1.0 5:1.0). "
            "Hours not listed use weight=1.0."
        ),
    )
    parser.add_argument(
        "--channel-loss-weights",
        type=str,
        nargs="*",
        default=None,
        metavar="CHAN:W",
        help=(
            "Optional per-variable train loss weights. "
            "Supported names: temperature|u_component_of_wind|v_component_of_wind|"
            "specific_humidity|geopotential or aliases t|u|v|q|z."
        ),
    )
    parser.add_argument("--lambda-anchor", type=float, default=0.0)
    parser.add_argument("--anchor-loss", type=str, default="smooth_l1", choices=["smooth_l1", "mse", "l1"])
    parser.add_argument(
        "--anchor-every-n-batches",
        type=int,
        default=1,
        help="Apply anchor loss every N train batches (1 = every batch).",
    )
    parser.add_argument("--lambda-tau-smooth", type=float, default=0.0)
    parser.add_argument("--tau-smooth-delta", type=float, default=1.0 / 12.0)
    parser.add_argument("--tau-smooth-loss", type=str, default="l1", choices=["l1", "mse", "smooth_l1"])
    parser.add_argument("--lambda-residual", type=float, default=0.0,
                        help="Direct supervision on decoder output to predict (target - bilinear).")
    parser.add_argument("--residual-loss", type=str, default="smooth_l1",
                        choices=["smooth_l1", "mse", "l1"])
    parser.add_argument("--residual-scale-floor", type=float, default=0.0,
                        help="Minimum absolute value for residual_scale. Prevents collapse to 0.")
    parser.add_argument("--direct-prediction", action="store_true", default=False,
                        help="If set, model directly predicts target (no bilinear baseline).")
    parser.add_argument("--lat-weighted-loss", action="store_true", default=False,
                        help="Apply cos(lat) area weighting in training loss (standard for global ERA5).")
    parser.add_argument("--multi-level", action="store_true", default=False,
                        help="Use all pressure levels (5 vars × N levels = 5N PL channels).")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--lambda-latent", type=float, default=0.0)
    parser.add_argument("--lambda-reg-d", type=float, default=0.0)
    parser.add_argument("--use-latent-loss", action="store_true", default=False)
    parser.add_argument("--no-latent-loss", action="store_false", dest="use_latent_loss")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-grad-norm-every", type=int, default=200)

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--resume-from-last",
        action="store_true",
        default=False,
        help="Resume from <log-dir>/<run-name>/last.ckpt when it exists.",
    )
    parser.add_argument(
        "--resume-ckpt",
        type=str,
        default=None,
        help="Explicit checkpoint path to resume from (overrides --resume-from-last).",
    )
    parser.add_argument(
        "--sanity-val-steps",
        type=int,
        default=0,
        help="Short validation warm-start before epoch 1 (set >0 to enable).",
    )
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-every-train-steps", type=int, default=None)
    parser.add_argument(
        "--grid-aware-arch",
        action="store_true",
        default=False,
        help="If set, adapt model depth to grid width to keep even longitudes after downsampling.",
    )
    parser.add_argument(
        "--cache-in-ram",
        action="store_true",
        default=False,
        help="Pre-load all selected zarr channels into a contiguous numpy array per year. "
             "Required for sane DataLoader speed when reading from s3fs.",
    )
    parser.add_argument("--limit-train-batches", type=float, default=None,
                        help="Cap train batches per epoch (Lightning limit_train_batches).")
    parser.add_argument("--limit-val-batches", type=float, default=None,
                        help="Cap val batches per epoch (Lightning limit_val_batches).")
    return parser.parse_args()


def _normalize_variables(variables: List[str]) -> List[str]:
    norm = [v.upper() for v in variables]
    if len(norm) != 5:
        raise ValueError(
            f"WeatherHermiteModel requires exactly 5 variables, got {len(norm)}: {norm}"
        )
    return norm

def _parse_hour_loss_weights(values: List[str] | None) -> Dict[int, float]:
    if not values:
        return {}
    out: Dict[int, float] = {}
    for item in values:
        if ":" not in item:
            raise ValueError(f"Invalid --hour-loss-weights entry '{item}', expected HOUR:WEIGHT")
        hour_s, weight_s = item.split(":", 1)
        hour = int(hour_s.strip())
        weight = float(weight_s.strip())
        if weight <= 0.0:
            raise ValueError(f"Hour weight must be > 0, got {item}")
        out[hour] = weight
    return out

def _parse_channel_loss_weights(values: List[str] | None) -> Dict[str, float]:
    if not values:
        return {}
    out: Dict[str, float] = {}
    alias = {
        "t": "temperature",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
        "q": "specific_humidity",
        "z": "geopotential",
    }
    for item in values:
        if ":" not in item:
            raise ValueError(f"Invalid --channel-loss-weights entry '{item}', expected NAME:WEIGHT")
        name_s, weight_s = item.split(":", 1)
        name = name_s.strip().lower()
        name = alias.get(name, name)
        weight = float(weight_s.strip())
        if weight <= 0.0:
            raise ValueError(f"Channel weight must be > 0, got {item}")
        out[name] = weight
    return out


def _apply_preset(args: argparse.Namespace) -> None:
    presets = {
        "fast": {
            "epochs": 12,
            "batch_size": 16,
            "lr": 2e-4,
            "weight_decay": 1e-5,
            "gradient_clip_val": 0.8,
            "latent_channels": 96,
            "block_out_channels": [128, 128, 192, 192],
            "layers_per_block": [2, 2, 2, 2],
            "hidden_channels": 128,
            "cond_mlp_dim": 32,
            "precision": "16-mixed",
        },
        "stable": {
            "epochs": 24,
            "batch_size": 12,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "gradient_clip_val": 0.6,
            "latent_channels": 128,
            "block_out_channels": [128, 128, 256, 256],
            "layers_per_block": [3, 3, 3, 3],
            "hidden_channels": 192,
            "cond_mlp_dim": 48,
            "precision": "16-mixed",
        },
        "quality": {
            "epochs": 36,
            "batch_size": 8,
            "lr": 7e-5,
            "weight_decay": 1e-5,
            "gradient_clip_val": 0.5,
            "latent_channels": 256,
            "block_out_channels": [128, 128, 256, 256],
            "layers_per_block": [4, 4, 4, 4],
            "hidden_channels": 256,
            "cond_mlp_dim": 64,
            "precision": "16-mixed",
        },
    }

    cfg = presets[args.preset]
    for key, value in cfg.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def _downsample_size(size: int) -> int:
    # Conv with kernel=3, stride=2, padding=1 in current encoder down blocks.
    return (size + 1) // 2


def _max_safe_downsamples_for_width(width: int, requested: int) -> int:
    """
    Keep width even before every SphereConv call.
    After each downsample, encoder still applies conv blocks, so the resulting width
    must stay even as well.
    """
    safe = 0
    cur = int(width)
    for _ in range(max(0, requested)):
        cur = _downsample_size(cur)
        if cur % 2 != 0:
            break
        safe += 1
    return safe


def _adapt_architecture_for_grid(
    block_out_channels: List[int],
    layers_per_block: List[int],
    width: int,
) -> Tuple[List[int], List[int], int, int]:
    if len(block_out_channels) != len(layers_per_block):
        raise ValueError(
            'block_out_channels and layers_per_block must have same length, '
            f'got {len(block_out_channels)} and {len(layers_per_block)}'
        )

    requested_downsamples = max(0, len(block_out_channels) - 1)
    safe_downsamples = _max_safe_downsamples_for_width(width=width, requested=requested_downsamples)
    keep_levels = safe_downsamples + 1

    if keep_levels < 1:
        raise ValueError('Invalid architecture depth after grid adaptation')

    return (
        list(block_out_channels[:keep_levels]),
        list(layers_per_block[:keep_levels]),
        requested_downsamples,
        safe_downsamples,
    )


class PlainTextProgressCallback(Callback):
    """Emit stable plain-text training progress lines for non-TTY logs."""

    def __init__(self, log_every_batches: int = 50) -> None:
        super().__init__()
        self.log_every_batches = max(1, int(log_every_batches))

    @staticmethod
    def _metric_to_float(value):
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                return float(value.detach().item())
            return float(value)
        except Exception:
            return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if not trainer.is_global_zero:
            return
        cur_batch = int(batch_idx) + 1
        if cur_batch > 5 and (cur_batch % self.log_every_batches) != 0:
            return
        step = int(trainer.global_step)

        total_batches = trainer.num_training_batches
        if isinstance(total_batches, float):
            total_batches = int(total_batches)
        if not isinstance(total_batches, int) or total_batches <= 0:
            total_batches = -1

        loss = self._metric_to_float(trainer.callback_metrics.get("train/loss_step"))
        recon = self._metric_to_float(trainer.callback_metrics.get("train/loss_recon_step"))
        latent = self._metric_to_float(trainer.callback_metrics.get("train/loss_latent_step"))
        reg_d = self._metric_to_float(trainer.callback_metrics.get("train/loss_reg_d_step"))
        anchor = self._metric_to_float(trainer.callback_metrics.get("train/loss_anchor_step"))

        if total_batches > 0:
            pct = 100.0 * float(batch_idx + 1) / float(total_batches)
            batch_part = f"batch={batch_idx + 1}/{total_batches} ({pct:.1f}%)"
        else:
            batch_part = f"batch={batch_idx + 1}"

        parts = [
            f"epoch={trainer.current_epoch + 1}/{trainer.max_epochs}",
            f"step={step}",
            batch_part,
        ]
        if loss is not None:
            parts.append(f"loss={loss:.6f}")
        if recon is not None:
            parts.append(f"recon={recon:.6f}")
        if latent is not None:
            parts.append(f"latent={latent:.6f}")
        if reg_d is not None:
            parts.append(f"reg_d={reg_d:.6f}")
        if anchor is not None:
            parts.append(f"anchor={anchor:.6f}")

        print("[train_progress] " + " | ".join(parts), flush=True)


def main() -> None:
    args = parse_args()
    _apply_preset(args)

    if args.accumulate_grad_batches < 1:
        raise ValueError("--accumulate-grad-batches must be >= 1")

    pl.seed_everything(args.seed, workers=True)

    run_name = args.run_name or datetime.now().strftime(f"weather_hermite_{args.preset}_%Y%m%d_%H%M")
    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    variables = _normalize_variables(args.variables)
    hour_loss_weights = _parse_hour_loss_weights(args.hour_loss_weights)
    channel_loss_weights = _parse_channel_loss_weights(args.channel_loss_weights)
    if not args.pressure_levels:
        raise ValueError("At least one pressure level must be provided.")
    # Multi-level support: use all pressure levels from --pressure-levels.
    # When --multi-level passed, all 4 levels (1000,950,900,850) are used → 5*4=20 PL channels.
    # Default backward-compat: single level [1000].
    if getattr(args, "multi_level", False):
        pressure_levels = list(args.pressure_levels)
    else:
        pressure_levels = [args.pressure_levels[0]]

    print("Model: WeatherHermiteModel")
    print(f"Model type: {args.model_type}")
    print(f"Preset: {args.preset}")
    print(f"Seed: {args.seed}")
    print(f"Input years: {args.years}")
    print(f"Variables: {variables}")
    print(f"Pressure levels used: {pressure_levels}")
    print("Conditioning: continuous tau + delta_t only (no hour-id features)")

    datamodule = WeatherHermiteDataModule(
        data_dir=args.data_dir,
        years=args.years,
        train_years=args.train_years,
        test_years=args.test_years,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_per_date=args.samples_per_date,
        variables=variables,
        pressure_levels=pressure_levels,
        val_split=args.val_split,
        static_path=args.static_path,
        stats_path=args.stats_path,
        delta_t_hours=args.delta_t_hours,
        train_hours=args.train_hours,
        eval_hours=args.eval_hours,
        cache_in_ram=args.cache_in_ram,
        surface_data_dir=args.surface_data_dir,
        surface_variables=args.surface_variables,
        surface_stats_path=args.surface_stats_path,
        use_analytic_tisr=args.analytic_tisr,
    )
    datamodule.setup("fit")

    print(
        "Resolved year split: "
        f"train={datamodule.resolved_train_years}, "
        f"epoch_test={datamodule.resolved_test_years}"
    )
    print(
        "Resolved hour split: "
        f"train_hours={datamodule.resolved_train_hours}, "
        f"eval_hours={datamodule.resolved_eval_hours}"
    )
    print(f"Train samples: {len(datamodule.train_dataset)}")
    print(f"Epoch-test samples: {len(datamodule.val_dataset)}")
    print(f"Input channels: {datamodule.in_channels}")

    sample = datamodule.train_dataset[0]
    input_h = int(sample["x0"].shape[-2])
    input_w = int(sample["x0"].shape[-1])
    print(f"Input grid (H x W): {input_h} x {input_w}")

    if args.grid_aware_arch:
        (
            args.block_out_channels,
            args.layers_per_block,
            requested_downsamples,
            safe_downsamples,
        ) = _adapt_architecture_for_grid(
            block_out_channels=list(args.block_out_channels),
            layers_per_block=list(args.layers_per_block),
            width=input_w,
        )
        if safe_downsamples < requested_downsamples:
            print(
                "[grid-aware] Reduced encoder/decoder depth to keep longitude width even: "
                f"requested_downsamples={requested_downsamples}, used={safe_downsamples}, "
                f"block_out_channels={args.block_out_channels}, layers_per_block={args.layers_per_block}"
            )
        else:
            print(
                "[grid-aware] Architecture depth is compatible with current grid: "
                f"downsamples={safe_downsamples}"
            )
    else:
        print(
            "[grid-aware] Disabled: keeping requested architecture depth; "
            "odd latent sizes will be trimmed inside the encoder."
        )

    model = WeatherHermiteLightningModule(
        latent_channels=args.latent_channels,
        cond_dim=args.cond_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_latent=args.lambda_latent,
        lambda_reg_d=args.lambda_reg_d,
        use_latent_loss=args.use_latent_loss,
        log_every=args.log_every,
        channel_groups=datamodule.channel_groups,
        seen_hours=datamodule.resolved_train_hours,
        max_tau_hours=int(round(args.delta_t_hours)),
        model_type=args.model_type,
        block_out_channels=tuple(args.block_out_channels),
        layers_per_block=tuple(args.layers_per_block),
        hidden_channels=args.hidden_channels,
        cond_mlp_dim=args.cond_mlp_dim,
        lat_crop=args.lat_crop,
        residual_scale_init=args.residual_scale_init,
        residual_scale_learnable=args.residual_scale_learnable,
        residual_clip=args.residual_clip,
        recon_loss=args.recon_loss,
        smooth_l1_beta=args.smooth_l1_beta,
        use_physical_scales_loss=args.use_physical_scales_loss,
        use_aurora_weights=args.use_aurora_weights,
        physical_loss_scales=tuple(args.physical_loss_scales),
        hour_loss_weights=hour_loss_weights,
        channel_loss_weights=channel_loss_weights,
        lambda_anchor=args.lambda_anchor,
        anchor_loss=args.anchor_loss,
        anchor_every_n_batches=max(1, int(args.anchor_every_n_batches)),
        lambda_tau_smooth=args.lambda_tau_smooth,
        tau_smooth_delta=args.tau_smooth_delta,
        tau_smooth_loss=args.tau_smooth_loss,
        lambda_residual=args.lambda_residual,
        residual_loss=args.residual_loss,
        residual_scale_floor=args.residual_scale_floor,
        direct_prediction=args.direct_prediction,
        lat_weighted_loss=args.lat_weighted_loss,
        n_static_features=args.n_static_features,
        n_surface_channels=len(args.surface_variables) if args.surface_variables else 0,
        n_pl_channels=len(variables) * len(pressure_levels),
        modafno_depth=args.modafno_depth,
        modafno_num_blocks=args.modafno_num_blocks,
        modafno_drop_rate=args.modafno_drop_rate,
        sdyff_num_layers=args.sdyff_num_layers,
        sdyff_n_modes_lat=args.sdyff_n_modes_lat,
        sdyff_n_modes_lon=args.sdyff_n_modes_lon,
        sdyff_dropout=args.sdyff_dropout,
        sdyff_drop_path=args.sdyff_drop_path,
        sdyff_inference_steps=args.sdyff_inference_steps,
        sdyff_train_refine_steps=args.sdyff_train_refine_steps,
        fuxi_depth=args.fuxi_depth,
        fuxi_num_heads=args.fuxi_num_heads,
        fuxi_window_size_h=args.fuxi_window_size_h,
        fuxi_window_size_w=args.fuxi_window_size_w,
        fuxi_patch_size=args.fuxi_patch_size,
        fuxi_drop_path=args.fuxi_drop_path,
        fm_ode_steps=args.fm_ode_steps,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        log_grad_norm_every=args.log_grad_norm_every,
    )

    loggers = []
    csv_logger = CSVLogger(save_dir=str(log_dir), name="lightning_logs")
    loggers.append(csv_logger)
    try:
        tb_logger = TensorBoardLogger(save_dir=str(log_dir), name="tensorboard")
        loggers.append(tb_logger)
    except Exception as exc:
        print(f"TensorBoard logger is disabled due to environment error: {exc}")

    checkpoint_kwargs = dict(
        dirpath=str(log_dir),
        save_top_k=-1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    if args.save_every_train_steps is not None:
        checkpoint_kwargs.update(
            filename="model_step_{step:07d}",
            every_n_train_steps=args.save_every_train_steps,
            every_n_epochs=None,
        )
    else:
        checkpoint_kwargs.update(
            filename="model_epoch_{epoch:03d}",
            every_n_epochs=args.save_every,
        )
    checkpoint = ModelCheckpoint(**checkpoint_kwargs)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    plain_progress = PlainTextProgressCallback(log_every_batches=50)

    resume_ckpt: Path | None = None
    if args.resume_ckpt:
        candidate = Path(args.resume_ckpt).expanduser()
        resume_ckpt = candidate if candidate.is_absolute() else Path.cwd() / candidate
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"--resume-ckpt does not exist: {resume_ckpt}")
    elif args.resume_from_last:
        candidate = log_dir / "last.ckpt"
        if candidate.exists():
            resume_ckpt = candidate
        else:
            print(f"Resume requested but checkpoint not found: {candidate}")

    print(f"Starting training: {args.epochs} epochs, precision={args.precision}")
    print(f"Batch size: {args.batch_size}, accumulate_grad_batches={args.accumulate_grad_batches}")
    print(f"Optimizer lr={args.lr}, weight_decay={args.weight_decay}")
    print(
        "Physical loss scales (T/U/V/Q/Z): "
        f"{args.physical_loss_scales} | enabled={args.use_physical_scales_loss}"
    )
    print(f"Hour loss weights: {hour_loss_weights if hour_loss_weights else 'default(1.0)'}")
    print(f"Channel loss weights: {channel_loss_weights if channel_loss_weights else 'default(1.0)'}")
    print(
        f"Anchor: lambda={args.lambda_anchor}, every_n_batches={max(1, int(args.anchor_every_n_batches))}"
    )
    print(f"Sanity validation steps: {max(0, int(args.sanity_val_steps))}")
    print(f"Logs will be saved to: {log_dir}")
    if resume_ckpt is not None:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    else:
        print("Resuming from checkpoint: disabled")

    trainer_kwargs = dict(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        callbacks=[checkpoint, lr_monitor, plain_progress],
        logger=loggers,
        default_root_dir=str(log_dir),
        log_every_n_steps=1,
        enable_progress_bar=False,
        num_sanity_val_steps=max(0, int(args.sanity_val_steps)),
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=False,
    )
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = (
            int(args.limit_train_batches) if args.limit_train_batches > 1 else args.limit_train_batches
        )
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = (
            int(args.limit_val_batches) if args.limit_val_batches > 1 else args.limit_val_batches
        )
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(resume_ckpt) if resume_ckpt is not None else None,
    )

    print("Training completed.")
    print(f"Checkpoint dir: {log_dir}")
    print(f"Last checkpoint: {log_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()

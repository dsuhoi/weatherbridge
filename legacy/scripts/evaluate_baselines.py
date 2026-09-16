"""Evaluate baselines (bilinear/bicubic via F.interpolate) and optional model on test set.
Writes TensorBoard logs and saves metrics to JSON.

Usage:
  python evaluate_baselines.py --data-dir data/zarr --years 2020 --output metrics/baselines_test.json
  python evaluate_baselines.py --data-dir data/zarr --years 2020 --model-checkpoint logs/run_x/model_epoch_10.pth
  python evaluate_baselines.py --data-dir data/zarr --years 2017 --log-dir logs/baselines_2017 --name bilinear_bicubic
  python evaluate_baselines.py --data-dir data/zarr --years 2017 --per-hour --output metrics/baselines_2017_per_hour.json
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from weather_time_interp.config import HOURS_PER_TAU_UNIT, DEFAULT_VARIABLES, DEFAULT_PRESSURE_LEVELS
from dataset import ERA5ResNetODEDataset
from metrics import WeatherMetrics
from models.resnet.resnet_ode_model import ResNetODEModel


def _is_weather_hermite_ckpt(ckpt: Dict[str, Any]) -> bool:
    """Проверить, является ли чекпоинт от WeatherHermite."""
    hparams = ckpt.get("hyper_parameters", {})
    return "latent_channels" in hparams or "cond_dim" in hparams


class NaNReplacementEncoder(json.JSONEncoder):
    """JSON encoder that replaces NaN/Inf with null."""
    def encode(self, obj):
        obj = self._replace_nan(obj)
        return super().encode(obj)
    
    def _replace_nan(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: self._replace_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_nan(v) for v in obj]
        return obj


def interpolate_time_with_f_interpolate(
    x0: torch.Tensor,
    x1: torch.Tensor,
    tau: torch.Tensor,
    mode: str,
    steps: int = 6,
) -> torch.Tensor:
    """Linear or cubic interpolation in TIME (not spatial) between x0 and xT at τ ∈ [0,1].

    Bilinear: closed-form (1-τ)·x0 + τ·xT — exact per-sample.
    Bicubic:  not well-defined for just 2 control points; we use cubic Hermite with
              zero end-tangents which collapses to bilinear (better than the prior
              align_corners trick that mis-quantized τ to a 6-bin grid).
    """
    if mode not in {"bilinear", "bicubic"}:
        raise ValueError("mode must be 'bilinear' or 'bicubic'")

    tau_ = tau.float().view(-1)
    B = x0.size(0)
    if tau_.numel() == 1 and B > 1:
        tau_ = tau_.expand(B)
    tau_b = tau_.view(B, 1, 1, 1)

    # bilinear / bicubic with 2 control points reduce to the same closed-form.
    # For real bicubic we'd need 4 anchors (x_{-1}, x0, x1, x2) which we don't have.
    pred = (1.0 - tau_b) * x0 + tau_b * x1
    return pred


def evaluate_per_hour_all(
    loader: DataLoader,
    model: torch.nn.Module,
    model_type: str,
    device: torch.device,
    channel_groups: Dict[str, List[int]],
    writer: SummaryWriter,
    n_ensemble_passes: int = 1,
    max_tau_hours: int = 6,
    skip_bicubic: bool = False,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Compare pred_h vs truth_h for each h ∈ {1..max_tau-1}.

    Requires dataset to set `eval_all_hours=True` so each batch contains
    `target_all_hours` (B, n_hours, C, H, W). Otherwise fallback to the OLD
    buggy behaviour: comparing pred_h vs single batch target (wrong hour).
    """
    per_hour_results: Dict[int, Dict[str, Dict[str, float]]] = {}
    per_hour_counts: Dict[int, int] = {}
    eval_hours = list(range(1, max_tau_hours))   # interior only — endpoints exact
    print(f"Evaluating per hour {eval_hours} (max_tau={max_tau_hours}, skip_bicubic={skip_bicubic})...")

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="per-hour-all", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            x0 = batch["x0"].to(device)
            x1 = batch["x1"].to(device)
            time_emb = batch["time_emb"].to(device)
            static = batch.get("static")
            if static is not None:
                static = static.to(device)
            bsz = x0.size(0)

            target_all = batch.get("target_all_hours")
            if target_all is not None:
                target_all = target_all.to(device)   # (B, n_hours, C, H, W)
            else:
                target_all = None
                target_single = batch["target"].to(device)

            # Vectorized: build all tau values in one tensor, compute bilinear in one call
            tau_all = torch.tensor([h / float(max_tau_hours) for h in eval_hours],
                                   device=device, dtype=torch.float32)
            # bilinear pred: shape (B, n_hours, C, H, W) via broadcast
            tau_b = tau_all.view(1, -1, 1, 1, 1)                    # (1, n_h, 1, 1, 1)
            x0_b = x0.unsqueeze(1)                                  # (B, 1, C, H, W)
            x1_b = x1.unsqueeze(1)
            pred_bi_all = (1.0 - tau_b) * x0_b + tau_b * x1_b       # (B, n_h, C, H, W)

            if not skip_bicubic:
                # Bicubic uses spatial F.interpolate per hour — keep loop for it
                pred_bic_all_list = []
                for h in eval_hours:
                    tau_h = torch.full((bsz,), h / float(max_tau_hours), device=device)
                    pred_bic_all_list.append(
                        interpolate_time_with_f_interpolate(x0, x1, tau_h, mode="bicubic")
                    )
                pred_bic_all = torch.stack(pred_bic_all_list, dim=1)
            else:
                pred_bic_all = None

            # Model: one forward per hour (model is τ-conditional)
            if model is not None:
                pred_model_list = []
                for h in eval_hours:
                    tau_h = torch.full((bsz,), h / float(max_tau_hours), device=device)
                    pred_m, _ = forward_ensemble(
                        model, model_type, x0, x1, tau_h, time_emb, static, device,
                        n_passes=n_ensemble_passes,
                    )
                    pred_model_list.append(pred_m)
                pred_model_all = torch.stack(pred_model_list, dim=1)
            else:
                pred_model_all = None

            for h_idx, hour in enumerate(eval_hours):
                target_h = target_all[:, h_idx] if target_all is not None else target_single

                metrics_bi = WeatherMetrics.all(pred_bi_all[:, h_idx], target_h, channel_groups, exclude_channels=["tisr"])
                metrics_bic = (WeatherMetrics.all(pred_bic_all[:, h_idx], target_h, channel_groups, exclude_channels=["tisr"])
                               if pred_bic_all is not None else {})

                if hour not in per_hour_results:
                    per_hour_results[hour] = {"bilinear": {}, "bicubic": {}}
                    per_hour_counts[hour] = 0
                    if model is not None:
                        per_hour_results[hour]["model"] = {}

                for k, v in metrics_bi.items():
                    per_hour_results[hour]["bilinear"][k] = per_hour_results[hour]["bilinear"].get(k, 0.0) + v
                for k, v in metrics_bic.items():
                    per_hour_results[hour]["bicubic"][k] = per_hour_results[hour]["bicubic"].get(k, 0.0) + v
                if pred_model_all is not None:
                    metrics_m = WeatherMetrics.all(pred_model_all[:, h_idx], target_h, channel_groups, exclude_channels=["tisr"])
                    for k, v in metrics_m.items():
                        per_hour_results[hour]["model"][k] = per_hour_results[hour]["model"].get(k, 0.0) + v
                per_hour_counts[hour] += bsz

            if (batch_idx + 1) % 10 == 0:
                pbar.set_postfix(done=f"{batch_idx + 1}/{len(loader)}")

    for hour in per_hour_results:
        count = per_hour_counts[hour]
        for method in per_hour_results[hour]:
            for k in per_hour_results[hour][method]:
                per_hour_results[hour][method][k] /= count
    return per_hour_results


def _extract_model_state(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]

    state_dict = ckpt.get("state_dict", ckpt)
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            normalized[key[len("model."):]] = value
    return normalized if normalized else state_dict


def _extract_config(ckpt: Dict[str, Any], in_channels_fallback: int) -> Dict[str, Any]:
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        return ckpt["config"]

    hparams = ckpt.get("hyper_parameters", {})
    return {
        "in_channels": hparams.get("in_channels", in_channels_fallback),
        "static_channels": hparams.get("static_channels", 0),
        "latent_dim": hparams.get("latent_dim", 4),
        "depth": hparams.get("depth", 1),
    }


def load_model(checkpoint_path: Path, device: torch.device, in_channels_fallback: int, channel_groups: Dict[str, List[int]]) -> torch.nn.Module:
    """Загрузить модель из чекпоинта (поддерживает ResNetODE и WeatherHermite)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Проверяем тип модели
    if _is_weather_hermite_ckpt(ckpt):
        return load_weather_hermite_model(checkpoint_path, device, channel_groups)
    else:
        return load_resnet_ode_model(checkpoint_path, device, in_channels_fallback)


def load_weather_hermite_model(
    checkpoint_path: Path,
    device: torch.device,
    channel_groups: Dict[str, List[int]],
) -> torch.nn.Module:
    """Загрузить WeatherHermiteModel из чекпоинта Lightning."""
    from trainer_weather_hermite import WeatherHermiteLightningModule
    
    # Загружаем чекпоинт для извлечения гиперпараметров
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    
    # Извлекаем параметры архитектуры из чекпоинта
    model = WeatherHermiteLightningModule.load_from_checkpoint(
        str(checkpoint_path),
        channel_groups=channel_groups,
        latent_channels=hparams.get("latent_channels", 64),
        cond_dim=hparams.get("cond_dim", 1),
        block_out_channels=tuple(hparams.get("block_out_channels", (128, 128, 256, 256))),
        layers_per_block=tuple(hparams.get("layers_per_block", (2, 2, 2, 2))),
        hidden_channels=hparams.get("hidden_channels", 128),
        cond_mlp_dim=hparams.get("cond_mlp_dim", 32),
        strict=False,  # Не требовать точного совпадения всех параметров
    )
    model.to(device)
    model.eval()
    return model


def load_resnet_ode_model(
    checkpoint_path: Path,
    device: torch.device,
    in_channels_fallback: int,
) -> ResNetODEModel:
    """Загрузить ResNetODEModel из чекпоинта."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = _extract_config(ckpt, in_channels_fallback)
    model = ResNetODEModel(
        in_channels=cfg.get("in_channels", in_channels_fallback),
        static_channels=cfg.get("static_channels", 0),
        latent_dim=cfg.get("latent_dim", 4),
        depth=cfg.get("depth", 1),
        use_film=True,
        film_dim=4,
        ode_solver="rk4",
    )
    state_dict = _extract_model_state(ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def enable_dropout(model: torch.nn.Module) -> int:
    """Switch every Dropout / DropPath module back to train() while keeping rest in eval().
    Returns count of modules toggled (0 ⇒ no MC randomness possible)."""
    n = 0
    for module in model.modules():
        cls_name = module.__class__.__name__
        if isinstance(module, torch.nn.Dropout) or "DropPath" in cls_name or "drop_path" in cls_name.lower():
            module.train(True)
            n += 1
    return n


def _forward_model_once(model, model_type, x0, x1, tau, time_emb, static, device):
    bsz = x0.size(0)
    if model_type == "weather_hermite":
        tau_float = tau.float()
        cond = torch.tensor([6.0] * bsz, dtype=torch.float32, device=device)
        x_hat, _ = model(x0, x1, tau_float, cond, static=static)
        return x_hat
    out = model(x0, x1, time_emb, tau, static=static, return_endpoint=False)
    return out.pred if hasattr(out, "pred") else out


def forward_ensemble(model, model_type, x0, x1, tau, time_emb, static, device, n_passes: int):
    """Average n_passes forward passes; returns (mean_pred, std_pred). For n_passes=1 std is None."""
    if n_passes <= 1:
        with torch.no_grad():
            return _forward_model_once(model, model_type, x0, x1, tau, time_emb, static, device), None
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(_forward_model_once(model, model_type, x0, x1, tau, time_emb, static, device))
    stack = torch.stack(preds, dim=0)
    return stack.mean(dim=0), stack.std(dim=0)


def update_sums(sums: Dict[str, float], metrics: Dict[str, float], weight: int) -> None:
    for k, v in metrics.items():
        sums[k] = sums.get(k, 0.0) + v * weight


def finalize_avg(sums: Dict[str, float], total: int) -> Dict[str, float]:
    """Вычислить средние метрики, заменяя NaN на 0."""
    result = {}
    for k, v in sums.items():
        avg = v / max(total, 1)
        # Замена NaN и Inf на 0
        if not torch.isfinite(torch.tensor(avg)):
            result[k] = 0.0
            print(f"Warning: Metric {k} was NaN/Inf, replaced with 0")
        else:
            result[k] = avg
    return result


def choose_best(results: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    best: Dict[str, str] = {}
    if not results:
        return best
    metrics_keys = list(next(iter(results.values())).keys())
    for key in metrics_keys:
        values = {name: res[key] for name, res in results.items() if key in res}
        if not values:
            continue
        if key.startswith("silhouette") or key == "psnr":
            best[key] = max(values, key=values.get)
        elif key.startswith("bias"):
            best[key] = min(values, key=lambda k: abs(values[k]))
        else:
            best[key] = min(values, key=values.get)
    return best


def overall_winner(results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    metrics_main = ["rmse", "mae", "bias", "psnr", "silhouette"]
    counts: Dict[str, int] = {k: 0 for k in results.keys()}
    winners: Dict[str, str] = {}
    for m in metrics_main:
        vals = {name: res[m] for name, res in results.items() if m in res}
        if not vals:
            continue
        if m in {"silhouette", "psnr"}:
            w = max(vals, key=vals.get)
        elif m == "bias":
            w = min(vals, key=lambda k: abs(vals[k]))
        else:
            w = min(vals, key=vals.get)
        winners[m] = w
        counts[w] = counts.get(w, 0) + 1
    if not counts:
        return {"winner": None, "wins": {}, "by_metric": winners}
    max_wins = max(counts.values())
    top = [k for k, v in counts.items() if v == max_wins]
    winner = top[0] if len(top) == 1 else "tie"
    return {"winner": winner, "wins": counts, "by_metric": winners}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baselines (bilinear/bicubic) on test set.")
    parser.add_argument("--data-dir", type=str, default="~/data/weather_data/time_interpolation/")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--samples-per-date", type=int, default=3)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--variables", type=str, nargs="+", default=DEFAULT_VARIABLES)
    parser.add_argument("--pressure-levels", type=int, nargs="+", default=DEFAULT_PRESSURE_LEVELS)
    parser.add_argument("--static-path", type=str, default=None)
    parser.add_argument("--stats-path", type=str, default="data/json_stats.nc")
    parser.add_argument("--model-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="metrics/baselines_test.json")
    parser.add_argument("--log-dir", type=str, default="logs/baselines")
    parser.add_argument("--name", type=str, default="bilinear_bicubic")
    parser.add_argument("--per-hour", action="store_true", help="Оценивать качество для каждого часа (1-6) отдельно")
    parser.add_argument("--per-hour-all", action="store_true", help="Оценивать качество для ВСЕХ часов 1-6 (требует больше памяти)")
    parser.add_argument("--eval-hours", type=int, nargs="+", default=None,
                        help="Explicit list of tau hours to evaluate (e.g. 0 1 2 3 4 5 6).")
    parser.add_argument("--cache-in-ram", action="store_true", default=False,
                        help="Pre-load zarr arrays into a contiguous numpy array for fast random access.")
    parser.add_argument("--surface-data-dir", type=str, default=None,
                        help="Optional dir with surface_<year>.zarr (for kitchen-sink models).")
    parser.add_argument("--surface-variables", type=str, nargs="*", default=None,
                        help="Surface variable short names (t2m u10 v10 mslp tisr).")
    parser.add_argument("--surface-stats-path", type=str, default=None,
                        help="JSON with mean/std for surface variables.")
    parser.add_argument("--analytic-tisr", action="store_true", default=False,
                        help="Compute TISR analytically instead of reading zarr (matches train-side flag).")
    parser.add_argument("--delta-t-hours", type=int, default=6,
                        help="Interpolation gap. Must match the model's training delta_t.")
    parser.add_argument("--skip-bicubic", action="store_true", default=False,
                        help="Skip bicubic baseline computation (~30% faster eval).")
    parser.add_argument("--eval-days-per-month", type=int, default=None,
                        help="Economy eval: subsample to ~K days/month with full start-hour coverage. "
                             "K=3→days [1,11,21]; K=4→[1,8,15,22]; K=5→[1,7,14,21,28]. None=full eval.")
    parser.add_argument("--ensemble-passes", type=int, default=1,
                        help="Number of forward passes to average per sample. >1 implies MC-dropout style ensembling.")
    parser.add_argument("--mc-dropout", action="store_true", default=False,
                        help="Force Dropout / DropPath layers to remain active during eval (model.eval() but dropout in train mode).")
    args = parser.parse_args()
    if args.ensemble_passes > 1 and not args.mc_dropout:
        print("Warning: --ensemble-passes > 1 without --mc-dropout will collapse to deterministic forward; auto-enabling --mc-dropout.")
        args.mc_dropout = True

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Определяем тип чекпоинта заранее, чтобы корректно собрать датасет.
    model_type = None
    if args.model_checkpoint:
        ckpt_hint = torch.load(Path(args.model_checkpoint), map_location="cpu", weights_only=False)
        if _is_weather_hermite_ckpt(ckpt_hint):
            model_type = "weather_hermite"
            # Multi-level / kitchen-sink ckpts use multiple pressure levels and
            # explicit --in-channels. Only force legacy single-level fallback when
            # user did not explicitly supply --pressure-levels (>1 means kitchen sink).
            if len(args.pressure_levels) <= 1:
                if args.in_channels is None:
                    args.in_channels = 5
                    print("Detected WeatherHermite checkpoint (legacy single-level): forcing --in-channels=5")
        else:
            model_type = "resnet_ode"

    # Инициализация TensorBoard
    log_dir = Path(args.log_dir) / args.name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")

    dataset = ERA5ResNetODEDataset(
        data_dir=args.data_dir,
        years=args.years,
        in_channels=args.in_channels,
        variables=args.variables,
        pressure_levels=args.pressure_levels,
        max_tau_hours=int(args.delta_t_hours),
        samples_per_date=args.samples_per_date,
        train=False,
        static_path=args.static_path,
        stats_path=args.stats_path,
        eval_hours=args.eval_hours,
        cache_in_ram=args.cache_in_ram,
        surface_data_dir=args.surface_data_dir,
        surface_variables=args.surface_variables,
        surface_stats_path=args.surface_stats_path,
        use_analytic_tisr=args.analytic_tisr,
        eval_all_hours=bool(args.per_hour_all),
    )

    if args.eval_days_per_month is not None:
        K = max(1, int(args.eval_days_per_month))
        day_picks = {3: [1, 11, 21], 4: [1, 8, 15, 22], 5: [1, 7, 14, 21, 28]}.get(
            K, sorted({1 + i * (30 // K) for i in range(K)})
        )
        allowed_days = set(day_picks)
        import datetime as _dt
        filtered = []
        for entry in dataset.index:
            y, t0, _, _ = entry
            doy = t0 // 24  # day-of-year (0-based)
            try:
                date = _dt.date(int(y), 1, 1) + _dt.timedelta(days=int(doy))
            except Exception:
                continue
            if date.day in allowed_days:
                filtered.append(entry)
        kept, total_idx = len(filtered), len(dataset.index)
        print(
            f"[economy-eval] K={K} days/month {sorted(allowed_days)} → "
            f"kept {kept}/{total_idx} index entries ({100*kept/max(total_idx,1):.1f}%)"
        )
        dataset.index = filtered

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    model = None
    if args.model_checkpoint:
        model = load_model(
            Path(args.model_checkpoint),
            device,
            in_channels_fallback=dataset.in_channels,
            channel_groups=dataset.channel_groups,
        )
        if model_type is None:
            model_type = "resnet_ode"
        if args.mc_dropout:
            n_dropout = enable_dropout(model)
            print(f"MC-dropout: enabled train mode on {n_dropout} Dropout/DropPath modules; ensemble_passes={args.ensemble_passes}")
            if n_dropout == 0:
                print("Warning: model contains no Dropout/DropPath layers — ensemble passes will return identical outputs.")

    results_sums: Dict[str, Dict[str, float]] = {
        "bilinear": {},
        "bicubic": {},
    }
    if model is not None:
        results_sums["model"] = {}

    total = 0

    # Для почасовой оценки (старый режим - только для существующих tau в датасете)
    per_hour_results: Dict[int, Dict[str, Dict[str, float]]] = {}
    per_hour_counts: Dict[int, int] = {}

    # Если --per-hour-all, используем отдельную функцию
    if args.per_hour_all:
        per_hour_avg = evaluate_per_hour_all(
            loader, model, model_type, device, dataset.channel_groups, writer,
            n_ensemble_passes=args.ensemble_passes,
            max_tau_hours=int(args.delta_t_hours),
            skip_bicubic=bool(args.skip_bicubic),
        )
        # Сохраняем для JSON
        payload: Dict[str, Any] = {
            "num_samples": len(dataset),
            "years": args.years,
            "per_hour": per_hour_avg,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, cls=NaNReplacementEncoder)
        print(f"Per-hour results saved to: {out_path}")

        # Вывод
        print("\n" + "=" * 60)
        print("PER-HOUR RESULTS (hours 1-6)")
        print("=" * 60)
        for hour in sorted(per_hour_avg.keys()):
            print(f"\nHour {hour}:")
            for method, metrics in per_hour_avg[hour].items():
                print(f"  {method}:")
                for k in ["rmse", "mae", "bias", "psnr", "silhouette"]:
                    if k in metrics:
                        print(f"    {k}: {metrics[k]:.6f}")
        writer.close()
        return  # Завершаем, не выполняем обычный цикл

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="eval-baselines", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            x0 = batch["x0"].to(device)
            x1 = batch["x1"].to(device)
            tau = batch["tau"].to(device)
            target = batch["target"].to(device)
            time_emb = batch["time_emb"].to(device)
            static = batch.get("static")
            if static is not None:
                static = static.to(device)


            bsz = x0.size(0)
            total += bsz

            pred_bilinear = interpolate_time_with_f_interpolate(x0, x1, tau, mode="bilinear")
            pred_bicubic = interpolate_time_with_f_interpolate(x0, x1, tau, mode="bicubic")


            metrics_bilinear = WeatherMetrics.all(pred_bilinear, target, dataset.channel_groups)
            metrics_bicubic = WeatherMetrics.all(pred_bicubic, target, dataset.channel_groups)
            if "rmse" in metrics_bilinear and "rmse" in metrics_bicubic:
                pbar.set_postfix(
                    rmse_bi=f"{metrics_bilinear['rmse']:.4f}",
                    rmse_bic=f"{metrics_bicubic['rmse']:.4f}",
                )


            # Запись в TensorBoard
            for k, v in metrics_bilinear.items():
                writer.add_scalar(f"bilinear/{k}", v, total)
            for k, v in metrics_bicubic.items():
                writer.add_scalar(f"bicubic/{k}", v, total)

            update_sums(results_sums["bilinear"], metrics_bilinear, bsz)
            update_sums(results_sums["bicubic"], metrics_bicubic, bsz)

            if model is not None:
                pred_model, _pred_std = forward_ensemble(
                    model, model_type, x0, x1, tau, time_emb, static, device,
                    n_passes=args.ensemble_passes,
                )

                metrics_model = WeatherMetrics.all(pred_model, target, dataset.channel_groups)
                for k, v in metrics_model.items():
                    writer.add_scalar(f"model/{k}", v, total)
                update_sums(results_sums["model"], metrics_model, bsz)
            
            # Почасовая оценка (только если включён args.per_hour)
            if args.per_hour:
                # tau_hours для каждого сэмпла в батче
                tau_hours_batch = (tau * HOURS_PER_TAU_UNIT).round().clamp(1, 6).long().cpu().numpy()

                for i in range(bsz):
                    hour = int(tau_hours_batch[i])
                    if hour < 1 or hour > 6:
                        continue

                    if hour not in per_hour_results:
                        per_hour_results[hour] = {"bilinear": {}, "bicubic": {}, "model": {}}
                        per_hour_counts[hour] = 0

                    # Метрики для этого часа
                    pred_bi = pred_bilinear[i:i+1]
                    pred_bic = pred_bicubic[i:i+1]
                    tgt = target[i:i+1]

                    metrics_bi = WeatherMetrics.all(pred_bi, tgt, dataset.channel_groups)
                    metrics_bic = WeatherMetrics.all(pred_bic, tgt, dataset.channel_groups)

                    for k, v in metrics_bi.items():
                        per_hour_results[hour]["bilinear"][k] = per_hour_results[hour]["bilinear"].get(k, 0.0) + v
                    for k, v in metrics_bic.items():
                        per_hour_results[hour]["bicubic"][k] = per_hour_results[hour]["bicubic"].get(k, 0.0) + v

                    if model is not None:
                        if model_type == "weather_hermite":
                            pred_m = x_hat[i:i+1]
                        else:
                            pred_m = pred_model[i:i+1]
                        metrics_m = WeatherMetrics.all(pred_m, tgt, dataset.channel_groups)
                        for k, v in metrics_m.items():
                            per_hour_results[hour]["model"][k] = per_hour_results[hour]["model"].get(k, 0.0) + v

                    per_hour_counts[hour] += 1

    results_avg = {name: finalize_avg(sums, total) for name, sums in results_sums.items()}
    best = choose_best(results_avg)
    overall = overall_winner(results_avg)

    # Почасовые результаты (усреднение)
    per_hour_avg: Dict[int, Dict[str, Dict[str, float]]] = {}
    if args.per_hour and per_hour_results:
        for hour in sorted(per_hour_results.keys()):
            count = per_hour_counts.get(hour, 1)
            per_hour_avg[hour] = {
                method: {k: v / count for k, v in metrics.items()}
                for method, metrics in per_hour_results[hour].items()
            }

    # Запись итоговых метрик в TensorBoard
    for name, metrics in results_avg.items():
        for k, v in metrics.items():
            writer.add_scalars(f"final/{k}", {name: v}, 0)
        writer.add_scalar("final/samples", total, 0)
    
    writer.close()

    payload: Dict[str, Any] = {
        "num_samples": total,
        "years": args.years,
        "methods": results_avg,
        "best": best,
        "overall": overall,
    }
    
    if args.per_hour and per_hour_avg:
        payload["per_hour"] = per_hour_avg

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, cls=NaNReplacementEncoder)

    print("Results saved to:", out_path)
    for name, metrics in results_avg.items():
        print(f"\n{name}")
        for k in ["rmse", "mae", "bias", "psnr", "silhouette"]:
            if k in metrics:
                print(f"  {k}: {metrics[k]:.6f}")
    if best:
        print("\nBest by metric:")
        for k, v in best.items():
            print(f"  {k}: {v}")
    if overall:
        print("\nOverall winner:")
        print("  winner:", overall.get("winner"))
        print("  wins:", overall.get("wins"))
    
    # Почасовые результаты
    if args.per_hour and per_hour_avg:
        print("\n" + "=" * 60)
        print("PER-HOUR RESULTS")
        print("=" * 60)
        for hour in sorted(per_hour_avg.keys()):
            print(f"\nHour {hour}:")
            for method, metrics in per_hour_avg[hour].items():
                print(f"  {method}:")
                for k in ["rmse", "mae", "bias", "psnr", "silhouette"]:
                    if k in metrics:
                        print(f"    {k}: {metrics[k]:.6f}")


if __name__ == "__main__":
    main()

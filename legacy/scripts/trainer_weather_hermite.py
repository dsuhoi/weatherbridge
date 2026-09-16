import math
from typing import Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader, Dataset

from weather_time_interp.config import DEFAULT_PRESSURE_LEVELS, DEFAULT_VARIABLES
from dataset import ERA5ResNetODEDataset
from metrics import WeatherMetrics
try:
    from models.koopman import WeatherKoopmanLinearModel
except ModuleNotFoundError:
    WeatherKoopmanLinearModel = None
from weather_time_interp.model.WeatherInterpModel import (
    WeatherHermiteModel,
    WeatherUNetResidualLinearModel,
    WeatherUNetDirectModel,
    WeatherDCAEResidualLinearModel,
    WeatherSFNOResidualLinearModel,
    WeatherSDyffusionResidualLinearModel,
    WeatherTrueUNetResidualLinearModel,
)


class ERA5WeatherHermiteDataset(Dataset):
    """Adapter over ERA5ResNetODEDataset for WeatherHermiteModel input format."""

    def __init__(self, base_dataset: Dataset, delta_t_hours: float = 6.0) -> None:
        self.base_dataset = base_dataset
        self.delta_t_hours = float(delta_t_hours)
        self._grouped_indices: Optional[List[List[int]]] = None
        base_index = getattr(base_dataset, "index", None)
        if isinstance(base_index, list) and base_index:
            grouped: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
            for idx, item in enumerate(base_index):
                if not isinstance(item, tuple) or len(item) < 3:
                    grouped = {}
                    break
                key = (int(item[0]), int(item[1]))
                tau_hour = int(item[2])
                grouped.setdefault(key, []).append((tau_hour, idx))
            if grouped:
                self._grouped_indices = []
                for _key, pairs in grouped.items():
                    pairs.sort(key=lambda x: x[0])
                    self._grouped_indices.append([idx for _, idx in pairs])

    def __len__(self) -> int:
        if self._grouped_indices is not None:
            return len(self._grouped_indices)
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._grouped_indices is not None:
            samples = [self.base_dataset[i] for i in self._grouped_indices[idx]]
            first = samples[0]
            tau = torch.stack([s["tau"].float().view(1) for s in samples], dim=0)
            tau_hour = torch.stack([s["tau_hour"].long().view(1) for s in samples], dim=0)
            target = torch.stack([s["target"] for s in samples], dim=0)
            out = {
                "x0": first["x0"],
                "xT": first["x1"],
                "target": target,
                "tau": tau,
                "tau_hour": tau_hour,
                "cond": torch.tensor([self.delta_t_hours], dtype=torch.float32),
            }
            # Pass through static features if present in base sample
            if "static" in first:
                out["static"] = first["static"]
            return out
        sample = self.base_dataset[idx]
        out = {
            "x0": sample["x0"],
            "xT": sample["x1"],
            "target": sample["target"],
            "tau": sample["tau"].float().view(1),
            "tau_hour": sample["tau_hour"].long().view(1),
            "cond": torch.tensor([self.delta_t_hours], dtype=torch.float32),
        }
        if "static" in sample:
            out["static"] = sample["static"]
        return out


class WeatherHermiteDataModule(pl.LightningDataModule):
    """DataModule for training on one year set and evaluating on a held-out year set each epoch."""

    def __init__(
        self,
        data_dir: str,
        years: Optional[List[int]] = None,
        train_years: Optional[List[int]] = None,
        test_years: Optional[List[int]] = None,
        batch_size: int = 12,
        num_workers: int = 4,
        samples_per_date: int = 4,
        variables: Optional[List[str]] = None,
        pressure_levels: Optional[List[int]] = None,
        val_split: float = 0.1,
        static_path: Optional[str] = None,
        stats_path: str = "data/json_stats.nc",
        delta_t_hours: float = 6.0,
        train_hours: Optional[List[int]] = None,
        eval_hours: Optional[List[int]] = None,
        cache_in_ram: bool = False,
        surface_data_dir: Optional[str] = None,
        surface_variables: Optional[List[str]] = None,
        surface_stats_path: Optional[str] = None,
        use_analytic_tisr: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.years = self._normalize_years(years)
        self.train_years = self._normalize_years(train_years)
        self.test_years = self._normalize_years(test_years)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_per_date = samples_per_date
        self.variables = variables or DEFAULT_VARIABLES
        self.pressure_levels = pressure_levels or DEFAULT_PRESSURE_LEVELS
        self.val_split = val_split
        self.static_path = static_path
        self.stats_path = stats_path
        self.delta_t_hours = delta_t_hours
        self.train_hours = self._normalize_hours(train_hours)
        self.eval_hours = self._normalize_hours(eval_hours)
        self.cache_in_ram = cache_in_ram
        self.surface_data_dir = surface_data_dir
        self.surface_variables = list(surface_variables) if surface_variables else []
        self.surface_stats_path = surface_stats_path
        self.use_analytic_tisr = bool(use_analytic_tisr)

        self.resolved_train_years, self.resolved_test_years = self._resolve_year_splits()
        self.resolved_train_hours, self.resolved_eval_hours = self._resolve_hour_splits()

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.in_channels: Optional[int] = None
        self.channel_groups: Optional[Dict[str, List[int]]] = None

    @staticmethod
    def _normalize_years(years: Optional[List[int]]) -> List[int]:
        if not years:
            return []
        return sorted(dict.fromkeys(int(y) for y in years))

    @staticmethod
    def _normalize_hours(hours: Optional[List[int]]) -> List[int]:
        if not hours:
            return []
        return sorted(dict.fromkeys(int(h) for h in hours))

    def _resolve_year_splits(self) -> Tuple[List[int], List[int]]:
        if self.train_years or self.test_years:
            if not self.train_years:
                raise ValueError('Explicit split requires at least one --train-years value.')
            if not self.test_years:
                raise ValueError('Explicit split requires at least one --test-years value.')
            overlap = set(self.train_years) & set(self.test_years)
            if overlap:
                raise ValueError(f'Train/test year splits must be disjoint, overlap={sorted(overlap)}')
            return list(self.train_years), list(self.test_years)

        years = list(self.years)
        if len(years) < 2:
            raise ValueError(
                'Need at least two years for a held-out evaluation split. '
                'Provide --train-years/--test-years explicitly or pass >=2 values to --years.'
            )
        return years[:-1], [years[-1]]

    def _resolve_hour_splits(self) -> Tuple[List[int], List[int]]:
        max_tau_hours = int(round(self.delta_t_hours))
        default_train = list(range(0, max_tau_hours + 1))
        train_hours = self.train_hours if self.train_hours else default_train
        train_hours = sorted({max(0, min(max_tau_hours, h)) for h in train_hours})
        if not train_hours:
            raise ValueError("No valid train hours after normalization.")

        if self.eval_hours:
            eval_hours = sorted({max(0, min(max_tau_hours, h)) for h in self.eval_hours})
        else:
            eval_hours = list(range(0, max_tau_hours + 1))
        if not eval_hours:
            raise ValueError("No valid eval hours after normalization.")
        return train_hours, eval_hours

    def _build_base_dataset(self, years: List[int], train: bool) -> ERA5ResNetODEDataset:
        # Multi-level: N_pl = N_vars × N_levels (5 vars × N levels).
        n_pl = len(self.variables) * len(self.pressure_levels)
        dataset = ERA5ResNetODEDataset(
            data_dir=self.data_dir,
            years=years,
            in_channels=n_pl,
            variables=self.variables,
            pressure_levels=self.pressure_levels,
            max_tau_hours=int(round(self.delta_t_hours)),
            samples_per_date=self.samples_per_date,
            train=train,
            static_path=self.static_path,
            stats_path=self.stats_path,
            train_hours=self.resolved_train_hours if train else None,
            eval_hours=self.resolved_eval_hours if not train else None,
            cache_in_ram=self.cache_in_ram and train,
            surface_data_dir=self.surface_data_dir,
            surface_variables=self.surface_variables,
            surface_stats_path=self.surface_stats_path,
            use_analytic_tisr=self.use_analytic_tisr,
        )
        if dataset.in_channels != n_pl:
            raise ValueError(
                f'Pressure-level dataset expected {n_pl} channels (5 vars × {len(self.pressure_levels)} levels), '
                f'got {dataset.in_channels}.'
            )
        return dataset

    def _register_schema(self, dataset: ERA5ResNetODEDataset) -> None:
        if self.in_channels is None:
            self.in_channels = dataset.in_channels
            self.channel_groups = dataset.channel_groups
            return
        if dataset.in_channels != self.in_channels:
            raise ValueError(
                'Inconsistent in_channels across splits: '
                f'expected {self.in_channels}, got {dataset.in_channels}.'
            )
        if dataset.channel_groups != self.channel_groups:
            raise ValueError('Inconsistent channel_groups across splits.')

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit"):
            return

        if self.train_dataset is None:
            train_base = self._build_base_dataset(self.resolved_train_years, train=True)
            self._register_schema(train_base)
            self.train_dataset = ERA5WeatherHermiteDataset(
                train_base,
                delta_t_hours=self.delta_t_hours,
            )

        if self.val_dataset is None:
            test_base = self._build_base_dataset(self.resolved_test_years, train=False)
            self._register_schema(test_base)
            self.val_dataset = ERA5WeatherHermiteDataset(
                test_base,
                delta_t_hours=self.delta_t_hours,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule is not set up. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule is not set up. Call setup('fit') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )


class _RunningWeatherMetrics:
    """Streaming accumulator for weather metrics without storing full epoch tensors."""

    def __init__(
        self,
        channel_groups: Optional[Dict[str, List[int]]],
        include_lat_weighted: bool = False,
    ) -> None:
        self.channel_groups = channel_groups or {}
        self.include_lat_weighted = include_lat_weighted
        self.reset()

    @staticmethod
    def _corr_stats(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, int]:
        pred_flat = pred.flatten(2)
        target_flat = target.flatten(2)
        pred_centered = pred_flat - pred_flat.mean(dim=2, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=2, keepdim=True)
        num = (pred_centered * target_centered).sum(dim=2)
        den = (pred_centered.pow(2).sum(dim=2).sqrt() * target_centered.pow(2).sum(dim=2).sqrt()).clamp(min=1e-8)
        corr = torch.nan_to_num(num / den, nan=0.0, posinf=0.0, neginf=0.0)
        return float(corr.sum().item()), int(corr.numel())

    @staticmethod
    def _latitude_weights(height: int, dtype: torch.dtype) -> torch.Tensor:
        lats = torch.linspace(-90.0, 90.0, steps=height, dtype=dtype)
        return torch.cos(torch.deg2rad(lats)).clamp_min(0.0).view(1, 1, height, 1)

    def reset(self) -> None:
        self.sse = 0.0
        self.sae = 0.0
        self.sdiff = 0.0
        self.n = 0
        self.target_abs_max = 0.0
        self.corr_sum = 0.0
        self.corr_count = 0

        self.weighted_sse = 0.0
        self.weighted_sae = 0.0
        self.weighted_den = 0.0

        self.per_param: Dict[str, Dict[str, float | int | List[int]]] = {}
        for name, idxs in self.channel_groups.items():
            self.per_param[name] = {
                "idxs": list(idxs),
                "sse": 0.0,
                "sae": 0.0,
                "sdiff": 0.0,
                "n": 0,
                "corr_sum": 0.0,
                "corr_count": 0,
                "weighted_sse": 0.0,
                "weighted_sae": 0.0,
                "weighted_den": 0.0,
            }

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.numel() == 0:
            return

        pred = pred.detach().float().cpu()
        target = target.detach().float().cpu()
        diff = pred - target

        self.sse += float(diff.pow(2).sum().item())
        self.sae += float(diff.abs().sum().item())
        self.sdiff += float(diff.sum().item())
        self.n += int(diff.numel())
        self.target_abs_max = max(self.target_abs_max, float(target.abs().max().item()))

        corr_sum, corr_count = self._corr_stats(pred, target)
        self.corr_sum += corr_sum
        self.corr_count += corr_count

        weight = None
        if self.include_lat_weighted:
            weight = self._latitude_weights(diff.size(-2), dtype=diff.dtype)
            self.weighted_sse += float((diff.pow(2) * weight).sum().item())
            self.weighted_sae += float((diff.abs() * weight).sum().item())
            self.weighted_den += float(weight.sum().item() * diff.size(0) * diff.size(1) * diff.size(-1))

        for name, stat in self.per_param.items():
            idxs = stat["idxs"]
            if not idxs:
                continue
            p = pred[:, idxs]
            t = target[:, idxs]
            d = p - t
            stat["sse"] += float(d.pow(2).sum().item())
            stat["sae"] += float(d.abs().sum().item())
            stat["sdiff"] += float(d.sum().item())
            stat["n"] += int(d.numel())
            csum, ccount = self._corr_stats(p, t)
            stat["corr_sum"] += csum
            stat["corr_count"] += ccount

            if self.include_lat_weighted and weight is not None:
                stat["weighted_sse"] += float((d.pow(2) * weight).sum().item())
                stat["weighted_sae"] += float((d.abs() * weight).sum().item())
                stat["weighted_den"] += float(weight.sum().item() * d.size(0) * d.size(1) * d.size(-1))

    def compute(self) -> Dict[str, float]:
        if self.n == 0:
            return {}

        result: Dict[str, float] = {}

        mse = self.sse / float(self.n)
        result["rmse"] = math.sqrt(max(0.0, mse))
        result["mae"] = self.sae / float(self.n)
        result["bias"] = self.sdiff / float(self.n)

        if mse == 0.0:
            result["psnr"] = 100.0
        elif self.target_abs_max < 1e-8:
            result["psnr"] = float("nan")
        else:
            result["psnr"] = 20.0 * math.log10(self.target_abs_max) - 10.0 * math.log10(mse)

        result["silhouette"] = self.corr_sum / float(max(1, self.corr_count))

        for name, stat in self.per_param.items():
            n = int(stat["n"])
            if n <= 0:
                continue
            sse = float(stat["sse"])
            sae = float(stat["sae"])
            sdiff = float(stat["sdiff"])
            corr_sum = float(stat["corr_sum"])
            corr_count = int(stat["corr_count"])
            result[f"rmse_{name}"] = math.sqrt(max(0.0, sse / float(n)))
            result[f"mae_{name}"] = sae / float(n)
            result[f"bias_{name}"] = sdiff / float(n)
            result[f"silhouette_{name}"] = corr_sum / float(max(1, corr_count))

        if self.include_lat_weighted and self.weighted_den > 0.0:
            result["rmse_wb2"] = math.sqrt(max(0.0, self.weighted_sse / self.weighted_den))
            result["mae_wb2"] = self.weighted_sae / self.weighted_den
            for name, stat in self.per_param.items():
                den = float(stat["weighted_den"])
                if den <= 0.0:
                    continue
                result[f"rmse_wb2_{name}"] = math.sqrt(max(0.0, float(stat["weighted_sse"]) / den))
                result[f"mae_wb2_{name}"] = float(stat["weighted_sae"]) / den

        return result


class WeatherHermiteLightningModule(pl.LightningModule):
    """Lightning wrapper for WeatherHermiteModel."""

    def __init__(
        self,
        latent_channels: int = 64,
        cond_dim: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_latent: float = 0.0,
        lambda_reg_d: float = 0.0,
        use_latent_loss: bool = True,
        log_every: int = 100,
        channel_groups: Optional[Dict[str, List[int]]] = None,
        seen_hours: Optional[List[int]] = None,
        max_tau_hours: int = 6,
        model_type: str = "hermite",
        block_out_channels: Tuple[int, ...] = (128, 128, 256, 256),
        layers_per_block: Tuple[int, ...] = (2, 2, 2, 2),
        hidden_channels: int = 128,
        cond_mlp_dim: int = 32,
        lat_crop: int = 1,
        residual_scale_init: float = 0.10,
        residual_scale_learnable: bool = True,
        residual_clip: Optional[float] = None,
        recon_loss: str = "smooth_l1",
        smooth_l1_beta: float = 0.02,
        use_physical_scales_loss: bool = True,
        physical_loss_scales: Tuple[float, ...] = (1.0, 2.0, 2.0, 1e-3, 98.1),
        use_aurora_weights: bool = False,
        hour_loss_weights: Optional[Dict[int, float]] = None,
        channel_loss_weights: Optional[Dict[str, float]] = None,
        lambda_tau_smooth: float = 0.0,
        tau_smooth_delta: float = 1.0 / 12.0,
        tau_smooth_loss: str = "l1",
        lambda_anchor: float = 0.0,
        anchor_loss: str = "smooth_l1",
        anchor_every_n_batches: int = 1,
        lambda_residual: float = 0.0,
        residual_loss: str = "smooth_l1",
        residual_scale_floor: float = 0.0,
        direct_prediction: bool = False,
        lat_weighted_loss: bool = False,
        n_static_features: int = 0,
        n_surface_channels: int = 0,
        n_pl_channels: int = 5,
        modafno_depth: int = 8,
        modafno_num_blocks: int = 8,
        modafno_drop_rate: float = 0.0,
        sdyff_num_layers: int = 4,
        sdyff_n_modes_lat: int = 16,
        sdyff_n_modes_lon: int = 32,
        sdyff_dropout: float = 0.1,
        sdyff_drop_path: float = 0.1,
        sdyff_inference_steps: int = 5,
        sdyff_train_refine_steps: int = 1,
        fuxi_depth: int = 8,
        fuxi_num_heads: int = 8,
        fuxi_window_size_h: int = 5,
        fuxi_window_size_w: int = 9,
        fuxi_patch_size: int = 4,
        fuxi_drop_path: float = 0.1,
        fm_ode_steps: int = 1,
        warmup_epochs: int = 3,
        min_lr_ratio: float = 0.05,
        log_grad_norm_every: int = 200,
        # Novelty A: τ-conditional skip gates (DC-AE Skip only)
        tau_conditional_gates: bool = False,
        # Novelty C: spectral consistency loss (weight HF differences)
        lambda_spectral: float = 0.0,
        spectral_k_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if model_type == "hermite":
            self.model = WeatherHermiteModel(
                latent_channels=latent_channels,
                cond_dim=cond_dim,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                hidden_channels=hidden_channels,
                cond_mlp_dim=cond_mlp_dim,
                lat_crop=lat_crop,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
            )
        elif model_type == "unet_residual_linear":
            self.model = WeatherUNetResidualLinearModel(
                latent_channels=latent_channels,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                lat_crop=lat_crop,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                direct_prediction=direct_prediction,
                n_static_features=n_static_features,
            )
        elif model_type == "unet_direct":
            # Always direct, ignore --direct-prediction flag.
            self.model = WeatherUNetDirectModel(
                latent_channels=latent_channels,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                lat_crop=lat_crop,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
            )
        elif model_type == "dcae_residual_linear":
            # DC-AE has stricter constraints than UNet (group_size invariant +
            # pixel_unshuffle requires even spatial). Use wrapper's defaults
            # (3 stages, lat_crop=5 → height 176, 2 downsamples for 360 lon).
            self.model = WeatherDCAEResidualLinearModel(
                latent_channels=latent_channels,
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                direct_prediction=direct_prediction,
                n_static_features=n_static_features,
            )
        elif model_type == "dcae_swin_skip_residual_linear":
            from weather_time_interp.model.dcae_swin_skip_model import WeatherDCAESwinSkipModel
            self.model = WeatherDCAESwinSkipModel(
                latent_channels=latent_channels,
                block_out_channels=tuple(block_out_channels) if len(block_out_channels) >= 3 else (128, 256, 512),
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                direct_prediction=direct_prediction,
                n_static_features=n_static_features,
                skip_gate_init=0.0,
                skip_lateral_rank=0,
                bottleneck_window_size=(4, 9),
                bottleneck_num_heads=8,
                bottleneck_drop_path=0.1,
            )
        elif model_type == "dcae_bilinear_xattn_residual_linear":
            from weather_time_interp.model.dcae_bilinear_xattn_model import WeatherDCAEBilinearXAttnModel
            self.model = WeatherDCAEBilinearXAttnModel(
                latent_channels=latent_channels,
                block_out_channels=tuple(block_out_channels) if len(block_out_channels) >= 3 else (128, 256, 512),
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                direct_prediction=direct_prediction,
                n_static_features=n_static_features,
                skip_gate_init=0.0,
                skip_lateral_rank=0,
            )
        elif model_type == "dcae_adaln_skip_residual_linear":
            # DC-AE v9 AdaLN + U-Net-style gated zero-init skip connections.
            from weather_time_interp.model.dcae_adaln_skip_model import WeatherDCAEAdaLNSkipModel
            _sk_bo = tuple(block_out_channels) if len(block_out_channels) >= 3 else (128, 256, 512)
            _sk_ly = tuple(layers_per_block) if len(layers_per_block) >= 3 else (2, 2, 2)
            # Auto-broadcast block_type / qkv_multiscales to match the effective
            # #stages so 4-stage (or deeper) ckpts (e.g. DC-AE Skip 14M with
            # block_out=(128,128,256,256) + layers=(2,2,2) — the legacy
            # zip-truncated layout) load cleanly. We anchor on
            # min(len(block_out), len(layers)) because that is what
            # `zip(block_out, layers)` produces inside the Encoder/Decoder.
            _n_stages = min(len(_sk_bo), len(_sk_ly))
            _sk_block_type = ("ResBlock",) * (_n_stages - 1) + ("EfficientViTBlock",)
            _sk_qkv = ((),) * (_n_stages - 1) + ((5,),)
            self.model = WeatherDCAEAdaLNSkipModel(
                latent_channels=latent_channels,
                block_out_channels=_sk_bo,
                layers_per_block=_sk_ly,
                block_type=_sk_block_type,
                qkv_multiscales=_sk_qkv,
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                lat_crop=lat_crop,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                direct_prediction=direct_prediction,
                n_static_features=n_static_features,
                skip_gate_init=0.0,
                skip_lateral_rank=0,
                tau_conditional_gates=tau_conditional_gates,
            )
        elif model_type == "fuxi_swinv2_residual_linear":
            # FuXi-style SwinV2 transformer + FiLM time conditioning.
            from weather_time_interp.model.fuxi_swinv2_model import WeatherFuXiSwinV2Model
            self.model = WeatherFuXiSwinV2Model(
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                n_static_features=n_static_features,
                embed_dim=latent_channels,
                depth=int(fuxi_depth),
                num_heads=int(fuxi_num_heads),
                window_size=(int(fuxi_window_size_h), int(fuxi_window_size_w)),
                mlp_ratio=4.0,
                patch_size=int(fuxi_patch_size),
                stochastic_depth_prob=float(fuxi_drop_path),
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                direct_prediction=direct_prediction,
            )
        elif model_type == "dcae_adaln_residual_linear":
            # DC-AE v9: τ via sinusoidal+MLP → native FiLM (temb_channels), no τ-map channel.
            from weather_time_interp.model.dcae_adaln_model import WeatherDCAEAdaLNModel
            _ns_bo = tuple(block_out_channels) if len(block_out_channels) >= 3 else (128, 256, 512)
            _ns_ly = tuple(layers_per_block) if len(layers_per_block) >= 3 else (2, 2, 2)
            # Auto-broadcast block_type / qkv_multiscales to the effective
            # #stages (`zip(block_out, layers)` truncates to the shorter), so
            # 4-stage (or deeper) NoSkip ckpts — including the legacy ckpts
            # saved with mismatched block_out vs layers tuples — load cleanly.
            _ns_n_stages = min(len(_ns_bo), len(_ns_ly))
            _ns_block_type = ("ResBlock",) * (_ns_n_stages - 1) + ("EfficientViTBlock",)
            _ns_qkv = ((),) * (_ns_n_stages - 1) + ((5,),)
            self.model = WeatherDCAEAdaLNModel(
                latent_channels=latent_channels,
                block_out_channels=_ns_bo,
                layers_per_block=_ns_ly,
                block_type=_ns_block_type,
                qkv_multiscales=_ns_qkv,
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                direct_prediction=direct_prediction,
                n_static_features=n_static_features,
                lat_crop=lat_crop,
            )
        elif model_type == "weatherbridge_hybrid":
            # WeatherDCAE-Hybrid: DC-AE backbone + cross-frame attention +
            # τ-conditional adaptive skip + per-channel residual scale.
            from weather_time_interp.model.weatherbridge_hybrid_model import (
                WeatherDCAEHybridModel,
            )
            _wh_bo = tuple(block_out_channels) if len(block_out_channels) >= 3 else (96, 192, 384)
            _wh_ly = tuple(layers_per_block) if len(layers_per_block) >= 3 else (2, 2, 2)
            _wh_n_stages = min(len(_wh_bo), len(_wh_ly))
            _wh_block_type = ("ResBlock",) * (_wh_n_stages - 1) + ("EfficientViTBlock",)
            _wh_qkv = ((),) * (_wh_n_stages - 1) + ((5,),)
            self.model = WeatherDCAEHybridModel(
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                n_static_features=n_static_features,
                latent_channels=latent_channels,
                block_out_channels=_wh_bo,
                layers_per_block=_wh_ly,
                block_type=_wh_block_type,
                qkv_multiscales=_wh_qkv,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_scale_floor=residual_scale_floor,
                residual_clip=residual_clip,
                direct_prediction=direct_prediction,
                lat_crop=lat_crop,
                max_tau_hours=int(round(self.delta_t_hours)) if hasattr(self, "delta_t_hours") else 6,
            )
        elif model_type == "channels_residual_linear":
            # Multi-hour output: one forward predicts ALL hours h ∈ {1..max_tau-1}.
            # Same DC-AE backbone, but decoder out_channels = C * (max_tau - 1).
            from weather_time_interp.model.channels_baseline_model import (
                WeatherChannelsResidualLinearModel,
            )
            max_tau = int(round(self.delta_t_hours))
            self.model = WeatherChannelsResidualLinearModel(
                latent_channels=latent_channels,
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                max_tau_hours=max_tau,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                direct_prediction=direct_prediction,
                n_static_features=n_static_features,
            )
            self._is_channels_model = True
        elif model_type == "sfno_residual_linear":
            if WeatherSFNOResidualLinearModel is None:
                raise RuntimeError(
                    "SFNO model requires neuraloperator + torch-harmonics. "
                    "Install via: pip install neuraloperator torch-harmonics"
                )
            self.model = WeatherSFNOResidualLinearModel(
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                hidden_channels=latent_channels,
                lat_crop=lat_crop,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                direct_prediction=direct_prediction,
                n_static_features=n_static_features,
            )
        elif model_type == "true_unet_residual_linear":
            self.model = WeatherTrueUNetResidualLinearModel(
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                base_channels=latent_channels,
                num_levels=3,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                direct_prediction=direct_prediction,
                n_static_features=n_static_features,
            )
        elif model_type == "sdyff_dyffusion_residual_linear":
            # S-DYff v9: DYffusion-style iterative refinement (I_φ + R_θ).
            from weather_time_interp.model.sdyff_dyffusion_model import (
                WeatherSDyffusionDYffusionModel,
            )
            import os as _os
            _sdyff_nlat = int(_os.environ.get("SDYFF_NLAT", "180"))
            _sdyff_nlon = int(_os.environ.get("SDYFF_NLON", "360"))
            _sdyff_lat_crop = int(_os.environ.get("SDYFF_LAT_CROP", "1"))
            self.model = WeatherSDyffusionDYffusionModel(
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                embed_dim=latent_channels,
                num_layers=int(sdyff_num_layers),
                refiner_num_layers=max(2, int(sdyff_num_layers) // 2),
                n_modes_lat=int(sdyff_n_modes_lat),
                n_modes_lon=int(sdyff_n_modes_lon),
                dropout=float(sdyff_dropout),
                drop_path=float(sdyff_drop_path),
                n_inference_steps=int(sdyff_inference_steps),
                n_train_refine_steps=int(sdyff_train_refine_steps),
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                n_static_features=n_static_features,
                nlat=_sdyff_nlat,
                nlon=_sdyff_nlon,
                lat_crop=_sdyff_lat_crop,
            )
        elif model_type == "sdyff_residual_linear":
            if WeatherSDyffusionResidualLinearModel is None:
                raise RuntimeError(
                    "Spherical-DYffusion model requires torch-harmonics. "
                    "Install via: pip install torch-harmonics"
                )
            import os as _os
            _sdyff_nlat = int(_os.environ.get("SDYFF_NLAT", "180"))
            _sdyff_nlon = int(_os.environ.get("SDYFF_NLON", "360"))
            _sdyff_lat_crop = int(_os.environ.get("SDYFF_LAT_CROP", "1"))
            # Wrapper has its own defaults (embed=128, num_layers=6, n_modes=(32,64), nlat=180).
            self.model = WeatherSDyffusionResidualLinearModel(
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                embed_dim=latent_channels,
                num_layers=int(sdyff_num_layers),
                n_modes_lat=int(sdyff_n_modes_lat),
                n_modes_lon=int(sdyff_n_modes_lon),
                dropout=float(sdyff_dropout),
                drop_path=float(sdyff_drop_path),
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                nlat=_sdyff_nlat,
                nlon=_sdyff_nlon,
                lat_crop=_sdyff_lat_crop,
            )
        elif model_type in (
            "modafno_residual_linear",
            "modafno_official_residual_linear",
            "afno_official_residual_linear",
        ):
            from weather_time_interp.model.modafno_baseline_model import (
                WeatherModAFNOResidualLinearModel,
                WeatherModAFNOOfficialResidualLinearModel,
            )
            cls = (
                WeatherModAFNOResidualLinearModel
                if model_type == "modafno_residual_linear"
                else WeatherModAFNOOfficialResidualLinearModel
            )
            modulate = model_type != "afno_official_residual_linear"
            import os as _os
            _inp_h = int(_os.environ.get("MODAFNO_INP_H", "182"))
            _inp_w = int(_os.environ.get("MODAFNO_INP_W", "360"))
            _native_h = int(_os.environ.get("MODAFNO_NATIVE_H", "181"))
            _native_w = int(_os.environ.get("MODAFNO_NATIVE_W", "360"))
            self.model = cls(
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                embed_dim=latent_channels,
                mod_dim=64,
                depth=int(modafno_depth),
                patch_size=(2, 2),
                inp_shape=(_inp_h, _inp_w),
                native_shape=(_native_h, _native_w),
                num_blocks=int(modafno_num_blocks),
                mlp_ratio=2.0,
                drop_rate=float(modafno_drop_rate),
                modulate_filter=modulate,
                modulate_mlp=modulate,
                scale_shift_mode="complex",
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
                residual_scale_floor=residual_scale_floor,
                direct_prediction=direct_prediction,
                n_static_features=n_static_features,
            )
        elif model_type == "fm_residual_linear":
            from weather_time_interp.model.fm_baseline_model import (
                WeatherFlowMatchingResidualModel,
            )
            self.model = WeatherFlowMatchingResidualModel(
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                embed_dim=latent_channels,
                mod_dim=64,
                depth=int(modafno_depth),
                patch_size=(2, 2),
                num_blocks=int(modafno_num_blocks),
                mlp_ratio=2.0,
                drop_rate=float(modafno_drop_rate),
                modulate_filter=True,
                modulate_mlp=True,
                scale_shift_mode="complex",
                n_static_features=n_static_features,
                ode_steps=int(fm_ode_steps),
            )
        elif model_type == "koopman_linear":
            if WeatherKoopmanLinearModel is None:
                raise RuntimeError(
                    "Koopman baseline sources are not available in this checkout"
                )
            self.model = WeatherKoopmanLinearModel(
                latent_channels=latent_channels,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                in_channels=n_pl_channels + n_surface_channels,
                out_channels=n_pl_channels + n_surface_channels,
                lat_crop=lat_crop,
                residual_scale_init=residual_scale_init,
                residual_scale_learnable=residual_scale_learnable,
                residual_clip=residual_clip,
            )
        else:
            raise ValueError(f"Unknown model_type={model_type}")

        self.channel_groups = channel_groups
        self.seen_hours = sorted(dict.fromkeys(int(h) for h in (seen_hours or [])))
        self.max_tau_hours = int(max_tau_hours)
        self.hour_loss_weights = self._normalize_hour_loss_weights(hour_loss_weights)
        self.channel_loss_weights = self._normalize_channel_loss_weights(channel_loss_weights)
        # TISR is analytic (closed-form from datetime) — exclude from target loss
        # so model doesn't waste capacity learning a deterministic function.
        # Acts as INPUT signal only. Override only if user explicitly set a non-zero weight.
        if "tisr" not in self.channel_loss_weights:
            self.channel_loss_weights["tisr"] = 0.0
            print("  [tisr] automatic exclude from loss (analytic input-only)")
        # Aurora-style per-level weights (Bodnar 2024, Table 21 approximation).
        # Applied on top of normalized channels (assumes data already z-scored).
        # Toggled via --use-aurora-weights; otherwise no-op.
        if bool(getattr(self.hparams, "use_aurora_weights", False)):
            aurora_pl = {
                # var: [w@1000, w@925, w@850, w@700]
                "T": [1.0, 1.0, 1.0, 1.0],
                "U": [0.5, 0.7, 0.9, 1.2],
                "V": [0.5, 0.7, 0.9, 1.2],
                "Q": [1.5, 1.3, 1.0, 0.6],
                "Z": [1.0, 0.8, 0.5, 0.3],
            }
            levels = [1000, 925, 850, 700]
            for var, weights in aurora_pl.items():
                for lvl, w in zip(levels, weights):
                    key = f"{var}{lvl}"
                    self.channel_loss_weights.setdefault(key, float(w))
            aurora_surf = {
                "t2m": 3.0, "u10": 0.77, "v10": 0.66,
                "mslp": 1.5, "sst": 1.0, "tcc": 0.5,
            }
            for k, w in aurora_surf.items():
                self.channel_loss_weights.setdefault(k, float(w))
            print(f"  [aurora] weights applied: PL per-level + surface. tisr stays at 0.")
            print(f"  [aurora] channel_loss_weights = {self.channel_loss_weights}")

        print(f"Model type: {model_type}")
        print("WeatherHermiteModel architecture:")
        print(f"  latent_channels={latent_channels}")
        print(f"  block_out_channels={block_out_channels}")
        print(f"  layers_per_block={layers_per_block}")
        print(f"  hidden_channels={hidden_channels}")
        print(f"  cond_mlp_dim={cond_mlp_dim}")
        print(f"  lat_crop={lat_crop}")
        print(f"  residual_scale_init={residual_scale_init}")
        print(f"  residual_scale_learnable={residual_scale_learnable}")
        print(f"  residual_clip={residual_clip}")
        print(f"  recon_loss={recon_loss}")
        print(f"  use_physical_scales_loss={use_physical_scales_loss}")
        print(f"  physical_loss_scales={physical_loss_scales}")
        print(f"  hour_loss_weights={self.hour_loss_weights}")
        print(f"  channel_loss_weights={self.channel_loss_weights}")
        print(f"  lambda_tau_smooth={lambda_tau_smooth}")
        print(f"  tau_smooth_delta={tau_smooth_delta}")
        print(f"  tau_smooth_loss={tau_smooth_loss}")
        print(f"  lambda_anchor={lambda_anchor}")
        print(f"  anchor_loss={anchor_loss}")
        print(f"  anchor_every_n_batches={max(1, int(anchor_every_n_batches))}")
        print(f"  lambda_residual={lambda_residual}")
        print(f"  residual_loss={residual_loss}")
        print(f"  residual_scale_floor={residual_scale_floor}")
        print(f"  Total params: {sum(p.numel() for p in self.model.parameters()):,}")

        self._train_metrics_acc = _RunningWeatherMetrics(self.channel_groups, include_lat_weighted=False)
        self._val_metrics_acc = _RunningWeatherMetrics(self.channel_groups, include_lat_weighted=True)
        self._val_seen_metrics_acc = _RunningWeatherMetrics(self.channel_groups, include_lat_weighted=True)
        self._val_unseen_metrics_acc = _RunningWeatherMetrics(self.channel_groups, include_lat_weighted=True)
        self._val_hour_metrics_acc: Dict[int, _RunningWeatherMetrics] = {
            hour: _RunningWeatherMetrics(self.channel_groups, include_lat_weighted=True)
            for hour in range(0, self.max_tau_hours + 1)
        }

        self._val_loss_sum = 0.0
        self._val_loss_count = 0
        self._val_loss_recon_sum = 0.0
        self._val_loss_recon_count = 0
        self._val_loss_latent_sum = 0.0
        self._val_loss_latent_count = 0
        self._val_loss_reg_d_sum = 0.0
        self._val_loss_reg_d_count = 0
        self._train_batch_counter = 0
        self._batch_anchor_counter = 0

    def _normalize_hour_loss_weights(
        self,
        hour_loss_weights: Optional[Dict[int, float]],
    ) -> Dict[int, float]:
        if not hour_loss_weights:
            return {}
        out: Dict[int, float] = {}
        for hour, weight in hour_loss_weights.items():
            h = int(hour)
            if h < 0 or h > self.max_tau_hours:
                continue
            w = float(weight)
            if w <= 0.0:
                raise ValueError(f"hour_loss_weights[{h}] must be > 0, got {w}")
            out[h] = w
        return out

    @staticmethod
    def _normalize_channel_loss_weights(
        channel_loss_weights: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        if not channel_loss_weights:
            return {}
        alias = {
            "t": "temperature",
            "u": "u_component_of_wind",
            "v": "v_component_of_wind",
            "q": "specific_humidity",
            "z": "geopotential",
        }
        out: Dict[str, float] = {}
        for name, weight in channel_loss_weights.items():
            key = alias.get(str(name).strip().lower(), str(name).strip().lower())
            w = float(weight)
            if w <= 0.0:
                raise ValueError(f"channel_loss_weights['{key}'] must be > 0, got {w}")
            out[key] = w
        return out

    def _channel_loss_scales(self, pred: torch.Tensor) -> torch.Tensor:
        c = int(pred.size(1))
        scales = torch.ones(c, dtype=pred.dtype, device=pred.device)
        if not bool(self.hparams.use_physical_scales_loss):
            return scales

        base = tuple(float(v) for v in self.hparams.physical_loss_scales)
        if any(v <= 0.0 for v in base):
            raise ValueError(f"physical_loss_scales must be positive, got {base}")

        # Default order for this pipeline is T, U, V, Q, Z.
        if c == 5:
            scales[:] = torch.tensor(base, dtype=pred.dtype, device=pred.device)

        groups = self.channel_groups or {}
        alias_map = {
            "temperature": 0,
            "u_component_of_wind": 1,
            "v_component_of_wind": 2,
            "specific_humidity": 3,
            "geopotential": 4,
        }
        for name, idx in alias_map.items():
            for ch in groups.get(name, []):
                if 0 <= int(ch) < c:
                    scales[int(ch)] = float(base[idx])
        return scales

    def _channel_loss_multipliers(self, pred: torch.Tensor) -> torch.Tensor:
        c = int(pred.size(1))
        multipliers = torch.ones(c, dtype=pred.dtype, device=pred.device)
        if not self.channel_loss_weights:
            return multipliers

        groups = self.channel_groups or {}
        alias_idx = {
            "temperature": 0,
            "u_component_of_wind": 1,
            "v_component_of_wind": 2,
            "specific_humidity": 3,
            "geopotential": 4,
        }
        for name, weight in self.channel_loss_weights.items():
            idxs = groups.get(name, [])
            if idxs:
                for ch in idxs:
                    ch_i = int(ch)
                    if 0 <= ch_i < c:
                        multipliers[ch_i] *= float(weight)
                continue
            if name in alias_idx and alias_idx[name] < c:
                multipliers[alias_idx[name]] *= float(weight)
        return multipliers

    def _sample_hour_multipliers(
        self,
        tau_hour: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        stage: str,
    ) -> torch.Tensor:
        weights = torch.ones(batch_size, dtype=dtype, device=device)
        if stage != "train" or tau_hour is None or not self.hour_loss_weights:
            return weights
        tau_hour = tau_hour.detach().view(-1).to(device=device)
        for hour, weight in self.hour_loss_weights.items():
            weights = torch.where(tau_hour == int(hour), torch.tensor(float(weight), device=device, dtype=dtype), weights)
        return weights

    def forward(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau: torch.Tensor,
        cond: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ):
        # Pass static features only to models that accept them.
        if static is not None and getattr(self.model, "n_static_features", 0) > 0:
            return self.model(x0, xT, tau, cond, static=static)
        return self.model(x0, xT, tau, cond)

    def _lat_weights(self, pred: torch.Tensor) -> torch.Tensor:
        """Latitude cosine weights, normalized to mean=1. Standard for global ERA5 loss.

        Used by GraphCast / Pangu / FourCastNet / SFNO / S-DYff. Polar rows cover
        much smaller surface area than equatorial; uniform weighting over-fits poles.
        """
        H = int(pred.size(-2))
        # Cache per (H, device, dtype).
        cache_key = (H, pred.device, pred.dtype)
        cached = getattr(self, "_lat_weights_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        # ERA5 lat grid: [+90, ..., -90], H rows. Use π/2 → -π/2 range.
        lats = torch.linspace(
            math.pi / 2, -math.pi / 2, H, device=pred.device, dtype=pred.dtype
        )
        w = torch.cos(lats)
        w = w / w.mean()  # normalize so a uniform field has mean weight 1.0
        w = w.view(1, 1, H, 1)
        self._lat_weights_cache = (cache_key, w)
        return w

    def _spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L2 on 2D FFT magnitude difference, weighted by wavenumber to emphasize HF.

        Encourages the model to preserve the power spectrum of target, not just
        per-pixel MSE. Important for atmospheric fields where high-freq turbulent
        structure carries physical information but is averaged out by recon_loss.
        """
        # FFT over (lat, lon) — operate in fp32 for cuFFT compatibility.
        pred_spec = torch.fft.rfft2(pred.float(), dim=(-2, -1)).abs()
        tgt_spec = torch.fft.rfft2(target.float(), dim=(-2, -1)).abs()
        # Wavenumber-magnitude weighting (1 + α·|k|) — emphasises HF differences.
        H, Wh = pred_spec.shape[-2], pred_spec.shape[-1]
        ky = torch.fft.fftfreq(pred.shape[-2], device=pred.device).abs()  # (H,)
        kx = torch.fft.rfftfreq(pred.shape[-1], device=pred.device).abs()  # (Wh,)
        k_mag = (ky[:, None] ** 2 + kx[None, :] ** 2).sqrt()               # (H, Wh)
        alpha = float(getattr(self.hparams, "spectral_k_scale", 10.0))
        weight = 1.0 + alpha * k_mag
        return ((pred_spec - tgt_spec).pow(2) * weight[None, None]).mean()

    def _recon_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        tau_hour: Optional[torch.Tensor] = None,
        stage: str = "train",
    ) -> torch.Tensor:
        scales = self._channel_loss_scales(pred).view(1, -1, 1, 1)
        pred_norm = pred / scales
        target_norm = target / scales
        if self.hparams.recon_loss == "mse":
            point_loss = (pred_norm - target_norm).pow(2)
        elif self.hparams.recon_loss == "smooth_l1":
            point_loss = F.smooth_l1_loss(
                pred_norm,
                target_norm,
                beta=float(self.hparams.smooth_l1_beta),
                reduction="none",
            )
        else:
            raise ValueError(f"Unknown recon_loss={self.hparams.recon_loss}")

        channel_mul = self._channel_loss_multipliers(pred).view(1, -1, 1, 1)
        hour_mul = self._sample_hour_multipliers(
            tau_hour=tau_hour,
            batch_size=int(pred.size(0)),
            device=pred.device,
            dtype=pred.dtype,
            stage=stage,
        ).view(-1, 1, 1, 1)
        weights = channel_mul * hour_mul

        # Lat-cos weighting (HIGH priority fix from paper review).
        if getattr(self.hparams, "lat_weighted_loss", False):
            lat_w = self._lat_weights(pred)  # (1, 1, H, 1)
            weights = weights * lat_w

        weighted = point_loss * weights
        return weighted.sum() / weights.sum().clamp_min(1e-12)

    def _safe_check(self, t: torch.Tensor, name: str) -> None:
        if not torch.isfinite(t).all():
            raise RuntimeError(f"Non-finite tensor detected: {name}")

    def _pointwise_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        if loss_type == "mse":
            return F.mse_loss(pred, target)
        if loss_type == "l1":
            return F.l1_loss(pred, target)
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target, beta=float(self.hparams.smooth_l1_beta))
        raise ValueError(f"Unknown loss_type={loss_type}")

    @staticmethod
    def _to_float(x: torch.Tensor | float | None) -> float | None:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return float(x.detach().item())
        return float(x)

    def _rmse_summary(self, metrics: Dict[str, float]) -> Dict[str, float]:
        key_map = {
            "rmse": "rmse",
            "t": "rmse_temperature",
            "u": "rmse_u_component_of_wind",
            "v": "rmse_v_component_of_wind",
            "q": "rmse_specific_humidity",
            "z": "rmse_geopotential",
        }
        return {
            short: float(metrics[full])
            for short, full in key_map.items()
            if full in metrics
        }

    @rank_zero_only
    def _print_epoch_summary(
        self,
        stage: str,
        rmse_summary: Dict[str, float],
        loss: torch.Tensor | float | None = None,
        loss_recon: torch.Tensor | float | None = None,
        loss_latent: torch.Tensor | float | None = None,
        loss_reg_d: torch.Tensor | float | None = None,
    ) -> None:
        parts = [
            f"epoch={self.current_epoch + 1}",
            f"stage={stage}",
        ]
        loss_f = self._to_float(loss)
        if loss_f is not None:
            parts.append(f"loss={loss_f:.6f}")
        recon_f = self._to_float(loss_recon)
        if recon_f is not None:
            parts.append(f"recon={recon_f:.6f}")
        latent_f = self._to_float(loss_latent)
        if latent_f is not None:
            parts.append(f"latent={latent_f:.6f}")
        reg_f = self._to_float(loss_reg_d)
        if reg_f is not None:
            parts.append(f"reg_d={reg_f:.6f}")
        if rmse_summary:
            parts.append(
                " ".join(
                    [
                        f"RMSE={rmse_summary.get('rmse', float('nan')):.6f}",
                        f"T={rmse_summary.get('t', float('nan')):.6f}",
                        f"U={rmse_summary.get('u', float('nan')):.6f}",
                        f"V={rmse_summary.get('v', float('nan')):.6f}",
                        f"Q={rmse_summary.get('q', float('nan')):.6f}",
                        f"Z={rmse_summary.get('z', float('nan')):.6f}",
                    ]
                )
            )
        print("[epoch_summary] " + " | ".join(parts))

    def _flatten_multi_tau_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x0 = batch["x0"]
        xT = batch["xT"]
        target = batch["target"]
        tau = batch["tau"].float()
        cond = batch["cond"].float()
        tau_hour = batch.get("tau_hour")

        if target.ndim == 5:
            bsz, n_tau = target.shape[:2]
            x0 = x0.unsqueeze(1).expand(-1, n_tau, -1, -1, -1).reshape(bsz * n_tau, *x0.shape[1:])
            xT = xT.unsqueeze(1).expand(-1, n_tau, -1, -1, -1).reshape(bsz * n_tau, *xT.shape[1:])
            target = target.reshape(bsz * n_tau, *target.shape[2:])
            tau = tau.reshape(bsz * n_tau, 1)
            cond = cond.unsqueeze(1).expand(-1, n_tau, -1).reshape(bsz * n_tau, cond.shape[-1])
            if tau_hour is not None:
                tau_hour = tau_hour.reshape(bsz * n_tau)
        else:
            tau = tau.view(tau.size(0), -1)
            if tau.size(1) != 1:
                tau = tau[:, :1]
            if tau_hour is not None:
                tau_hour = tau_hour.view(-1)
        return x0, xT, target, tau, cond, tau_hour

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        if stage == "train" and "target" in batch and "tau" in batch:
            target_full = batch["target"]
            tau_full = batch["tau"]
            if target_full.ndim == 5 and tau_full.ndim >= 2:
                n_tau = int(target_full.shape[1])
                if n_tau > 0:
                    # Sample 1-2 tau values per iteration (or fewer if unavailable).
                    k = int(torch.randint(1, min(2, n_tau) + 1, (1,), device=tau_full.device).item())
                    sel = torch.randperm(n_tau, device=tau_full.device)[:k]
                    sel, _ = torch.sort(sel)
                    batch = dict(batch)
                    batch["target"] = target_full.index_select(1, sel)
                    batch["tau"] = tau_full.index_select(1, sel)
                    if "tau_hour" in batch:
                        batch["tau_hour"] = batch["tau_hour"].index_select(1, sel)

        x0, xT, target, tau, cond, tau_hour = self._flatten_multi_tau_batch(batch)
        static = batch.get("static") if isinstance(batch, dict) else None
        # If multi-tau flattened x0 to B*n_tau but static is still B (or 3,H,W),
        # let model.forward broadcast — it handles 3D and B<x0_size cases.
        # But trainer.forward needs static to broadcast properly. Pass as-is.

        x_hat, aux = self(x0, xT, tau, cond, static=static)
        self._safe_check(x_hat, f"{stage}/x_hat")

        loss_recon = self._recon_loss(x_hat, target, tau_hour=tau_hour, stage=stage)
        loss = loss_recon
        log_dict = {f"{stage}/loss_recon": loss_recon}

        # Anchor loss: penalize non-zero residual at tau=0 and tau=1 (endpoints).
        # Bilinear is exact at endpoints, so the model should learn zero correction there.
        lambda_anchor = float(self.hparams.lambda_anchor)
        if stage == "train" and lambda_anchor > 0.0:
            anchor_every = max(1, int(self.hparams.anchor_every_n_batches))
            if self._batch_anchor_counter % anchor_every == 0:
                B = x0.size(0)
                device = x0.device
                tau_zero = torch.zeros(B, 1, device=device, dtype=tau.dtype)
                tau_one = torch.ones(B, 1, device=device, dtype=tau.dtype)
                cond_anchor = cond[:B]
                x_hat_0, _ = self(x0, xT, tau_zero, cond_anchor, static=static)
                x_hat_1, _ = self(x0, xT, tau_one, cond_anchor, static=static)
                loss_anchor = 0.5 * (
                    self._pointwise_loss(x_hat_0, x0, self.hparams.anchor_loss)
                    + self._pointwise_loss(x_hat_1, xT, self.hparams.anchor_loss)
                )
                loss = loss + lambda_anchor * loss_anchor
                log_dict[f"{stage}/loss_anchor"] = loss_anchor
            self._batch_anchor_counter += 1

        # Tau-smooth loss: penalize large change between adjacent tau values.
        lambda_tau_smooth = float(self.hparams.lambda_tau_smooth)
        if stage == "train" and lambda_tau_smooth > 0.0:
            eps = float(self.hparams.tau_smooth_delta)
            tau_eps = torch.clamp(tau + eps, 0.0, 1.0)
            x_hat_eps, _ = self(x0, xT, tau_eps, cond, static=static)
            loss_smooth = self._pointwise_loss(x_hat, x_hat_eps.detach(), self.hparams.tau_smooth_loss)
            loss = loss + lambda_tau_smooth * loss_smooth
            log_dict[f"{stage}/loss_tau_smooth"] = loss_smooth

        # Direct residual supervision: train decoder to predict (target - bilinear) directly.
        # We compute loss on (bilinear + decoder_out) vs target — equivalent to predicting
        # the residual directly, but uses the same physical-scale weighting as recon_loss
        # so lambda_residual=1.0 puts equal weight on raw decoder output vs full model.
        # This bypasses the residual_scale chain rule that drives scale -> 0 when decoder
        # output starts noisy, letting decoder learn useful residuals before scale collapses.
        lambda_residual = float(self.hparams.lambda_residual)
        if stage == "train" and lambda_residual > 0.0 and "decoder_out" in aux:
            decoder_out = aux["decoder_out"]
            x_bilin = aux["x_bilinear"]
            x_pred_unit_scale = x_bilin + decoder_out  # what model would predict if residual_scale=1
            loss_residual = self._recon_loss(
                x_pred_unit_scale, target, tau_hour=tau_hour, stage=stage
            )
            loss = loss + lambda_residual * loss_residual
            log_dict[f"{stage}/loss_residual"] = loss_residual

        # Novelty C: spectral consistency loss — penalize differences in 2D FFT
        # magnitude, weighted by wavenumber to emphasize high-frequency preservation.
        lambda_spectral = float(getattr(self.hparams, "lambda_spectral", 0.0))
        if stage == "train" and lambda_spectral > 0.0:
            loss_spectral = self._spectral_loss(x_hat, target)
            loss = loss + lambda_spectral * loss_spectral
            log_dict[f"{stage}/loss_spectral"] = loss_spectral

        self._safe_check(loss, f"{stage}/loss")
        log_dict[f"{stage}/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=(stage == "train"),
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        if stage == "train":
            self._train_metrics_acc.update(x_hat, target)
            if self.global_step % self.hparams.log_every == 0:
                step_metrics = WeatherMetrics.all(
                    x_hat.detach(),
                    target.detach(),
                    self.channel_groups,
                )
                for key, value in step_metrics.items():
                    self.log(
                        f"train/{key}",
                        value,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                        sync_dist=False,
                    )
                residual_scale = aux.get("residual_scale")
                if residual_scale is not None:
                    self.log(
                        "train/residual_scale",
                        residual_scale.mean(),
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                        sync_dist=False,
                    )
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.global_step % max(1, int(self.hparams.log_grad_norm_every)) != 0:
            return
        total = torch.zeros((), device=self.device)
        for p in self.parameters():
            if p.grad is not None:
                total = total + p.grad.detach().pow(2).sum()
        self.log(
            "train/grad_norm_2",
            torch.sqrt(total),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )

    def on_train_epoch_start(self) -> None:
        self._train_metrics_acc.reset()
        self._train_batch_counter = 0
        self._batch_anchor_counter = 0

    def on_train_epoch_end(self) -> None:
        train_metrics = self._train_metrics_acc.compute()
        if not train_metrics:
            return

        for key, value in train_metrics.items():
            self.log(f"train/{key}", value, on_epoch=True, prog_bar=False, sync_dist=False)
        rmse_summary = self._rmse_summary(train_metrics)
        self._print_epoch_summary(stage="train", rmse_summary=rmse_summary)

    def validation_step(self, batch: Dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        x0, xT, target, tau, cond, tau_hour_batch = self._flatten_multi_tau_batch(batch)
        static = batch.get("static") if isinstance(batch, dict) else None

        x_hat, aux = self(x0, xT, tau, cond, static=static)
        self._safe_check(x_hat, "test/x_hat")

        loss_recon = self._recon_loss(x_hat, target, tau_hour=tau_hour_batch, stage="val")
        loss = loss_recon

        # Keep validation objective strictly reconstruction-only.
        loss_latent = None
        loss_reg_d = None

        self._safe_check(loss, "test/loss")

        self._val_loss_sum += float(loss.detach().item())
        self._val_loss_count += 1
        self._val_loss_recon_sum += float(loss_recon.detach().item())
        self._val_loss_recon_count += 1
        if loss_latent is not None:
            self._val_loss_latent_sum += float(loss_latent.detach().item())
            self._val_loss_latent_count += 1
        if loss_reg_d is not None:
            self._val_loss_reg_d_sum += float(loss_reg_d.detach().item())
            self._val_loss_reg_d_count += 1

        pred_cpu = x_hat.detach().float().cpu()
        target_cpu = target.detach().float().cpu()
        tau_cpu = tau.detach().float().cpu().view(-1)

        self._val_metrics_acc.update(pred_cpu, target_cpu)

        if tau_hour_batch is not None:
            tau_hours = tau_hour_batch.detach().view(-1).long().cpu().clamp(0, self.max_tau_hours)
        else:
            tau_hours = (tau_cpu * float(self.max_tau_hours)).round().clamp(0, self.max_tau_hours).long()

        for hour in range(0, self.max_tau_hours + 1):
            mask = tau_hours == hour
            if bool(mask.any()):
                self._val_hour_metrics_acc[hour].update(pred_cpu[mask], target_cpu[mask])

        if self.seen_hours:
            seen_mask = torch.zeros_like(tau_hours, dtype=torch.bool)
            for hour in self.seen_hours:
                seen_mask |= tau_hours == int(hour)
        else:
            seen_mask = torch.ones_like(tau_hours, dtype=torch.bool)
        unseen_mask = ~seen_mask
        if bool(seen_mask.any()):
            self._val_seen_metrics_acc.update(pred_cpu[seen_mask], target_cpu[seen_mask])
        if bool(unseen_mask.any()):
            self._val_unseen_metrics_acc.update(pred_cpu[unseen_mask], target_cpu[unseen_mask])

        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_metrics_acc.reset()
        self._val_seen_metrics_acc.reset()
        self._val_unseen_metrics_acc.reset()
        self._val_hour_metrics_acc = {
            hour: _RunningWeatherMetrics(self.channel_groups, include_lat_weighted=True)
            for hour in range(0, self.max_tau_hours + 1)
        }

        self._val_loss_sum = 0.0
        self._val_loss_count = 0
        self._val_loss_recon_sum = 0.0
        self._val_loss_recon_count = 0
        self._val_loss_latent_sum = 0.0
        self._val_loss_latent_count = 0
        self._val_loss_reg_d_sum = 0.0
        self._val_loss_reg_d_count = 0

    def on_validation_epoch_end(self) -> None:
        if self._val_loss_count == 0:
            return

        avg_loss = self._val_loss_sum / float(self._val_loss_count)
        avg_recon = self._val_loss_recon_sum / float(max(1, self._val_loss_recon_count))
        self.log("test/loss", avg_loss, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("test/loss_recon", avg_recon, on_epoch=True, prog_bar=False, sync_dist=False)

        avg_latent = None
        avg_reg_d = None
        if self._val_loss_latent_count > 0:
            avg_latent = self._val_loss_latent_sum / float(self._val_loss_latent_count)
            self.log("test/loss_latent", avg_latent, on_epoch=True, prog_bar=False, sync_dist=False)
        if self._val_loss_reg_d_count > 0:
            avg_reg_d = self._val_loss_reg_d_sum / float(self._val_loss_reg_d_count)
            self.log("test/loss_reg_d", avg_reg_d, on_epoch=True, prog_bar=False, sync_dist=False)

        test_metrics = self._val_metrics_acc.compute()
        progress_keys = {
            "rmse",
            "rmse_temperature",
            "rmse_u_component_of_wind",
            "rmse_v_component_of_wind",
            "rmse_specific_humidity",
            "rmse_geopotential",
            "rmse_wb2",
        }
        for key, value in test_metrics.items():
            self.log(
                f"test/{key}",
                value,
                on_epoch=True,
                prog_bar=(key in progress_keys),
                sync_dist=False,
            )

        for hour in range(0, self.max_tau_hours + 1):
            hour_metrics = self._val_hour_metrics_acc[hour].compute()
            for key, value in hour_metrics.items():
                self.log(
                    f"test/hour_{hour}/{key}",
                    value,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=False,
                )

        seen_metrics = self._val_seen_metrics_acc.compute()
        for key, value in seen_metrics.items():
            self.log(
                f"test/seen/{key}",
                value,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )

        unseen_metrics = self._val_unseen_metrics_acc.compute()
        for key, value in unseen_metrics.items():
            self.log(
                f"test/unseen/{key}",
                value,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )

        rmse_summary = self._rmse_summary(test_metrics)
        self._print_epoch_summary(
            stage="test",
            rmse_summary=rmse_summary,
            loss=avg_loss,
            loss_recon=avg_recon,
            loss_latent=avg_latent,
            loss_reg_d=avg_reg_d,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        warmup_epochs = max(0, int(self.hparams.warmup_epochs))
        min_lr_ratio = float(self.hparams.min_lr_ratio)

        def lr_lambda(epoch: int) -> float:
            if self.trainer.max_epochs <= 1:
                return 1.0
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return max(1e-3, float(epoch + 1) / float(warmup_epochs))

            cosine_steps = max(1, self.trainer.max_epochs - warmup_epochs - 1)
            progress = float(max(0, epoch - warmup_epochs)) / float(cosine_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }

"""
Weather metrics: RMSE, MAE, bias, PSNR, and correlation ("silhouette" name kept for compatibility).
Correlation is mean Pearson correlation over spatial dims per sample/channel.
"""
from typing import Dict, List, Optional

import torch


class WeatherMetrics:
    """RMSE, MAE, bias, PSNR, and correlation (silhouette) globally and per variable."""

    @staticmethod
    def _latitude_weights(height: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # WeatherBench2-style latitude weighting: w(lat) = cos(lat).
        lats = torch.linspace(-90.0, 90.0, steps=height, device=device, dtype=dtype)
        return torch.cos(torch.deg2rad(lats)).clamp_min(0.0)

    @classmethod
    def weighted_rmse(cls, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Latitude-weighted RMSE with cosine(latitude) weights."""
        err2 = (pred - target).pow(2)
        h = err2.size(-2)
        w = cls._latitude_weights(h, device=err2.device, dtype=err2.dtype).view(1, 1, h, 1)
        num = (err2 * w).sum()
        den = w.sum() * err2.size(0) * err2.size(1) * err2.size(-1)
        return torch.sqrt(num / den.clamp_min(1e-12)).item()

    @classmethod
    def weighted_mae(cls, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Latitude-weighted MAE with cosine(latitude) weights."""
        err = (pred - target).abs()
        h = err.size(-2)
        w = cls._latitude_weights(h, device=err.device, dtype=err.dtype).view(1, 1, h, 1)
        num = (err * w).sum()
        den = w.sum() * err.size(0) * err.size(1) * err.size(-1)
        return (num / den.clamp_min(1e-12)).item()

    @classmethod
    def weighted_rmse_per_param(
        cls,
        pred: torch.Tensor,
        target: torch.Tensor,
        channel_groups: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, float]:
        if not channel_groups:
            return {}
        pred = pred.detach()
        target = target.detach()
        out: Dict[str, float] = {}
        for name, idxs in channel_groups.items():
            out[name] = cls.weighted_rmse(pred[:, idxs], target[:, idxs])
        return out

    @classmethod
    def weighted_mae_per_param(
        cls,
        pred: torch.Tensor,
        target: torch.Tensor,
        channel_groups: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, float]:
        if not channel_groups:
            return {}
        pred = pred.detach()
        target = target.detach()
        out: Dict[str, float] = {}
        for name, idxs in channel_groups.items():
            out[name] = cls.weighted_mae(pred[:, idxs], target[:, idxs])
        return out

    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Global RMSE over all channels and pixels."""
        return torch.sqrt(((pred - target) ** 2).mean()).item()

    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Global MAE over all channels and pixels."""
        return (pred - target).abs().mean().item()

    @staticmethod
    def bias(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Global bias: mean(pred - target)."""
        return (pred - target).mean().item()

    @staticmethod
    def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Peak Signal-to-Noise Ratio using target max as dynamic range."""
        mse = ((pred - target) ** 2).mean()
        if mse.item() == 0:
            return 100.0
        max_val = target.abs().max()
        if max_val.item() < 1e-8:
            return float('nan')  # Все значения target ~ 0
        psnr_val = 20.0 * torch.log10(max_val) - 10.0 * torch.log10(mse)
        return psnr_val.item()

    @staticmethod
    def silhouette(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Correlation: mean Pearson correlation between pred and target over spatial dims (H, W)
        for each sample and channel. Value in [-1, 1], higher is better.
        """
        pred_flat = pred.flatten(2)
        target_flat = target.flatten(2)
        pred_centered = pred_flat - pred_flat.mean(dim=2, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=2, keepdim=True)
        num = (pred_centered * target_centered).sum(dim=2)
        den = (pred_centered.pow(2).sum(dim=2).sqrt() * target_centered.pow(2).sum(dim=2).sqrt()).clamp(min=1e-8)
        corr = (num / den).mean()
        # Проверка на NaN (может возникнуть при constant predictions)
        result = corr.item()
        if not torch.isfinite(corr):
            return 0.0  # Возвращаем 0 вместо NaN
        return result

    @staticmethod
    def rmse_per_param(
        pred: torch.Tensor,
        target: torch.Tensor,
        channel_groups: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, float]:
        """RMSE per variable based on channel groups."""
        if not channel_groups:
            return {}
        pred = pred.detach()
        target = target.detach()
        out: Dict[str, float] = {}
        for name, idxs in channel_groups.items():
            p = pred[:, idxs]
            t = target[:, idxs]
            out[name] = torch.sqrt(((p - t) ** 2).mean()).item()
        return out

    @staticmethod
    def mae_per_param(
        pred: torch.Tensor,
        target: torch.Tensor,
        channel_groups: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, float]:
        """MAE per variable based on channel groups."""
        if not channel_groups:
            return {}
        pred = pred.detach()
        target = target.detach()
        out: Dict[str, float] = {}
        for name, idxs in channel_groups.items():
            p = pred[:, idxs]
            t = target[:, idxs]
            out[name] = (p - t).abs().mean().item()
        return out

    @staticmethod
    def bias_per_param(
        pred: torch.Tensor,
        target: torch.Tensor,
        channel_groups: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, float]:
        """Bias per variable based on channel groups."""
        if not channel_groups:
            return {}
        pred = pred.detach()
        target = target.detach()
        out: Dict[str, float] = {}
        for name, idxs in channel_groups.items():
            p = pred[:, idxs]
            t = target[:, idxs]
            out[name] = (p - t).mean().item()
        return out

    @staticmethod
    def silhouette_per_param(
        pred: torch.Tensor,
        target: torch.Tensor,
        channel_groups: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, float]:
        """Correlation per variable based on channel groups."""
        if not channel_groups:
            return {}
        pred = pred.detach()
        target = target.detach()
        out: Dict[str, float] = {}
        for name, idxs in channel_groups.items():
            p = pred[:, idxs].flatten(2)
            t = target[:, idxs].flatten(2)
            p_c = p - p.mean(dim=2, keepdim=True)
            t_c = t - t.mean(dim=2, keepdim=True)
            num = (p_c * t_c).sum(dim=2)
            den = (p_c.pow(2).sum(dim=2).sqrt() * t_c.pow(2).sum(dim=2).sqrt()).clamp(min=1e-8)
            out[name] = (num / den).mean().item()
        return out

    @classmethod
    def all(
        cls,
        pred: torch.Tensor,
        target: torch.Tensor,
        channel_groups: Optional[Dict[str, List[int]]] = None,
        include_lat_weighted: bool = False,
        exclude_channels: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """All metrics (global + per variable).

        exclude_channels: list of channel-group names whose indices are skipped
            from OVERALL metrics (e.g. ["tisr"] for analytic-input-only).
            Per-channel metrics (rmse_temperature etc) remain inspectable.
        """
        # Build keep-mask if exclude_channels given
        pred_overall, target_overall = pred, target
        if exclude_channels and channel_groups:
            excluded_idx = set()
            for name in exclude_channels:
                for ch in channel_groups.get(name, []):
                    excluded_idx.add(int(ch))
            if excluded_idx:
                C = int(pred.size(1))
                keep = [i for i in range(C) if i not in excluded_idx]
                if keep:
                    keep_t = torch.tensor(keep, device=pred.device, dtype=torch.long)
                    pred_overall = pred.index_select(1, keep_t)
                    target_overall = target.index_select(1, keep_t)
        result: Dict[str, float] = {
            "rmse": cls.rmse(pred_overall, target_overall),
            "mae": cls.mae(pred_overall, target_overall),
            "bias": cls.bias(pred_overall, target_overall),
            "psnr": cls.psnr(pred_overall, target_overall),
            "silhouette": cls.silhouette(pred_overall, target_overall),
        }
        for k, v in cls.rmse_per_param(pred, target, channel_groups).items():
            result[f"rmse_{k}"] = v
        for k, v in cls.mae_per_param(pred, target, channel_groups).items():
            result[f"mae_{k}"] = v
        for k, v in cls.bias_per_param(pred, target, channel_groups).items():
            result[f"bias_{k}"] = v
        for k, v in cls.silhouette_per_param(pred, target, channel_groups).items():
            result[f"silhouette_{k}"] = v

        if include_lat_weighted:
            result["rmse_wb2"] = cls.weighted_rmse(pred, target)
            result["mae_wb2"] = cls.weighted_mae(pred, target)
            for k, v in cls.weighted_rmse_per_param(pred, target, channel_groups).items():
                result[f"rmse_wb2_{k}"] = v
            for k, v in cls.weighted_mae_per_param(pred, target, channel_groups).items():
                result[f"mae_wb2_{k}"] = v

        return result

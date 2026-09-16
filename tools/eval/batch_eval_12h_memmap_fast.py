#!/usr/bin/env python3
"""Fast journal evaluator for the model-versus-linear comparison.

This wrapper preserves the canonical evaluator's model loading, dataset,
metric definitions, and artifact schema. It omits the unused Catmull--Rom
baseline and batches GPU-to-CPU reductions instead of synchronising once per
sample, channel, and metric.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import torch

from tools.eval import batch_eval_12h_memmap as base


def grouped_channel_sums(
    values: np.ndarray,
    tau_values: np.ndarray,
    valid_taus: set[int],
) -> Dict[int, np.ndarray]:
    """Sum ``(sample, channel)`` values by query hour on the CPU."""
    if values.ndim != 2 or tau_values.shape != (values.shape[0],):
        raise ValueError("metric values and tau vector have incompatible shapes")
    return {
        tau: values[tau_values == tau].sum(axis=0, dtype=np.float64)
        for tau in sorted(valid_taus)
        if np.any(tau_values == tau)
    }


def batch4_reference_order(
    years: np.ndarray,
    starts: np.ndarray,
    taus: np.ndarray,
) -> np.ndarray:
    """Return the historical batch-4 evaluator order for window rows."""
    if not (years.ndim == starts.ndim == taus.ndim == 1):
        raise ValueError("window index arrays must be one-dimensional")
    if not (years.size == starts.size == taus.size):
        raise ValueError("window index arrays must have equal lengths")

    anchor_rank: Dict[tuple[int, int], int] = {}
    for year, start in zip(years.tolist(), starts.tolist(), strict=True):
        anchor = (int(year), int(start))
        if anchor not in anchor_rank:
            anchor_rank[anchor] = len(anchor_rank)
    tau_rank = {
        int(tau): rank for rank, tau in enumerate(sorted(set(taus.tolist())))
    }
    keys = []
    for row, (year, start, tau) in enumerate(
        zip(years.tolist(), starts.tolist(), taus.tolist(), strict=True)
    ):
        rank = anchor_rank[(int(year), int(start))]
        keys.append((rank // 4, tau_rank[int(tau)], rank % 4, row))
    return np.asarray([key[3] for key in sorted(keys)], dtype=np.int64)


class FastBatchModelRunner12h(base.BatchModelRunner12h):
    """Canonical runner with batched reductions and no unused cubic method."""

    def method_names(self) -> List[str]:
        return ["model", "bilinear"]

    def predictions_for_metrics(self, x0, xT, tau_h, cond, static):
        prediction = self._forward_model(
            x0=x0, xT=xT, tau_h=tau_h, cond=cond, static=static,
        )
        return prediction, base._bilinear_2anchor(x0, xT, tau_h)

    def _canonicalize_window_order(self) -> None:
        if getattr(self, "_window_order_is_canonical", False):
            return
        order = batch4_reference_order(
            np.asarray(self._window_year),
            np.asarray(self._window_t0),
            np.asarray(self._window_tau),
        )

        def reordered(values: list) -> list:
            return [values[int(index)] for index in order]

        self._window_year = reordered(self._window_year)
        self._window_t0 = reordered(self._window_t0)
        self._window_tau = reordered(self._window_tau)
        for method in self.method_names():
            self._window_mse_norm[method] = reordered(
                self._window_mse_norm[method]
            )
            self._window_acc[method] = reordered(self._window_acc[method])
        self._window_order_is_canonical = True

    def _finalize_payload(self) -> Dict[str, object]:
        if self.save_window_metrics:
            self._canonicalize_window_order()
        return super()._finalize_payload()

    def _process_one_batch(
        self,
        *,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        wrapped_ds: base.ERA5WeatherHermiteDataset,
        base_ds: base.ERA5MemmapDataset,
        grouped_idx: Optional[List[List[int]]],
        acc_idx_t: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        cond_value: float,
        max_tau: float,
    ) -> None:
        if self.save_physical_metrics or self.save_temporal_metrics:
            raise ValueError(
                "fast journal evaluation supports RMSE, ACC, and window metrics only"
            )

        device = self.device
        x0 = batch["x0"].to(device, non_blocking=True)
        xT = batch["xT"].to(device, non_blocking=True)
        tau_hour_all = batch["tau_hour"].long()
        target_all = batch["target"].to(device, non_blocking=True)
        static = batch.get("static")
        if static is not None:
            static = static.to(device, non_blocking=True)

        model_x0 = x0
        model_xT = xT
        if self.input_n_channels is not None:
            model_x0 = x0[:, : self.input_n_channels].contiguous()
            model_xT = xT[:, : self.input_n_channels].contiguous()
        if self.keep_n_channels is not None and x0.size(1) > self.keep_n_channels:
            n = self.keep_n_channels
            x0 = x0[:, :n].contiguous()
            xT = xT[:, :n].contiguous()
            target_all = target_all[:, :, :n].contiguous()
            if self.input_n_channels is None:
                model_x0 = x0
                model_xT = xT

        batch_size = x0.size(0)
        cond = torch.full(
            (batch_size,), cond_value, device=device, dtype=torch.float32
        )
        channel_names = list(self.channel_names)
        if self.keep_n_channels is not None:
            channel_names = channel_names[: self.keep_n_channels]
            mu = mu[:, : self.keep_n_channels]
            sigma = sigma[:, : self.keep_n_channels]

        w_lat = self.w_lat
        spatial_scale = 1.0 / float(self.W) if self.proper_rmse else 1.0
        method_order = self.method_names()
        valid_taus = set(self._n_per_tau)

        for h_idx in range(tau_hour_all.size(1)):
            tau_hour_h = tau_hour_all[:, h_idx, 0]
            tau_values = tau_hour_h.numpy().astype(np.int16, copy=False)
            target_h = target_all[:, h_idx]
            tau_h = tau_hour_h.float().to(device) / max_tau

            predictions = self.predictions_for_metrics(
                x0=model_x0,
                xT=model_xT,
                tau_h=tau_h,
                cond=cond,
                static=static,
            )
            if self.keep_n_channels is not None:
                predictions = tuple(
                    prediction[:, : self.keep_n_channels].contiguous()
                    for prediction in predictions
                )
            norm_errors_t = torch.stack(
                [
                    (((prediction - target_h).square() * w_lat).sum(dim=(-2, -1)))
                    * spatial_scale
                    for prediction in predictions
                ]
            )
            target_phys = target_h * sigma + mu
            predictions_phys = tuple(
                prediction * sigma + mu for prediction in predictions
            )
            phys_errors_t = torch.stack(
                [
                    (((prediction - target_phys).square() * w_lat).sum(dim=(-2, -1)))
                    * spatial_scale
                    for prediction in predictions_phys
                ]
            )

            sample_meta = []
            for index, tau in enumerate(tau_values.tolist()):
                if tau not in valid_taus:
                    continue
                wrapped_index = batch_idx * self.loader.batch_size + index
                if wrapped_index >= len(wrapped_ds):
                    break
                year, t0 = base._extract_year_t0(
                    wrapped_index=wrapped_index,
                    grouped_idx=grouped_idx,
                    base_ds=base_ds,
                )
                sample_meta.append((index, int(tau), year, t0))

            if len(sample_meta) != batch_size:
                raise ValueError("fast evaluator requires every batch sample to be valid")

            window_acc_t = torch.full(
                (len(method_order), batch_size, len(self.acc_indices)),
                float("nan"),
                dtype=torch.float32,
                device=device,
            )
            if self.climatology is not None:
                climatology = torch.stack(
                    [
                        self.climatology.lookup(
                            (base_ds.time_starts[year] + timedelta(hours=t0 + tau))
                            .timetuple()
                            .tm_yday,
                            float(
                                (
                                    base_ds.time_starts[year]
                                    + timedelta(hours=t0 + tau)
                                ).hour
                            ),
                            year,
                        )
                        for _, tau, year, t0 in sample_meta
                    ]
                ).to(device)
                target_anomaly = target_phys.index_select(1, acc_idx_t) - climatology
                prediction_anomalies = tuple(
                    prediction.index_select(1, acc_idx_t) - climatology
                    for prediction in predictions_phys
                )
                target_energy = (
                    w_lat * target_anomaly.square()
                ).sum(dim=(-2, -1))
                for method_index, (method, anomaly) in enumerate(
                    zip(method_order, prediction_anomalies, strict=True)
                ):
                    cross = (w_lat * anomaly * target_anomaly).sum(dim=(-2, -1))
                    prediction_energy = (
                        w_lat * anomaly.square()
                    ).sum(dim=(-2, -1))
                    window_acc_t[method_index] = cross / torch.sqrt(
                        prediction_energy * target_energy + 1e-12
                    )
                    for tau in sorted(set(tau_values.tolist()) & valid_taus):
                        mask = torch.from_numpy(tau_values == tau).to(device)
                        self._acc_sxy[method][tau] += (
                            cross[mask].double().sum(dim=0)
                        )
                        self._acc_sxx[method][tau] += (
                            prediction_energy[mask].double().sum(dim=0)
                        )
                        self._acc_syy[method][tau] += (
                            target_energy[mask].double().sum(dim=0)
                        )

            # Three transfers replace hundreds of scalar CUDA synchronisations.
            norm_errors = norm_errors_t.float().cpu().numpy()
            phys_errors = phys_errors_t.float().cpu().numpy()
            window_acc = window_acc_t.cpu().numpy()

            for method_index, method in enumerate(method_order):
                norm_sums = grouped_channel_sums(
                    norm_errors[method_index], tau_values, valid_taus
                )
                phys_sums = grouped_channel_sums(
                    phys_errors[method_index], tau_values, valid_taus
                )
                for tau, sums in norm_sums.items():
                    self._rmse_norm_sum_sq[method][tau].update(
                        {
                            name: self._rmse_norm_sum_sq[method][tau].get(name, 0.0)
                            + float(value)
                            for name, value in zip(channel_names, sums, strict=True)
                        }
                    )
                for tau, sums in phys_sums.items():
                    self._rmse_phys_sum_sq[method][tau].update(
                        {
                            name: self._rmse_phys_sum_sq[method][tau].get(name, 0.0)
                            + float(value)
                            for name, value in zip(channel_names, sums, strict=True)
                        }
                    )

            for index, tau, year, t0 in sample_meta:
                self._n_per_tau[tau] += 1
                if self.save_window_metrics:
                    self._window_year.append(year)
                    self._window_t0.append(t0)
                    self._window_tau.append(tau)
                    for method_index, method in enumerate(method_order):
                        self._window_mse_norm[method].append(
                            norm_errors[method_index, index]
                        )
                        self._window_acc[method].append(
                            window_acc[method_index, index]
                        )

        if batch_idx % 25 == 0:
            print(f"  batch {batch_idx}/{len(self.loader)}")


def main() -> None:
    base.BatchModelRunner12h = FastBatchModelRunner12h
    base.main()


if __name__ == "__main__":
    main()

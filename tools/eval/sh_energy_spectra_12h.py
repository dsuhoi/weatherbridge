"""Spherical-harmonic diagnostics on the ERA5 cell-centred latitude grid.

For each ckpt and each requested τ ∈ {2, 3, 5, 8}, runs the model on the
2020 test set, computes the SHT-based angular power spectrum
``E(ℓ) = sum_m |a_{ℓm}|^2`` per channel for both predictions and ground
truth, and writes a ``.npz`` payload to
``metrics/sh_spectra_12h_ep10/<model_name>_tau<τ>.npz``.

The scalar transform uses the actual latitude midpoints implied by the source
2x2 block average and spherical strip-area quadrature. Paired U/V fields also
receive a vector SHT, which
separates spheroidal and toroidal wind energy without treating coordinate
components as independent scalar fields. The journal protocol retains degrees
0 through 180 and reports the diagnostic band 80 through 180.

This script reuses ``ERA5MemmapDataset`` + ``ERA5WeatherHermiteDataset``
identical to ``batch_eval_12h_memmap.py``. It supports WeatherDCAE,
WeatherBridge, FuXi, ModAFNO, S-DYff, ATM-VFI and the analytic temporal
linear baseline under the common 24-channel protocol.

Output ``.npz`` schema::

    {
      "ell":             (lmax+1,) integer ℓ axis,
      "channel_names":   list of 5 channel strings,
      "pred_El":         (C, lmax+1) E(ℓ) for predictions,
      "gt_El":           (C, lmax+1) E(ℓ) for ground truth,
      "n_samples":       int (windows × tau-samples used),
      "tau":             int,
      "H":               int (latitude grid size),
      "W":               int,
      "lmax":            int,
      "model_name":      str,
      "ckpt":            str,
    }

Usage::

    python tools/eval/sh_energy_spectra_12h.py \\
        --memmap-dir /tmp/wb2_0p5_cache \\
        --ckpt /workspace/code/wti/logs/exp_12h_oddskip_dcae_noskip_3yr_fibo/epoch=9-step=10940.ckpt \\
        --model-name DC-AE_NoSkip_3yr_12h_fibo \\
        --model-kind hermite \\
        --keep-n-channels 24 \\
        --taus 2,3,5,8 \\
        --out-dir metrics/sh_spectra_12h_ep10
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.metrics.spherical_spectra import (
    CellCenteredRealSHT,
    CellCenteredRealVectorSHT,
    coefficient_cross,
    coefficient_power,
)
from weather_time_interp.grid import (
    WB2_BLOCK_GRID_NAME,
    wb2_block_average_latitudes,
)


SPECTRAL_SCHEMA_VERSION = 6


def _hash_required_sources(
    paths: tuple[Path, ...],
    repo_root: Path,
) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing spectral source files: " + ", ".join(missing)
        )
    return {
        str(path.relative_to(repo_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }


def _load_legacy_trainer_class(repo_root: Path):
    """Load the frozen trainer without relying on an ambiguous top-level import."""
    trainer_path = repo_root / "legacy" / "scripts" / "trainer_weather_hermite.py"
    spec = importlib.util.spec_from_file_location(
        "_weatherbridge_spectral_legacy_trainer",
        trainer_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy trainer from {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WeatherHermiteLightningModule


def _load_model_hermite(ckpt_path: str, device: torch.device,
                        channel_groups: Dict[str, List[int]], envs: str = ""):
    import os as _os
    for kv in envs.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            _os.environ[k] = v
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    state = ckpt.get("state_dict", ckpt)
    mt = hparams.get("model_type", "")
    if mt == "dcae_adaln_residual_linear":
        from tools.eval.dcae_checkpoint_loader import load_dcae_checkpoint

        return load_dcae_checkpoint(ckpt_path, device)

    import weather_time_interp.model.WeatherInterpModel as model_exports

    if not hasattr(model_exports, "WeatherHermiteModel"):
        class _UnavailableLegacyHermite(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                raise RuntimeError(
                    "The retired WeatherHermiteModel implementation is not "
                    "present in this evaluation mirror."
                )

        model_exports.WeatherHermiteModel = _UnavailableLegacyHermite

    repo_root = Path(__file__).resolve().parents[2]
    WeatherHermiteLightningModule = _load_legacy_trainer_class(repo_root)
    if "dcae" in mt:
        boc = tuple(hparams.get("block_out_channels", ()))
        if not boc:
            boc = (128, 256, 512)
        hparams["block_out_channels"] = boc
        lpb = tuple(hparams.get("layers_per_block", ()))
        if not lpb:
            lpb = (3, 3, 3)
        hparams["layers_per_block"] = lpb
        # Infer true encoder widths from the state dictionary when legacy
        # Lightning metadata does not describe the current module exactly.
        try:
            import re as _re
            idx_chan: dict[int, int] = {}
            for k, v in state.items():
                m = _re.match(r"model\.encoder\.down_blocks\.(\d+)\.conv1\.weight$", k)
                if m and v.dim() == 4:
                    idx_chan[int(m.group(1))] = int(v.shape[0])
                m2 = _re.match(
                    r"model\.encoder\.down_blocks\.(\d+)\.attn\.to_qkv_multiscale\.0\.proj_in\.weight$",
                    k,
                )
                if m2 and v.dim() == 4:
                    idx_chan[int(m2.group(1))] = int(v.shape[0]) // 3
            if idx_chan:
                widths = []
                for idx in sorted(idx_chan):
                    w = idx_chan[idx]
                    if not widths or widths[-1] != w:
                        widths.append(w)
                if (
                    len(widths) >= 3
                    and tuple(widths) != tuple(hparams["block_out_channels"])
                    and len(widths) != len(hparams["block_out_channels"])
                ):
                    print(f"  [legacy-compat] block_out {hparams['block_out_channels']} → {tuple(widths)}")
                    hparams["block_out_channels"] = tuple(widths)
                    hparams["layers_per_block"] = (2,) * len(widths)
        except Exception as _e:
            print(f"  [legacy-compat] sniff failed: {_e}")
    for k in ("channel_groups", "_class_path", "model_type_save"):
        hparams.pop(k, None)
    hparams["channel_groups"] = channel_groups
    # Drop any hparams the current trainer signature doesn't accept (e.g.
    # `freeze_skip_gates`, `freq_cond_film`, `lambda_pyramid`, `pyramid_levels`
    # — added during interim research branches, removed/renamed since).
    # Single inspect-based filter handles all such legacy keys uniformly.
    import inspect as _inspect
    _ok = set(_inspect.signature(WeatherHermiteLightningModule.__init__).parameters)
    for _k in [k for k in hparams if k not in _ok and k != "channel_groups"]:
        hparams.pop(_k, None)

    # The current ``WeatherHermiteLightningModule`` enforces ``len(boc)==3`` and
    # ``len(lpb)==3`` for the dcae_adaln_skip branch (otherwise falls back to
    # ``(128, 256, 512)`` / ``(2, 2, 2)``). The DC-AE Skip 3-yr fibo ckpt was
    # trained with a 4-stage `(128, 128, 256, 256)` / `(2, 2, 2, 2)` shape —
    # going through the strict trainer rebuilds a 3-stage model and the load
    # fails with size-mismatch on every conv. Bypass the trainer for that
    # specific model_type and instantiate the underlying model class directly.
    if mt == "dcae_adaln_skip_residual_linear":
        from weather_time_interp.model.dcae_adaln_skip_model import (  # type: ignore
            WeatherDCAEAdaLNSkipModel,
        )
        n_pl = int(hparams.get("n_pl_channels", 20))
        n_sf = int(hparams.get("n_surface_channels", 4))
        boc_tup = tuple(hparams["block_out_channels"])
        lpb_tup = tuple(hparams["layers_per_block"])
        # Recorded hparams for the 12h DC-AE Skip 3yr fibo ckpt are
        # ``(128, 128, 256, 256)`` / ``(2, 2, 2, 2)`` but the actual ckpt
        # was trained with only 3 active stages (``lpb=(2, 2, 2, 0)``) —
        # there are 9 encoder.down_blocks entries vs 11 a true 4-stage
        # network would produce. We patch ``lpb`` here only for that
        # specific (BOC=(128,128,256,256), declared lpb=(2,2,2,2))
        # combination so it loads correctly. NoSkip 3yr ckpt has its own
        # branch and is unaffected.
        if len(boc_tup) == 4 and lpb_tup == (2, 2, 2, 2):
            lpb_tup = (2, 2, 2, 0)
            print(f"  [skip-ckpt-patch] lpb (2,2,2,2)→(2,2,2,0) (matches 9 saved down_blocks)")
        # block_type / qkv_multiscales aren't recorded in hparams. Use the
        # convention "ResBlocks for shallow stages + EfficientViTBlock for
        # the last *non-empty* stage". For the patched (2,2,2,0) layout,
        # the attention sits at stage index 2 (the last with layers>0).
        last_used = max(i for i, n in enumerate(lpb_tup) if n > 0) if any(lpb_tup) else 0
        n_stages = len(boc_tup)
        block_type = tuple(
            "EfficientViTBlock" if i == last_used else "ResBlock"
            for i in range(n_stages)
        )
        qkv_multiscales = tuple(
            (5,) if i == last_used else ()
            for i in range(n_stages)
        )
        skip_model = WeatherDCAEAdaLNSkipModel(
            latent_channels=int(hparams.get("latent_channels", 256)),
            block_out_channels=boc_tup,
            layers_per_block=lpb_tup,
            block_type=block_type,
            qkv_multiscales=qkv_multiscales,
            in_channels=n_pl + n_sf,
            out_channels=n_pl + n_sf,
            lat_crop=int(hparams.get("lat_crop", 0)),
            residual_scale_init=float(hparams.get("residual_scale_init", 0.1)),
            residual_scale_learnable=bool(hparams.get("residual_scale_learnable", True)),
            residual_clip=hparams.get("residual_clip"),
            residual_scale_floor=float(hparams.get("residual_scale_floor", 0.0)),
            direct_prediction=bool(hparams.get("direct_prediction", False)),
            n_static_features=int(hparams.get("n_static_features", 0)),
            skip_gate_init=0.0,
            skip_lateral_rank=0,
            tau_conditional_gates=bool(hparams.get("tau_conditional_gates", False)),
        )
        # Strip the "model." prefix in state_dict — it was saved as the
        # LightningModule's submodule attribute.
        prefix = "model."
        sub_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        missing, unexpected = skip_model.load_state_dict(sub_state, strict=False)
        if missing or unexpected:
            print(f"  state load: missing={len(missing)}, unexpected={len(unexpected)}")
        skip_model.to(device).eval()
        # Wrap so the outer forward(x0, xT, tau, cond, static) call works the
        # same as the LightningModule.
        class _SkipWrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m
            def forward(self, x0, xT, tau_norm, cond=None, static=None):
                return self.model(x0, xT, tau_norm, cond, static=static)
        return _SkipWrap(skip_model).to(device).eval(), mt

    model = WeatherHermiteLightningModule(**hparams)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  state load: missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device).eval()
    return model, mt


def _load_model_atmvfi(ckpt_path: str, device: torch.device):
    """Load PixelAttentionVFI with the same surgical asymmetric-I/O patch as
    tools/eval/batch_eval_memmap.py. 6h v2 ckpts are 27-in / 24-out
    (lat-cos / lsm / orog static features concatenated inside forward)."""
    from tools.eval.batch_eval_memmap import _load_atmvfi_model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    state = ckpt.get("state_dict", ckpt)
    model = _load_atmvfi_model(ckpt_path, state, hparams, device)
    return model, "atm_vfi_pixel_attn"


class _BilinearTimeInterp(torch.nn.Module):
    """Trivial baseline: linear blend between x0 and xT in normalised space.

    Matches the bilinear/lerp baseline used in `tools/eval/batch_eval_12h_memmap.py`
    so the SH spectra are comparable across models + bilinear floor.
    """

    def forward(self, x0, xT, tau_norm, cond=None, static=None):
        # tau_norm: (B,) in [0, 1]
        t = tau_norm.view(-1, 1, 1, 1).to(x0.dtype)
        return x0 + (xT - x0) * t


# Standard 24-channel order produced by ERA5MemmapDataset for the WTI paper:
#   PL (T, U, V, Q, Z) × (1000, 925, 850, 700) hPa = 20 channels
#   surface (t2m, u10, v10, mslp)                  =  4 channels
ALL_24_CHANNELS = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]

WIND_CHANNEL_PAIRS = (
    ("wind1000", "U1000", "V1000"),
    ("wind925", "U925", "V925"),
    ("wind850", "U850", "V850"),
    ("wind700", "U700", "V700"),
    ("wind10", "u10", "v10"),
)
VECTOR_COMPONENT_NAMES = ("spheroidal", "toroidal")


def _select_channel_indices(channel_names: Sequence[str],
                            selected: Sequence[str]) -> List[int]:
    idx: List[int] = []
    for s in selected:
        try:
            idx.append(channel_names.index(s))
        except ValueError:
            raise SystemExit(
                f"channel {s!r} not in dataset (got {list(channel_names)[:6]} ...)"
            )
    return idx


def _selected_wind_pairs(
    selected_channels: Sequence[str],
) -> tuple[list[str], list[tuple[int, int]]]:
    names: list[str] = []
    indices: list[tuple[int, int]] = []
    for pair_name, u_name, v_name in WIND_CHANNEL_PAIRS:
        if u_name in selected_channels and v_name in selected_channels:
            names.append(pair_name)
            indices.append(
                (selected_channels.index(u_name), selected_channels.index(v_name))
            )
    return names, indices


def _build_sht(H: int, W: int, lmax: int, device: torch.device):
    """Create an SHT on the actual WB2 block-average latitude rows."""
    if lmax < 1 or lmax >= H:
        raise ValueError(
            f"lmax must satisfy 1 <= lmax < H, got {lmax} for H={H}"
        )
    mmax = lmax  # square truncation
    return CellCenteredRealSHT(
        H,
        W,
        lmax=lmax + 1,
        mmax=mmax + 1,
        latitude_degrees=wb2_block_average_latitudes(H),
    ).to(device)


def _build_vector_sht(H: int, W: int, lmax: int, device: torch.device):
    """Create the matching vector SHT for paired zonal/meridional winds."""
    if lmax < 1 or lmax >= H:
        raise ValueError(
            f"lmax must satisfy 1 <= lmax < H, got {lmax} for H={H}"
        )
    return CellCenteredRealVectorSHT(
        H,
        W,
        lmax=lmax + 1,
        mmax=lmax + 1,
        latitude_degrees=wb2_block_average_latitudes(H),
    ).to(device)


def _angular_power(field: torch.Tensor, sht) -> torch.Tensor:
    """Compute ``sum_m |a_{ℓ m}|^2`` for each ℓ.

    field: (B, C, H, W) — accumulates per (C, ℓ) summed over batch.
    Returns (C, lmax+1) (real, float64).
    """
    # Feed fp64 input to keep SHT entirely in double precision. On B300/sm_103
    # mixing fp32 inputs with fp64 SHT buffers triggers an NVRTC JIT compile
    # for an arch the toolchain doesn't recognise; staying fully fp64 avoids it.
    coeffs = sht(field.double())  # (B, C, lmax+1, mmax+1) complex128
    return coefficient_power(coeffs).sum(dim=0)


def _spectral_stats(
    pred: torch.Tensor,
    truth: torch.Tensor,
    sht,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return prediction power, truth power, and complex cross-spectrum."""
    pred_coeff = sht(pred.double())
    truth_coeff = sht(truth.double())
    return (
        coefficient_power(pred_coeff).sum(dim=0),
        coefficient_power(truth_coeff).sum(dim=0),
        coefficient_cross(pred_coeff, truth_coeff).sum(dim=0),
    )


def _rescale_to_physical_anomalies(
    field: torch.Tensor,
    channel_std: torch.Tensor,
) -> torch.Tensor:
    """Convert standardized fields to physical anomaly units for the SHT.

    The fixed training mean is deliberately not restored: it belongs to the
    constant mode but can leak into the high-degree band under discrete
    source-grid quadrature. Per-channel standard deviations are required so
    vector U/V spectra are not distorted by independent standardization.
    """
    if field.ndim != 4:
        raise ValueError("spectral fields must have shape (B,C,H,W)")
    scale = channel_std.to(device=field.device, dtype=field.dtype)
    if scale.ndim == 1:
        scale = scale.view(1, -1, 1, 1)
    if scale.shape != (1, field.size(1), 1, 1):
        raise ValueError("channel_std must provide one scale per field channel")
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("channel_std must be finite and strictly positive")
    return field * scale


def _vector_spectral_stats(
    pred_wind: torch.Tensor,
    truth_wind: torch.Tensor,
    vector_sht,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return spheroidal/toroidal power and cross-spectra for wind pairs."""
    pred_coeff = vector_sht(pred_wind.double())
    truth_coeff = vector_sht(truth_wind.double())
    return (
        coefficient_power(pred_coeff).sum(dim=0),
        coefficient_power(truth_coeff).sum(dim=0),
        coefficient_cross(pred_coeff, truth_coeff).sum(dim=0),
    )


def _band_window_diagnostics(
    pred_power: torch.Tensor,
    truth_power: torch.Tensor,
    cross_power: torch.Tensor,
    ell_min: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Separate band energy, shape, coherence, and signed cospectrum."""
    pred_band = pred_power[..., ell_min:]
    truth_band = truth_power[..., ell_min:]
    cross_band = cross_power[..., ell_min:]
    pred_energy = pred_band.sum(dim=-1)
    truth_energy = truth_band.sum(dim=-1)
    energy_ratio = pred_energy / truth_energy.clamp_min(1.0e-30)

    # Normalize each spectrum before comparing shape. Without this step a
    # uniform amplitude bias is counted twice, as both energy and shape error.
    pred_shape = pred_band / pred_energy.clamp_min(1.0e-30).unsqueeze(-1)
    truth_shape = truth_band / truth_energy.clamp_min(1.0e-30).unsqueeze(-1)
    shape_error = (
        torch.log(pred_shape.clamp_min(1.0e-12))
        - torch.log(truth_shape.clamp_min(1.0e-12))
    ).abs().mean(dim=-1)

    cross_energy = cross_band.sum(dim=-1)
    coherence = (
        cross_energy.real.square() + cross_energy.imag.square()
    ) / (pred_energy * truth_energy).clamp_min(1.0e-30)
    signed_cospectrum = cross_energy.real / torch.sqrt(
        (pred_energy * truth_energy).clamp_min(1.0e-30)
    )
    return (
        energy_ratio,
        shape_error,
        coherence.clamp(0.0, 1.0),
        signed_cospectrum.clamp(-1.0, 1.0),
    )


def _selection_sha256(index: Sequence[tuple]) -> str:
    keys = sorted({(int(entry[0]), int(entry[1])) for entry in index})
    text = "\n".join(f"{year},{t0}" for year, t0 in keys)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _filter_index_by_days_of_month(
    index: Sequence[tuple],
    days_of_month: Sequence[int],
) -> list[tuple]:
    """Select an explicit, provenance-stable set of calendar days."""
    allowed = {int(day) for day in days_of_month}
    if not allowed or any(day < 1 or day > 31 for day in allowed):
        raise ValueError("days_of_month must contain days in [1, 31]")
    return [
        entry
        for entry in index
        if (
            dt.date(int(entry[0]), 1, 1)
            + dt.timedelta(days=int(entry[1]) // 24)
        ).day
        in allowed
    ]


def _cached_spectrum_is_current(
    path: Path,
    *,
    tau: int,
    lmax: int,
    channels: Sequence[str],
    expected_n_samples: int,
    expected_metadata: dict,
) -> bool:
    """Fail closed unless a cached spectrum is complete and protocol-identical."""
    required = {
        "ell",
        "pred_El",
        "gt_El",
        "cross_El_real",
        "cross_El_imag",
        "coherence_l",
        "signed_cospectrum_l",
        "channel_names",
        "n_samples",
        "window_year",
        "window_t0",
        "window_hf_energy_ratio",
        "window_hf_log_shape_error",
        "window_hf_coherence",
        "window_hf_signed_cospectrum",
        "wind_pair_names",
        "vector_component_names",
        "vector_pred_El",
        "vector_gt_El",
        "vector_cross_El_real",
        "vector_cross_El_imag",
        "vector_coherence_l",
        "vector_signed_cospectrum_l",
        "window_vector_hf_energy_ratio",
        "window_vector_hf_log_shape_error",
        "window_vector_hf_coherence",
        "window_vector_hf_signed_cospectrum",
        "tau",
        "H",
        "W",
        "lmax",
        "model_name",
        "metadata_json",
    }
    try:
        with np.load(path, allow_pickle=False) as cached:
            if not required.issubset(cached.files):
                return False
            metadata = json.loads(str(cached["metadata_json"].item()))
            if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                return False
            n_samples = int(cached["n_samples"])
            n_channels = len(channels)
            wind_names, _ = _selected_wind_pairs(channels)
            n_wind = len(wind_names)
            spectral_shape = (n_channels, lmax + 1)
            window_shape = (expected_n_samples, n_channels)
            vector_shape = (n_wind, len(VECTOR_COMPONENT_NAMES), lmax + 1)
            vector_window_shape = (
                expected_n_samples,
                n_wind,
                len(VECTOR_COMPONENT_NAMES),
            )
            if (
                int(cached["tau"]) != tau
                or int(cached["lmax"]) != lmax
                or n_samples != expected_n_samples
                or str(cached["model_name"].item())
                != str(expected_metadata["model_name"])
                or [str(value) for value in cached["channel_names"]] != list(channels)
                or not np.array_equal(
                    np.asarray(cached["ell"]),
                    np.arange(lmax + 1, dtype=np.int32),
                )
                or np.asarray(cached["pred_El"]).shape != spectral_shape
                or np.asarray(cached["gt_El"]).shape != spectral_shape
                or np.asarray(cached["cross_El_real"]).shape != spectral_shape
                or np.asarray(cached["cross_El_imag"]).shape != spectral_shape
                or np.asarray(cached["coherence_l"]).shape != spectral_shape
                or np.asarray(cached["signed_cospectrum_l"]).shape
                != spectral_shape
                or np.asarray(cached["window_year"]).shape != (expected_n_samples,)
                or np.asarray(cached["window_t0"]).shape != (expected_n_samples,)
                or np.asarray(cached["window_hf_energy_ratio"]).shape != window_shape
                or np.asarray(
                    cached["window_hf_log_shape_error"]
                ).shape != window_shape
                or np.asarray(cached["window_hf_coherence"]).shape != window_shape
                or np.asarray(cached["window_hf_signed_cospectrum"]).shape
                != window_shape
                or [str(value) for value in cached["wind_pair_names"]]
                != wind_names
                or [str(value) for value in cached["vector_component_names"]]
                != list(VECTOR_COMPONENT_NAMES)
                or np.asarray(cached["vector_pred_El"]).shape != vector_shape
                or np.asarray(cached["vector_gt_El"]).shape != vector_shape
                or np.asarray(cached["vector_cross_El_real"]).shape
                != vector_shape
                or np.asarray(cached["vector_cross_El_imag"]).shape
                != vector_shape
                or np.asarray(cached["vector_coherence_l"]).shape
                != vector_shape
                or np.asarray(cached["vector_signed_cospectrum_l"]).shape
                != vector_shape
                or np.asarray(cached["window_vector_hf_energy_ratio"]).shape
                != vector_window_shape
                or np.asarray(
                    cached["window_vector_hf_log_shape_error"]
                ).shape
                != vector_window_shape
                or np.asarray(cached["window_vector_hf_coherence"]).shape
                != vector_window_shape
                or np.asarray(
                    cached["window_vector_hf_signed_cospectrum"]
                ).shape
                != vector_window_shape
                or int(cached["H"]) <= lmax
                or int(cached["W"]) <= 0
            ):
                return False
            finite_arrays = (
                "pred_El",
                "gt_El",
                "cross_El_real",
                "cross_El_imag",
                "coherence_l",
                "signed_cospectrum_l",
                "window_hf_energy_ratio",
                "window_hf_log_shape_error",
                "window_hf_coherence",
                "window_hf_signed_cospectrum",
                "vector_pred_El",
                "vector_gt_El",
                "vector_cross_El_real",
                "vector_cross_El_imag",
                "vector_coherence_l",
                "vector_signed_cospectrum_l",
                "window_vector_hf_energy_ratio",
                "window_vector_hf_log_shape_error",
                "window_vector_hf_coherence",
                "window_vector_hf_signed_cospectrum",
            )
            if any(
                not np.isfinite(np.asarray(cached[name])).all()
                for name in finite_arrays
            ):
                return False
            pred_energy = np.asarray(cached["pred_El"])
            truth_energy = np.asarray(cached["gt_El"])
            coherence = np.asarray(cached["coherence_l"])
            window_ratio = np.asarray(cached["window_hf_energy_ratio"])
            window_shape_error = np.asarray(
                cached["window_hf_log_shape_error"]
            )
            window_coherence = np.asarray(cached["window_hf_coherence"])
            signed = np.asarray(cached["signed_cospectrum_l"])
            window_signed = np.asarray(cached["window_hf_signed_cospectrum"])
            vector_coherence = np.asarray(cached["vector_coherence_l"])
            vector_signed = np.asarray(cached["vector_signed_cospectrum_l"])
            vector_pred_energy = np.asarray(cached["vector_pred_El"])
            vector_truth_energy = np.asarray(cached["vector_gt_El"])
            window_vector_ratio = np.asarray(
                cached["window_vector_hf_energy_ratio"]
            )
            window_vector_shape_error = np.asarray(
                cached["window_vector_hf_log_shape_error"]
            )
            window_vector_coherence = np.asarray(
                cached["window_vector_hf_coherence"]
            )
            window_vector_signed = np.asarray(
                cached["window_vector_hf_signed_cospectrum"]
            )
            return bool(
                (pred_energy >= 0).all()
                and (truth_energy >= 0).all()
                and (window_ratio >= 0).all()
                and (window_shape_error >= 0).all()
                and ((coherence >= 0) & (coherence <= 1)).all()
                and ((window_coherence >= 0) & (window_coherence <= 1)).all()
                and ((signed >= -1) & (signed <= 1)).all()
                and ((window_signed >= -1) & (window_signed <= 1)).all()
                and ((vector_coherence >= 0) & (vector_coherence <= 1)).all()
                and ((vector_signed >= -1) & (vector_signed <= 1)).all()
                and (vector_pred_energy >= 0).all()
                and (vector_truth_energy >= 0).all()
                and (window_vector_ratio >= 0).all()
                and (window_vector_shape_error >= 0).all()
                and (
                    (window_vector_coherence >= 0)
                    & (window_vector_coherence <= 1)
                ).all()
                and (
                    (window_vector_signed >= -1)
                    & (window_vector_signed <= 1)
                ).all()
            )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _atomic_savez(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-name", required=True,
                    help="Used as the output .npz file prefix.")
    ap.add_argument("--model-kind",
                    choices=["hermite", "capmatched", "atm_vfi", "atmvfi", "bilinear"],
                    default="hermite",
                    help="`bilinear` ignores --ckpt and uses linear interpolation.")
    ap.add_argument("--envs", default="")
    ap.add_argument("--out-dir", default="metrics/sh_spectra_12h_ep10")
    ap.add_argument("--taus", default="2,3,5,8",
                    help="Comma-separated τ values to evaluate.")
    ap.add_argument("--channels", default="t2m,mslp,u10,v10,T850",
                    help="Comma-separated channel names to keep (must be in dataset), "
                         "or `all` for the full 24-channel set "
                         "(T,U,V,Q,Z × {1000,925,850,700} + t2m,u10,v10,mslp).")
    ap.add_argument("--lmax", type=int, default=180)
    ap.add_argument("--keep-n-channels", type=int, default=24,
                    help="Slice x0/xT/target to first N channels (24 for keep_24ch).")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--samples-per-date", type=int, default=2)
    ap.add_argument("--eval-days-per-month", type=int, default=2,
                    help="Sub-sample test set (default 2 days/month for speed).")
    ap.add_argument(
        "--days-of-month",
        default=None,
        help=(
            "Explicit comma-separated calendar days, for example 1,8,15,22. "
            "Recorded in the artifact and mutually exclusive with --full-year."
        ),
    )
    ap.add_argument("--full-year", action="store_true",
                    help="Evaluate all valid anchor windows instead of monthly subsampling.")
    ap.add_argument("--hf-ell-min", type=int, default=80,
                    help="Minimum degree of the journal diagnostic band.")
    ap.add_argument("--max-tau-hours", type=int, default=12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip τ values whose output .npz already exists in --out-dir.")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    taus_requested = sorted({int(x) for x in args.taus.split(",") if x.strip()})
    if not taus_requested:
        raise SystemExit("--taus must contain at least one hour")
    if any(tau < 1 or tau >= args.max_tau_hours for tau in taus_requested):
        raise SystemExit(
            "--taus must be interior hours in [1, --max-tau-hours)"
        )
    if args.lmax < 1:
        raise SystemExit("--lmax must be positive")
    if not 1 <= args.hf_ell_min <= args.lmax:
        raise SystemExit("--hf-ell-min must be in [1, --lmax]")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.samples_per_date < 1 or 24 % args.samples_per_date:
        raise SystemExit("--samples-per-date must be a positive divisor of 24")
    if args.eval_days_per_month < 1:
        raise SystemExit("--eval-days-per-month must be positive")
    days_of_month = (
        None
        if args.days_of_month is None
        else sorted({int(value) for value in args.days_of_month.split(",") if value.strip()})
    )
    if args.full_year and days_of_month is not None:
        raise SystemExit("--full-year and --days-of-month are mutually exclusive")
    if days_of_month is not None and (
        not days_of_month or any(day < 1 or day > 31 for day in days_of_month)
    ):
        raise SystemExit("--days-of-month must contain days in [1, 31]")
    if args.keep_n_channels is not None and args.keep_n_channels < 1:
        raise SystemExit("--keep-n-channels must be positive")
    if args.channels.strip().lower() == "all":
        sel_channels = list(ALL_24_CHANNELS)
    else:
        sel_channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    if not sel_channels or len(sel_channels) != len(set(sel_channels)):
        raise SystemExit("--channels must contain unique channel names")

    from weather_time_interp.normalization import (
        STATIC_FEATURES_3,
        file_provenance,
        static_feature_provenance,
    )
    from tools.train.training_protocol import memmap_dataset_provenance

    if args.model_kind == "bilinear":
        expected_provenance = {"kind": "analytic_linear_interpolation"}
    else:
        expected_provenance = file_provenance(args.ckpt)
    evaluation_dataset_provenance = memmap_dataset_provenance(
        args.memmap_dir,
        [args.test_year],
    )
    evaluation_input_provenance = {
        "static_features": static_feature_provenance(args.static_path),
        "pressure_level_stats": file_provenance(args.stats_path),
        "surface_stats": file_provenance(args.surface_stats_path),
    }
    evaluation_script_sha256 = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    repo_root = Path(__file__).resolve().parents[2]
    supporting_paths = (
        repo_root / "dataset.py",
        repo_root / "metrics" / "__init__.py",
        repo_root / "metrics" / "weather.py",
        repo_root / "legacy" / "scripts" / "trainer_weather_hermite.py",
        repo_root / "tools" / "eval" / "capmatched_loader.py",
        repo_root
        / "tools"
        / "train"
        / "train_capacity_matched_6h.py",
        repo_root / "tools" / "train" / "training_protocol.py",
        repo_root
        / "weather_time_interp"
        / "grid.py",
        repo_root / "weather_time_interp" / "memmap_dataset.py",
        repo_root / "weather_time_interp" / "eval_datasets.py",
        repo_root
        / "weather_time_interp"
        / "metrics"
        / "spherical_spectra.py",
        repo_root
        / "weather_time_interp"
        / "model"
        / "weatherbridge_flow_model.py",
        repo_root
        / "weather_time_interp"
        / "model"
        / "dcae_adaln_model.py",
        repo_root
        / "weather_time_interp"
        / "model"
        / "dcae_adaln_skip_model.py",
        repo_root
        / "weather_time_interp"
        / "model"
        / "fuxi_swinv2_model.py",
        repo_root
        / "weather_time_interp"
        / "model"
        / "modafno_baseline_model.py",
        repo_root
        / "weather_time_interp"
        / "model"
        / "sdyff_dyffusion_model.py",
        repo_root
        / "legacy"
        / "scripts"
        / "train_atm_vfi_12h_oddskip.py",
    )
    supporting_code_sha256 = _hash_required_sources(
        supporting_paths,
        repo_root,
    )

    # Build the exact sampling index before accepting cached artifacts.
    from weather_time_interp.memmap_dataset import ERA5MemmapDataset
    from weather_time_interp.eval_datasets import ERA5WeatherHermiteDataset
    ds_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[args.test_year],
        max_tau_hours=args.max_tau_hours,
        samples_per_date=args.samples_per_date,
        train=False,
        eval_hours=taus_requested,
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
    )
    if days_of_month is not None:
        ds_base.index = _filter_index_by_days_of_month(
            ds_base.index,
            days_of_month,
        )
    elif not args.full_year:
        k_days = max(1, int(args.eval_days_per_month))
        days_by_month: Dict[int, set[int]] = {}
        for entry in ds_base.index:
            year, t0_h, _, _ = entry
            date = dt.date(int(year), 1, 1) + dt.timedelta(days=int(t0_h) // 24)
            days_by_month.setdefault(date.month, set()).add(int(t0_h) // 24)
        selected_days = set()
        for month in sorted(days_by_month):
            days = sorted(days_by_month[month])
            positions = np.rint(
                np.linspace(0, len(days) - 1, min(k_days, len(days)))
            ).astype(int)
            selected_days.update(days[index] for index in positions)
        ds_base.index = [
            entry for entry in ds_base.index
            if int(entry[1]) // 24 in selected_days
        ]
    selection_hash = _selection_sha256(ds_base.index)
    expected_n_samples = {
        tau: sum(int(entry[2]) == tau for entry in ds_base.index)
        for tau in taus_requested
    }
    missing_taus = [
        tau for tau, count in expected_n_samples.items() if count < 1
    ]
    if missing_taus:
        raise RuntimeError(f"selected dataset has no windows for taus={missing_taus}")
    print(
        f"  windows after selection: "
        f"{len(ds_base.index) // len(taus_requested)}; "
        f"sha256={selection_hash}"
    )

    channel_groups = ds_base.channel_groups
    channel_names = list(ds_base.channel_names) + list(ds_base.surface_variables)
    print(f"  full channels: {len(channel_names)}; first 6 = {channel_names[:6]}")
    if args.keep_n_channels is not None:
        channel_names = channel_names[: args.keep_n_channels]
        print(f"  sliced to {len(channel_names)} channels: {channel_names}")
    sel_idx = _select_channel_indices(channel_names, sel_channels)
    print(f"  selected channel indices: {sel_idx}")
    channel_std = torch.cat((ds_base.sigma, ds_base.surface_sigma))
    if args.keep_n_channels is not None:
        channel_std = channel_std[: args.keep_n_channels]
    selected_channel_std = (
        channel_std[sel_idx]
        .to(device=device)
        .view(1, len(sel_idx), 1, 1)
    )
    wind_pair_names, wind_pair_indices = _selected_wind_pairs(sel_channels)
    print(f"  vector wind pairs: {wind_pair_names}")

    sample_strategy = (
        "all_valid_anchor_windows"
        if args.full_year
        else (
            "explicit_calendar_days"
            if days_of_month is not None
            else "uniform_days_within_each_month"
        )
    )
    expected_metadata = {
        "schema_version": SPECTRAL_SCHEMA_VERSION,
        "year": args.test_year,
        "model_name": args.model_name,
        "model_kind": args.model_kind,
        "checkpoint_provenance": expected_provenance,
        "sample_strategy": sample_strategy,
        "eval_days_per_month": (
            None
            if args.full_year or days_of_month is not None
            else int(args.eval_days_per_month)
        ),
        "days_of_month": days_of_month,
        "samples_per_date": int(args.samples_per_date),
        "selection_sha256": selection_hash,
        "hf_ell_min": int(args.hf_ell_min),
        "band_energy_metric": "absolute_log_total_power_ratio",
        "band_shape_metric": "mean_absolute_log_normalized_power",
        "band_coherence_metric": "integrated_complex_coherence",
        "band_phase_metric": "signed_normalized_integrated_cospectrum",
        "sht_grid": WB2_BLOCK_GRID_NAME + "_latitude_strip_area_sht",
        "wind_metric": "vector_sht_spheroidal_toroidal_energy",
        "spectral_field_units": "physical_anomaly_units_via_channel_std",
        "max_tau_hours": int(args.max_tau_hours),
        "keep_n_channels": (
            None if args.keep_n_channels is None else int(args.keep_n_channels)
        ),
        "evaluation_script_sha256": evaluation_script_sha256,
        "supporting_code_sha256": supporting_code_sha256,
        "evaluation_input_provenance": evaluation_input_provenance,
        "evaluation_dataset_provenance": evaluation_dataset_provenance,
        "static_feature_names": list(STATIC_FEATURES_3),
    }

    # Resume only from complete artifacts produced under this exact protocol.
    out_dir_path = Path(args.out_dir)
    taus = []
    for tau in taus_requested:
        cand = out_dir_path / f"{args.model_name}_tau{tau}.npz"
        if args.skip_existing and cand.exists():
            if _cached_spectrum_is_current(
                cand,
                tau=tau,
                lmax=args.lmax,
                channels=sel_channels,
                expected_n_samples=expected_n_samples[tau],
                expected_metadata=expected_metadata,
            ):
                print(f"  [skip-existing] validated τ={tau} at {cand}")
                continue
            print(f"  [skip-existing] stale τ={tau}; recomputing {cand}")
        taus.append(tau)
    if not taus:
        print(
            f"  [skip-existing] all τ already done for "
            f"{args.model_name}; nothing to do"
        )
        return

    active_taus = set(taus)
    ds_base.index = [
        entry for entry in ds_base.index if int(entry[2]) in active_taus
    ]
    print(
        f"device={device}  taus={taus}  "
        f"channels({len(sel_channels)})={sel_channels}"
    )

    wrapped = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=float(args.max_tau_hours))
    loader = DataLoader(wrapped, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=True)

    if args.model_kind == "hermite":
        model, mt = _load_model_hermite(args.ckpt, device, channel_groups, args.envs)
    elif args.model_kind == "capmatched":
        from tools.eval.capmatched_loader import load_capmatched_checkpoint

        model, mt = load_capmatched_checkpoint(
            args.ckpt,
            device,
            static_path=args.static_path,
        )
    elif args.model_kind in ("atm_vfi", "atmvfi"):
        model, mt = _load_model_atmvfi(args.ckpt, device)
    elif args.model_kind == "bilinear":
        model, mt = _BilinearTimeInterp().to(device).eval(), "bilinear"
    else:
        raise SystemExit(f"unknown --model-kind {args.model_kind!r}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  model loaded: type={mt}, params={n_params:.1f}M")
    model_input_channels = int(
        getattr(
            model,
            "in_channels",
            getattr(getattr(model, "net", None), "in_channels", 0),
        )
    )
    if model_input_channels and model_input_channels != args.keep_n_channels:
        print(
            f"  model anchor input: {model_input_channels} channels; "
            f"spectral targets: first {args.keep_n_channels}"
        )

    # Probe H, W from a sample.
    sample = next(iter(loader))
    H, W = sample["x0"].shape[-2:]
    print(f"  H×W = {H}×{W}")
    sht = _build_sht(H, W, args.lmax, device)
    vector_sht = (
        _build_vector_sht(H, W, args.lmax, device)
        if wind_pair_indices
        else None
    )
    lmax_plus_1 = args.lmax + 1

    # Per-τ accumulators.
    pred_acc = {tau: torch.zeros(len(sel_idx), lmax_plus_1, dtype=torch.float64,
                                 device=device) for tau in taus}
    gt_acc = {tau: torch.zeros(len(sel_idx), lmax_plus_1, dtype=torch.float64,
                               device=device) for tau in taus}
    cross_acc = {
        tau: torch.zeros(
            len(sel_idx),
            lmax_plus_1,
            dtype=torch.complex128,
            device=device,
        )
        for tau in taus
    }
    n_per_tau = {tau: 0 for tau in taus}
    window_year = {tau: [] for tau in taus}
    window_t0 = {tau: [] for tau in taus}
    window_hf_ratio = {tau: [] for tau in taus}
    window_hf_shape_error = {tau: [] for tau in taus}
    window_hf_coherence = {tau: [] for tau in taus}
    window_hf_signed_cospectrum = {tau: [] for tau in taus}
    vector_shape = (len(wind_pair_indices), 2, lmax_plus_1)
    vector_pred_acc = {
        tau: torch.zeros(vector_shape, dtype=torch.float64, device=device)
        for tau in taus
    }
    vector_gt_acc = {
        tau: torch.zeros(vector_shape, dtype=torch.float64, device=device)
        for tau in taus
    }
    vector_cross_acc = {
        tau: torch.zeros(vector_shape, dtype=torch.complex128, device=device)
        for tau in taus
    }
    window_vector_hf_ratio = {tau: [] for tau in taus}
    window_vector_hf_shape_error = {tau: [] for tau in taus}
    window_vector_hf_coherence = {tau: [] for tau in taus}
    window_vector_hf_signed_cospectrum = {tau: [] for tau in taus}
    max_tau = float(args.max_tau_hours)
    hf_ell_min = int(args.hf_ell_min)
    grouped_indices = getattr(wrapped, "_grouped_indices", None)
    window_offset = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x0 = batch["x0"].to(device, non_blocking=True)
            xT = batch["xT"].to(device, non_blocking=True)
            tau_hour_all = batch["tau_hour"].long()  # (B, nH, 1)
            target_all = batch["target"].to(device, non_blocking=True)
            static = batch.get("static")
            if static is not None:
                static = static.to(device, non_blocking=True)
            model_x0 = x0
            model_xT = xT
            if model_input_channels:
                model_x0 = x0[:, :model_input_channels].contiguous()
                model_xT = xT[:, :model_input_channels].contiguous()
            if args.keep_n_channels is not None:
                K_ch = args.keep_n_channels
                x0 = x0[:, :K_ch].contiguous()
                xT = xT[:, :K_ch].contiguous()
                target_all = target_all[:, :, :K_ch].contiguous()
                if not model_input_channels:
                    model_x0 = x0
                    model_xT = xT

            B = x0.size(0)
            nH = tau_hour_all.size(1)
            cond = torch.full((B,), max_tau, device=device, dtype=torch.float32)
            for h_idx in range(nH):
                tau_h_int = tau_hour_all[:, h_idx, 0]
                tau_norm = tau_h_int.float().to(device) / max_tau
                target_h = target_all[:, h_idx]

                if args.model_kind in ("hermite", "capmatched"):
                    out = model(
                        model_x0,
                        model_xT,
                        tau_norm,
                        cond,
                        static=static,
                    )
                    pred = out[0] if isinstance(out, tuple) else out
                elif args.model_kind in ("atm_vfi", "atmvfi"):
                    pred = model.net(x0, xT, tau_norm)
                else:  # bilinear
                    pred = model(x0, xT, tau_norm)

                pred_sel = _rescale_to_physical_anomalies(
                    pred[:, sel_idx],
                    selected_channel_std,
                )
                tgt_sel = _rescale_to_physical_anomalies(
                    target_h[:, sel_idx],
                    selected_channel_std,
                )

                # Group by integer τ (uniform within batch by construction).
                # Each batch element has the same τ_h_int; but be safe per-i.
                for i in range(B):
                    tau_i = int(tau_h_int[i].item())
                    if tau_i not in pred_acc:
                        continue
                    p = pred_sel[i:i + 1]
                    g = tgt_sel[i:i + 1]
                    p_power, g_power, cross = _spectral_stats(p, g, sht)
                    pred_acc[tau_i] += p_power
                    gt_acc[tau_i] += g_power
                    cross_acc[tau_i] += cross
                    n_per_tau[tau_i] += 1
                    wrapped_index = window_offset + i
                    base_index = (
                        grouped_indices[wrapped_index][0]
                        if grouped_indices is not None
                        else wrapped_index
                    )
                    year_i, t0_i, *_ = ds_base.index[base_index]
                    window_year[tau_i].append(int(year_i))
                    window_t0[tau_i].append(int(t0_i))
                    (
                        ratio,
                        shape_error,
                        coherence,
                        signed_cospectrum,
                    ) = _band_window_diagnostics(
                        p_power,
                        g_power,
                        cross,
                        hf_ell_min,
                    )
                    window_hf_ratio[tau_i].append(
                        ratio.cpu().numpy().astype(np.float32)
                    )
                    window_hf_shape_error[tau_i].append(
                        shape_error.cpu().numpy().astype(np.float32)
                    )
                    window_hf_coherence[tau_i].append(
                        coherence.cpu().numpy().astype(np.float32)
                    )
                    window_hf_signed_cospectrum[tau_i].append(
                        signed_cospectrum.cpu().numpy().astype(np.float32)
                    )
                    if vector_sht is not None:
                        p_wind = torch.stack(
                            [
                                torch.stack((p[:, u], p[:, v]), dim=1)
                                for u, v in wind_pair_indices
                            ],
                            dim=1,
                        )
                        g_wind = torch.stack(
                            [
                                torch.stack((g[:, u], g[:, v]), dim=1)
                                for u, v in wind_pair_indices
                            ],
                            dim=1,
                        )
                        (
                            vector_p_power,
                            vector_g_power,
                            vector_cross,
                        ) = _vector_spectral_stats(
                            p_wind,
                            g_wind,
                            vector_sht,
                        )
                        vector_pred_acc[tau_i] += vector_p_power
                        vector_gt_acc[tau_i] += vector_g_power
                        vector_cross_acc[tau_i] += vector_cross
                        (
                            vector_ratio,
                            vector_shape_error,
                            vector_coherence,
                            vector_signed_cospectrum,
                        ) = _band_window_diagnostics(
                            vector_p_power,
                            vector_g_power,
                            vector_cross,
                            hf_ell_min,
                        )
                        window_vector_hf_ratio[tau_i].append(
                            vector_ratio.cpu().numpy().astype(np.float32)
                        )
                        window_vector_hf_shape_error[tau_i].append(
                            vector_shape_error.cpu().numpy().astype(np.float32)
                        )
                        window_vector_hf_coherence[tau_i].append(
                            vector_coherence.cpu().numpy().astype(np.float32)
                        )
                        window_vector_hf_signed_cospectrum[tau_i].append(
                            vector_signed_cospectrum.cpu().numpy().astype(
                                np.float32
                            )
                        )
            window_offset += B
            if bi % 10 == 0:
                print(f"  batch {bi}/{len(loader)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ell = np.arange(lmax_plus_1, dtype=np.int32)
    if args.model_kind == "bilinear":
        checkpoint_provenance = {"kind": "analytic_linear_interpolation"}
    else:
        checkpoint_provenance = file_provenance(args.ckpt)
    for tau in taus:
        if n_per_tau[tau] != expected_n_samples[tau]:
            raise RuntimeError(
                f"tau={tau}: evaluated {n_per_tau[tau]} windows, "
                f"expected {expected_n_samples[tau]}"
            )
        n = n_per_tau[tau]
        pred_El = (pred_acc[tau] / n).cpu().numpy()
        gt_El = (gt_acc[tau] / n).cpu().numpy()
        cross_El = (cross_acc[tau] / n).cpu().numpy()
        coherence_l = np.clip(
            np.abs(cross_El) ** 2
            / np.clip(pred_El * gt_El, 1e-30, None),
            0.0,
            1.0,
        )
        signed_cospectrum_l = np.clip(
            cross_El.real / np.sqrt(np.clip(pred_El * gt_El, 1e-30, None)),
            -1.0,
            1.0,
        )
        vector_pred_El = (vector_pred_acc[tau] / n).cpu().numpy()
        vector_gt_El = (vector_gt_acc[tau] / n).cpu().numpy()
        vector_cross_El = (vector_cross_acc[tau] / n).cpu().numpy()
        vector_coherence_l = np.clip(
            np.abs(vector_cross_El) ** 2
            / np.clip(vector_pred_El * vector_gt_El, 1e-30, None),
            0.0,
            1.0,
        )
        vector_signed_cospectrum_l = np.clip(
            vector_cross_El.real
            / np.sqrt(np.clip(vector_pred_El * vector_gt_El, 1e-30, None)),
            -1.0,
            1.0,
        )
        window_ratio = (
            np.stack(window_hf_ratio[tau])
            if window_hf_ratio[tau]
            else np.empty((0, len(sel_idx)), dtype=np.float32)
        )
        window_shape_error = (
            np.stack(window_hf_shape_error[tau])
            if window_hf_shape_error[tau]
            else np.empty((0, len(sel_idx)), dtype=np.float32)
        )
        window_coherence = (
            np.stack(window_hf_coherence[tau])
            if window_hf_coherence[tau]
            else np.empty((0, len(sel_idx)), dtype=np.float32)
        )
        window_signed_cospectrum = np.stack(
            window_hf_signed_cospectrum[tau]
        ).astype(np.float32)
        vector_window_shape = (n, len(wind_pair_indices), 2)
        window_vector_ratio = (
            np.stack(window_vector_hf_ratio[tau]).astype(np.float32)
            if window_vector_hf_ratio[tau]
            else np.empty(vector_window_shape, dtype=np.float32)
        )
        window_vector_shape_error = (
            np.stack(window_vector_hf_shape_error[tau]).astype(np.float32)
            if window_vector_hf_shape_error[tau]
            else np.empty(vector_window_shape, dtype=np.float32)
        )
        window_vector_coherence = (
            np.stack(window_vector_hf_coherence[tau]).astype(np.float32)
            if window_vector_hf_coherence[tau]
            else np.empty(vector_window_shape, dtype=np.float32)
        )
        window_vector_signed_cospectrum = (
            np.stack(window_vector_hf_signed_cospectrum[tau]).astype(np.float32)
            if window_vector_hf_signed_cospectrum[tau]
            else np.empty(vector_window_shape, dtype=np.float32)
        )
        if checkpoint_provenance != expected_provenance:
            raise RuntimeError("checkpoint provenance changed during evaluation")
        metadata = dict(expected_metadata)
        payload = dict(
            ell=ell,
            pred_El=pred_El,
            gt_El=gt_El,
            cross_El_real=cross_El.real,
            cross_El_imag=cross_El.imag,
            coherence_l=coherence_l,
            signed_cospectrum_l=signed_cospectrum_l,
            channel_names=np.array(sel_channels),
            n_samples=int(n_per_tau[tau]),
            window_year=np.asarray(window_year[tau], dtype=np.int16),
            window_t0=np.asarray(window_t0[tau], dtype=np.int32),
            window_hf_energy_ratio=window_ratio,
            window_hf_log_shape_error=window_shape_error,
            window_hf_coherence=window_coherence,
            window_hf_signed_cospectrum=window_signed_cospectrum,
            wind_pair_names=np.asarray(wind_pair_names),
            vector_component_names=np.asarray(VECTOR_COMPONENT_NAMES),
            vector_pred_El=vector_pred_El,
            vector_gt_El=vector_gt_El,
            vector_cross_El_real=vector_cross_El.real,
            vector_cross_El_imag=vector_cross_El.imag,
            vector_coherence_l=vector_coherence_l,
            vector_signed_cospectrum_l=vector_signed_cospectrum_l,
            window_vector_hf_energy_ratio=window_vector_ratio,
            window_vector_hf_log_shape_error=window_vector_shape_error,
            window_vector_hf_coherence=window_vector_coherence,
            window_vector_hf_signed_cospectrum=(
                window_vector_signed_cospectrum
            ),
            tau=int(tau),
            H=int(H),
            W=int(W),
            lmax=int(args.lmax),
            model_name=args.model_name,
            ckpt=args.ckpt,
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        out_path = out_dir / f"{args.model_name}_tau{tau}.npz"
        _atomic_savez(out_path, payload)
        print(f"  wrote {out_path}  (n={n_per_tau[tau]})")

    print(f"=== DONE in {(time.time() - t0) / 60:.1f} min ===")


if __name__ == "__main__":
    main()

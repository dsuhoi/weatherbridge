"""Evaluate interpolators on forecast anchors with ERA5 or forecast truth.

For each init:
  - anchor pairs: every (lead_a, lead_b) with lead_b - lead_a = ΔT
  - truth: ERA5 at the matching valid time, or an available intermediate
    forecast lead from the same initialisation
  - interpolators: capacity-matched checkpoints, bare blobs, or linear baseline
  - baseline: Linear Interp.  =  (1-τ/6)·anchor_k + (τ/6)·anchor_{k+1}

Notes on the memmap contracts we consume here:
  * ``preprocess_forecast_to_memmap.py`` writes 24-channel fp32 memmaps in
    the paper's canonical order (see CHANNELS_ORDER). Channels not present
    in the forecast source (Aurora: 1000/925 hPa) are NaN-filled — we mask
    those out per pair and only accumulate valid channels.
  * ``preprocess_0p5_to_memmap.py`` writes 27-channel ERA5 memmaps
    (20 PL + 7 surface: t2m u10 v10 mslp sst tcc tcwv). We select the
    first 24 = 20 PL + {t2m u10 v10 mslp} to match the forecast contract.

Output JSON:
  per_tau[str(tau)] -> {"rmse_norm_<CH>": float, "n_pairs": int}
  model inputs use the exact training mean/std files; metric errors are
  divided by the same per-channel training std.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

from tools.train.training_protocol import memmap_dataset_provenance
from weather_time_interp.grid import (
    WB2_BLOCK_GRID_NAME,
    latitude_strip_weights,
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)
from weather_time_interp.normalization import (
    PAPER_CHANNELS_24,
    ChannelStats,
    file_provenance,
    load_channel_stats,
    static_feature_provenance,
)

try:
    import torch  # only needed for BareModel path
except ImportError:  # pragma: no cover — linear-only environments (CPU dev boxes)
    torch = None

CHANNELS_ORDER = list(PAPER_CHANNELS_24)
FORECAST_LEAD_BINS = (
    ("fresh", 0, 48),
    ("medium", 48, 120),
    ("long", 120, 240),
)
LEGACY_HRES_QC_POLICY = (
    "finite_only_2x2_mean_require_at_least_3_of_4_native_values"
)
HRES_QC_POLICY = (
    "finite_only_2x2_mean_require_at_least_2_of_4_native_values_"
    "and_nonfinite_fraction_le_1e-6"
)
MAX_HRES_SOURCE_NONFINITE_FRACTION = 1.0e-6


def forecast_evaluation_source_paths() -> dict[str, Path]:
    """Return code paths whose contents can change forecast-anchor outputs."""
    repo_root = Path(__file__).resolve().parents[2]
    model_root = repo_root / "weather_time_interp" / "model"
    return {
        "batch_eval_forecast_anchor.py": Path(__file__).resolve(),
        "capmatched_loader.py": (
            repo_root / "tools" / "eval" / "capmatched_loader.py"
        ),
        "normalization.py": (
            repo_root / "weather_time_interp" / "normalization.py"
        ),
        "grid.py": repo_root / "weather_time_interp" / "grid.py",
        "training_protocol.py": (
            repo_root / "tools" / "train" / "training_protocol.py"
        ),
        "train_capacity_matched_6h.py": (
            repo_root / "tools" / "train" / "train_capacity_matched_6h.py"
        ),
        "weatherbridge_flow_model.py": (
            model_root / "weatherbridge_flow_model.py"
        ),
        "dcae_adaln_model.py": model_root / "dcae_adaln_model.py",
        "dcae_adaln_skip_model.py": (
            model_root / "dcae_adaln_skip_model.py"
        ),
        "train_atm_vfi_12h_oddskip.py": (
            repo_root / "legacy" / "scripts" / "train_atm_vfi_12h_oddskip.py"
        ),
    }


# ------ IO ------

def read_init(memmap_bin: Path, memmap_json: Path):
    meta = json.loads(memmap_json.read_text())
    if meta.get("channel_order") != CHANNELS_ORDER:
        raise ValueError(
            f"{memmap_json}: channel_order does not match canonical 24 fields"
        )
    if meta.get("dtype") != "float32":
        raise ValueError(f"{memmap_json}: expected dtype=float32")
    arr = np.memmap(str(memmap_bin), dtype="float32", mode="r",
                    shape=tuple(meta["shape"]))
    return np.asarray(arr), meta["lead_hours"], meta


class Era5Cache:
    def __init__(self, era5_dir: Path):
        self.era5_dir = era5_dir
        self._maps: dict[int, tuple[np.memmap, dict]] = {}

    def _open(self, year: int):
        if year in self._maps:
            return self._maps[year]
        bin_p = self.era5_dir / f"wb2_{year}.bin"
        json_p = self.era5_dir / f"wb2_{year}.json"
        meta = json.loads(json_p.read_text())
        if meta.get("sparse_file") is True:
            selected = meta.get("selected_relative_hours")
            if (
                not isinstance(selected, list)
                or not selected
                or len(selected) != len(set(selected))
                or any(
                    not isinstance(hour, int)
                    or hour < 0
                    or hour >= int(meta["shape"][0])
                    for hour in selected
                )
            ):
                raise ValueError(f"{json_p}: invalid sparse-hour index")
            meta["_selected_relative_hours_set"] = frozenset(selected)
        mm = np.memmap(str(bin_p), dtype="float32", mode="r",
                       shape=tuple(meta["shape"]))
        self._maps[year] = (mm, meta)
        return self._maps[year]

    def hour(self, valid_time: np.datetime64) -> np.ndarray | None:
        """Return ERA5 (24, H, W) at valid_time; None if out of range."""
        y = int(str(valid_time)[:4])
        try:
            mm, meta = self._open(y)
        except FileNotFoundError:
            return None
        t0 = np.datetime64(f"{y}-01-01T00", "h")
        hi = int((np.datetime64(valid_time, "h") - t0) / np.timedelta64(1, "h"))
        if not (0 <= hi < meta["shape"][0]):
            return None
        if (
            meta.get("sparse_file") is True
            and hi not in meta["_selected_relative_hours_set"]
        ):
            return None
        return np.asarray(mm[hi, :24])  # first 24 ch = 20 PL + t2m u10 v10 mslp


# ------ Norm ------


def normalize_forecast_anchors(
    values: np.ndarray,
    stats: ChannelStats,
) -> np.ndarray:
    """Apply the training transform, imputing missing channels at the mean."""
    shape = (1,) * (values.ndim - 3) + (-1, 1, 1)
    mean = stats.mean.reshape(shape)
    finite_values = np.where(np.isfinite(values), values, mean)
    return stats.normalize(finite_values).astype(np.float32, copy=False)


def area_weighted_channel_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Return latitude-area-weighted normalised MSE for every channel."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have shape (C, H, W)")
    height, width = prediction.shape[-2:]
    if height == 360:
        latitude = wb2_block_average_latitudes(height)
        weights = latitude_strip_weights(latitude)
    else:
        latitude = np.linspace(
            90.0 - 90.0 / height,
            -90.0 + 90.0 / height,
            height,
            dtype=np.float64,
        )
        weights = np.cos(np.deg2rad(latitude))
    weights /= weights.sum()
    squared = (
        (prediction.astype(np.float64) - target.astype(np.float64))
        / std[:, None, None]
    ) ** 2
    return (
        (squared * weights[None, :, None]).sum(axis=(-2, -1))
        / width
    )


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp.npz"
    )
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _forecast_manifest_provenance(
    forecast_dir: Path,
    init_files: list[Path],
    *,
    sampled_bytes_per_binary: int = 1024 * 1024,
) -> dict[str, object]:
    """Validate the canonical HRES archive and fingerprint selected files.

    Full binary hashes are computed once by the archive builder. Repeated
    model evaluations bind to those hashes through immutable size/mtime
    records and fresh beginning/middle/end content samples.
    """
    if sampled_bytes_per_binary <= 0:
        raise ValueError("sampled_bytes_per_binary must be positive")

    manifest_path = forecast_dir / "forecast_archive_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"{forecast_dir}: missing canonical forecast archive manifest"
        )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 2
        or manifest.get("archive_kind")
        != "weatherbench2_ifs_hres_forecast_anchors"
        or manifest.get("source_qc_policy")
        not in {LEGACY_HRES_QC_POLICY, HRES_QC_POLICY}
    ):
        raise ValueError(f"{manifest_path}: unsupported archive manifest")
    expected_latitude_hash = _sha256_arrays(wb2_block_average_latitudes())
    expected_longitude_hash = _sha256_arrays(
        wb2_block_average_longitudes()
    )
    grid = manifest.get("grid", {})
    if (
        grid.get("name") != WB2_BLOCK_GRID_NAME
        or grid.get("latitude_order") != "north_to_south"
        or grid.get("latitude_sha256") != expected_latitude_hash
        or grid.get("longitude_sha256") != expected_longitude_hash
    ):
        raise ValueError(f"{manifest_path}: wrong forecast grid")
    builder = manifest.get("builder", {})
    builder_path_raw = builder.get("path")
    builder_sha256 = builder.get("sha256")
    if (
        not isinstance(builder_path_raw, str)
        or not builder_path_raw
        or not isinstance(builder_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", builder_sha256) is None
    ):
        raise ValueError(f"{manifest_path}: invalid archive builder provenance")
    builder_path = Path(builder_path_raw)
    current_builder = (
        file_provenance(builder_path) if builder_path.is_file() else None
    )
    builder_provenance = {
        "path": builder_path_raw,
        "sha256": builder_sha256,
        "current_path_exists": current_builder is not None,
        "current_path_sha256": (
            current_builder["sha256"] if current_builder is not None else None
        ),
        "current_path_matches_recorded_sha256": (
            current_builder is not None
            and current_builder["sha256"] == builder_sha256
        ),
    }
    manifest_files = manifest.get("files", {})
    if not isinstance(manifest_files, dict):
        raise TypeError(f"{manifest_path}: missing archive file records")

    def sampled_binary(path: Path) -> dict[str, object]:
        stat = path.stat()
        chunk_size = min(sampled_bytes_per_binary, stat.st_size)
        offsets = sorted(
            {
                0,
                max(0, (stat.st_size - chunk_size) // 2),
                max(0, stat.st_size - chunk_size),
            }
        )
        content = hashlib.sha256()
        with path.open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                chunk = handle.read(chunk_size)
                content.update(str(offset).encode("ascii"))
                content.update(chunk)
        return {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sampled_offsets": offsets,
            "sampled_sha256": content.hexdigest(),
        }

    digest = hashlib.sha256()
    total_size = 0
    files: dict[str, object] = {}
    for bin_path in init_files:
        json_path = bin_path.with_suffix(".json")
        record = manifest_files.get(bin_path.name)
        if not isinstance(record, dict):
            raise TypeError(f"{manifest_path}: unregistered {bin_path.name}")
        if record.get("sidecar") != json_path.name:
            raise ValueError(f"{manifest_path}: wrong sidecar for {bin_path.name}")
        sidecar_record = file_provenance(json_path)
        if sidecar_record["sha256"] != record.get("sidecar_sha256"):
            raise ValueError(f"{json_path}: changed since archive creation")
        sidecar = json.loads(json_path.read_text())
        sidecar_binary = sidecar.get("binary", {})
        sidecar_grid = sidecar.get("grid", {})
        sidecar_policy = sidecar.get("source_qc", {}).get("policy")
        valid_sidecar_schema = (
            sidecar.get("schema_version") == 3
            and sidecar_policy == LEGACY_HRES_QC_POLICY
        ) or (
            sidecar.get("schema_version") == 4
            and sidecar_policy == HRES_QC_POLICY
        )
        if (
            not valid_sidecar_schema
            or sidecar.get("channel_order") != CHANNELS_ORDER
            or sidecar.get("channels_available") != CHANNELS_ORDER
            or sidecar_grid != grid
            or sidecar_binary.get("sha256") != record.get("binary_sha256")
        ):
            raise ValueError(f"{json_path}: invalid canonical sidecar")
        source_qc = sidecar.get("source_qc", {})
        channel_qc = source_qc.get("channels", {})
        channel_missing = sum(
            int(item.get("source_nonfinite_values", -1))
            for item in channel_qc.values()
        )
        channel_affected = sum(
            int(item.get("affected_output_values", -1))
            for item in channel_qc.values()
        )
        channel_output_nonfinite = sum(
            int(item.get("output_nonfinite_values", -1))
            for item in channel_qc.values()
        )
        minimum_required = (
            3 if sidecar_policy == LEGACY_HRES_QC_POLICY else 2
        )
        fraction_invalid = (
            sidecar_policy == HRES_QC_POLICY
            and any(
                float(item.get("source_nonfinite_fraction", float("inf")))
                > MAX_HRES_SOURCE_NONFINITE_FRACTION
                for item in channel_qc.values()
            )
        )
        if (
            set(channel_qc) != set(CHANNELS_ORDER)
            or int(source_qc.get("minimum_finite_native_values_per_output", -1))
            < minimum_required
            or int(source_qc.get("output_nonfinite_values", -1)) != 0
            or channel_missing
            != int(source_qc.get("source_nonfinite_values", -2))
            or channel_affected
            != int(source_qc.get("affected_output_values", -2))
            or channel_output_nonfinite != 0
            or fraction_invalid
            or any(
                int(item.get("minimum_finite_native_values_per_output", -1))
                < minimum_required
                or int(item.get("output_nonfinite_values", -1)) != 0
                for item in channel_qc.values()
            )
        ):
            raise ValueError(f"{json_path}: invalid source-QC record")
        stat = bin_path.stat()
        if (
            stat.st_size != int(record.get("binary_size_bytes", -1))
            or stat.st_mtime_ns != int(record.get("binary_mtime_ns", -1))
            or stat.st_size != int(sidecar_binary.get("size_bytes", -1))
            or stat.st_mtime_ns != int(sidecar_binary.get("mtime_ns", -1))
        ):
            raise ValueError(f"{bin_path}: binary identity changed")
        binary = sampled_binary(bin_path)
        total_size += int(binary["size_bytes"]) + int(sidecar_record["size_bytes"])
        entry = {
            "binary": binary,
            "binary_full_sha256": record["binary_sha256"],
            "sidecar": sidecar_record,
        }
        files[bin_path.name] = entry
        digest.update(bin_path.name.encode("utf-8"))
        digest.update(
            json.dumps(entry, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    return {
        "path": str(forecast_dir.resolve()),
        "init_count": len(init_files),
        "total_size_bytes": total_size,
        "sampled_bytes_per_binary": sampled_bytes_per_binary,
        "files": files,
        "manifest_sha256": digest.hexdigest(),
        "archive_manifest": file_provenance(manifest_path),
        "source": manifest.get("source"),
        "grid": grid,
        "builder": builder_provenance,
        "source_qc_policy": manifest["source_qc_policy"],
    }


def select_init_files(
    init_files: list[Path],
    max_inits: int,
) -> list[Path]:
    """Select an evenly spaced, deterministic subset across the archive."""
    if max_inits <= 0 or max_inits >= len(init_files):
        return init_files
    indices = np.rint(
        np.linspace(0, len(init_files) - 1, max_inits)
    ).astype(np.int64)
    if np.unique(indices).size != max_inits:
        raise RuntimeError("stratified init selection produced duplicates")
    return [init_files[int(index)] for index in indices]


def select_anchor_pairs(
    lead_hours: list[int],
    delta_t_hours: int,
    *,
    lead_stride_hours: int | None = None,
    maximum_left_lead_hours: int | None = None,
) -> list[tuple[int, int]]:
    """Return index pairs whose forecast leads are exactly delta_t apart."""
    if lead_stride_hours is not None and lead_stride_hours <= 0:
        raise ValueError("lead_stride_hours must be positive")
    if maximum_left_lead_hours is not None and maximum_left_lead_hours <= 0:
        raise ValueError("maximum_left_lead_hours must be positive")
    leads = [int(lead) for lead in lead_hours]
    if len(leads) != len(set(leads)):
        raise ValueError("forecast lead_hours must be unique")
    if leads != sorted(leads):
        raise ValueError("forecast lead_hours must be sorted")
    index_by_lead = {lead: index for index, lead in enumerate(leads)}
    return [
        (left_index, index_by_lead[left_lead + delta_t_hours])
        for left_index, left_lead in enumerate(leads)
        if left_lead + delta_t_hours in index_by_lead
        and (
            lead_stride_hours is None
            or left_lead % lead_stride_hours == 0
        )
        and (
            maximum_left_lead_hours is None
            or left_lead < maximum_left_lead_hours
        )
    ]


def select_forecast_target_index(
    lead_hours: list[int],
    left_index: int,
    tau_hours: int,
) -> int | None:
    """Return the same-trajectory target index when that lead is available."""
    leads = [int(lead) for lead in lead_hours]
    target_lead = leads[left_index] + int(tau_hours)
    return {lead: index for index, lead in enumerate(leads)}.get(target_lead)


def forecast_lead_bin(lead_hours: int) -> str | None:
    """Return the declared forecast-age bin for an anchor lead."""
    for name, lower, upper in FORECAST_LEAD_BINS:
        if lower <= lead_hours < upper:
            return name
    return None


# ------ Model wrap ------

class LinearInterp:
    def __call__(self, x0: np.ndarray, xT: np.ndarray, tau_h: int, dt: int = 6):
        w = tau_h / dt
        return (1 - w) * x0 + w * xT


class BareModel:
    def __init__(self, blob_path: str, device: str, stats: ChannelStats,
                 static_path: str | None = None):
        if torch is None:
            raise RuntimeError("torch not available — install torch to use non-linear models")
        import sys
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from examples._bare_loader import load_bare
        self.net = load_bare(blob_path, device)
        self.device = device
        self.stats = stats
        self.mean = torch.from_numpy(stats.mean).to(device).view(1, -1, 1, 1)
        self.std = torch.from_numpy(stats.std).to(device).view(1, -1, 1, 1)
        self.static = None
        n_static = getattr(self.net, "n_static_features", 0)
        if n_static > 0:
            sp = static_path or str(root / "data" / "static_features_0p5.pt")
            s = torch.load(sp, weights_only=False).float()
            if s.dim() == 3:
                s = s.unsqueeze(0)
            self.static = s[:, :n_static].to(device)

    def __call__(self, x0: np.ndarray, xT: np.ndarray, tau_h: int, dt: int = 6):
        with torch.no_grad():
            x0_norm = normalize_forecast_anchors(x0, self.stats)
            xT_norm = normalize_forecast_anchors(xT, self.stats)
            x0_t = torch.from_numpy(x0_norm[None]).to(self.device)
            xT_t = torch.from_numpy(xT_norm[None]).to(self.device)
            tau_t = torch.tensor([[tau_h / dt]], device=self.device, dtype=torch.float32)
            # ATM-VFI uses model.net(x0, xT, tau) — 3 args, static baked in.
            if type(self.net).__name__ == "PixelAttentionVFI":
                out = self.net.net(x0_t, xT_t, tau_t)
            else:
                cond_t = torch.tensor([float(dt)], device=self.device, dtype=torch.float32)
                kwargs = {"static": self.static} if self.static is not None else {}
                out = self.net(x0_t, xT_t, tau_t, cond_t, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            return (out * self.std + self.mean).cpu().numpy()[0]


class CheckpointModel:
    """Run a checkpoint through the provenance-bound inference loader."""

    def __init__(
        self,
        checkpoint_path: str,
        arch: str,
        device: str,
        stats: ChannelStats,
        *,
        static_path: str,
        delta_t_hours: int,
        taus: list[int],
    ):
        if torch is None:
            raise RuntimeError("torch is required for checkpoint evaluation")
        from tools.eval.capmatched_loader import load_capmatched_checkpoint

        self.device = device
        self.stats = stats
        self.mean = torch.from_numpy(stats.mean).to(device).view(1, -1, 1, 1)
        self.std = torch.from_numpy(stats.std).to(device).view(1, -1, 1, 1)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        hyper_parameters = checkpoint.get("hyper_parameters", {})
        checkpoint_arch = hyper_parameters.get("arch")
        if checkpoint_arch != arch:
            raise ValueError(
                f"{checkpoint_path}: requested arch={arch!r}, "
                f"checkpoint records arch={checkpoint_arch!r}"
            )
        try:
            checkpoint_delta_t = float(hyper_parameters["delta_t"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{checkpoint_path}: missing or invalid delta_t"
            ) from exc
        if not math.isclose(
            checkpoint_delta_t,
            float(delta_t_hours),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"{checkpoint_path}: requested delta_t={delta_t_hours}, "
                f"checkpoint records delta_t={checkpoint_delta_t}"
            )
        self.checkpoint_metadata = {
            "epoch": int(checkpoint.get("epoch", -1)),
            "global_step": int(checkpoint.get("global_step", -1)),
            "training_protocol": hyper_parameters.get("training_protocol"),
            "training_input_provenance": hyper_parameters.get(
                "training_input_provenance"
            ),
            "training_code_sha256": hyper_parameters.get(
                "training_code_sha256"
            ),
            "resume_lineage": hyper_parameters.get("resume_lineage"),
        }
        del checkpoint
        self.model, self.model_type = load_capmatched_checkpoint(
            checkpoint_path,
            torch.device(device),
            static_path=static_path,
        )

    def __call__(
        self,
        x0: np.ndarray,
        xT: np.ndarray,
        tau_h: int,
        dt: int = 6,
    ) -> np.ndarray:
        return self.predict_taus(x0, xT, [tau_h], dt=dt)[0]

    def predict_taus(
        self,
        x0: np.ndarray,
        xT: np.ndarray,
        tau_hours: list[int],
        *,
        dt: int = 6,
        chunk_size: int = 2,
        anchor_lead_hours: int | None = None,
    ) -> np.ndarray:
        """Evaluate several query times while reusing normalized anchors."""
        if not tau_hours or chunk_size < 1:
            raise ValueError("tau_hours and chunk_size must be positive")
        x0_norm = normalize_forecast_anchors(x0, self.stats)
        xT_norm = normalize_forecast_anchors(xT, self.stats)
        device_type = torch.device(self.device).type
        predictions: list[np.ndarray] = []
        for start in range(0, len(tau_hours), chunk_size):
            chunk = tau_hours[start : start + chunk_size]
            size = len(chunk)
            x0_tensor = torch.from_numpy(x0_norm[None]).to(self.device).expand(
                size, -1, -1, -1
            )
            xT_tensor = torch.from_numpy(xT_norm[None]).to(self.device).expand(
                size, -1, -1, -1
            )
            tau = torch.tensor(
                [tau_h / dt for tau_h in chunk],
                device=self.device,
                dtype=torch.float32,
            )
            with torch.no_grad(), torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=device_type == "cuda",
            ):
                if anchor_lead_hours is None:
                    output = self.model(x0_tensor, xT_tensor, tau)
                else:
                    lead = torch.full(
                        (size,),
                        float(anchor_lead_hours),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    output = self.model(x0_tensor, xT_tensor, tau, lead)
            if isinstance(output, tuple):
                output = output[0]
            physical = (output.float() * self.std + self.mean).cpu().numpy()
            predictions.extend(physical)
        return np.stack(predictions)


class ResidualBlendAdapter:
    """Blend a frozen model correction with the exact linear scaffold."""

    def __init__(
        self,
        base: CheckpointModel,
        adapter_path: str | Path,
        checkpoint_path: str | Path,
        arch: str,
        delta_t_hours: int,
    ) -> None:
        self.base = base
        self.checkpoint_metadata = base.checkpoint_metadata
        self.path = Path(adapter_path)
        payload = json.loads(self.path.read_text())
        checkpoint_record = payload.get("base_checkpoint", {})
        schema_version = payload.get("schema_version")
        if (
            schema_version not in (1, 2)
            or payload.get("kind") != "nwp_linear_residual_blend"
            or payload.get("arch") != arch
            or int(payload.get("delta_t_hours", -1)) != delta_t_hours
            or payload.get("channel_order") != CHANNELS_ORDER
            or checkpoint_record.get("sha256")
            != file_provenance(checkpoint_path)["sha256"]
        ):
            raise ValueError(f"{self.path}: incompatible NWP blend adapter")
        raw_gates = payload.get("gates", {})
        raw_biases = payload.get("bias_normalized", {})
        self.gates = {
            int(tau): np.asarray(values, dtype=np.float32).reshape(-1, 1, 1)
            for tau, values in raw_gates.items()
        }
        self.biases = {
            int(tau): np.asarray(values, dtype=np.float32).reshape(-1, 1, 1)
            for tau, values in raw_biases.items()
        }
        expected_taus = set(range(1, delta_t_hours))
        if (
            set(self.gates) != expected_taus
            or set(self.biases) != expected_taus
            or any(
            gate.shape != (len(CHANNELS_ORDER), 1, 1)
            or not np.isfinite(gate).all()
            or (gate < 0.0).any()
            or (gate > 1.0).any()
            for gate in self.gates.values()
            )
            or any(
                bias.shape != (len(CHANNELS_ORDER), 1, 1)
                or not np.isfinite(bias).all()
                for bias in self.biases.values()
            )
        ):
            raise ValueError(f"{self.path}: invalid channelwise blend gates")
        self.physical_bias_scale = base.stats.std.reshape(-1, 1, 1)
        if schema_version == 1:
            self.lead_scales = None
        else:
            raw_lead_scales = payload.get("lead_scale_bins", {})
            expected = {
                name: {"lower_hours": lower, "upper_hours": upper}
                for name, lower, upper in FORECAST_LEAD_BINS
            }
            if set(raw_lead_scales) != set(expected):
                raise ValueError(f"{self.path}: invalid lead-scale bins")
            self.lead_scales = {}
            for name, bounds in expected.items():
                record = raw_lead_scales[name]
                scale = float(record.get("scale", float("nan")))
                if (
                    record.get("lower_hours") != bounds["lower_hours"]
                    or record.get("upper_hours") != bounds["upper_hours"]
                    or not np.isfinite(scale)
                    or not 0.0 <= scale <= 1.0
                ):
                    raise ValueError(f"{self.path}: invalid lead scale {name}")
                self.lead_scales[name] = scale

    def __call__(
        self,
        x0: np.ndarray,
        xT: np.ndarray,
        tau_h: int,
        dt: int = 6,
        anchor_lead_hours: int | None = None,
    ) -> np.ndarray:
        return self.predict_taus(
            x0,
            xT,
            [tau_h],
            dt=dt,
            chunk_size=1,
            anchor_lead_hours=anchor_lead_hours,
        )[0]

    def predict_taus(
        self,
        x0: np.ndarray,
        xT: np.ndarray,
        tau_hours: list[int],
        *,
        dt: int = 6,
        chunk_size: int = 2,
        anchor_lead_hours: int | None = None,
    ) -> np.ndarray:
        linear = np.stack(
            [LinearInterp()(x0, xT, tau_h, dt) for tau_h in tau_hours]
        )
        lead_scale = 1.0
        if self.lead_scales is not None:
            if anchor_lead_hours is None:
                raise ValueError("lead-aware adapter requires anchor_lead_hours")
            lead_bin = forecast_lead_bin(anchor_lead_hours)
            if lead_bin is None:
                raise ValueError(
                    f"anchor lead {anchor_lead_hours} is outside adapter bins"
                )
            lead_scale = self.lead_scales[lead_bin]
        if lead_scale == 0.0:
            return linear

        active_indices = [
            index
            for index, tau_h in enumerate(tau_hours)
            if np.any(self.gates[int(tau_h)] != 0.0)
            or np.any(self.biases[int(tau_h)] != 0.0)
        ]
        if not active_indices:
            return linear
        active_taus = [tau_hours[index] for index in active_indices]
        predict_kwargs = {
            "dt": dt,
            "chunk_size": min(chunk_size, len(active_taus)),
        }
        parameters = inspect.signature(self.base.predict_taus).parameters
        if "anchor_lead_hours" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            predict_kwargs["anchor_lead_hours"] = anchor_lead_hours
        model = self.base.predict_taus(x0, xT, active_taus, **predict_kwargs)
        result = linear.copy()
        for model_index, output_index in enumerate(active_indices):
            tau_h = int(tau_hours[output_index])
            gate = self.gates[tau_h]
            bias = self.biases[tau_h] * self.physical_bias_scale
            result[output_index] += lead_scale * (
                gate * (model[model_index] - linear[output_index]) + bias
            )
        return result


# ------ Main ------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast-dir", required=True,
                    help="dir of init_*.bin/json from preprocess_forecast_to_memmap")
    ap.add_argument(
        "--target-source",
        choices=("era5", "same_forecast"),
        default="era5",
        help=(
            "intermediate truth source: ERA5 at the valid time or an "
            "available lead from the same forecast initialisation"
        ),
    )
    ap.add_argument(
        "--era5-memmap-dir",
        default=None,
        help="ERA5 hourly memmap dir; required when --target-source=era5",
    )
    model_group = ap.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-blob",
        help="*.pt bare blob; use 'linear' for baseline only",
    )
    model_group.add_argument(
        "--checkpoint",
        help="CapMatchedLit checkpoint evaluated through the training wrapper",
    )
    ap.add_argument(
        "--arch",
        default=None,
        help="internal architecture name required with --checkpoint",
    )
    ap.add_argument(
        "--blend-adapter",
        default=None,
        help="provenance-bound channelwise Linear/model blend JSON",
    )
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument(
        "--out-npz",
        default=None,
        help="paired per-window artifact; defaults beside --out-json",
    )
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument(
        "--surface-stats-path",
        default="data/surface_stats_0p5.json",
    )
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--delta-t-hours", type=int, default=6)
    ap.add_argument("--forecast-lead-stride-hours", type=int, default=None)
    ap.add_argument(
        "--forecast-maximum-left-lead-hours",
        type=int,
        default=None,
    )
    ap.add_argument("--taus", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument(
        "--tau-batch-size",
        type=int,
        default=1,
        help="number of query times evaluated in one checkpoint forward pass",
    )
    ap.add_argument("--max-inits", type=int, default=-1,
                    help="cap #inits (for smoke)")
    args = ap.parse_args()
    if args.delta_t_hours < 2:
        raise SystemExit("--delta-t-hours must be at least 2")
    if args.tau_batch_size < 1:
        raise SystemExit("--tau-batch-size must be positive")
    if (
        args.forecast_lead_stride_hours is not None
        and args.forecast_lead_stride_hours <= 0
    ):
        raise SystemExit("--forecast-lead-stride-hours must be positive")
    if (
        args.forecast_maximum_left_lead_hours is not None
        and args.forecast_maximum_left_lead_hours <= 0
    ):
        raise SystemExit(
            "--forecast-maximum-left-lead-hours must be positive"
        )
    if (
        not args.taus
        or len(args.taus) != len(set(args.taus))
        or any(not 1 <= tau < args.delta_t_hours for tau in args.taus)
    ):
        raise SystemExit(
            "--taus must be unique interior hours of --delta-t-hours"
        )
    if args.checkpoint and not args.arch:
        raise SystemExit("--arch is required with --checkpoint")
    if args.arch and not args.checkpoint:
        raise SystemExit("--arch is only valid with --checkpoint")
    if args.blend_adapter and not args.checkpoint:
        raise SystemExit("--blend-adapter requires --checkpoint")
    if args.target_source == "era5" and not args.era5_memmap_dir:
        raise SystemExit(
            "--era5-memmap-dir is required when --target-source=era5"
        )

    forecast_dir = Path(args.forecast_dir)
    era5_cache = (
        Era5Cache(Path(args.era5_memmap_dir))
        if args.target_source == "era5"
        else None
    )
    stats = load_channel_stats(
        args.stats_path,
        args.surface_stats_path,
        CHANNELS_ORDER,
    )
    stds = stats.std

    if args.model_blob == "linear":
        interp = LinearInterp()
        model_key = args.model_name or "linear"
    elif args.checkpoint:
        interp = CheckpointModel(
            args.checkpoint,
            args.arch,
            args.device,
            stats,
            static_path=args.static_path,
            delta_t_hours=args.delta_t_hours,
            taus=args.taus,
        )
        if args.blend_adapter:
            interp = ResidualBlendAdapter(
                interp,
                args.blend_adapter,
                args.checkpoint,
                args.arch,
                args.delta_t_hours,
            )
        model_key = args.model_name or args.arch
    else:
        interp = BareModel(
            args.model_blob,
            args.device,
            stats,
            static_path=args.static_path,
        )
        model_key = args.model_name or "model"

    # Aggregators: (tau, bucket) → sxx accumulator per channel
    LEAD_BINS = FORECAST_LEAD_BINS
    SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    n_ch = len(CHANNELS_ORDER)

    def _new_accum():
        return {tau: {"sxx": np.zeros(n_ch, dtype=np.float64),
                       "cnt": np.zeros(n_ch, dtype=np.int64),
                       "n": 0} for tau in args.taus}
    acc_all   = _new_accum()
    acc_lead  = {name: _new_accum() for name, _, _ in LEAD_BINS}
    acc_season = {s: _new_accum() for s in SEASONS}

    def _bucket_lead(lead_a: int) -> str | None:
        for name, lo, hi in LEAD_BINS:
            if lo <= lead_a < hi:
                return name
        return None

    def _season_for(dt64: np.datetime64) -> str | None:
        month = int(str(dt64)[5:7])
        for s, months in SEASONS.items():
            if month in months:
                return s
        return None

    init_files = sorted(forecast_dir.glob("init_*.bin"))
    init_files = select_init_files(init_files, args.max_inits)
    if not init_files:
        raise SystemExit(f"no init_*.bin files in {forecast_dir}")
    print(f"[init] {len(init_files)} forecast inits, model={model_key}", flush=True)

    window_init_time: list[int] = []
    window_lead: list[int] = []
    window_tau: list[int] = []
    window_squared_error: list[np.ndarray] = []
    window_valid_channel: list[np.ndarray] = []
    target_years: set[int] = set()
    for fi, bin_p in enumerate(init_files):
        json_p = bin_p.with_suffix(".json")
        arr, leads, meta = read_init(bin_p, json_p)
        init_time = np.datetime64(meta["init_time"])
        avail_mask = np.array([c in meta["channels_available"] for c in CHANNELS_ORDER])
        n_processed = 0
        for left_index, right_index in select_anchor_pairs(
            leads,
            args.delta_t_hours,
            lead_stride_hours=args.forecast_lead_stride_hours,
            maximum_left_lead_hours=(
                args.forecast_maximum_left_lead_hours
            ),
        ):
            lead_a = int(leads[left_index])
            lead_bucket = _bucket_lead(lead_a)
            a = arr[left_index]      # (24, H, W)
            b = arr[right_index]
            if isinstance(interp, ResidualBlendAdapter):
                predictions = interp.predict_taus(
                    a,
                    b,
                    args.taus,
                    dt=args.delta_t_hours,
                    chunk_size=args.tau_batch_size,
                    anchor_lead_hours=lead_a,
                )
            elif hasattr(interp, "predict_taus"):
                predictions = interp.predict_taus(
                    a,
                    b,
                    args.taus,
                    dt=args.delta_t_hours,
                    chunk_size=args.tau_batch_size,
                    anchor_lead_hours=lead_a,
                )
            else:
                predictions = np.stack(
                    [
                        interp(a, b, tau, dt=args.delta_t_hours)
                        for tau in args.taus
                    ]
                )
            for tau, pred in zip(args.taus, predictions, strict=True):
                valid = init_time + np.timedelta64(lead_a + tau, "h")
                if args.target_source == "same_forecast":
                    target_index = select_forecast_target_index(
                        leads,
                        left_index,
                        tau,
                    )
                    if target_index is None:
                        continue
                    tgt = arr[target_index]
                else:
                    assert era5_cache is not None
                    tgt = era5_cache.hour(valid)
                    if tgt is None:
                        continue
                    target_years.add(int(str(valid)[:4]))
                season = _season_for(valid)
                mean_sq_ch = area_weighted_channel_mse(pred, tgt, stds)
                finite_anchors = (
                    np.isfinite(a).all(axis=(1, 2))
                    & np.isfinite(b).all(axis=(1, 2))
                )
                valid_ch_mask = (
                    avail_mask
                    & finite_anchors
                    & np.isfinite(mean_sq_ch)
                )
                paired_squared_error = np.where(
                    valid_ch_mask,
                    mean_sq_ch,
                    np.nan,
                ).astype(np.float32)
                mean_sq_ch = np.where(valid_ch_mask, mean_sq_ch, 0.0)

                acc_all[tau]["sxx"] += mean_sq_ch
                acc_all[tau]["cnt"] += valid_ch_mask.astype(np.int64)
                acc_all[tau]["n"] += 1
                if lead_bucket is not None:
                    acc_lead[lead_bucket][tau]["sxx"] += mean_sq_ch
                    acc_lead[lead_bucket][tau]["cnt"] += valid_ch_mask.astype(np.int64)
                    acc_lead[lead_bucket][tau]["n"] += 1
                if season is not None:
                    acc_season[season][tau]["sxx"] += mean_sq_ch
                    acc_season[season][tau]["cnt"] += valid_ch_mask.astype(np.int64)
                    acc_season[season][tau]["n"] += 1
                window_init_time.append(
                    int(init_time.astype("datetime64[h]").astype(np.int64))
                )
                window_lead.append(int(lead_a))
                window_tau.append(int(tau))
                window_squared_error.append(paired_squared_error)
                window_valid_channel.append(valid_ch_mask)
                n_processed += 1
        if (fi + 1) % 20 == 0 or fi == 0:
            print(f"[{fi+1}/{len(init_files)}] init {str(init_time)[:13]} "
                  f"pairs+={n_processed}", flush=True)

    def _pack(accum) -> dict:
        out = {}
        for tau in args.taus:
            rec = {}
            for ci, c in enumerate(CHANNELS_ORDER):
                n = int(accum[tau]["cnt"][ci])
                rec[f"rmse_norm_{c}"] = (float(np.sqrt(accum[tau]["sxx"][ci] / max(n, 1)))
                                          if n else None)
                rec[f"n_pairs_{c}"] = n
            rec["n_pairs_any"] = int(accum[tau]["n"])
            out[str(tau)] = {model_key: rec}
        return out

    per_tau = _pack(acc_all)
    per_lead = {name: _pack(acc_lead[name]) for name, _, _ in LEAD_BINS}
    per_season = {s: _pack(acc_season[s]) for s in SEASONS}

    if not window_squared_error:
        raise SystemExit(
            "no matched forecast-anchor/target windows for the requested protocol"
        )
    paired_init = np.asarray(window_init_time, dtype=np.int64)
    paired_lead = np.asarray(window_lead, dtype=np.int16)
    paired_tau = np.asarray(window_tau, dtype=np.int8)
    paired_squared = np.stack(window_squared_error).astype(np.float32)
    paired_valid = np.stack(window_valid_channel).astype(np.bool_)
    index_sha256 = _sha256_arrays(paired_init, paired_lead, paired_tau)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npz = (
        Path(args.out_npz)
        if args.out_npz
        else out_json.with_name(f"{out_json.stem}.paired.npz")
    )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(
        out_npz,
        init_time_hours=paired_init,
        anchor_lead_hours=paired_lead,
        tau_hours=paired_tau,
        squared_error_norm=paired_squared,
        valid_channel=paired_valid,
        channel_names=np.asarray(CHANNELS_ORDER),
    )

    if args.model_blob == "linear":
        model_provenance = {"kind": "linear"}
    elif args.checkpoint:
        model_provenance = {
            "kind": (
                "capmatched_checkpoint_with_nwp_blend"
                if args.blend_adapter
                else "capmatched_checkpoint"
            ),
            "arch": args.arch,
            "artifact": file_provenance(args.checkpoint),
            "checkpoint_metadata": interp.checkpoint_metadata,
            "blend_adapter": (
                file_provenance(args.blend_adapter)
                if args.blend_adapter
                else None
            ),
        }
    else:
        model_provenance = {
            "kind": "bare_blob",
            "artifact": file_provenance(args.model_blob),
        }
    forecast_provenance = _forecast_manifest_provenance(
        forecast_dir,
        init_files,
    )
    out = {
        "schema_version": 2,
        "model_name": model_key,
        "per_tau": per_tau,
        "per_lead_bin": per_lead,
        "per_season": per_season,
        "lead_bins": [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in LEAD_BINS],
        "seasons": SEASONS,
        "channels": CHANNELS_ORDER,
        "protocol": {
            "delta_t_hours": args.delta_t_hours,
            "taus": args.taus,
            "anchor_pairing": "same_initialization_forecast_leads",
            "target_source": (
                "same_initialization_forecast_lead"
                if args.target_source == "same_forecast"
                else "ERA5_at_intermediate_valid_time"
            ),
            "future_analysis_as_input": False,
            "area_weighting": "spherical_latitude_strip_area",
            "latitude_grid": WB2_BLOCK_GRID_NAME,
            "normalization": "(x - training_mean) / training_std",
            "missing_anchor_channel_imputation": "training_mean",
            "missing_channels_excluded_from_metrics": True,
            "init_selection": (
                "all"
                if args.max_inits <= 0
                else "evenly_spaced_over_sorted_archive"
            ),
            "max_inits": args.max_inits,
            "inference_tau_batch_size": args.tau_batch_size,
            "forecast_lead_stride_hours": (
                args.forecast_lead_stride_hours
            ),
            "maximum_left_forecast_lead_hours_exclusive": (
                args.forecast_maximum_left_lead_hours
            ),
        },
        "provenance": {
            "evaluator": file_provenance(__file__),
            "evaluation_code": {
                name: file_provenance(path)
                for name, path in forecast_evaluation_source_paths().items()
            },
            "model": model_provenance,
            "forecast_anchors": forecast_provenance,
            "forecast_targets": (
                {"same_artifact_as": "forecast_anchors"}
                if args.target_source == "same_forecast"
                else None
            ),
            "era5_truth": (
                memmap_dataset_provenance(
                    args.era5_memmap_dir,
                    sorted(target_years),
                )
                if args.target_source == "era5"
                else None
            ),
            "pressure_stats": file_provenance(args.stats_path),
            "surface_stats": file_provenance(args.surface_stats_path),
            "static_features": (
                static_feature_provenance(args.static_path)
                if args.model_blob != "linear" or args.checkpoint
                else None
            ),
        },
        "paired_artifact": {
            **file_provenance(out_npz),
            "n_windows": int(paired_tau.size),
            "window_index_sha256": index_sha256,
        },
    }
    _atomic_write_json(out_json, out)
    print(f"[write] {out_json}")
    print(f"[write] {out_npz}")


if __name__ == "__main__":
    main()

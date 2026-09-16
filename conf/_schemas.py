"""Structured config schemas for Hydra 1.3 train.py / eval.py entry points.

Type-safe dataclasses backing the YAML group composition.  Each group's
YAML maps onto one of these dataclasses; Hydra validates at instantiation
time.  We register these with `ConfigStore` in `train.py` / `eval.py` so
the CLI gets type checking + autocompletion via `python train.py --help`.

The actual model objects are constructed in the entry-point scripts via
`hydra.utils.instantiate(cfg.model)` against `_target_` fields in the
YAML — this preserves type safety for the schema (channel counts, paths,
hyper-params) without forcing the model class itself to be a dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


# -------------------------------------------------------------------- DATA #
@dataclass
class DataConfig:
    """ERA5 dataset (memmap-backed)."""
    _target_: str = "weather_time_interp.memmap_dataset.ERA5MemmapDataset"
    name: str = "era5_0p5_6h"
    memmap_dir: str = "/tmp/wb2_0p5_cache"
    static_path: str = "data/static_features_0p5.pt"
    stats_path: str = "data/json_stats_0p5.nc"
    surface_stats_path: str = "data/surface_stats_0p5.json"
    years: List[int] = field(default_factory=lambda: [2018])
    val_years: List[int] = field(default_factory=lambda: [2020])
    max_tau_hours: int = 6
    delta_t_hours: float = 6.0
    samples_per_date: int = 4
    samples_per_date_train: int = 4
    samples_per_date_val: int = 1
    train_hours: List[int] = field(default_factory=lambda: [1, 3, 5])
    eval_hours: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    keep_24ch: bool = True
    # Number of pressure-level channels in the dataset (T,U,V,Q,Z × 4 levels = 20).
    n_pl_channels: int = 20
    # Surface channel count after keep_24ch filter (4 if 24ch else 7).
    n_surface_channels: int = 4
    # Static features: land-sea mask, normalized orography, cosine latitude.
    n_static_features: int = 3


# -------------------------------------------------------------------- MODEL #
@dataclass
class ModelConfig:
    """Base model config — discriminator on `kind`.

    kind in {hermite, atm_vfi, corrdiff_fm}.

    - hermite: routed through WeatherHermiteLightningModule + model_type
    - atm_vfi: PixelAttentionVFI Lightning class (own class)
    - corrdiff_fm: CorrDiffFMVFI Lightning class (own class), needs `base_ckpt`
    """
    name: str = "weatherdcae"
    kind: str = "hermite"
    # ----- hermite: forwarded to WeatherHermiteLightningModule -----
    model_type: str = "dcae_adaln_residual_linear"
    latent_channels: int = 256
    lat_crop: int = -8
    # DC-AE-family knobs (only used if `dcae` in model_type)
    block_out_channels: Optional[List[int]] = None
    layers_per_block: Optional[List[int]] = None
    freeze_skip_gates: bool = False
    direct_prediction: bool = False
    # FuXi SwinV2 knobs
    fuxi_depth: int = 8
    fuxi_num_heads: int = 8
    fuxi_window_size_h: int = 5
    fuxi_window_size_w: int = 9
    fuxi_patch_size: int = 4
    fuxi_drop_path: float = 0.1
    # S-DYff knobs
    sdyff_num_layers: int = 6
    sdyff_n_modes_lat: int = 16
    sdyff_n_modes_lon: int = 32
    sdyff_dropout: float = 0.1
    sdyff_drop_path: float = 0.1
    sdyff_inference_steps: int = 5
    sdyff_train_refine_steps: int = 1
    # ModAFNO knobs
    modafno_depth: int = 12
    modafno_num_blocks: int = 8
    modafno_drop_rate: float = 0.0
    # Env-var overrides (set in train.py from these before model instantiation).
    env_vars: dict = field(default_factory=dict)
    # Loss knobs
    lat_weighted_loss: bool = True
    lambda_residual: float = 1.0
    residual_scale_floor: float = 0.05
    residual_scale_init: float = 0.30
    lambda_anchor: float = 0.5
    anchor_every_n_batches: int = 4
    use_aurora_weights: bool = True
    use_physical_scales_loss: bool = False
    seen_hours: Optional[List[int]] = None
    # ----- atm_vfi: PixelAttentionVFI -----
    hidden: int = 64
    n_levels: int = 3
    # ----- corrdiff_fm: CorrDiffFMVFI -----
    base_ckpt: Optional[str] = None
    n_ode_steps: int = 5
    scale_init: float = 0.10
    # ----- shared -----
    lr: float = 1e-4
    weight_decay: float = 1e-5


# ------------------------------------------------------------------ TRAINER #
@dataclass
class TrainerConfig:
    """Lightning Trainer + run params."""
    name: str = "default"
    max_epochs: int = 8
    batch_size: int = 1
    val_batch_size: int = 1
    num_workers: int = 4
    val_num_workers: int = 2
    precision: str = "bf16-mixed"
    devices: Any = 1
    strategy: str = "auto"
    accelerator: str = "gpu"
    log_every_n_steps: int = 20
    ckpt_every_n_epochs: int = 2
    save_top_k: int = -1
    val_every_n_epochs: int = 1
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    sanity_val_steps: int = 0
    resume_ckpt: Optional[str] = None
    enable_progress_bar: bool = False
    gradient_clip_val: Optional[float] = None


# --------------------------------------------------------------------- EVAL #
@dataclass
class EvalConfig:
    """Discriminator on `kind` ∈ {memmap_6h, memmap_12h, crps_ensemble, sh_spectra}.

    For `memmap_*`: writes `metrics/${out_dir}/${model_name}.json` via
    `batch_eval_memmap.main()`.  Multiple models accepted via `models` list
    (NAME:CKPT:ENVS strings).
    """
    name: str = "memmap_6h"
    kind: str = "memmap_6h"
    # Shared params
    memmap_dir: str = "/tmp/wb2_0p5_cache"
    test_year: int = 2020
    climatology: str = (
        "/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/"
        "climatology_1990-2019_0p5_canonical_v2.zarr"
    )
    stats_path: str = "data/json_stats_0p5.nc"
    surface_stats_path: str = "data/surface_stats_0p5.json"
    static_path: str = "data/static_features_0p5.pt"
    batch_size: int = 4
    num_workers: int = 2
    samples_per_date: int = 4
    eval_days_per_month: int = 4
    max_tau_hours: int = 6
    keep_n_channels: Optional[int] = 24
    out_dir: str = "metrics/eval_0p5_2020_fast"
    out_acc_dir: Optional[str] = "metrics/acc_0p5_2020_fast"
    paper_tag: Optional[str] = None
    eval_hours: Optional[str] = None
    seen_tau: Optional[str] = None
    unseen_tau: Optional[str] = None
    no_acc: bool = False
    device: Optional[str] = None
    # Models list: comma-separated NAME:CKPT:ENVS triples (matches
    # batch_eval_memmap.py CLI). Set per-eval-job.
    models: str = ""
    # crps_ensemble-specific
    crps_mode: str = "corrdiff_fm"   # corrdiff_fm | sdyff_mc
    ckpt: Optional[str] = None
    fm_ckpt: Optional[str] = None
    base_ckpt: Optional[str] = None
    n_ensemble: int = 16
    model_name: Optional[str] = None
    out_rmse: Optional[str] = None
    out_crps: Optional[str] = None
    # sh_spectra-specific
    sh_n_dates: int = 24
    sh_tau_hours: int = 3
    sh_channels: str = "t2m,u10,v10,mslp"
    sh_lmax: int = 180
    sh_envs: Optional[str] = None
    # sh_spectra_12h-specific (sh_energy_spectra_12h.py)
    sh_taus: Optional[str] = None              # comma-separated τ list, e.g. "2,3,5,8"
    sh_model_kind: Optional[str] = None        # hermite | atm_vfi | bilinear
    sh_skip_existing: bool = False
    # bootstrap_ci-specific (post-eval CI on existing JSONs)
    horizon: str = "6h"                        # 6h | 12h | legacy
    bs_metrics_dir: Optional[str] = None       # input dir with eval JSONs
    bs_out: Optional[str] = None
    bs_out_24ch: Optional[str] = None
    bs_out_27ch: Optional[str] = None
    bs_bootstraps: int = 2000
    bs_seed: int = 20260610
    bs_write_tex: bool = False
    bs_tex_out: Optional[str] = None


# ---------------------------------------------------------------- TOP LEVEL #
@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    exp_name: str = "wti_run"
    out_dir: str = "logs/${exp_name}"
    seed: int = 42


@dataclass
class EvaluateConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    exp_name: str = "wti_eval"

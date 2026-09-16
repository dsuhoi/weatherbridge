"""Backward-compatible model exports."""

from .encoder_decoder import WeatherEncoder, WeatherDecoder
from .hermite_model import (
    FiLM,
    HermiteLatentInterpolator,
    HermiteParamNetFiLM,
    WeatherHermiteModel,
)
from .unet_baseline_model import (
    WeatherUNetBaselineModel,
    WeatherUNetResidualLinearModel,
    WeatherUNetDirectModel,
)
from .dcae_baseline_model import WeatherDCAEResidualLinearModel
from .true_unet_baseline_model import WeatherTrueUNetResidualLinearModel

# SFNO is optional (requires neuraloperator + torch-harmonics).
try:
    from .sfno_baseline_model import WeatherSFNOResidualLinearModel
    _SFNO_AVAILABLE = True
except ImportError:
    WeatherSFNOResidualLinearModel = None
    _SFNO_AVAILABLE = False

# Spherical DYffusion (Stage 1) — requires torch_harmonics.
try:
    from .sdyff_baseline_model import WeatherSDyffusionResidualLinearModel
    _SDYFF_AVAILABLE = True
except ImportError:
    WeatherSDyffusionResidualLinearModel = None
    _SDYFF_AVAILABLE = False

# ModAFNO — standalone re-implementation, no extra deps.
from .modafno_baseline_model import (
    WeatherModAFNOResidualLinearModel,
    WeatherModAFNOOfficialResidualLinearModel,
)

__all__ = [
    "WeatherEncoder",
    "WeatherDecoder",
    "FiLM",
    "HermiteParamNetFiLM",
    "HermiteLatentInterpolator",
    "WeatherHermiteModel",
    "WeatherUNetBaselineModel",
    "WeatherUNetResidualLinearModel",
    "WeatherUNetDirectModel",
    "WeatherDCAEResidualLinearModel",
    "WeatherTrueUNetResidualLinearModel",
    "WeatherSFNOResidualLinearModel",
    "WeatherSDyffusionResidualLinearModel",
    "WeatherModAFNOResidualLinearModel",
    "WeatherModAFNOOfficialResidualLinearModel",
]

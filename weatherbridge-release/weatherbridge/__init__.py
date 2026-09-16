"""WeatherBridge — released checkpoints for global weather time interpolation.

    import torch
    from weatherbridge import load_model, load_static_features

    model = load_model("weatherbridge-6h", device="cuda")
    x_hat = model(x0, xT, tau_norm, delta_t, static=static)

``list_models()`` names everything in the archive. Channel order, tensor
shapes and the forward signature of each family are documented in README.md.
"""
from ._loader import (  # noqa: F401
    list_models,
    load_model,
    load_static_features,
    model_info,
    normalize_decoder_state_dict,
)
from .predict import linear_interpolation, predict  # noqa: F401

__all__ = [
    "linear_interpolation",
    "predict",
    "list_models",
    "load_model",
    "load_static_features",
    "model_info",
    "normalize_decoder_state_dict",
]
__version__ = "1.0.0"

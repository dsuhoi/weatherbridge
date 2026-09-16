"""Neutral re-export shim for the WTI training-side Lightning module.

The class is still defined in ``trainer_weather_hermite.py`` (legacy
filename retained so the cluster-side eval pipeline keeps working);
this shim exposes it under a clean name for new code paths:

    from trainer import WTIModelModule
"""
from trainer_weather_hermite import WeatherHermiteLightningModule as WTIModelModule

__all__ = ["WTIModelModule"]

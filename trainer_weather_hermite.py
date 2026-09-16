"""Compatibility import for legacy Lightning checkpoints.

The maintained evaluation code imports this historical module name. Keep the
implementation under ``legacy/scripts`` so the loaded source is versioned and
included in reproducibility manifests.
"""

from legacy.scripts.trainer_weather_hermite import *  # noqa: F403

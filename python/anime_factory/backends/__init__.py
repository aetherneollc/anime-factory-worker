"""Production image and video backends for the stills pipeline.

Env:
  IMAGE_BACKEND=flux2_klein4b
  VIDEO_BACKEND=skyreels_v3_r2v

``STILL_BACKEND=kolors`` does not override IMAGE_BACKEND.
``CONTROL_BACKEND`` is an asset-generation knob, not part of the video lease.
H3 / LongLive / Kolors are not production selections.
"""

from __future__ import annotations

import os
from typing import Mapping

from anime_factory.backends.control import (
    CONTROL_BACKENDS,
    DEFAULT_CONTROL_BACKEND,
    ControlBackend,
    ControlBackendError,
    get_control_backend,
    select_control_backend,
)
from anime_factory.backends.image import (
    DEFAULT_IMAGE_BACKEND,
    IMAGE_BACKENDS,
    ImageBackend,
    ImageBackendError,
    get_image_backend,
    select_image_backend,
)
from anime_factory.backends.video_ext import (
    DEFAULT_PLUGGABLE_VIDEO_BACKEND,
    PLUGGABLE_VIDEO_BACKENDS,
    VideoGenerateBackend,
    get_video_generate_backend,
    select_pluggable_video_backend,
)

__all__ = [
    "CONTROL_BACKENDS",
    "DEFAULT_CONTROL_BACKEND",
    "DEFAULT_IMAGE_BACKEND",
    "DEFAULT_PLUGGABLE_VIDEO_BACKEND",
    "IMAGE_BACKENDS",
    "PLUGGABLE_VIDEO_BACKENDS",
    "ControlBackend",
    "ControlBackendError",
    "ImageBackend",
    "ImageBackendError",
    "VideoGenerateBackend",
    "get_control_backend",
    "get_image_backend",
    "get_video_generate_backend",
    "select_control_backend",
    "select_image_backend",
    "select_pluggable_video_backend",
    "resolve_backend_env",
]


def resolve_backend_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Production IMAGE / VIDEO snapshot. Control is not a video-path setting."""
    mapping = env if env is not None else os.environ
    return {
        "IMAGE_BACKEND": select_image_backend(env=mapping),
        "VIDEO_BACKEND": select_pluggable_video_backend(env=mapping),
    }

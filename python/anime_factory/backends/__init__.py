"""Pluggable image / control / video backends for the stills strategy pipeline.

Env (preferred):
  IMAGE_BACKEND=hunyuan21|kolors
  CONTROL_BACKEND=kolors_ipadapter|ip_adapter|composite|none
  VIDEO_BACKEND=hunyuan15|h3|longlive

Legacy adapters (do not break existing tests):
  IMAGE_BACKEND unset → STILL_BACKEND=kolors → kolors; else hunyuan21
  VIDEO_BACKEND unset → AF_VIDEO_BACKEND (via anime_factory.video_backend)
  CHARACTER_STILL_BACKEND remains character-sheet TokenHub/Qwen path — not IMAGE_BACKEND
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
    """Snapshot of active IMAGE / CONTROL / VIDEO backend names."""
    mapping = env if env is not None else os.environ
    return {
        "IMAGE_BACKEND": select_image_backend(env=mapping),
        "CONTROL_BACKEND": select_control_backend(env=mapping),
        "VIDEO_BACKEND": select_pluggable_video_backend(env=mapping),
    }

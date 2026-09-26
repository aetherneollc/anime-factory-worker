"""Legacy Kolors / H3 / LongLive adapters.

These classes stay so frozen worker images can still boot. Production
selection (``IMAGE_BACKENDS``, ``VIDEO_BACKENDS``, ``normalize_video_backend``,
``select_video_backend``, the shot planner, and the produce lease) does not
return them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from anime_factory.backends.image import ImageBackend, ImageBackendError, ImageGenerateRequest, ImageGenerateResult
from anime_factory.backends.video_ext import (
    VideoGenerateBackend,
    VideoGenerateRequest,
    VideoGenerateResult,
)

LEGACY_IMAGE_BACKENDS = ("kolors",)
LEGACY_VIDEO_BACKENDS = ("h3", "longlive")
LONGLIVE_ALIASES = frozenset(
    {
        "longlive",
        "longlive2",
        "longlive-2",
        "longlive-2.0",
        "longlive_2",
        "longlive_2.0",
        "longlive-2.0-5b",
    }
)
H3_ALIASES = frozenset({"h3", "wan", "minimax", "minimax-h3", "minimaxh3"})
FACTORY_JSON_NAME = "factory.json"


class KolorsImageBackend(ImageBackend):
    """Legacy Kolors T2I stub. Not a production IMAGE_BACKEND."""

    name = "kolors"

    def generate(self, request: ImageGenerateRequest) -> ImageGenerateResult:
        raise ImageBackendError(
            "KolorsImageBackend is legacy and is not a production IMAGE_BACKEND; "
            "use IMAGE_BACKEND=flux2_klein4b"
        )


class H3VideoGenerateBackend(VideoGenerateBackend):
    """Historical H3 wrapper. Real work stays in gpu_worker.h3."""

    name = "h3"

    def generate(self, request: VideoGenerateRequest) -> VideoGenerateResult:
        return VideoGenerateResult(
            path=str(request.meta.get("out_path") or f"video/{request.shot_id}.mp4"),
            backend=self.name,
            status="pending",
            meta={"delegated": "gpu_worker.h3", **dict(request.meta)},
        )


class LongliveVideoGenerateBackend(VideoGenerateBackend):
    name = "longlive"

    def generate(self, request: VideoGenerateRequest) -> VideoGenerateResult:
        return VideoGenerateResult(
            path=str(request.meta.get("out_path") or f"video/{request.shot_id}.mp4"),
            backend=self.name,
            status="pending",
            meta={"delegated": "gpu_worker.longlive", **dict(request.meta)},
        )


def normalize_legacy_video_backend(value: Any) -> str:
    """Worker-only name fold. Production normalize refuses these names."""
    raw = str(value or "").strip().lower().replace(" ", "")
    if not raw:
        return "skyreels_v3_r2v"
    if raw.startswith("longlive") or raw in LONGLIVE_ALIASES:
        return "longlive"
    if raw in H3_ALIASES or raw == "h3":
        return "h3"
    folded = raw.replace("-", "_")
    if folded in {"skyreels_v3_r2v", "skyreels", "skyreels_v3", "skyreelsv3", "sr3", "r2v"} or folded.startswith(
        "skyreels"
    ):
        return "skyreels_v3_r2v"
    return "skyreels_v3_r2v"


def _read_factory_raw(root: Path | None) -> str | None:
    if root is None:
        return None
    path = Path(root) / FACTORY_JSON_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("video_backend")
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw)


def select_legacy_video_backend(
    segment: Mapping[str, Any] | None = None,
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Historical worker line. Not used by the shot planner or produce lease."""
    mapping = env if env is not None else os.environ
    locked = str(mapping.get("AF_VIDEO_BACKEND_LOCKED") or "").strip()
    if locked:
        return normalize_legacy_video_backend(locked)
    if segment:
        explicit = segment.get("video_backend")
        if explicit is not None and str(explicit).strip():
            return normalize_legacy_video_backend(explicit)
    from_file = _read_factory_raw(root)
    if from_file:
        return normalize_legacy_video_backend(from_file)
    preferred = str(mapping.get("VIDEO_BACKEND") or "").strip()
    if preferred:
        return normalize_legacy_video_backend(preferred)
    legacy = str(mapping.get("AF_VIDEO_BACKEND") or "").strip()
    if legacy:
        return normalize_legacy_video_backend(legacy)
    image_cap = str(mapping.get("AF_IMAGE_CAPABILITY") or "").strip().lower()
    if image_cap in {"h3", "longlive", "skyreels_v3_r2v"}:
        return normalize_legacy_video_backend(image_cap)
    return "skyreels_v3_r2v"

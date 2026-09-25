"""Pluggable video generate backends (H3 kept; HunyuanVideo-1.5 scaffold).

Selection prefers VIDEO_BACKEND, then AF_VIDEO_BACKEND via video_backend.normalize.
H3 path is never deleted — hunyuan15 is an additional backend.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from anime_factory.contracts import (
    KEYFRAME_FINAL_NAME,
    KeyframeContract,
    VideoContract,
    assert_keyframe_allows_video,
)

PLUGGABLE_VIDEO_BACKENDS = ("h3", "longlive", "hunyuan15")
DEFAULT_PLUGGABLE_VIDEO_BACKEND = "h3"
HUNYUAN15_ALIASES = frozenset(
    {
        "hunyuan15",
        "hunyuan-15",
        "hunyuan_1.5",
        "hunyuan-1.5",
        "hunyuanvideo15",
        "hunyuanvideo-1.5",
        "hunyuanvideo_1.5",
        "hy15",
    }
)


class VideoGenerateBackendError(ValueError):
    """VIDEO_BACKEND invalid or generate prerequisites failed."""


@dataclass
class VideoGenerateRequest:
    shot_id: str
    keyframe_final: str
    keyframe_png: bytes | None = None
    duration_s: float = 5.0
    prompt: str = ""
    rife: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoGenerateResult:
    path: str
    backend: str
    status: str = "ready"
    rife_applied: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class VideoGenerateBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, request: VideoGenerateRequest) -> VideoGenerateResult:
        raise NotImplementedError


def select_pluggable_video_backend(env: Mapping[str, str] | None = None) -> str:
    """VIDEO_BACKEND → AF_VIDEO_BACKEND → h3. Recognizes hunyuan15 aliases."""
    mapping = env if env is not None else os.environ
    raw = str(mapping.get("VIDEO_BACKEND") or "").strip().lower().replace(" ", "")
    if raw:
        return normalize_pluggable_video_backend(raw)
    # Legacy adapter — keep AF_VIDEO_BACKEND / lock behavior intact.
    from anime_factory.video_backend import select_video_backend

    return select_video_backend(env=mapping)


def normalize_pluggable_video_backend(value: Any) -> str:
    raw = str(value or "").strip().lower().replace(" ", "")
    if raw in HUNYUAN15_ALIASES or raw.startswith("hunyuan15") or raw.startswith("hunyuanvideo"):
        return "hunyuan15"
    from anime_factory.video_backend import normalize_video_backend

    return normalize_video_backend(raw)


def _assert_keyframe_final_only(path: str) -> str:
    text = str(path or "").replace("\\", "/").strip()
    if not text:
        raise VideoGenerateBackendError("hunyuan15 requires keyframe_final.png path")
    name = PurePosixPath(text).name
    if name != KEYFRAME_FINAL_NAME:
        raise VideoGenerateBackendError(
            f"hunyuan15 consumes only {KEYFRAME_FINAL_NAME}, got {name!r}"
        )
    return text


class H3VideoGenerateBackend(VideoGenerateBackend):
    """Thin wrapper marking H3 as the Phase-A video path (real work stays in session)."""

    name = "h3"

    def generate(self, request: VideoGenerateRequest) -> VideoGenerateResult:
        # Scaffold only — GPU session still owns MiniMax H3 submit.
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


class Hunyuan15VideoGenerateBackend(VideoGenerateBackend):
    """HunyuanVideo-1.5 I2V scaffold: keyframe_final only → 720p → optional RIFE."""

    name = "hunyuan15"

    def generate(self, request: VideoGenerateRequest) -> VideoGenerateResult:
        final = _assert_keyframe_final_only(request.keyframe_final)
        out = str(request.meta.get("out_path") or f"video/{request.shot_id}_hy15.mp4")
        rife_meta = rife_hook(enabled=bool(request.rife), video_path=out)
        return VideoGenerateResult(
            path=out,
            backend=self.name,
            status="pending",
            rife_applied=bool(rife_meta.get("applied")),
            meta={
                "stub": True,
                "keyframe_final": final,
                "target": "720p_i2v",
                "rife": rife_meta,
                **dict(request.meta),
            },
        )


def rife_hook(*, enabled: bool, video_path: str) -> dict[str, Any]:
    """Optional RIFE interpolation stub (Phase C). Does not run weights."""
    if not enabled:
        return {"enabled": False, "applied": False, "path": video_path}
    # Scaffold: pretend a sibling *_rife.mp4 without touching disk.
    stem = Path(video_path)
    out = str(stem.with_name(stem.stem + "_rife" + stem.suffix)) if stem.suffix else video_path + "_rife"
    return {"enabled": True, "applied": False, "stub": True, "path": out}


def gated_video_generate(
    keyframe: KeyframeContract | Mapping[str, Any],
    request: VideoGenerateRequest,
    *,
    backend: VideoGenerateBackend | None = None,
    env: Mapping[str, str] | None = None,
) -> VideoGenerateResult:
    """QC gate then video_backend.generate(). FAIL never reaches GPU video."""
    assert_keyframe_allows_video(keyframe)
    impl = backend or get_video_generate_backend(env=env)
    if impl.name == "hunyuan15":
        request.keyframe_final = _assert_keyframe_final_only(
            request.keyframe_final
            or (keyframe.final_path if isinstance(keyframe, KeyframeContract) else "")
            or KEYFRAME_FINAL_NAME
        )
    return impl.generate(request)


_BACKENDS: dict[str, type[VideoGenerateBackend]] = {
    "h3": H3VideoGenerateBackend,
    "longlive": LongliveVideoGenerateBackend,
    "hunyuan15": Hunyuan15VideoGenerateBackend,
}


def get_video_generate_backend(
    name: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> VideoGenerateBackend:
    chosen = normalize_pluggable_video_backend(name or select_pluggable_video_backend(env=env))
    if chosen not in _BACKENDS:
        raise VideoGenerateBackendError(
            f"VIDEO_BACKEND must be one of {PLUGGABLE_VIDEO_BACKENDS}, got {chosen!r}"
        )
    return _BACKENDS[chosen]()


def video_contract_from_result(
    shot_id: str,
    keyframe_final: str,
    result: VideoGenerateResult,
) -> VideoContract:
    return VideoContract(
        shot_id=shot_id,
        keyframe_final=keyframe_final,
        path=result.path,
        status=result.status if result.status in {"blocked", "pending", "generating", "ready", "failed"} else "pending",  # type: ignore[arg-type]
        backend=result.backend,
        rife=bool(result.rife_applied),
        meta=dict(result.meta),
    )

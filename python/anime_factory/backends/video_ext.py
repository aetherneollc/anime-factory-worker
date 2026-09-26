"""Production video generate backend: SkyReels V3 R2V only.

H3 and LongLive live in ``anime_factory.legacy_backends`` and are not registered
here. A missing reference pack is a refusal; ``keyframe_final`` is never used
as a substitute image.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from anime_factory.contracts import (
    KeyframeContract,
    RefPackContract,
    VideoContract,
    assert_ref_pack_allows_video,
    ref_pack_from_keyframe,
)

PLUGGABLE_VIDEO_BACKENDS = ("skyreels_v3_r2v",)
DEFAULT_PLUGGABLE_VIDEO_BACKEND = "skyreels_v3_r2v"
SKYREELS_ALIASES = frozenset(
    {
        "skyreels_v3_r2v",
        "skyreels",
        "skyreels_v3",
        "skyreels-v3",
        "skyreels-v3-r2v",
        "r2v",
    }
)


class VideoGenerateBackendError(ValueError):
    """VIDEO_BACKEND invalid or generate prerequisites failed."""


@dataclass
class VideoGenerateRequest:
    shot_id: str
    prompt: str = ""
    camera: str = ""
    motion: str = ""
    duration: float = 5.0
    resolution: str = "720P"
    seed: int | None = None
    references: list[Any] = field(default_factory=list)
    keyframe_final: str = ""
    keyframe_png: bytes | None = None
    ref_images: list[str] = field(default_factory=list)
    duration_s: float = 5.0
    aspect: str = "16:9"
    rife: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.duration != self.duration_s:
            if self.duration != 5.0:
                self.duration_s = float(self.duration)
            else:
                self.duration = float(self.duration_s)
        else:
            self.duration = float(self.duration)
            self.duration_s = self.duration
        if self.seed is not None:
            self.seed = int(self.seed)


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
    """VIDEO_BACKEND → AF_VIDEO_BACKEND → skyreels_v3_r2v."""
    mapping = env if env is not None else os.environ
    raw = str(mapping.get("VIDEO_BACKEND") or "").strip().lower().replace(" ", "")
    if raw:
        return normalize_pluggable_video_backend(raw)
    from anime_factory.video_backend import select_video_backend

    return select_video_backend(env=mapping)


def normalize_pluggable_video_backend(value: Any) -> str:
    raw = str(value or "").strip().lower().replace(" ", "")
    folded = raw.replace("-", "_")
    if folded in SKYREELS_ALIASES or folded.startswith("skyreels") or folded == "r2v":
        return "skyreels_v3_r2v"
    from anime_factory.video_backend import normalize_video_backend

    return normalize_video_backend(raw)


class SkyReelsV3R2VBackend(VideoGenerateBackend):
    """Official generate_video.py --task_type reference_to_video. Frees Klein first."""

    name = "skyreels_v3_r2v"

    def generate(self, request: VideoGenerateRequest) -> VideoGenerateResult:
        from anime_factory.skyreels_r2v import generate_reference_video, truncate_references

        source: list[Any] = list(request.references or [])
        if not source:
            raw = request.ref_images or request.meta.get("references") or request.meta.get("ref_images") or []
            source = list(raw)
        kept, trunc = truncate_references(source)
        dry = request.meta.get("dry_run")
        seed = request.seed if request.seed is not None else request.meta.get("seed")
        seed_i = 42 if seed is None else int(seed)
        resolution = str(request.resolution or "720P")
        produced = generate_reference_video(
            ref_imgs=kept,
            prompt=request.prompt,
            duration_s=float(request.duration or request.duration_s or 5),
            resolution=resolution,
            aspect=request.aspect or "16:9",
            seed=seed_i,
            out_path=request.meta.get("out_path"),
            dry_run=None if dry is None else bool(dry),
        )
        return VideoGenerateResult(
            path=produced.path,
            backend=self.name,
            status="ready",
            meta={
                "argv": list(produced.argv),
                "dry_run": produced.dry_run,
                "aspect": produced.aspect,
                "fps": produced.fps,
                "resolution": produced.resolution,
                "duration_s": produced.duration_s,
                "duration": float(request.duration or produced.duration_s),
                "camera": request.camera,
                "motion": request.motion,
                "seed": seed_i,
                "low_vram": produced.low_vram,
                "offload": produced.offload,
                "model_id": produced.model_id,
                "ref_images": kept,
                "references": kept,
                **dict(produced.meta),
                **trunc,
            },
        )


def _pack_from_subject(subject: RefPackContract | KeyframeContract | Mapping[str, Any]) -> RefPackContract:
    if isinstance(subject, RefPackContract):
        return subject
    if isinstance(subject, Mapping) and (
        "references" in subject or "images" in subject or "ref_images" in subject
    ):
        return RefPackContract.from_dict(subject)
    from anime_factory.contracts import ref_pack_from_keyframe

    kf = subject if isinstance(subject, KeyframeContract) else KeyframeContract.from_dict(subject)  # type: ignore[arg-type]
    return ref_pack_from_keyframe(kf)


def gated_video_generate(
    subject: RefPackContract | KeyframeContract | Mapping[str, Any],
    request: VideoGenerateRequest,
    *,
    backend: VideoGenerateBackend | None = None,
    env: Mapping[str, str] | None = None,
) -> VideoGenerateResult:
    """QC-passed ref pack, then video_backend.generate()."""
    pack = assert_ref_pack_allows_video(_pack_from_subject(subject))
    request.references = list(pack.references)
    request.ref_images = [img.path for img in pack.references]
    request.meta["references"] = [img.to_dict() for img in pack.references]
    request.meta["ref_images"] = list(request.ref_images)
    impl = backend or get_video_generate_backend(env=env)
    return impl.generate(request)


_BACKENDS: dict[str, type[VideoGenerateBackend]] = {
    "skyreels_v3_r2v": SkyReelsV3R2VBackend,
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
    *,
    ref_images: list[str] | None = None,
    request: VideoGenerateRequest | None = None,
) -> VideoContract:
    refs = list(ref_images or result.meta.get("references") or result.meta.get("ref_images") or [])
    seed = request.seed if request is not None else result.meta.get("seed")
    try:
        seed_i = int(seed) if seed is not None and str(seed).strip() != "" else None
    except (TypeError, ValueError):
        seed_i = None
    duration = float(request.duration if request is not None else result.meta.get("duration") or result.meta.get("duration_s") or 5)
    return VideoContract(
        shot_id=shot_id,
        prompt=request.prompt if request is not None else str(result.meta.get("prompt") or ""),
        camera=request.camera if request is not None else str(result.meta.get("camera") or ""),
        motion=request.motion if request is not None else str(result.meta.get("motion") or ""),
        duration=duration,
        resolution=request.resolution if request is not None else str(result.meta.get("resolution") or "720P"),
        seed=seed_i,
        references=[str(x) for x in refs],
        keyframe_final=keyframe_final,
        path=result.path,
        status=result.status if result.status in {"blocked", "pending", "generating", "ready", "failed"} else "pending",  # type: ignore[arg-type]
        backend=result.backend,
        rife=bool(result.rife_applied),
        meta=dict(result.meta),
    )

"""Production image backends: fal.ai (default) and local Qwen-Image-2.1."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from anime_factory.models import DEFAULT_IMAGE_BACKEND, IMAGE_BACKENDS


class ImageBackendError(ValueError):
    """IMAGE_BACKEND is missing or not in the allow-list."""


@dataclass
class ImageGenerateRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1280
    height: int = 720
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageGenerateResult:
    png: bytes
    backend: str
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ImageBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, request: ImageGenerateRequest) -> ImageGenerateResult:
        raise NotImplementedError


def select_image_backend(env: Mapping[str, str] | None = None) -> str:
    """Production T2I defaults to fal.ai. ``STILL_BACKEND=kolors`` does not override it."""
    mapping = env if env is not None else os.environ
    raw = str(mapping.get("IMAGE_BACKEND") or "").strip().lower()
    if not raw:
        return DEFAULT_IMAGE_BACKEND
    if raw not in IMAGE_BACKENDS:
        raise ImageBackendError(f"IMAGE_BACKEND must be one of {IMAGE_BACKENDS}, got {raw!r}")
    return raw


def _env_flag(mapping: Mapping[str, str], name: str) -> bool | None:
    if name not in mapping:
        return None
    return str(mapping.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


class FalImageBackend(ImageBackend):
    """fal.ai text-to-image. Dry-run when FAL_DRY_RUN=1. No local GPU."""

    name = "fal"

    def __init__(self, *, dry_run: bool | None = None) -> None:
        self.dry_run = dry_run

    def generate(self, request: ImageGenerateRequest) -> ImageGenerateResult:
        from anime_factory.fal_models import generate_fal_image_t2i, resolve_fal_value

        png = generate_fal_image_t2i(
            request.prompt,
            request.width,
            request.height,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            dry_run=self.dry_run,
        )
        return ImageGenerateResult(
            png=png,
            backend=self.name,
            seed=request.seed,
            meta={"model": resolve_fal_value("image_t2i_model"), **dict(request.meta)},
        )


class QwenImage21ImageBackend(ImageBackend):
    """Local QwenImage21Pipeline. Dry-run when QWEN_IMAGE_21_DRY_RUN=1."""

    name = "qwen_image_21"

    def __init__(self, *, dry_run: bool | None = None) -> None:
        self.dry_run = dry_run

    def generate(self, request: ImageGenerateRequest) -> ImageGenerateResult:
        from anime_factory.qwen_image_21 import generate_qwen_image_21

        png = generate_qwen_image_21(
            request.prompt,
            request.width,
            request.height,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            dry_run=self.dry_run,
        )
        return ImageGenerateResult(
            png=png,
            backend=self.name,
            seed=request.seed,
            meta={"model": "Qwen/Qwen-Image-2.1", **dict(request.meta)},
        )


class Flux2KleinImageBackend(ImageBackend):
    """FLUX.2 Klein 4B T2I. Dry-run when FLUX2_KLEIN_DRY_RUN=1."""

    name = "flux2_klein4b"

    def __init__(self, *, dry_run: bool | None = None) -> None:
        if dry_run is None:
            dry_run = str(os.environ.get("FLUX2_KLEIN_DRY_RUN") or "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
        self.dry_run = dry_run

    def generate(self, request: ImageGenerateRequest) -> ImageGenerateResult:
        from anime_factory.flux2_klein import generate_klein_t2i

        png = generate_klein_t2i(
            request.prompt,
            request.width,
            request.height,
            seed=request.seed,
            dry_run=self.dry_run,
        )
        return ImageGenerateResult(
            png=png,
            backend=self.name,
            seed=request.seed,
            meta=dict(request.meta),
        )


_BACKENDS: dict[str, type[ImageBackend]] = {
    "fal": FalImageBackend,
    "qwen_image_21": QwenImage21ImageBackend,
}


def get_image_backend(name: str | None = None, *, env: Mapping[str, str] | None = None) -> ImageBackend:
    chosen = (name or select_image_backend(env=env)).strip().lower()
    cls = _BACKENDS.get(chosen)
    if cls is None:
        raise ImageBackendError(f"IMAGE_BACKEND must be one of {IMAGE_BACKENDS}, got {chosen!r}")
    mapping = env if env is not None else os.environ
    if chosen == "fal":
        return FalImageBackend(dry_run=_env_flag(mapping, "FAL_DRY_RUN"))
    if chosen == "qwen_image_21":
        return QwenImage21ImageBackend(dry_run=_env_flag(mapping, "QWEN_IMAGE_21_DRY_RUN"))
    return cls()

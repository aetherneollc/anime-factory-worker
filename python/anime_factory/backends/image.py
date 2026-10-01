"""Production image backend: local Qwen-Image-2.1 only."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

IMAGE_BACKENDS = ("qwen_image_21",)
DEFAULT_IMAGE_BACKEND = "qwen_image_21"


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
    """Production T2I is Qwen-Image-2.1. ``STILL_BACKEND=kolors`` does not override it."""
    mapping = env if env is not None else os.environ
    raw = str(mapping.get("IMAGE_BACKEND") or "").strip().lower()
    if not raw:
        return DEFAULT_IMAGE_BACKEND
    if raw not in IMAGE_BACKENDS:
        raise ImageBackendError(f"IMAGE_BACKEND must be one of {IMAGE_BACKENDS}, got {raw!r}")
    return raw


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
    "qwen_image_21": QwenImage21ImageBackend,
}


def get_image_backend(name: str | None = None, *, env: Mapping[str, str] | None = None) -> ImageBackend:
    chosen = (name or select_image_backend(env=env)).strip().lower()
    if chosen not in _BACKENDS:
        raise ImageBackendError(f"IMAGE_BACKEND must be one of {IMAGE_BACKENDS}, got {chosen!r}")
    dry: bool | None = None
    mapping = env if env is not None else os.environ
    if "QWEN_IMAGE_21_DRY_RUN" in mapping:
        dry = str(mapping.get("QWEN_IMAGE_21_DRY_RUN") or "").strip().lower() in {"1", "true", "yes", "on"}
    return QwenImage21ImageBackend(dry_run=dry)

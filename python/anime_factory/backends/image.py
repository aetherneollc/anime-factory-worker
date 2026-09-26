"""Production image backend: FLUX.2 Klein 4B only."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

IMAGE_BACKENDS = ("flux2_klein4b",)
DEFAULT_IMAGE_BACKEND = "flux2_klein4b"


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
    """Production T2I is Klein. ``STILL_BACKEND=kolors`` does not override it."""
    mapping = env if env is not None else os.environ
    raw = str(mapping.get("IMAGE_BACKEND") or "").strip().lower()
    if not raw:
        return DEFAULT_IMAGE_BACKEND
    if raw not in IMAGE_BACKENDS:
        raise ImageBackendError(f"IMAGE_BACKEND must be one of {IMAGE_BACKENDS}, got {raw!r}")
    return raw


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
    "flux2_klein4b": Flux2KleinImageBackend,
}


def get_image_backend(name: str | None = None, *, env: Mapping[str, str] | None = None) -> ImageBackend:
    chosen = (name or select_image_backend(env=env)).strip().lower()
    if chosen not in _BACKENDS:
        raise ImageBackendError(f"IMAGE_BACKEND must be one of {IMAGE_BACKENDS}, got {chosen!r}")
    if chosen == "flux2_klein4b":
        dry: bool | None = None
        if env is not None and "FLUX2_KLEIN_DRY_RUN" in env:
            dry = str(env.get("FLUX2_KLEIN_DRY_RUN") or "").strip().lower() in {"1", "true", "yes"}
        return Flux2KleinImageBackend(dry_run=dry)
    return _BACKENDS[chosen]()

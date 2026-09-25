"""Image backends: HunyuanImage-2.1 (primary T2I) and Kolors (legacy / fallback)."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

IMAGE_BACKENDS = ("hunyuan21", "kolors")
DEFAULT_IMAGE_BACKEND = "hunyuan21"


class ImageBackendError(ValueError):
    """IMAGE_BACKEND is missing or not in the allow-list."""


@dataclass
class ImageGenerateRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1344
    height: int = 768
    seed: int | None = None
    # Tier-A identity lives in the prompt; refs are for Control (B/C), not multi-ref T2I.
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
    """IMAGE_BACKEND wins; else STILL_BACKEND=kolors → kolors; else hunyuan21."""
    mapping = env if env is not None else os.environ
    raw = str(mapping.get("IMAGE_BACKEND") or "").strip().lower()
    if not raw:
        still = str(mapping.get("STILL_BACKEND") or "").strip().lower()
        if still == "kolors":
            return "kolors"
        return DEFAULT_IMAGE_BACKEND
    if raw not in IMAGE_BACKENDS:
        raise ImageBackendError(f"IMAGE_BACKEND must be one of {IMAGE_BACKENDS}, got {raw!r}")
    return raw


class Hunyuan21ImageBackend(ImageBackend):
    """HunyuanImage-2.1 T2I. Real GPU path via hunyuan21 module; stub when dry."""

    name = "hunyuan21"

    def __init__(self, *, dry_run: bool | None = None) -> None:
        if dry_run is None:
            dry_run = str(os.environ.get("HUNYUAN21_DRY_RUN") or "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
        self.dry_run = dry_run

    def generate(self, request: ImageGenerateRequest) -> ImageGenerateResult:
        from anime_factory.hunyuan21 import generate_hunyuan21_t2i

        png = generate_hunyuan21_t2i(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            seed=request.seed,
            dry_run=self.dry_run,
        )
        return ImageGenerateResult(
            png=png,
            backend=self.name,
            seed=request.seed,
            meta=dict(request.meta),
        )


class KolorsImageBackend(ImageBackend):
    """Kolors T2I adapter stub for the pluggable IMAGE_BACKEND slot.

    Production Kolors stills remain on the Comfy / STILL_BACKEND path; this
    backend exists so IMAGE_BACKEND=kolors resolves without rewriting Docker.
    """

    name = "kolors"

    def generate(self, request: ImageGenerateRequest) -> ImageGenerateResult:
        raise ImageBackendError(
            "KolorsImageBackend is a selection stub; use STILL_BACKEND=kolors "
            "Comfy path or set IMAGE_BACKEND=hunyuan21 for Phase-A T2I"
        )


_BACKENDS: dict[str, type[ImageBackend]] = {
    "hunyuan21": Hunyuan21ImageBackend,
    "kolors": KolorsImageBackend,
}


def get_image_backend(name: str | None = None, *, env: Mapping[str, str] | None = None) -> ImageBackend:
    chosen = (name or select_image_backend(env=env)).strip().lower()
    if chosen not in _BACKENDS:
        raise ImageBackendError(f"IMAGE_BACKEND must be one of {IMAGE_BACKENDS}, got {chosen!r}")
    return _BACKENDS[chosen]()

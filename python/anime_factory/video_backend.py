"""Production video line: SkyReels V3 R2V only.

``normalize_video_backend`` and ``select_video_backend`` reject H3 and LongLive.
Frozen worker images still lock their own line through
``normalize_legacy_video_backend`` inside ``lock_video_backend``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from anime_factory.models import (
    H3_MAX_SECONDS,
    LONGLIVE_MAX_SECONDS,
    LONGLIVE_SHORT_MAX_SECONDS,
    SKYREELS_MAX_SECONDS,
)

VIDEO_BACKENDS = ("skyreels_v3_r2v",)
DEFAULT_VIDEO_BACKEND = "skyreels_v3_r2v"
# Image-capability labels the worker lock still understands. Not production enums.
_WORKER_IMAGE_LINES = frozenset({"h3", "longlive", "skyreels_v3_r2v"})
FACTORY_JSON_NAME = "factory.json"
LOCKED_ENV = "AF_VIDEO_BACKEND_LOCKED"
CAPABILITY_MISMATCH = "capability_mismatch:video_backend"
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
SKYREELS_ALIASES = frozenset(
    {
        "skyreels_v3_r2v",
        "skyreels",
        "skyreels-v3",
        "skyreels_v3",
        "skyreelsv3",
        "sr3",
        "r2v",
    }
)
H3_ALIASES = frozenset({"h3", "wan", "minimax", "minimax-h3", "minimaxh3"})


class VideoBackendLockError(RuntimeError):
    """Requested video backend disagrees with the running image."""


class ProductionBackendError(VideoBackendLockError):
    """Production selection was asked for a legacy video backend."""


def _is_skyreels(raw: str) -> bool:
    folded = raw.replace("-", "_")
    return folded in SKYREELS_ALIASES or folded.startswith("skyreels") or folded == "r2v"


def normalize_video_backend(value: Any) -> str:
    """Production normalize. Empty selects SkyReels. H3 and LongLive are refused."""
    raw = str(value or "").strip().lower().replace(" ", "")
    if not raw or _is_skyreels(raw):
        return DEFAULT_VIDEO_BACKEND
    raise ProductionBackendError(
        f"production VIDEO_BACKEND rejects {raw!r}; allowed {VIDEO_BACKENDS}"
    )


def max_seconds_for_backend(backend: str | None = None) -> float:
    if backend is None:
        chosen = select_video_backend()
    else:
        from anime_factory.legacy_backends import normalize_legacy_video_backend

        chosen = normalize_legacy_video_backend(backend)
    if chosen == "longlive":
        raw = (os.environ.get("LONGLIVE_MAX_SECONDS") or "").strip()
        try:
            return max(H3_MAX_SECONDS, float(raw) if raw else LONGLIVE_MAX_SECONDS)
        except ValueError:
            return LONGLIVE_MAX_SECONDS
    if chosen == "skyreels_v3_r2v":
        return SKYREELS_MAX_SECONDS
    return H3_MAX_SECONDS


def short_take_max_seconds(backend: str | None = None) -> float:
    if backend is None:
        chosen = select_video_backend()
    else:
        from anime_factory.legacy_backends import normalize_legacy_video_backend

        chosen = normalize_legacy_video_backend(backend)
    if chosen == "longlive":
        return min(max_seconds_for_backend("longlive"), LONGLIVE_SHORT_MAX_SECONDS)
    if chosen == "skyreels_v3_r2v":
        return SKYREELS_MAX_SECONDS
    return H3_MAX_SECONDS


def factory_json_path(root: Path | None) -> Path | None:
    if root is None:
        return None
    return Path(root) / FACTORY_JSON_NAME


def read_factory_backend(root: Path | None) -> str | None:
    path = factory_json_path(root)
    if path is None or not path.is_file():
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
    from anime_factory.legacy_backends import normalize_legacy_video_backend

    return normalize_legacy_video_backend(raw)


def write_factory_backend(root: Path, backend: str) -> Path:
    dest = Path(root) / FACTORY_JSON_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    from anime_factory.legacy_backends import normalize_legacy_video_backend

    payload = {"video_backend": normalize_legacy_video_backend(backend)}
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def image_video_capability(env: Mapping[str, str] | None = None) -> str:
    """Explicit image video line, or empty when unknown (do not guess h3)."""
    mapping = env if env is not None else os.environ
    explicit = str(mapping.get("AF_IMAGE_CAPABILITY") or "").strip().lower()
    if explicit in _WORKER_IMAGE_LINES:
        return explicit
    try:
        from gpu_worker.images import running_image_name, video_capability, image_capabilities

        image = running_image_name()
        if image:
            return video_capability(image_capabilities(image))
    except Exception:  # noqa: BLE001 — lock must still work in unit tests
        pass
    return ""


def _first_explicit_backend(
    segment: Mapping[str, Any] | None,
    root: Path | None,
    mapping: Mapping[str, str],
) -> str:
    locked = str(mapping.get(LOCKED_ENV) or "").strip()
    if locked:
        return locked
    if segment:
        explicit = segment.get("video_backend")
        if explicit is not None and str(explicit).strip():
            return str(explicit)
    path = factory_json_path(root)
    if path is not None and path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict) and str(data.get("video_backend") or "").strip():
            return str(data.get("video_backend"))
    preferred = str(mapping.get("VIDEO_BACKEND") or "").strip()
    if preferred:
        return preferred
    return str(mapping.get("AF_VIDEO_BACKEND") or "").strip()


def select_video_backend(
    segment: Mapping[str, Any] | None = None,
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Production selection. Unset env is SkyReels. H3 and LongLive raise."""
    mapping = env if env is not None else os.environ
    explicit = _first_explicit_backend(segment, root, mapping)
    if explicit:
        return normalize_video_backend(explicit)
    return DEFAULT_VIDEO_BACKEND


def lock_video_backend(
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
    requested: str | None = None,
) -> str:
    """Resolve AF_VIDEO_BACKEND once and refuse image/capability mismatch.

    Must run before any H3 or LongLive weight download.
    """
    from anime_factory.legacy_backends import normalize_legacy_video_backend, select_legacy_video_backend

    mapping = dict(env) if env is not None else dict(os.environ)
    already = str(mapping.get(LOCKED_ENV) or "").strip()
    if already:
        backend = normalize_legacy_video_backend(already)
        image_cap = image_video_capability(mapping)
        if image_cap and backend != image_cap:
            raise VideoBackendLockError(
                f"{CAPABILITY_MISMATCH}: locked backend {backend!r} disagrees with "
                f"image capability {image_cap!r}"
            )
        os.environ["AF_VIDEO_BACKEND"] = backend
        os.environ[LOCKED_ENV] = backend
        return backend
    requested_norm = (
        normalize_legacy_video_backend(requested)
        if requested is not None and str(requested).strip()
        else select_legacy_video_backend(root=root, env=mapping)
    )
    # No explicit request: a frozen H3/LongLive image keeps its line.
    # Otherwise the default is SkyReels V3 R2V.
    env_set = str(mapping.get("AF_VIDEO_BACKEND") or "").strip()
    factory = read_factory_backend(root)
    image_cap = image_video_capability(mapping)
    if not factory and not env_set and image_cap:
        requested_norm = image_cap
    if image_cap and requested_norm != image_cap:
        raise VideoBackendLockError(
            f"{CAPABILITY_MISMATCH}: requested backend {requested_norm!r} disagrees "
            f"with image capability {image_cap!r}"
        )
    os.environ["AF_VIDEO_BACKEND"] = requested_norm
    os.environ[LOCKED_ENV] = requested_norm
    if root is not None:
        try:
            write_factory_backend(Path(root), requested_norm)
        except OSError:
            pass
    return requested_norm


def locked_video_backend() -> str | None:
    from anime_factory.legacy_backends import normalize_legacy_video_backend

    raw = (os.environ.get(LOCKED_ENV) or "").strip()
    return normalize_legacy_video_backend(raw) if raw else None

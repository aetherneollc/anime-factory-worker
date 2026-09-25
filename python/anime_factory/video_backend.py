"""Per-story video line: MiniMax H3 (default), LongLive 2.0, or HunyuanVideo-1.5.

Selection is story-scoped so an in-flight H3 shoot cannot be flipped by a
global Worker secret. Missing factory.json / env → h3.

Preferred env: VIDEO_BACKEND (stills strategy). Legacy adapter: AF_VIDEO_BACKEND.
AF_VIDEO_BACKEND must be resolved and locked before any video-weight task.
Image capability (AF_IMAGE_CAPABILITY / catalog) that disagrees with the
requested backend fails closed — H3 images never pull LongLive, LongLive
images never pull H3. hunyuan15 is scaffolded alongside; H3 path stays.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from anime_factory.models import H3_MAX_SECONDS, LONGLIVE_MAX_SECONDS, LONGLIVE_SHORT_MAX_SECONDS

VIDEO_BACKENDS = ("h3", "longlive", "hunyuan15")
DEFAULT_VIDEO_BACKEND = "h3"
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


class VideoBackendLockError(RuntimeError):
    """Requested video backend disagrees with the running image."""


def normalize_video_backend(value: Any) -> str:
    raw = str(value or "").strip().lower().replace(" ", "")
    if raw.startswith("longlive") or raw in LONGLIVE_ALIASES:
        return "longlive"
    if raw in HUNYUAN15_ALIASES or raw.startswith("hunyuan15"):
        return "hunyuan15"
    # Prefer VIDEO_BACKEND env alias when callers pass through select_video_backend.
    return DEFAULT_VIDEO_BACKEND


def max_seconds_for_backend(backend: str | None = None) -> float:
    chosen = normalize_video_backend(backend) if backend is not None else select_video_backend()
    if chosen == "longlive":
        raw = (os.environ.get("LONGLIVE_MAX_SECONDS") or "").strip()
        try:
            return max(H3_MAX_SECONDS, float(raw) if raw else LONGLIVE_MAX_SECONDS)
        except ValueError:
            return LONGLIVE_MAX_SECONDS
    return H3_MAX_SECONDS


def short_take_max_seconds(backend: str | None = None) -> float:
    chosen = normalize_video_backend(backend) if backend is not None else select_video_backend()
    if chosen == "longlive":
        return min(max_seconds_for_backend("longlive"), LONGLIVE_SHORT_MAX_SECONDS)
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
    return normalize_video_backend(raw)


def write_factory_backend(root: Path, backend: str) -> Path:
    dest = Path(root) / FACTORY_JSON_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"video_backend": normalize_video_backend(backend)}
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def image_video_capability(env: Mapping[str, str] | None = None) -> str:
    """Explicit image video line, or empty when unknown (do not guess h3)."""
    mapping = env if env is not None else os.environ
    explicit = str(mapping.get("AF_IMAGE_CAPABILITY") or "").strip().lower()
    if explicit in VIDEO_BACKENDS:
        return explicit
    try:
        from gpu_worker.images import running_image_name, video_capability, image_capabilities

        image = running_image_name()
        if image:
            return video_capability(image_capabilities(image))
    except Exception:  # noqa: BLE001 — lock must still work in unit tests
        pass
    return ""


def select_video_backend(
    segment: Mapping[str, Any] | None = None,
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Locked env → shot override → story factory.json → VIDEO_BACKEND → AF_VIDEO_BACKEND → h3."""
    mapping = env if env is not None else os.environ
    locked = str(mapping.get(LOCKED_ENV) or "").strip()
    if locked:
        return normalize_video_backend(locked)
    if segment:
        explicit = segment.get("video_backend")
        if explicit is not None and str(explicit).strip():
            return normalize_video_backend(explicit)
    from_file = read_factory_backend(root)
    if from_file:
        return from_file
    # New stills-strategy name first; AF_VIDEO_BACKEND remains the legacy adapter.
    preferred = str(mapping.get("VIDEO_BACKEND") or "").strip()
    if preferred:
        return normalize_video_backend(preferred)
    return normalize_video_backend(mapping.get("AF_VIDEO_BACKEND"))


def lock_video_backend(
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
    requested: str | None = None,
) -> str:
    """Resolve AF_VIDEO_BACKEND once and refuse image/capability mismatch.

    Must run before any H3 or LongLive weight download.
    """
    mapping = dict(env) if env is not None else dict(os.environ)
    already = str(mapping.get(LOCKED_ENV) or "").strip()
    if already:
        backend = normalize_video_backend(already)
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
        normalize_video_backend(requested)
        if requested is not None and str(requested).strip()
        else select_video_backend(root=root, env=mapping)
    )
    # Bare env default is h3; a LongLive image with no explicit request stays longlive.
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
    raw = (os.environ.get(LOCKED_ENV) or "").strip()
    return normalize_video_backend(raw) if raw else None

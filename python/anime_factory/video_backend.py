"""Per-story video line: MiniMax H3 (default) or LongLive 2.0.

Selection is story-scoped so an in-flight H3 shoot cannot be flipped by a
global Worker secret. Missing factory.json / env → h3.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from anime_factory.models import H3_MAX_SECONDS, LONGLIVE_MAX_SECONDS

VIDEO_BACKENDS = ("h3", "longlive")
DEFAULT_VIDEO_BACKEND = "h3"
FACTORY_JSON_NAME = "factory.json"
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


def normalize_video_backend(value: Any) -> str:
    raw = str(value or "").strip().lower().replace(" ", "")
    if raw.startswith("longlive") or raw in LONGLIVE_ALIASES:
        return "longlive"
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


def select_video_backend(
    segment: Mapping[str, Any] | None = None,
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Shot override → story factory.json → AF_VIDEO_BACKEND → h3."""
    if segment:
        explicit = segment.get("video_backend")
        if explicit is not None and str(explicit).strip():
            return normalize_video_backend(explicit)
    from_file = read_factory_backend(root)
    if from_file:
        return from_file
    mapping = env if env is not None else os.environ
    return normalize_video_backend(mapping.get("AF_VIDEO_BACKEND"))

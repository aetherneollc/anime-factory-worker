"""HARD asset isolation: every story object lives under stories/<story_id>/."""

from __future__ import annotations

import posixpath
from typing import Iterable

SHARED_PREFIXES = ("shared/sfx/", "shared/fonts/", "archives/")
FINAL_DIR = "final/"
CHARACTER_TURNAROUND = "sheet_turnaround.png"
CHARACTER_FRONT_ALIAS = "sheet_front.png"
SCENE_PLATE = "plate_base.png"


class PathIsolationError(ValueError):
    """Raised when a key would escape the owning story prefix."""


def story_prefix(story_id: str) -> str:
    if not story_id or "/" in story_id or ".." in story_id:
        raise PathIsolationError(f"invalid story_id: {story_id!r}")
    return f"stories/{story_id}/"


def join_story(story_id: str, *parts: str) -> str:
    prefix = story_prefix(story_id)
    rel = "/".join(str(p).replace("\\", "/").strip("/") for p in parts if p)
    if not rel:
        return prefix
    candidate = posixpath.normpath(prefix + rel).replace("\\", "/")
    if not candidate.startswith(prefix.rstrip("/")):
        raise PathIsolationError(
            f"path escapes story prefix {prefix!r}: {rel!r} -> {candidate!r}"
        )
    other = f"stories/"
    if candidate.startswith(other) and not candidate.startswith(prefix):
        raise PathIsolationError(f"cross-story path forbidden: {candidate}")
    return candidate if candidate.endswith("/") or rel.endswith("/") else candidate


def character_asset_rel(cid: str, filename: str = CHARACTER_TURNAROUND) -> str:
    name = str(filename or CHARACTER_TURNAROUND).replace("\\", "/").split("/")[-1]
    return f"assets/characters/{cid}/{name}"


def scene_asset_rel(lid: str, filename: str = SCENE_PLATE) -> str:
    name = str(filename or SCENE_PLATE).replace("\\", "/").split("/")[-1]
    return f"assets/scenes/{lid}/{name}"


def is_final_key(key: str) -> bool:
    return "/final/" in key.replace("\\", "/")


def purge_list(keys: Iterable[str], story_id: str) -> list[str]:
    """Keys eligible for heat-data purge. `final/` is never included."""
    prefix = story_prefix(story_id)
    out: list[str] = []
    for key in keys:
        normalized = key.replace("\\", "/")
        if not normalized.startswith(prefix):
            raise PathIsolationError(f"purge key outside story: {key}")
        rel = normalized[len(prefix) :]
        if rel.startswith(FINAL_DIR) or rel == "final" or "/final/" in f"/{rel}":
            continue
        out.append(normalized)
    return out


def never_delete_final(keys: Iterable[str]) -> list[str]:
    return [k for k in keys if not is_final_key(k)]

"""HARD asset isolation: every story object lives under stories/<story_id>/."""

from __future__ import annotations

import hashlib
import posixpath
import re
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


# Cue keys are author-controlled text. They are never trusted as a path
# component: a key that is not already a plain [a-z0-9_-] slug is rewritten to
# a sanitized prefix plus a stable sha256 suffix, so `../../x`, `a/b`, unicode
# and empty keys can never traverse a local cache dir or an R2 prefix, and the
# same cue_key always maps to the same object name.
_SLUG_SAFE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def safe_cue_slug(cue_key: str) -> str:
    """Stable filesystem/R2-safe name for a cue key. Pure keys pass through."""
    raw = str(cue_key or "")
    if _SLUG_SAFE_RE.match(raw):
        return raw
    cleaned = re.sub(r"[^a-z0-9_-]+", "_", raw.lower()).strip("_-")[:40]
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned}-{digest}" if cleaned else f"cue-{digest}"


def episode_sfx_key(story_id: str, episode_code: str, cue_key: str) -> str:
    """Episode-local SFX cue, isolated under the owning story: stories/<id>/episodes/<EP>/audio/sfx/."""
    ep = str(episode_code or "").strip()
    if not ep or "/" in ep or ".." in ep:
        raise PathIsolationError(f"invalid episode_code: {episode_code!r}")
    if not str(cue_key or "").strip():
        raise PathIsolationError(f"invalid cue_key: {cue_key!r}")
    return join_story(story_id, "episodes", ep, "audio", "sfx", f"{safe_cue_slug(cue_key)}.wav")


def shared_sfx_key(cue_key: str) -> str:
    """Reusable SFX/ambience asset shared across stories. Never a model-weight path.

    Lives directly under the top-level `shared/sfx/` prefix (see SHARED_PREFIXES),
    outside any `stories/<id>/` tree — it is not story-isolated by design.
    """
    if not str(cue_key or "").strip():
        raise PathIsolationError(f"invalid cue_key: {cue_key!r}")
    return f"{SHARED_PREFIXES[0]}{safe_cue_slug(cue_key)}.wav"


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

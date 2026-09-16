"""Cold archive: put/get/verify with sha256. github | r2 | none. none refuses purge."""

from __future__ import annotations

import hashlib
import json
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from anime_factory.config import load_settings
from anime_factory.models import (
    CHUNKING_VERSION,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    H3_MODEL_NAME,
    IMAGE_MODEL,
    PROMPT_TEMPLATE_VERSION,
    SCHEMA_VERSION,
    STYLE_PRESET_VERSION,
    TTS_MODEL,
    VECTOR_SCHEMA_VERSION,
)
from anime_factory.r2_paths import is_final_key, purge_list, story_prefix

log = logging.getLogger("anime_factory.archive")


class ArchiveError(RuntimeError):
    pass


@dataclass
class ArchiveResult:
    tag: str
    sha256: str
    status: str
    purged: list[str]
    backend: str


def archive_tag(story_id: str, version: str) -> str:
    v = version.lstrip("v")
    return f"story-{story_id}-v{v}"


def build_manifest(
    story_id: str,
    version: str,
    world_mode: str,
    character_seeds: dict[str, int] | None = None,
    voice_versions: dict | None = None,
    h3_model_name: str | None = None,
) -> dict:
    return {
        "story_id": story_id,
        "version": version,
        "world_mode": world_mode,
        "h3_model_name": h3_model_name or H3_MODEL_NAME,
        "kolors_model": IMAGE_MODEL,
        "image_model": IMAGE_MODEL,
        "tts_model": TTS_MODEL,
        "style_preset_version": STYLE_PRESET_VERSION,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "character_seeds": character_seeds or {},
        "voice_versions": voice_versions or {},
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunking_version": CHUNKING_VERSION,
        "vector_schema_version": VECTOR_SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def put_archive(data: bytes, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def get_archive(src: Path) -> bytes:
    return src.read_bytes()


def verify(data: bytes, expected_sha256: str) -> bool:
    return hashlib.sha256(data).hexdigest() == expected_sha256


def pack_story(root: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w") as tar:
        for path in root.rglob("*"):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(root)))
    return sha256_file(dest)


class ColdStore:
    def __init__(self, backend: str | None = None, put: Callable[[str, bytes], None] | None = None):
        self.backend = backend or load_settings().cold_storage_backend
        self._put = put
        self.objects: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        if self.backend == "none":
            raise ArchiveError("COLD_STORAGE_BACKEND=none refuses archive put that would enable purge")
        if self._put:
            self._put(key, data)
        self.objects[key] = data
        return digest

    def get(self, key: str) -> bytes:
        if key in self.objects:
            return self.objects[key]
        raise ArchiveError(f"missing archive object {key}")

    def verify(self, key: str, expected: str) -> bool:
        return hashlib.sha256(self.get(key)).hexdigest() == expected


def finish_and_archive(
    story_id: str,
    version: str,
    status: str,
    keys: list[str],
    pack_bytes: bytes,
    store: ColdStore,
    upload_ok: bool = True,
) -> ArchiveResult:
    """producing → finished → archived. Failed upload stays finished; heat data remains."""
    if status == "producing":
        status = "finished"
    tag = archive_tag(story_id, version)
    if store.backend == "none":
        log.warning("COLD_STORAGE_BACKEND=none: refuse purge of %s", story_prefix(story_id))
        raise ArchiveError("COLD_STORAGE_BACKEND=none refuses purge")
    if not upload_ok:
        return ArchiveResult(tag=tag, sha256="", status="finished", purged=[], backend=store.backend)
    digest = store.put(f"archives/{story_id}/{tag}.tar", pack_bytes)
    if not store.verify(f"archives/{story_id}/{tag}.tar", digest):
        return ArchiveResult(tag=tag, sha256=digest, status="finished", purged=[], backend=store.backend)
    purged = purge_list(keys, story_id)
    for k in purged:
        if is_final_key(k):
            raise ArchiveError("final/ leaked into purge list")
    return ArchiveResult(tag=tag, sha256=digest, status="archived", purged=purged, backend=store.backend)

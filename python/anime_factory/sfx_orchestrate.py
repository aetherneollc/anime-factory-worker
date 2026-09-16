"""Cue resolution: Freesound is always attempted before MOSS-SoundEffect v2.

One entry point, `resolve_cue`, drives both sources with one cache/provenance
shape (`anime_factory.sfx_common`) and persists the outcome to `sfx_cues` so
downstream compose/timeline stages and the R2 upload step can find it by
`(episode_code, cue_key)` without re-resolving.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path

from anime_factory.db import utcnow
from anime_factory.r2_paths import episode_sfx_key, shared_sfx_key
from anime_factory.sfx_common import SfxCue, SfxProvenance, SfxResult
from anime_factory.sfx_freesound import FreesoundClient, freesound_api_key, resolve_via_freesound
from anime_factory.sfx_moss import MossConfigError, MossSoundEffectClient, resolve_via_moss

log = logging.getLogger("anime_factory.sfx.orchestrate")


class SfxResolutionError(RuntimeError):
    """Neither Freesound nor MOSS produced an acceptable render for this cue."""


def r2_key_for(story_id: str, cue: SfxCue) -> str:
    if cue.shared:
        return shared_sfx_key(cue.cue_key)
    return episode_sfx_key(story_id, cue.episode_code or "EP001", cue.cue_key)


def _record_cue(
    conn: sqlite3.Connection,
    story_id: str,
    cue: SfxCue,
    result: SfxResult,
) -> None:
    prov = result.provenance
    r2_key = r2_key_for(story_id, cue) if result.status == "resolved" else None
    conn.execute(
        """
        INSERT INTO sfx_cues (id, episode_code, segment_id, cue_key, prompt, tags_json,
            duration_target, bus, source, status, license, creator, source_url, content_hash,
            sample_rate, duration_actual, r2_key, local_path, shared, provenance_json,
            created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            status = excluded.status,
            source = excluded.source,
            license = excluded.license,
            creator = excluded.creator,
            source_url = excluded.source_url,
            content_hash = excluded.content_hash,
            sample_rate = excluded.sample_rate,
            duration_actual = excluded.duration_actual,
            r2_key = excluded.r2_key,
            local_path = excluded.local_path,
            provenance_json = excluded.provenance_json,
            updated_at = excluded.updated_at
        """,
        (
            cue.cue_key,
            cue.episode_code or None,
            cue.segment_id,
            cue.cue_key,
            cue.query,
            json.dumps(list(cue.tags), ensure_ascii=False),
            cue.duration_target,
            cue.bus,
            prov.source if prov else None,
            "resolved" if result.status == "resolved" else "failed",
            prov.license if prov else None,
            prov.creator if prov else None,
            prov.source_url if prov else None,
            prov.content_hash if prov else None,
            prov.sample_rate if prov else None,
            prov.duration if prov else None,
            r2_key,
            result.local_path,
            1 if cue.shared else 0,
            json.dumps(_provenance_dict(prov), ensure_ascii=False) if prov else None,
            utcnow(),
            utcnow(),
        ),
    )
    conn.commit()


def _provenance_dict(prov: SfxProvenance | None) -> dict:
    if prov is None:
        return {}
    return asdict(prov)


def resolve_cue(
    cue: SfxCue,
    root: Path,
    *,
    story_id: str = "",
    conn: sqlite3.Connection | None = None,
    freesound_client: FreesoundClient | None = None,
    moss_client: MossSoundEffectClient | None = None,
) -> SfxResult:
    """Freesound first, MOSS-SoundEffect v2 only on a Freesound miss/QC-fail/license-reject.

    A cache hit on either path (episode-local or shared) short-circuits before
    any network call or subprocess spawn — see `sfx_common.cache_lookup`.
    """
    fs_client = freesound_client or FreesoundClient(freesound_api_key())
    freesound_error: Exception | None = None
    try:
        result = resolve_via_freesound(fs_client, cue, root)
        if conn is not None:
            _record_cue(conn, story_id, cue, result)
        return result
    except Exception as exc:  # noqa: BLE001 — any Freesound failure falls back to MOSS
        freesound_error = exc
        log.info("freesound miss for cue=%s (%s); falling back to MOSS", cue.cue_key, type(exc).__name__)

    moss = moss_client or MossSoundEffectClient()
    try:
        result = resolve_via_moss(moss, cue, root)
    except MossConfigError as exc:
        failure = SfxResult(
            cue_key=cue.cue_key,
            status="failed",
            reason=f"freesound: {freesound_error}; moss: {exc}",
        )
        if conn is not None:
            _record_cue(conn, story_id, cue, failure)
        raise SfxResolutionError(
            f"{cue.cue_key}: Freesound failed ({freesound_error}) and MOSS fallback is not "
            f"configured ({exc})"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surfaced as one resolution error
        failure = SfxResult(cue_key=cue.cue_key, status="failed", reason=str(exc))
        if conn is not None:
            _record_cue(conn, story_id, cue, failure)
        raise SfxResolutionError(f"{cue.cue_key}: Freesound and MOSS both failed: {exc}") from exc

    if conn is not None:
        _record_cue(conn, story_id, cue, result)
    return result

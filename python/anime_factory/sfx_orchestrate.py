"""Cue resolution: Freesound is always attempted before MOSS-SoundEffect v2.

One entry point, `resolve_cue`, drives both sources with one cache/provenance
shape (`anime_factory.sfx_common`) and persists the outcome to `sfx_cues` so
downstream compose/timeline stages and the R2 upload step can find it by
`(episode_code, cue_key)` without re-resolving.

`prepare_episode_sfx` is the single pre-compose entry point: it derives cues
strictly from explicit board/shot fields (never inventing a generic sound per
shot), resolves each one Freesound→MOSS, persists provenance, and returns the
exact `sfx_clips` / `ambience_clips` / `sfx_cues` shapes
`compose.compose_episode_audio` accepts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from anime_factory.db import utcnow
from anime_factory.r2_paths import episode_sfx_key, shared_sfx_key
from anime_factory.sfx_common import SfxCue, SfxProvenance, SfxResult, redact_secrets
from anime_factory.sfx_freesound import FreesoundClient, freesound_api_key, resolve_via_freesound
from anime_factory.sfx_moss import MossConfigError, MossSoundEffectClient, resolve_via_moss

log = logging.getLogger("anime_factory.sfx.orchestrate")


class SfxResolutionError(RuntimeError):
    """Neither Freesound nor MOSS produced an acceptable render for this cue."""


class SfxCueSpecError(ValueError):
    """A shot's explicit sfx/ambience field is malformed (e.g. no query)."""


def r2_key_for(story_id: str, cue: SfxCue) -> str:
    if cue.shared:
        return shared_sfx_key(cue.cue_key)
    return episode_sfx_key(story_id, cue.episode_code or "EP001", cue.cue_key)


def sfx_cue_db_id(cue: SfxCue) -> str:
    """Episode-scoped (or shared-scoped) primary key for `sfx_cues`.

    A bare `cue_key` collides across episodes — EP001's `door_creak` row would
    be silently overwritten by EP002's. Shared cues get one stable row.
    """
    scope = "shared" if cue.shared else (cue.episode_code or "EP001")
    return f"{scope}:{cue.cue_key}"


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
            sfx_cue_db_id(cue),
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
    before_moss: Callable[[], Any] | None = None,
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

    if before_moss is not None:
        try:
            before_moss()
        except Exception as exc:  # noqa: BLE001 — an uncleared GPU cannot safely load MOSS
            failure = SfxResult(
                cue_key=cue.cue_key,
                status="failed",
                reason=f"moss GPU handoff failed: {redact_secrets(str(exc))}",
            )
            if conn is not None:
                _record_cue(conn, story_id, cue, failure)
            raise SfxResolutionError(
                f"{cue.cue_key}: could not release the video stack before MOSS "
                f"({redact_secrets(str(exc))})"
            ) from exc

    moss = moss_client or MossSoundEffectClient()
    fs_reason = redact_secrets(str(freesound_error))
    try:
        result = resolve_via_moss(moss, cue, root)
    except MossConfigError as exc:
        failure = SfxResult(
            cue_key=cue.cue_key,
            status="failed",
            reason=f"freesound: {fs_reason}; moss: {redact_secrets(str(exc))}",
        )
        if conn is not None:
            _record_cue(conn, story_id, cue, failure)
        raise SfxResolutionError(
            f"{cue.cue_key}: Freesound failed ({fs_reason}) and MOSS fallback is not "
            f"configured ({redact_secrets(str(exc))})"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surfaced as one resolution error
        failure = SfxResult(cue_key=cue.cue_key, status="failed", reason=redact_secrets(str(exc)))
        if conn is not None:
            _record_cue(conn, story_id, cue, failure)
        raise SfxResolutionError(
            f"{cue.cue_key}: Freesound and MOSS both failed: {redact_secrets(str(exc))}"
        ) from exc

    if conn is not None:
        _record_cue(conn, story_id, cue, result)
    return result


# ---------------------------------------------------------------------------
# Pre-compose orchestration: explicit board cues -> resolved clips for
# compose.compose_episode_audio. This is the single helper session-level code
# calls to obtain sfx_clips / ambience_clips / sfx_cues; wiring it into
# gpu_worker.session itself is deliberately left to the session/Docker branch.
# ---------------------------------------------------------------------------


def _shot_start_times(shots: Sequence[dict], clip_durations: dict[str, float] | None) -> dict[str, float]:
    t = 0.0
    out: dict[str, float] = {}
    for shot in shots:
        sid = str(shot.get("id") or "")
        if sid:
            out[sid] = t
        t += float((clip_durations or {}).get(sid) or shot.get("duration") or 8.0)
    return out


def _episode_seconds(shots: Sequence[dict], clip_durations: dict[str, float] | None) -> float:
    return sum(
        float((clip_durations or {}).get(str(s.get("id") or "")) or s.get("duration") or 8.0)
        for s in shots
    )


def _cue_from_entry(
    entry: Any,
    *,
    shot: dict,
    index: int,
    episode_code: str,
    default_bus: str,
) -> tuple[SfxCue, float, bool]:
    """One explicit board entry -> (cue, offset_s within the shot, required)."""
    sid = str(shot.get("id") or "")
    if isinstance(entry, str):
        entry = {"query": entry}
    if not isinstance(entry, dict):
        raise SfxCueSpecError(f"{sid}: sfx entry #{index} must be a string or object, got {type(entry).__name__}")
    query = str(entry.get("query") or entry.get("prompt") or "").strip()
    if not query:
        raise SfxCueSpecError(f"{sid}: sfx entry #{index} has no query/prompt text")
    bus = str(entry.get("bus") or default_bus)
    duration = float(
        entry.get("duration_s")
        or entry.get("duration_target")
        or (shot.get("duration") if bus == "ambience" else 0)
        or 2.0
    )
    cue_key = str(entry.get("key") or entry.get("cue_key") or f"{sid}_{bus}{index}")
    cue = SfxCue(
        cue_key=cue_key,
        query=query,
        tags=tuple(str(t) for t in (entry.get("tags") or ())),
        duration_target=duration,
        duration_tolerance=float(entry.get("duration_tolerance") or 0.6),
        bus=bus,
        allow_cc_by=bool(entry.get("allow_cc_by", True)),
        episode_code=episode_code,
        segment_id=sid or None,
        shared=bool(entry.get("shared", False)),
        seed=entry.get("seed"),
    )
    offset_s = max(0.0, float(entry.get("offset_s") or 0.0))
    required = bool(entry.get("required", True))
    return cue, offset_s, required


def explicit_cues_from_shots(
    shots: Sequence[dict],
    episode_code: str,
    *,
    clip_durations: dict[str, float] | None = None,
) -> list[dict]:
    """Cues from explicit board/shot fields ONLY. No field, no cue.

    A shot may carry:
      - `sfx`: a list of strings (bare queries) or objects
        `{query, key?, tags?, duration_s?, offset_s?, bus?, required?, shared?,
        allow_cc_by?, seed?}`;
      - `ambience`: one string or object, bus forced to "ambience", spanning
        the shot by default.

    This function never invents a generic sound for a shot that declares
    nothing — silence is a valid, common shot state.
    """
    starts = _shot_start_times(shots, clip_durations)
    out: list[dict] = []
    for shot in shots:
        sid = str(shot.get("id") or "")
        base = starts.get(sid, 0.0)
        entries = shot.get("sfx") or []
        if isinstance(entries, (str, dict)):
            entries = [entries]
        for index, entry in enumerate(entries):
            cue, offset_s, required = _cue_from_entry(
                entry, shot=shot, index=index, episode_code=episode_code, default_bus="sfx"
            )
            out.append({"cue": cue, "onset_s": base + offset_s, "required": required})
        ambience = shot.get("ambience")
        if ambience:
            cue, offset_s, required = _cue_from_entry(
                ambience, shot=shot, index=0, episode_code=episode_code, default_bus="ambience"
            )
            out.append({"cue": cue, "onset_s": base + offset_s, "required": required})
    return out


def prepare_episode_sfx(
    shots: Sequence[dict],
    root: Path,
    episode_code: str,
    *,
    story_id: str = "",
    conn: sqlite3.Connection | None = None,
    clip_durations: dict[str, float] | None = None,
    freesound_client: FreesoundClient | None = None,
    moss_client: MossSoundEffectClient | None = None,
    before_moss: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Resolve every explicit board cue and return compose-ready inputs.

    Returns `{"sfx_clips", "ambience_clips", "sfx_cues", "results"}` where
    `sfx_clips`/`ambience_clips` are `(onset_s, wav_bytes)` pairs for
    `compose_episode_audio` and `sfx_cues` rows feed the frame timeline.

    A failed *required* cue raises `SfxResolutionError` — total SFX failure is
    never silently swallowed for a cue the board explicitly asked for. Cues
    marked `required: false` degrade to a warning. Only normalized PCM WAVs
    are ever referenced; model weights are never uploaded (see r2_key_for /
    r2_paths — every key is a `.wav` under audio/sfx or shared/sfx).
    """
    derived = explicit_cues_from_shots(shots, episode_code, clip_durations=clip_durations)
    episode_s = _episode_seconds(shots, clip_durations)
    sfx_clips: list[tuple[float, bytes]] = []
    ambience_clips: list[tuple[float, bytes]] = []
    cue_rows: list[dict] = []
    results: dict[str, SfxResult] = {}
    moss_handoff_done = False

    def handoff_to_moss_once() -> None:
        nonlocal moss_handoff_done
        if moss_handoff_done or before_moss is None:
            return
        before_moss()
        moss_handoff_done = True

    for item in derived:
        cue: SfxCue = item["cue"]
        onset_s = float(item["onset_s"])
        try:
            result = resolve_cue(
                cue,
                root,
                story_id=story_id,
                conn=conn,
                freesound_client=freesound_client,
                moss_client=moss_client,
                before_moss=handoff_to_moss_once if before_moss is not None else None,
            )
        except SfxResolutionError:
            if item["required"]:
                raise
            log.warning("optional cue %s failed; episode continues without it", cue.cue_key)
            continue
        results[cue.cue_key] = result
        blob = Path(result.local_path).read_bytes()
        measured_s = float(result.provenance.duration if result.provenance else cue.duration_target)
        # Clamp to the episode boundary so the frame timeline's fail-closed
        # validation (sfx end <= episode end ±1 frame) always holds.
        duration_s = max(0.0, min(measured_s, episode_s - onset_s))
        if duration_s <= 0.0:
            log.warning("cue %s onset %.2fs is at/past episode end %.2fs; dropped", cue.cue_key, onset_s, episode_s)
            continue
        target = ambience_clips if cue.bus == "ambience" else sfx_clips
        target.append((onset_s, blob))
        cue_rows.append(
            {
                "cue_key": cue.cue_key,
                "bus": cue.bus,
                "onset_s": onset_s,
                "duration_s": duration_s,
                "shared": cue.shared,
            }
        )
    return {
        "sfx_clips": sfx_clips,
        "ambience_clips": ambience_clips,
        "sfx_cues": cue_rows,
        "results": results,
    }

"""Bridge hosted (control-plane) pre-GPU TTS into the worker's line_audio table.

The private control plane's hosted preprod writes, per episode:

  - `stories/<id>/episodes/<EP>/audio/lines/<shot>.<lang>.wav` — line wavs;
  - `stories/<id>/episodes/<EP>/audio/tts_manifest.json` — a manifest whose
    `lines` map is keyed `<shot>.<lang>` with `{voice, model, fingerprint,
    reused?}` plus `voice_profiles`. Its per-line fingerprint is
    `sha256("v1|{model}|{voice_uri}|{text}")` truncated to 16 hex.

The worker's `produce_episode` gates line-wav reuse on a `line_audio` row whose
*local* fingerprint (`anime_factory.tts.line_fingerprint` — character, gender,
voice, model, speed, lang, text, lock_version) matches. Hosted wavs have no
such row, so after leasing the worker would re-synthesize audio that already
exists on disk/R2.

This module bridges the two safely instead of weakening either side:

  1. `seed_hosted_voice_lock` establishes the hosted voice_uri as the
     character's lock only where no lock exists yet and the hosted speaker
     passes the worker's own gender validation — an already-established lock
     always wins (hosted wavs in a different voice are rejected and re-made).
  2. `import_hosted_line_audio` verifies, per line: manifest fingerprint
     against the *current* board text + locked voice + model (stale text /
     voice / model → reject), measured on-disk duration plausibility, and
     speaker gender. Verified lines are recorded in `line_audio` under the
     worker's rich local fingerprint (the reuse gate), with the hosted 16-hex
     fingerprint stored alongside — both identities kept, deterministically.

Anything rejected simply has no `line_audio` row, so the normal TTS loop
re-synthesizes (or blocks) exactly as it would for a missing wav.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from anime_factory.models import TTS_MODEL
from anime_factory.tts import (
    LineDurationError,
    VoiceGenderError,
    assert_line_duration_plausible,
    assert_voice_gender_lock,
    existing_voice_lock,
    line_fingerprint,
    pcm_duration_seconds,
    record_line_audio,
    strip_stage_directions,
    wav_sha256,
)

log = logging.getLogger("anime_factory.tts_hosted")

# Mirrors packages/cf-control hosted_preprod.ts exactly.
HOSTED_FINGERPRINT_VERSION = "v1"
HOSTED_FINGERPRINT_HEX_LEN = 16


def hosted_line_fingerprint(text: str, voice_uri: str, model: str = TTS_MODEL) -> str:
    """sha256("v1|model|voice_uri|text")[:16] — the control plane's per-line id."""
    canonical = f"{HOSTED_FINGERPRINT_VERSION}|{model}|{voice_uri}|{text or ''}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:HOSTED_FINGERPRINT_HEX_LEN]


def hosted_manifest_path(root: Path, episode_code: str) -> Path:
    return Path(root) / "episodes" / episode_code / "audio" / "tts_manifest.json"


def hosted_line_wav_path(root: Path, episode_code: str, shot_id: str, lang: str) -> Path:
    return Path(root) / "episodes" / episode_code / "audio" / "lines" / f"{shot_id}.{lang}.wav"


def load_hosted_tts_manifest(root: Path, episode_code: str) -> dict[str, Any] | None:
    """Parsed manifest with a non-empty, well-formed `lines` map, else None.

    A deferred manifest (hosted TTS never completed) or a stale/invalid file
    yields None: the normal synthesis path then runs — never a silent reuse of
    audio the control plane itself did not vouch for.
    """
    path = hosted_manifest_path(root, episode_code)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("unreadable tts_manifest.json at %s; ignoring hosted TTS", path)
        return None
    if not isinstance(payload, dict) or payload.get("deferred"):
        return None
    lines = payload.get("lines")
    if not isinstance(lines, dict) or not lines:
        return None
    cleaned: dict[str, dict[str, str]] = {}
    for key, value in lines.items():
        if not isinstance(value, dict):
            continue
        fp = str(value.get("fingerprint") or "")
        voice = str(value.get("voice") or "")
        model = str(value.get("model") or "")
        if fp and voice and model:
            cleaned[str(key)] = {"fingerprint": fp, "voice": voice, "model": model}
    if not cleaned:
        return None
    return {
        "lines": cleaned,
        "voice_profiles": dict(payload.get("voice_profiles") or {}),
    }


def hosted_voice_for(
    manifest: dict[str, Any],
    character_id: str,
    shots: Sequence[dict],
    langs: Sequence[str],
) -> str | None:
    """The single hosted CosyVoice speaker used for this character, or None.

    Speakers are cross-lingual: a character whose hosted lines drift across
    voices is invalid and gets no seeded lock (the worker re-locks and
    re-synthesizes deterministically instead).
    """
    lines = manifest.get("lines") or {}
    voices: set[str] = set()
    for shot in shots:
        if str(shot.get("character_id") or "") != character_id:
            continue
        sid = str(shot.get("id") or "")
        for lang in langs:
            entry = lines.get(f"{sid}.{lang}")
            if entry:
                voices.add(entry["voice"])
    if len(voices) != 1:
        if len(voices) > 1:
            log.warning("hosted voices drift for %s: %s; not seeding a lock", character_id, sorted(voices))
        return None
    return voices.pop()


def seed_hosted_voice_lock(
    conn: sqlite3.Connection,
    character_id: str,
    voice_uri: str,
    langs: Sequence[str],
    *,
    gender: str,
) -> bool:
    """Establish the hosted voice as this character's lock — only when no lock
    exists yet and the hosted speaker passes the worker's gender validation.

    An already-established (non-empty) `character_voice.voice_uri` is never
    overwritten: `produce._lock_voice` / `resolve_locked_voice_uri` preserve
    it, and hosted wavs recorded in a different voice will simply fail the
    voice match in `import_hosted_line_audio` and be re-synthesized.
    """
    try:
        assert_voice_gender_lock(character_id, voice_uri, gender=gender)
    except VoiceGenderError as exc:
        log.warning("hosted voice rejected for %s: %s", character_id, exc)
        return False
    seeded = False
    for lang in langs:
        row = existing_voice_lock(conn, character_id, lang)
        if row is not None and str(row["voice_uri"] or ""):
            continue
        conn.execute(
            """
            INSERT INTO character_voice (character_id, lang, voice_uri, reference_audio,
                speed, emotion, version, lock_version)
            VALUES (?, ?, ?, ?, 1.0, 'neutral', 'v1', 1)
            ON CONFLICT(character_id, lang) DO UPDATE SET
                voice_uri = excluded.voice_uri
                WHERE character_voice.voice_uri IS NULL OR character_voice.voice_uri = ''
            """,
            (character_id, lang, voice_uri, f"assets/voice/{character_id}.{lang}.wav"),
        )
        seeded = True
    if seeded:
        conn.commit()
    return seeded


def _shot_line_text(line: dict, lang: str, primary: str) -> str:
    return str(line.get(lang) or line.get(primary) or line.get("text") or "").strip()


def import_hosted_line_audio(
    conn: sqlite3.Connection,
    root: Path,
    episode_code: str,
    shots: Sequence[dict],
    langs: Sequence[str],
    *,
    resolved_gender: dict[str, str],
    manifest: dict[str, Any],
    primary: str | None = None,
) -> dict[str, Any]:
    """Verify hosted line wavs and seed `line_audio` so produce reuses them.

    Per `<shot>.<lang>`: the manifest fingerprint must equal the hosted
    fingerprint recomputed from the *current* board text, manifest voice, and
    manifest model; the model must be the worker's TTS model; the wav must
    exist with a plausible measured duration for the text; the voice must
    match the character's locked voice AND the worker-resolved gender. Any
    failure rejects that line (recorded in the returned summary) and leaves it
    to the normal synthesis path — a stale manifest can never license reuse.
    """
    lines = manifest.get("lines") or {}
    primary_lang = primary or (langs[0] if langs else "zh")
    imported: list[str] = []
    rejected: dict[str, str] = {}
    for shot in shots:
        line = shot.get("line")
        sid = str(shot.get("id") or "")
        if not line or not sid:
            continue
        cid = str(shot.get("character_id") or "")
        gender = str(resolved_gender.get(cid) or "")
        for lang in langs:
            key = f"{sid}.{lang}"
            entry = lines.get(key)
            if not entry:
                continue
            text = strip_stage_directions(_shot_line_text(line, lang, primary_lang))
            if not text:
                continue
            if entry["model"] != TTS_MODEL:
                rejected[key] = f"model {entry['model']!r} != {TTS_MODEL!r}"
                continue
            voice = entry["voice"]
            expected_fp = hosted_line_fingerprint(text, voice, entry["model"])
            if entry["fingerprint"] != expected_fp:
                rejected[key] = "stale hosted fingerprint (text/voice/model drifted)"
                continue
            wav_path = hosted_line_wav_path(root, episode_code, sid, lang)
            if not wav_path.is_file() or wav_path.stat().st_size <= 100:
                rejected[key] = "hosted wav missing on disk"
                continue
            data = wav_path.read_bytes()
            duration = pcm_duration_seconds(data)
            try:
                assert_line_duration_plausible(duration, text, lang, sid=key)
            except LineDurationError as exc:
                rejected[key] = f"implausible duration: {exc}"
                continue
            try:
                assert_voice_gender_lock(cid, voice, gender=gender)
            except VoiceGenderError as exc:
                rejected[key] = f"gender mismatch: {exc}"
                continue
            lock = existing_voice_lock(conn, cid, lang)
            locked_uri = str(lock["voice_uri"]) if lock and lock["voice_uri"] else ""
            if locked_uri != voice:
                rejected[key] = f"hosted voice does not match locked voice for {cid}/{lang}"
                continue
            speed = float(lock["speed"] or 1.0)
            lock_version = int(lock["lock_version"] or 1)
            local_fp = line_fingerprint(cid, gender, voice, TTS_MODEL, speed, lang, text, lock_version)
            record_line_audio(
                conn,
                sid,
                lang,
                cid,
                local_fp,
                str(wav_path),
                duration,
                24000,
                wav_sha256(data),
                hosted_fingerprint=entry["fingerprint"],
            )
            imported.append(key)
    if imported or rejected:
        log.info(
            "hosted TTS bridge %s: imported=%d rejected=%d",
            episode_code,
            len(imported),
            len(rejected),
        )
    return {"imported": imported, "rejected": rejected}

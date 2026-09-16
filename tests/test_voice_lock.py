"""Gender resolution, voice-lock preservation, and line-wav fingerprint reuse."""

from __future__ import annotations

import pytest

from anime_factory.db import migrate, open_db
from anime_factory.tts import (
    GenderConflictError,
    GenderRequiredError,
    existing_voice_lock,
    line_fingerprint,
    line_wav_reusable,
    profile_fingerprint,
    record_line_audio,
    resolve_gender,
    resolve_locked_voice_uri,
    stock_voice_uri,
)


def _conn(tmp_path):
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    return conn


def test_unknown_speaker_no_longer_defaults_to_male():
    """The removed bug: a speaker with zero gender signal used to fall to male."""
    with pytest.raises(GenderRequiredError):
        resolve_gender("mystery_voice", gender=None, identity="a figure in the doorway", name=None)


def test_female_unknown_speaker_resolves_from_identity_marker_not_male_default():
    """No explicit `gender` field and no character_id hint — only a female textual
    marker in the identity — must resolve female, not fall back to male."""
    gender = resolve_gender("speaker_07", gender=None, identity="a woman steps out of the rain, she says nothing", name=None)
    assert gender == "female"


def test_female_unknown_speaker_resolves_from_1girl_tag():
    gender = resolve_gender("cast_12", gender=None, identity="1girl, short bob hair, school uniform", name=None)
    assert gender == "female"


def test_gender_field_and_1boy_1girl_tag_conflict_raises():
    with pytest.raises(GenderConflictError) as excinfo:
        resolve_gender("ke", gender="male", identity="1girl, ponytail, school uniform", name="阿柯")
    assert "ke" in str(excinfo.value)


def test_gender_field_and_identity_marker_conflict_raises():
    with pytest.raises(GenderConflictError):
        resolve_gender("li", gender="female", identity="he walked into the store, a tired man", name="阿李")


def test_explicit_gender_field_wins_when_no_conflicting_signal():
    assert resolve_gender("hero", gender="male", identity="a quiet figure", name="Hero") == "male"
    assert resolve_gender("heroine", gender="female", identity="a quiet figure", name="Heroine") == "female"


def test_cross_language_same_speaker_uses_one_stock_voice_regardless_of_lang():
    uri_zh = stock_voice_uri("ke", "zh", gender="male", identity="1boy, quiet hacker", name="阿柯")
    uri_en = stock_voice_uri("ke", "en", gender="male", identity="1boy, quiet hacker", name="阿柯")
    uri_ja = stock_voice_uri("ke", "ja", gender="male", identity="1boy, quiet hacker", name="阿柯")
    assert uri_zh == uri_en == uri_ja


def test_profile_and_line_fingerprint_are_deterministic_and_sensitive_to_inputs():
    fp1 = profile_fingerprint("ke", "male", "voice://ke/zh", "model-a", 1)
    fp2 = profile_fingerprint("ke", "male", "voice://ke/zh", "model-a", 1)
    assert fp1 == fp2
    fp3 = profile_fingerprint("ke", "male", "voice://ke/zh", "model-a", 2)
    assert fp3 != fp1

    lf1 = line_fingerprint("ke", "male", "voice://ke/zh", "model-a", 1.0, "zh", "你好世界", 1)
    lf2 = line_fingerprint("ke", "male", "voice://ke/zh", "model-a", 1.0, "zh", "你好世界", 1)
    assert lf1 == lf2
    # Any drift (text, speed, voice, lock_version) must change the fingerprint.
    assert line_fingerprint("ke", "male", "voice://ke/zh", "model-a", 1.0, "zh", "再见世界", 1) != lf1
    assert line_fingerprint("ke", "male", "voice://ke/zh", "model-a", 1.1, "zh", "你好世界", 1) != lf1
    assert line_fingerprint("ke", "male", "voice://ke/zh", "model-a", 1.0, "zh", "你好世界", 2) != lf1


def test_lock_preserved_across_episodes_unless_migrated(tmp_path):
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO character_voice (character_id, lang, voice_uri, speed, lock_version) VALUES (?, ?, ?, 1.0, 1)",
        ("ke", "zh", "voice://established/ke-zh"),
    )
    conn.commit()

    # Recomputing a candidate URI from a lightly-edited identity must not win.
    uri, version = resolve_locked_voice_uri(conn, "ke", "zh", "voice://freshly-recomputed/ke-zh")
    assert uri == "voice://established/ke-zh"
    assert version == 1

    # An explicit migration is the only way to force the new candidate in,
    # and it must bump lock_version so stale line wavs invalidate.
    uri2, version2 = resolve_locked_voice_uri(conn, "ke", "zh", "voice://freshly-recomputed/ke-zh", migrate=True)
    assert uri2 == "voice://freshly-recomputed/ke-zh"
    assert version2 == 2


def test_lock_uses_candidate_when_nothing_established_yet(tmp_path):
    conn = _conn(tmp_path)
    uri, version = resolve_locked_voice_uri(conn, "new_char", "zh", "voice://brand-new/zh")
    assert uri == "voice://brand-new/zh"
    assert version == 1


def test_stale_wav_invalidated_when_fingerprint_drifts(tmp_path):
    conn = _conn(tmp_path)
    wav_path = tmp_path / "s001.zh.wav"
    wav_path.write_bytes(b"RIFF" + b"\x00" * 200)  # size > 100 bytes, content irrelevant here

    fp_v1 = line_fingerprint("ke", "male", "voice://ke/zh", "model-a", 1.0, "zh", "你好", 1)
    record_line_audio(conn, "s001", "zh", "ke", fp_v1, str(wav_path), 1.0, 24000, "deadbeef")

    # Same fingerprint recomputed -> reusable.
    assert line_wav_reusable(conn, "s001", "zh", fp_v1, wav_path) is True

    # Voice migration bumps lock_version -> fingerprint changes -> stale wav rejected.
    fp_v2 = line_fingerprint("ke", "male", "voice://ke/zh", "model-a", 1.0, "zh", "你好", 2)
    assert line_wav_reusable(conn, "s001", "zh", fp_v2, wav_path) is False

    # A missing file is never reusable even with a matching fingerprint.
    missing = tmp_path / "s002.zh.wav"
    record_line_audio(conn, "s002", "zh", "ke", fp_v1, str(missing), 1.0, 24000, "deadbeef")
    assert line_wav_reusable(conn, "s002", "zh", fp_v1, missing) is False


def test_existing_voice_lock_reads_back_row(tmp_path):
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO character_voice (character_id, lang, voice_uri, speed, lock_version, profile_fingerprint) "
        "VALUES (?, ?, ?, 1.05, 3, 'fp123')",
        ("li", "en", "voice://li/en"),
    )
    conn.commit()
    row = existing_voice_lock(conn, "li", "en")
    assert row["voice_uri"] == "voice://li/en"
    assert row["lock_version"] == 3
    assert row["profile_fingerprint"] == "fp123"
    assert existing_voice_lock(conn, "nobody", "en") is None

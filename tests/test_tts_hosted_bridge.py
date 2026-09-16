"""Hosted pre-GPU TTS bridge: control-plane tts_manifest.json + R2 line wavs are
verified and reused by the worker without a synthesis call; stale text/voice/
model is rejected and re-synthesized. Mirrors the produce_episode wiring:
seed voice lock -> _lock_voice -> import_hosted_line_audio -> ensure_line_wavs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anime_factory.db import migrate, open_db
from anime_factory.models import TTS_MODEL
from anime_factory.produce import _lock_voice, ensure_line_wavs
from anime_factory.tts import CosyVoiceClient, line_fingerprint, voiced_dummy_wav
from anime_factory.tts_hosted import (
    hosted_line_fingerprint,
    hosted_voice_for,
    import_hosted_line_audio,
    load_hosted_tts_manifest,
    seed_hosted_voice_lock,
)

EP = "EP001"
HOSTED_VOICE = f"{TTS_MODEL}:alex"  # male stock speaker, matches gender below
TEXT = "你好世界一二三四五"


def _conn(tmp_path):
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    return conn


def _shots(text: str = TEXT) -> list[dict]:
    return [{"id": f"{EP}-01", "seq": 1, "duration": 8.0, "character_id": "ke", "line": {"zh": text}}]


def _write_hosted_artifacts(root: Path, *, text: str = TEXT, voice: str = HOSTED_VOICE, manifest_text: str | None = None):
    """Line wav + manifest exactly as the control plane's hosted preprod writes them."""
    lines_dir = root / "episodes" / EP / "audio" / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    wav = voiced_dummy_wav(2.2, freq=210.0)
    (lines_dir / f"{EP}-01.zh.wav").write_bytes(wav)
    fp = hosted_line_fingerprint(manifest_text if manifest_text is not None else text, voice)
    manifest = {
        "n_tts": 1,
        "shots": 1,
        "voice_profiles": {"ke": "aabbccdd00112233"},
        "lines": {f"{EP}-01.zh": {"voice": voice, "model": TTS_MODEL, "fingerprint": fp, "reused": True}},
    }
    (root / "episodes" / EP / "audio" / "tts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return wav


class ExplodingTts(CosyVoiceClient):
    """Any synthesis attempt is a test failure."""

    def __init__(self):
        super().__init__(["k"], live=False)

    def speech(self, text, voice_uri, speed=1.0):  # noqa: ARG002
        raise AssertionError("hosted TTS must be reused; synthesis must not be called")


class CountingTts(CosyVoiceClient):
    def __init__(self):
        super().__init__(["k"], live=False)
        self.calls = 0

    def speech(self, text, voice_uri, speed=1.0):
        self.calls += 1
        return super().speech(text, voice_uri, speed)


def test_hosted_line_fingerprint_matches_control_plane_formula():
    canonical = f"v1|{TTS_MODEL}|{HOSTED_VOICE}|{TEXT}"
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    assert hosted_line_fingerprint(TEXT, HOSTED_VOICE) == expected
    assert len(expected) == 16


def test_load_manifest_missing_deferred_or_empty_is_none(tmp_path):
    assert load_hosted_tts_manifest(tmp_path, EP) is None
    audio = tmp_path / "episodes" / EP / "audio"
    audio.mkdir(parents=True)
    (audio / "tts_manifest.json").write_text(json.dumps({"deferred": True, "n_tts": 0}), encoding="utf-8")
    assert load_hosted_tts_manifest(tmp_path, EP) is None
    (audio / "tts_manifest.json").write_text(json.dumps({"n_tts": 0, "lines": {}}), encoding="utf-8")
    assert load_hosted_tts_manifest(tmp_path, EP) is None
    (audio / "tts_manifest.json").write_text("{not json", encoding="utf-8")
    assert load_hosted_tts_manifest(tmp_path, EP) is None


def test_hosted_voice_seeded_only_when_no_established_lock(tmp_path):
    conn = _conn(tmp_path)
    assert seed_hosted_voice_lock(conn, "ke", HOSTED_VOICE, ("zh",), gender="male") is True
    row = conn.execute(
        "SELECT voice_uri, lock_version FROM character_voice WHERE character_id='ke' AND lang='zh'"
    ).fetchone()
    assert row["voice_uri"] == HOSTED_VOICE
    assert row["lock_version"] == 1
    # An established lock must never be overwritten by a hosted voice.
    conn2 = _conn(tmp_path.joinpath("other"))
    _lock_voice(conn2, "ke", ("zh",), {"zh": "voice://established/zh"}, gender="male")
    seed_hosted_voice_lock(conn2, "ke", HOSTED_VOICE, ("zh",), gender="male")
    kept = conn2.execute(
        "SELECT voice_uri FROM character_voice WHERE character_id='ke' AND lang='zh'"
    ).fetchone()
    assert kept["voice_uri"] == "voice://established/zh"


def test_hosted_voice_gender_mismatch_not_seeded(tmp_path):
    conn = _conn(tmp_path)
    female_voice = f"{TTS_MODEL}:claire"
    assert seed_hosted_voice_lock(conn, "ke", female_voice, ("zh",), gender="male") is False
    assert conn.execute("SELECT COUNT(*) AS n FROM character_voice").fetchone()["n"] == 0


def test_hosted_wav_reused_end_to_end_without_synthesis(tmp_path):
    """Full worker-side flow: manifest verify -> line_audio seed -> produce's
    ensure_line_wavs reuses the hosted wav; the TTS client is never called."""
    conn = _conn(tmp_path)
    shots = _shots()
    _write_hosted_artifacts(tmp_path)

    manifest = load_hosted_tts_manifest(tmp_path, EP)
    assert manifest is not None
    assert hosted_voice_for(manifest, "ke", shots, ("zh",)) == HOSTED_VOICE
    seed_hosted_voice_lock(conn, "ke", HOSTED_VOICE, ("zh",), gender="male")
    # produce_episode then runs _lock_voice with a freshly recomputed candidate;
    # the seeded hosted lock must survive it.
    _lock_voice(conn, "ke", ("zh",), {"zh": "voice://recomputed/zh"}, gender="male")

    summary = import_hosted_line_audio(
        conn, tmp_path, EP, shots, ("zh",), resolved_gender={"ke": "male"}, manifest=manifest
    )
    assert summary["imported"] == [f"{EP}-01.zh"]
    assert summary["rejected"] == {}
    row = conn.execute(
        "SELECT fingerprint, hosted_fingerprint FROM line_audio WHERE segment_id=? AND lang='zh'",
        (f"{EP}-01",),
    ).fetchone()
    lock = conn.execute(
        "SELECT voice_uri, speed, lock_version FROM character_voice WHERE character_id='ke' AND lang='zh'"
    ).fetchone()
    assert lock["voice_uri"] == HOSTED_VOICE
    # Both identities stored: rich local fingerprint is the reuse gate, the
    # hosted 16-hex fingerprint is recorded alongside — neither weakened.
    expected_local = line_fingerprint(
        "ke", "male", HOSTED_VOICE, TTS_MODEL, float(lock["speed"]), "zh", TEXT, int(lock["lock_version"])
    )
    assert row["fingerprint"] == expected_local
    assert row["hosted_fingerprint"] == hosted_line_fingerprint(TEXT, HOSTED_VOICE)

    audio_root = tmp_path / "episodes" / EP / "audio" / "lines"
    result = ensure_line_wavs(conn, ExplodingTts(), audio_root, shots, ("zh",), "zh", {"ke": "male"})
    assert result["reused"] == [f"{EP}-01.zh"]
    assert result["synthesized"] == []


def test_stale_board_text_rejected_and_resynthesized(tmp_path):
    """Manifest fingerprint was computed for old text; the board text changed —
    reuse must be rejected and the line re-synthesized."""
    conn = _conn(tmp_path)
    new_text = "完全不同的新台词哦"
    shots = _shots(new_text)
    _write_hosted_artifacts(tmp_path, manifest_text=TEXT)  # fingerprint of the OLD text

    manifest = load_hosted_tts_manifest(tmp_path, EP)
    seed_hosted_voice_lock(conn, "ke", HOSTED_VOICE, ("zh",), gender="male")
    summary = import_hosted_line_audio(
        conn, tmp_path, EP, shots, ("zh",), resolved_gender={"ke": "male"}, manifest=manifest
    )
    assert summary["imported"] == []
    assert "stale hosted fingerprint" in summary["rejected"][f"{EP}-01.zh"]

    tts = CountingTts()
    audio_root = tmp_path / "episodes" / EP / "audio" / "lines"
    result = ensure_line_wavs(conn, tts, audio_root, shots, ("zh",), "zh", {"ke": "male"})
    assert result["synthesized"] == [f"{EP}-01.zh"]
    assert tts.calls == 1  # re-synthesized exactly once


def test_wrong_model_or_voice_rejected(tmp_path):
    conn = _conn(tmp_path)
    shots = _shots()
    _write_hosted_artifacts(tmp_path)
    manifest = load_hosted_tts_manifest(tmp_path, EP)
    # Tamper: model drift.
    manifest["lines"][f"{EP}-01.zh"]["model"] = "SomeOther/TTS-Model"
    seed_hosted_voice_lock(conn, "ke", HOSTED_VOICE, ("zh",), gender="male")
    summary = import_hosted_line_audio(
        conn, tmp_path, EP, shots, ("zh",), resolved_gender={"ke": "male"}, manifest=manifest
    )
    assert summary["imported"] == []
    assert "model" in summary["rejected"][f"{EP}-01.zh"]

    # Voice drift vs the established lock: hosted wav in a different voice.
    conn2 = _conn(tmp_path / "locked")
    root2 = tmp_path / "locked_root"
    _write_hosted_artifacts(root2)
    manifest2 = load_hosted_tts_manifest(root2, EP)
    _lock_voice(conn2, "ke", ("zh",), {"zh": f"{TTS_MODEL}:charles"}, gender="male")
    summary2 = import_hosted_line_audio(
        conn2, root2, EP, shots, ("zh",), resolved_gender={"ke": "male"}, manifest=manifest2
    )
    assert summary2["imported"] == []
    assert "does not match locked voice" in summary2["rejected"][f"{EP}-01.zh"]


def test_missing_wav_and_implausible_duration_rejected(tmp_path):
    conn = _conn(tmp_path)
    shots = _shots()
    _write_hosted_artifacts(tmp_path)
    wav_path = tmp_path / "episodes" / EP / "audio" / "lines" / f"{EP}-01.zh.wav"
    wav_path.unlink()
    manifest = load_hosted_tts_manifest(tmp_path, EP)
    seed_hosted_voice_lock(conn, "ke", HOSTED_VOICE, ("zh",), gender="male")
    summary = import_hosted_line_audio(
        conn, tmp_path, EP, shots, ("zh",), resolved_gender={"ke": "male"}, manifest=manifest
    )
    assert summary["rejected"][f"{EP}-01.zh"] == "hosted wav missing on disk"

    # A wav far too long for its text must also be rejected, not trusted.
    wav_path.write_bytes(voiced_dummy_wav(25.0, freq=210.0))
    summary2 = import_hosted_line_audio(
        conn, tmp_path, EP, shots, ("zh",), resolved_gender={"ke": "male"}, manifest=manifest
    )
    assert "implausible duration" in summary2["rejected"][f"{EP}-01.zh"]

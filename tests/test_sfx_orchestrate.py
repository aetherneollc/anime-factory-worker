"""Cue resolution: Freesound is always attempted first, MOSS is the fallback."""

from __future__ import annotations

import json
import subprocess

import pytest

from anime_factory.db import migrate, open_db
from anime_factory.sfx_common import SfxCue
from anime_factory.sfx_freesound import FreesoundClient
from anime_factory.sfx_moss import MossRunnerConfig, MossSoundEffectClient
from anime_factory.sfx_orchestrate import SfxResolutionError, r2_key_for, resolve_cue
from anime_factory.tts import voiced_dummy_wav


def _fs_opener(rows, previews):
    def opener(req):
        url = req.full_url
        if "/search/" in url:
            return json.dumps({"results": rows}).encode("utf-8")
        for key, blob in previews.items():
            if key in url:
                return blob
        return b""

    return opener


def _candidate(sound_id, license_url, preview):
    return {
        "id": sound_id,
        "name": "door creak",
        "tags": ["door", "creak"],
        "duration": 1.5,
        "license": license_url,
        "username": "u",
        "previews": {"preview-hq-mp3": preview},
        "samplerate": 44100,
        "avg_rating": 4.0,
        "num_ratings": 5,
    }


def _moss_runner_writing(wav_bytes):
    def runner(args, env, timeout_s):
        out_path = args[args.index("--out") + 1]
        with open(out_path, "wb") as fh:
            fh.write(wav_bytes)
        return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=b"")

    return runner


def test_freesound_success_never_calls_moss(tmp_path):
    rows = [_candidate(1, "http://creativecommons.org/publicdomain/zero/1.0/", "https://cdn.freesound.org/p.mp3")]
    fs_client = FreesoundClient("k", opener=_fs_opener(rows, {"p.mp3": voiced_dummy_wav(1.5, freq=300.0)}))

    def exploding_runner(args, env, timeout_s):
        raise AssertionError("MOSS must not run when Freesound succeeds")

    moss_client = MossSoundEffectClient(
        config=MossRunnerConfig("py", "script.py", str(tmp_path / "w")), runner=exploding_runner
    )
    cue = SfxCue(cue_key="door_creak", query="door creak", duration_target=1.5, episode_code="EP001")
    result = resolve_cue(cue, tmp_path, freesound_client=fs_client, moss_client=moss_client)
    assert result.status == "resolved"
    assert result.provenance.source == "freesound"


def test_freesound_miss_falls_back_to_moss_and_persists(tmp_path):
    fs_client = FreesoundClient("k", opener=_fs_opener([], {}))  # no candidates at all
    wav = voiced_dummy_wav(2.0, freq=400.0)
    moss_client = MossSoundEffectClient(
        config=MossRunnerConfig("py", "script.py", str(tmp_path / "w")), runner=_moss_runner_writing(wav)
    )
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    cue = SfxCue(cue_key="thunder_rumble", query="thunder rumble", duration_target=2.0, episode_code="EP001", seed=1)
    result = resolve_cue(cue, tmp_path, story_id="story-abc-123", conn=conn, freesound_client=fs_client, moss_client=moss_client)
    assert result.status == "resolved"
    assert result.provenance.source == "moss"

    row = conn.execute("SELECT source, status, r2_key FROM sfx_cues WHERE cue_key = ?", (cue.cue_key,)).fetchone()
    assert row is not None
    assert row["source"] == "moss"
    assert row["status"] == "resolved"
    assert "audio/sfx" in row["r2_key"]


def test_both_sources_fail_raises_resolution_error(tmp_path, monkeypatch):
    for key in ("MOSS_SFX_PYTHON", "MOSS_SFX_SCRIPT", "MOSS_SFX_WEIGHTS_DIR"):
        monkeypatch.delenv(key, raising=False)
    fs_client = FreesoundClient("k", opener=_fs_opener([], {}))
    moss_client = MossSoundEffectClient()  # unconfigured: env vars absent
    cue = SfxCue(cue_key="nothing", query="nothing at all", duration_target=1.0, episode_code="EP001")
    with pytest.raises(SfxResolutionError):
        resolve_cue(cue, tmp_path, freesound_client=fs_client, moss_client=moss_client)


def test_r2_key_episode_local_vs_shared():
    episode_cue = SfxCue(cue_key="door_creak", query="door creak", episode_code="EP002", shared=False)
    shared_cue = SfxCue(cue_key="rain_ambience", query="rain", shared=True)
    ep_key = r2_key_for("story-abc-123", episode_cue)
    shared_key = r2_key_for("story-abc-123", shared_cue)
    assert ep_key == "stories/story-abc-123/episodes/EP002/audio/sfx/door_creak.wav"
    assert shared_key == "shared/sfx/rain_ambience.wav"

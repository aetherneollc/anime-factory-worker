"""Cue resolution: Freesound is always attempted first, MOSS is the fallback."""

from __future__ import annotations

import json
import subprocess

import pytest

from anime_factory.db import migrate, open_db
from anime_factory.sfx_common import SfxCue
from anime_factory.sfx_freesound import FreesoundClient
from anime_factory.sfx_moss import MossRunnerConfig, MossSoundEffectClient
from anime_factory.sfx_orchestrate import (
    SfxCueSpecError,
    SfxResolutionError,
    explicit_cues_from_shots,
    prepare_episode_sfx,
    r2_key_for,
    resolve_cue,
    sfx_cue_db_id,
)
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


def test_prepare_releases_video_gpu_once_before_moss_fallbacks(tmp_path):
    fs_client = FreesoundClient("k", opener=_fs_opener([], {}))
    moss_client = MossSoundEffectClient(
        config=MossRunnerConfig("py", "script.py", str(tmp_path / "weights")),
        runner=_moss_runner_writing(voiced_dummy_wav(1.0, freq=410.0)),
    )
    handoffs: list[str] = []
    out = prepare_episode_sfx(
        [
            {"id": "s001", "duration": 2.0, "sfx": [{"key": "a", "query": "sound a", "duration_s": 1.0}]},
            {"id": "s002", "duration": 2.0, "sfx": [{"key": "b", "query": "sound b", "duration_s": 1.0}]},
        ],
        tmp_path,
        "EP001",
        freesound_client=fs_client,
        moss_client=moss_client,
        before_moss=lambda: handoffs.append("released"),
    )
    assert handoffs == ["released"]
    assert len(out["sfx_clips"]) == 2
    assert {result.provenance.source for result in out["results"].values()} == {"moss"}


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


def test_r2_key_slugifies_hostile_cue_keys():
    evil = SfxCue(cue_key="../../weights/model.safetensors", query="q", episode_code="EP001")
    key = r2_key_for("story-abc-123", evil)
    assert key.startswith("stories/story-abc-123/episodes/EP001/audio/sfx/")
    assert ".." not in key
    assert key.endswith(".wav")
    shared_evil = SfxCue(cue_key="a/b\\c", query="q", shared=True)
    shared = r2_key_for("story-abc-123", shared_evil)
    assert shared.startswith("shared/sfx/")
    assert "/a/" not in shared and "\\" not in shared
    assert shared.endswith(".wav")


def test_db_ids_are_episode_scoped_no_cross_episode_collision(tmp_path):
    """`id=cue_key` collided across episodes; ids must be episode/shared-scoped."""
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    wav = voiced_dummy_wav(2.0, freq=380.0)
    for ep in ("EP001", "EP002"):
        moss_client = MossSoundEffectClient(
            config=MossRunnerConfig("py", "script.py", str(tmp_path / "w")), runner=_moss_runner_writing(wav)
        )
        cue = SfxCue(cue_key="door_creak", query="door creak", duration_target=2.0, episode_code=ep, seed=3)
        resolve_cue(
            cue,
            tmp_path,
            story_id="story-abc-123",
            conn=conn,
            freesound_client=FreesoundClient("k", opener=_fs_opener([], {})),
            moss_client=moss_client,
        )
    rows = conn.execute(
        "SELECT id, episode_code FROM sfx_cues WHERE cue_key = 'door_creak' ORDER BY episode_code"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["id"] == "EP001:door_creak"
    assert rows[1]["id"] == "EP002:door_creak"
    shared = SfxCue(cue_key="door_creak", query="door creak", shared=True)
    assert sfx_cue_db_id(shared) == "shared:door_creak"


def test_failure_reason_in_db_is_secret_redacted(tmp_path, monkeypatch):
    secret = "moss-hf-token-a1b2c3d4"
    monkeypatch.setenv("HF_TOKEN", secret)
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)

    import subprocess

    def leaky_runner(args, env, timeout_s):
        return subprocess.CompletedProcess(args, returncode=1, stdout=b"", stderr=f"denied for {secret}".encode())

    moss_client = MossSoundEffectClient(
        config=MossRunnerConfig("py", "script.py", str(tmp_path / "w")), runner=leaky_runner
    )
    cue = SfxCue(cue_key="leaky", query="anything", duration_target=1.0, episode_code="EP001")
    with pytest.raises(SfxResolutionError) as excinfo:
        resolve_cue(
            cue,
            tmp_path,
            story_id="s",
            conn=conn,
            freesound_client=FreesoundClient("k", opener=_fs_opener([], {})),
            moss_client=moss_client,
        )
    assert secret not in str(excinfo.value)
    row = conn.execute("SELECT provenance_json, status FROM sfx_cues WHERE cue_key = 'leaky'").fetchone()
    assert row["status"] == "failed"
    assert secret not in (row["provenance_json"] or "")


# ---------------------------------------------------------------------------
# Pre-compose orchestration: explicit cues only, compose-ready outputs.
# ---------------------------------------------------------------------------


def test_no_explicit_fields_derives_no_cues():
    shots = [
        {"id": "s001", "duration": 8.0, "line": {"zh": "你好"}},
        {"id": "s002", "duration": 8.0},
    ]
    assert explicit_cues_from_shots(shots, "EP001") == []


def test_explicit_sfx_and_ambience_fields_derive_cues_with_onsets():
    shots = [
        {"id": "s001", "duration": 8.0},
        {
            "id": "s002",
            "duration": 6.0,
            "sfx": [
                "door creak",
                {"query": "glass shatter", "key": "glass1", "offset_s": 2.0, "duration_s": 1.0, "tags": ["glass"]},
            ],
            "ambience": {"query": "night crickets", "shared": True},
        },
    ]
    derived = explicit_cues_from_shots(shots, "EP001")
    assert len(derived) == 3
    creak, glass, amb = derived
    assert creak["cue"].query == "door creak"
    assert creak["onset_s"] == 8.0  # shot s002 starts after the 8.0s s001
    assert creak["required"] is True
    assert glass["cue"].cue_key == "glass1"
    assert glass["onset_s"] == 10.0
    assert glass["cue"].tags == ("glass",)
    assert amb["cue"].bus == "ambience"
    assert amb["cue"].shared is True
    assert amb["cue"].duration_target == 6.0  # loopable source bounded to shot/bed
    assert amb["kind"] == "ambience_bed"
    assert amb["bed_duration_s"] == 6.0


def test_scene_ambience_merges_consecutive_same_scene_into_one_bed():
    shots = [
        {
            "id": "c01",
            "duration": 4.0,
            "scene_id": "classroom",
            "ambience": "quiet classroom room tone",
            "sfx": [{"query": "key jingle", "key": "keys", "offset_s": 1.0, "duration_s": 0.5}],
        },
        {
            "id": "c02",
            "duration": 4.0,
            "scene_id": "classroom",
            "ambience": "quiet classroom room tone",
        },
        {
            "id": "r01",
            "duration": 3.0,
            "scene_id": "rooftop",
            "ambience": "windy rooftop night",
        },
    ]
    derived = explicit_cues_from_shots(shots, "EP001")
    sfx = [d for d in derived if d["kind"] == "sfx"]
    beds = [d for d in derived if d["kind"] == "ambience_bed"]
    assert len(sfx) == 1
    assert sfx[0]["onset_s"] == 1.0  # precise cut onset inside first shot
    assert sfx[0]["cue"].cue_key == "keys"
    assert len(beds) == 2
    classroom, rooftop = beds
    assert classroom["onset_s"] == 0.0
    assert classroom["bed_duration_s"] == 8.0
    assert classroom["shot_ids"] == ["c01", "c02"]
    assert classroom["cue"].cue_key == "classroom_ambience"
    assert rooftop["onset_s"] == 8.0
    assert rooftop["bed_duration_s"] == 3.0
    assert rooftop["cue"].cue_key == "rooftop_ambience"
    with pytest.raises(SfxCueSpecError):
        explicit_cues_from_shots([{"id": "s001", "duration": 8.0, "sfx": [{"tags": ["x"]}]}], "EP001")


def test_prepare_episode_sfx_returns_compose_ready_shapes(tmp_path):
    rows = [_candidate(9, "http://creativecommons.org/publicdomain/zero/1.0/", "https://cdn.freesound.org/p.mp3")]
    fs_client = FreesoundClient("k", opener=_fs_opener(rows, {"p.mp3": voiced_dummy_wav(1.5, freq=320.0)}))
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    shots = [
        {"id": "s001", "duration": 8.0},
        {"id": "s002", "duration": 6.0, "sfx": [{"query": "door creak", "key": "door1", "duration_s": 1.5}]},
    ]
    out = prepare_episode_sfx(
        shots,
        tmp_path,
        "EP001",
        story_id="story-abc-123",
        conn=conn,
        freesound_client=fs_client,
    )
    assert len(out["sfx_clips"]) == 1
    onset, blob = out["sfx_clips"][0]
    assert onset == 8.0
    assert blob[:4] == b"RIFF"
    assert out["ambience_clips"] == []
    assert len(out["sfx_cues"]) == 1
    row = out["sfx_cues"][0]
    assert row["cue_key"] == "door1"
    assert row["bus"] == "sfx"
    assert row["onset_s"] == 8.0
    assert 0 < row["duration_s"] <= 14.0 - 8.0  # clamped inside the episode
    db_row = conn.execute("SELECT status, r2_key FROM sfx_cues WHERE cue_key = 'door1'").fetchone()
    assert db_row["status"] == "resolved"
    assert db_row["r2_key"].endswith(".wav")
    assert "audio/sfx" in db_row["r2_key"]

    # These shapes feed compose_episode_audio + the frame timeline directly.
    from anime_factory.timeline import build_episode_timeline

    timeline = build_episode_timeline("EP001", shots, sfx_cues=out["sfx_cues"])
    assert timeline.sfx[0].cue_key == "door1"


def test_prepare_episode_sfx_required_cue_failure_raises(tmp_path, monkeypatch):
    for key in ("MOSS_SFX_PYTHON", "MOSS_SFX_SCRIPT", "MOSS_SFX_WEIGHTS_DIR"):
        monkeypatch.delenv(key, raising=False)
    fs_client = FreesoundClient("k", opener=_fs_opener([], {}))  # freesound has nothing
    shots = [{"id": "s001", "duration": 8.0, "sfx": [{"query": "impossible sound"}]}]
    with pytest.raises(SfxResolutionError):
        prepare_episode_sfx(shots, tmp_path, "EP001", freesound_client=fs_client, moss_client=MossSoundEffectClient())


def test_prepare_episode_sfx_optional_cue_failure_continues(tmp_path, monkeypatch):
    for key in ("MOSS_SFX_PYTHON", "MOSS_SFX_SCRIPT", "MOSS_SFX_WEIGHTS_DIR"):
        monkeypatch.delenv(key, raising=False)
    fs_client = FreesoundClient("k", opener=_fs_opener([], {}))
    shots = [{"id": "s001", "duration": 8.0, "sfx": [{"query": "nice to have", "required": False}]}]
    out = prepare_episode_sfx(shots, tmp_path, "EP001", freesound_client=fs_client, moss_client=MossSoundEffectClient())
    assert out["sfx_clips"] == []
    assert out["sfx_cues"] == []


def test_prepare_episode_sfx_never_yields_weight_like_r2_keys(tmp_path):
    rows = [_candidate(10, "http://creativecommons.org/publicdomain/zero/1.0/", "https://cdn.freesound.org/p.mp3")]
    fs_client = FreesoundClient("k", opener=_fs_opener(rows, {"p.mp3": voiced_dummy_wav(1.5, freq=320.0)}))
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    shots = [{"id": "s001", "duration": 8.0, "sfx": [{"query": "door creak", "key": "model.safetensors"}]}]
    prepare_episode_sfx(shots, tmp_path, "EP001", story_id="story-abc-123", conn=conn, freesound_client=fs_client)
    for row in conn.execute("SELECT r2_key FROM sfx_cues WHERE r2_key IS NOT NULL").fetchall():
        key = row["r2_key"]
        assert key.endswith(".wav")
        assert "safetensors" not in key.rsplit(".", 1)[-1]
        assert "/audio/sfx/" in key or key.startswith("shared/sfx/")

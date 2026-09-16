"""produce.py integration: gender persistence and cross-episode voice-lock preservation."""

from __future__ import annotations

from pathlib import Path

from anime_factory.db import migrate, open_db
from anime_factory.produce import EpisodeSpec, _lock_voice, apply_script_world
from anime_factory.world import init_fiction_world


def _conn_with_story(tmp_path):
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    init_fiction_world(conn, "story-abc123", "Test Story", [], [], [], [])
    return conn


def _spec() -> EpisodeSpec:
    return EpisodeSpec(title="T", logline="L", langs=("zh", "en", "ja"))


def test_apply_script_world_persists_explicit_gender(tmp_path):
    conn = _conn_with_story(tmp_path)
    script = {
        "cast": [
            {"id": "ke", "name": "阿柯", "identity_prompt": "1boy, hacker", "gender": "male", "age": 24},
        ],
        "locations": [],
        "interiors": [],
    }
    apply_script_world(conn, "story-abc123", script, _spec(), tmp_path)
    row = conn.execute("SELECT gender FROM characters WHERE id = 'ke'").fetchone()
    assert row["gender"] == "male"


def test_apply_script_world_leaves_gender_null_when_unresolvable(tmp_path):
    """No explicit gender, no tag, no marker: ingest must not guess (male or otherwise);
    that is deferred to voice-lock time, which blocks the episode instead."""
    conn = _conn_with_story(tmp_path)
    script = {
        "cast": [{"id": "mystery", "name": "Mystery", "identity_prompt": "a figure", "age": 30}],
        "locations": [],
        "interiors": [],
    }
    apply_script_world(conn, "story-abc123", script, _spec(), tmp_path)
    row = conn.execute("SELECT gender FROM characters WHERE id = 'mystery'").fetchone()
    assert row["gender"] is None


def test_lock_voice_preserves_established_uri_across_calls(tmp_path):
    conn = _conn_with_story(tmp_path)
    _lock_voice(conn, "ke", ("zh", "en"), {"zh": "voice://original/zh", "en": "voice://original/en"}, gender="male")
    row_zh = conn.execute("SELECT voice_uri, lock_version FROM character_voice WHERE character_id='ke' AND lang='zh'").fetchone()
    assert row_zh["voice_uri"] == "voice://original/zh"
    assert row_zh["lock_version"] == 1

    # A later episode recomputes a *different* candidate URI (e.g. identity text
    # was lightly edited) — the established lock must win.
    _lock_voice(conn, "ke", ("zh", "en"), {"zh": "voice://recomputed/zh", "en": "voice://recomputed/en"}, gender="male")
    row_zh2 = conn.execute("SELECT voice_uri, lock_version FROM character_voice WHERE character_id='ke' AND lang='zh'").fetchone()
    assert row_zh2["voice_uri"] == "voice://original/zh"
    assert row_zh2["lock_version"] == 1


def test_lock_voice_migrate_flag_forces_new_uri_and_bumps_version(tmp_path):
    conn = _conn_with_story(tmp_path)
    _lock_voice(conn, "ke", ("zh",), {"zh": "voice://original/zh"}, gender="male")
    _lock_voice(conn, "ke", ("zh",), {"zh": "voice://swapped/zh"}, gender="male", migrate=True)
    row = conn.execute("SELECT voice_uri, lock_version FROM character_voice WHERE character_id='ke' AND lang='zh'").fetchone()
    assert row["voice_uri"] == "voice://swapped/zh"
    assert row["lock_version"] == 2


def test_lock_voice_records_profile_fingerprint(tmp_path):
    conn = _conn_with_story(tmp_path)
    _lock_voice(conn, "ke", ("zh",), {"zh": "voice://original/zh"}, gender="male")
    row = conn.execute("SELECT profile_fingerprint FROM character_voice WHERE character_id='ke' AND lang='zh'").fetchone()
    assert row["profile_fingerprint"]

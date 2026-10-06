"""Finished SkyReels shots upload to the story R2 prefix without an H3 last frame."""

from pathlib import Path

from anime_factory.contracts import ReferenceAsset
from gpu_worker import session


def test_skyreels_clip_seconds_caps_at_five():
    assert session._skyreels_clip_seconds({"duration": 5}) == 5
    assert session._skyreels_clip_seconds({"duration": 8}) == 5
    assert session._skyreels_clip_seconds({"duration": 2.5}) == 2.5
    assert session._skyreels_clip_seconds({}) == 5


def test_submit_skyreels_uses_stable_pack_and_five_second_duration(tmp_path, monkeypatch):
    captured: dict = {}
    ref = ReferenceAsset(
        path="assets/characters/hero/master.png",
        role="character",
        qc_status="pass",
        asset_id="hero",
    )

    def fake_generate(pack, req, **_kwargs):
        captured["duration"] = req.duration
        captured["refs"] = [item.path for item in pack.references]
        dest = Path(req.meta["out_path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * 64)
        return type("Result", (), {"path": str(dest)})()

    monkeypatch.setattr("anime_factory.backends.video_ext.gated_video_generate", fake_generate)
    dest = tmp_path / "shots" / "s001" / "generation-001.mp4"
    session._submit_skyreels(
        {"id": "s001", "duration": 8, "prompt": "walks"},
        tmp_path,
        dest,
        references=[ref],
    )
    assert captured["duration"] == 5
    assert captured["refs"] == ["assets/characters/hero/master.png"]
    assert dest.is_file()


def test_run_anim_skyreels_uploads_finished_shots_without_last_png(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "lock_video_backend", lambda **_k: "skyreels_v3_r2v")
    monkeypatch.setattr(
        session,
        "_board_shots",
        lambda _root: [
            {"id": "s001", "duration": 5, "prompt": "walk"},
            {"id": "s002", "duration": 5, "prompt": "turn"},
        ],
    )
    monkeypatch.setattr(session, "select_passing_generation", lambda *_a, **_k: None)
    ref = ReferenceAsset(
        path="assets/characters/hero/master.png",
        role="character",
        qc_status="pass",
        asset_id="hero",
    )
    monkeypatch.setattr(session, "_stable_skyreels_references", lambda _root: ([ref], "pass"))
    seen: list[list[str]] = []

    def submit(shot, _root, dest, references=None):
        seen.append([item.path for item in (references or [])])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * 5000)
        return dest

    monkeypatch.setattr(session, "_submit_skyreels", submit)
    uploads: list[tuple[str, str]] = []
    monkeypatch.setattr(
        session,
        "put_file",
        lambda key, _path, ctype="": uploads.append((key, ctype)) or {"ok": True, "key": key},
    )
    monkeypatch.setattr(session, "mark_completed_passing", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "record_generation_result", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "_checkpoint_story", lambda *_a, **_k: {"ok": True})
    events: list[str] = []
    from anime_factory.db import open_db, migrate
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    out = session.run_anim("story-1", tmp_path, conn, router=None, progress=events.append)
    assert out["generated"] == ["s001", "s002"]
    assert out["failed"] == []
    assert out["gpu_done"] is True
    assert uploads == [
        ("stories/story-1/shots/s001/generation-001.mp4", "video/mp4"),
        ("stories/story-1/shots/s002/generation-001.mp4", "video/mp4"),
    ]
    assert seen[0] == seen[1] == ["assets/characters/hero/master.png"]
    assert "shot_uploaded:s001" in events
    assert "shot_uploaded:s002" in events

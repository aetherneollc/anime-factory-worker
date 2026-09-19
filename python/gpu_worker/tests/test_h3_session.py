"""H3 session efficiency contracts: selective loads, prep overlap, flush."""

from __future__ import annotations

import inspect
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gpu_worker.h3 import FL2VA_UNET, REF2VA_UNET
from gpu_worker.h3_session import (
    H3PrepPool,
    H3SessionTracker,
    can_prefetch_staging,
    model_load_count_for_modes,
    modes_for_shots,
    plan_h3_mode_groups,
    unet_name_for_mode,
)
from gpu_worker import session


def test_unet_names_and_mode_load_budget():
    assert unet_name_for_mode("ref2va") == REF2VA_UNET
    assert unet_name_for_mode("fl2va_first") == FL2VA_UNET
    shots = [
        {"id": "a", "h3_mode": "ref2va", "character_id": "ke", "refs": ["char_ke_sheet"]},
        {"id": "b", "h3_mode": "fl2va_first", "first_frame_path": "f1.png"},
    ]
    modes = modes_for_shots(shots)
    assert modes == {"ref2va", "fl2va_first"}
    assert model_load_count_for_modes(modes) == 2


def test_mode_groups_merge_consecutive_same_dit():
    shots = [
        {"id": "a", "h3_mode": "ref2va", "character_id": "ke", "refs": ["sheet"]},
        {"id": "b", "h3_mode": "ref2va", "character_id": "ke", "refs": ["sheet"]},
        {"id": "c", "h3_mode": "fl2va_first", "first_frame_path": "f1.png"},
    ]
    groups = plan_h3_mode_groups(shots)
    assert [len(g["indices"]) for g in groups] == [2, 1]


def test_chain_tails_are_not_prefetched():
    assert not can_prefetch_staging({"id": "tail", "chain_index": 1})
    assert can_prefetch_staging({"id": "head", "chain_index": 0})


def test_session_tracker_counts_distinct_unets():
    tracker = H3SessionTracker()
    tracker.note_mode("ref2va")
    tracker.note_mode("fl2va_first_last")
    tracker.note_mode("ref2va")
    assert tracker.model_load_count == 2
    assert set(tracker.loaded_unets) == {REF2VA_UNET, FL2VA_UNET}


def test_prep_pool_overlaps_staging(tmp_path):
    seen: list[str] = []

    def stage(shot: dict) -> dict:
        seen.append(str(shot["id"]))
        return {**shot, "staged": True}

    pool = H3PrepPool(stage)
    first = pool.prime({"id": "s1"})
    assert first["staged"] is True
    pool.schedule({"id": "s2"})
    second = pool.prime({"id": "placeholder"})
    assert second["id"] == "s2"
    assert seen == ["s1", "s2"]
    pool.close()


def test_run_anim_h3_queues_upload_after_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(
        session,
        "_board_shots",
        lambda _root: [{"id": "E01-01", "duration": 8, "h3_mode": "fl2va_first", "first_frame_path": "f1.png"}],
    )
    monkeypatch.setattr(session, "lock_video_backend", lambda **_k: "h3")
    monkeypatch.setattr(session, "existing_generation_file", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "select_passing_generation", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "next_generation_path", lambda *_a, **_k: (1, "episodes/EP001/shots/E01-01/v001.mp4"))
    class _NoopPrepPool:
        def __init__(self, stage_fn):
            self._stage_fn = stage_fn

        def prime(self, shot):
            return self._stage_fn(shot)

        def schedule(self, shot):
            return None

        def close(self):
            return None

    monkeypatch.setattr(session, "H3PrepPool", _NoopPrepPool)
    uploads: list[str] = []

    def submit_gpu(_router, shot, _root, dest, progress=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * 5000)
        shot["last_frame_path"] = str(tmp_path / "last.png")
        (tmp_path / "last.png").write_bytes(b"\x89PNG" + b"x" * 64)
        return {"shot": shot["id"]}

    monkeypatch.setattr(session, "_submit_h3_gpu", submit_gpu)
    monkeypatch.setattr(session, "_normalize_h3_for_qc", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        session,
        "_probe_video",
        lambda _path: {"width": 1280, "height": 720, "duration": 8, "frames": 192, "size_bytes": 5000},
    )
    monkeypatch.setattr(session, "incremental_qc_segment", lambda *_args, **_kwargs: "pass")
    monkeypatch.setattr(session, "mark_completed_passing", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(session, "record_generation_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        session,
        "put_file",
        lambda key, _path, _content_type: uploads.append(key) or {"ok": True, "key": key},
    )
    monkeypatch.setattr(
        session,
        "checkpoint_and_upload",
        lambda _conn, dest: Path(dest),
    )
    monkeypatch.setattr(
        session,
        "_checkpoint_story",
        lambda *_args, **_kwargs: {"ok": True},
    )
    events: list[str] = []
    result = session.run_anim("story-1", tmp_path, object(), router=object(), progress=events.append)
    assert result["generated"] == ["E01-01"]
    assert result["model_load_count"] == 1
    assert result["gpu_done"] is True
    assert "gpu_flushed" not in events
    assert any(key.endswith("v001.mp4") for key in uploads)
    assert "shot_uploaded:E01-01" in events


def test_flush_gpu_artifacts_emits_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(
        session,
        "_checkpoint_story",
        lambda *_args, **_kwargs: {"ok": True},
    )
    events: list[str] = []
    session.flush_gpu_artifacts("story-1", tmp_path, object(), progress=events.append)
    assert events == ["gpu_flushed"]


def test_run_anim_h3_skips_only_passing_generation(tmp_path, monkeypatch):
    sid = "E01-01"
    shot_dir = tmp_path / "shots" / sid
    shot_dir.mkdir(parents=True)
    (shot_dir / "generation-001.mp4").write_bytes(b"x" * 5000)
    (shot_dir / "generation-002.mp4").write_bytes(b"y" * 5000)

    monkeypatch.setattr(
        session,
        "_board_shots",
        lambda _root: [{"id": sid, "duration": 8, "h3_mode": "fl2va_first", "first_frame_path": "f1.png"}],
    )
    monkeypatch.setattr(session, "lock_video_backend", lambda **_k: "h3")
    monkeypatch.setattr(
        session,
        "select_passing_generation",
        lambda _conn, root, segment_id: (root / "shots" / segment_id / "generation-002.mp4", 2, "pass"),
    )

    class _NoopPrepPool:
        def __init__(self, stage_fn):
            self._stage_fn = stage_fn

        def prime(self, shot):
            return shot

        def schedule(self, shot):
            return None

        def close(self):
            return None

    monkeypatch.setattr(session, "H3PrepPool", _NoopPrepPool)
    submits: list[bool] = []

    def _no_submit(*_a, **_k):
        submits.append(True)
        raise AssertionError("should resume passing generation, not resubmit")

    monkeypatch.setattr(session, "_submit_h3_gpu", _no_submit)
    monkeypatch.setattr(session, "put_file", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(session, "_ensure_last_frame", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "mark_completed_passing", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "_checkpoint_story", lambda *_a, **_k: {"ok": True})
    result = session._run_anim_h3("story-1", tmp_path, object(), router=object())
    assert sid in result["skipped_existing"]
    assert not submits


def test_h3_upload_files_does_not_require_db(tmp_path, monkeypatch):
    dest = tmp_path / "shots" / "E01-01" / "generation-001.mp4"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"x" * 5000)
    last = tmp_path / "last.png"
    last.write_bytes(b"\x89PNG" + b"x" * 64)
    monkeypatch.setattr(session, "put_file", lambda *a, **_k: {"ok": True, "key": a[0]})
    out = session._h3_upload_files(
        "story-1",
        tmp_path,
        "E01-01",
        dest,
        dest.relative_to(tmp_path).as_posix(),
        last,
    )
    assert out["sid"] == "E01-01"
    assert out["rel"].endswith("generation-001.mp4")


def test_h3_commit_passing_stays_on_caller_thread(tmp_path, monkeypatch):
    """Background CpuPostQueue may upload; sqlite commit must stay on the anim thread."""
    main_ident = threading.get_ident()
    commit_threads: list[int] = []
    upload_threads: list[int] = []

    monkeypatch.setattr(
        session,
        "_board_shots",
        lambda _root: [{"id": "E01-01", "duration": 8, "h3_mode": "fl2va_first", "first_frame_path": "f1.png"}],
    )
    monkeypatch.setattr(session, "lock_video_backend", lambda **_k: "h3")
    monkeypatch.setattr(session, "existing_generation_file", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "select_passing_generation", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "next_generation_path", lambda *_a, **_k: (1, "episodes/EP001/shots/E01-01/v001.mp4"))

    class _NoopPrepPool:
        def __init__(self, stage_fn):
            self._stage_fn = stage_fn

        def prime(self, shot):
            return self._stage_fn(shot)

        def schedule(self, shot):
            return None

        def close(self):
            return None

    monkeypatch.setattr(session, "H3PrepPool", _NoopPrepPool)

    def submit_gpu(_router, shot, _root, dest, progress=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * 5000)
        shot["last_frame_path"] = str(tmp_path / "last.png")
        (tmp_path / "last.png").write_bytes(b"\x89PNG" + b"x" * 64)
        return {"shot": shot["id"]}

    monkeypatch.setattr(session, "_submit_h3_gpu", submit_gpu)
    monkeypatch.setattr(session, "_normalize_h3_for_qc", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        session,
        "_probe_video",
        lambda _path: {"width": 1280, "height": 720, "duration": 8, "frames": 192, "size_bytes": 5000},
    )
    monkeypatch.setattr(session, "incremental_qc_segment", lambda *_args, **_kwargs: "pass")
    monkeypatch.setattr(
        session,
        "mark_completed_passing",
        lambda *_args, **_kwargs: commit_threads.append(threading.get_ident()),
    )
    monkeypatch.setattr(session, "record_generation_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        session,
        "put_file",
        lambda key, _path, _content_type: upload_threads.append(threading.get_ident()) or {"ok": True, "key": key},
    )
    monkeypatch.setattr(session, "checkpoint_and_upload", lambda _conn, dest: Path(dest))
    monkeypatch.setattr(session, "_checkpoint_story", lambda *_args, **_kwargs: {"ok": True})
    result = session._run_anim_h3("story-1", tmp_path, object(), router=object())
    assert result["generated"] == ["E01-01"]
    assert upload_threads, "upload should run (may be a worker thread)"
    assert commit_threads, "sqlite commit must run after drain"
    assert all(ident == main_ident for ident in commit_threads)


def test_h3_compose_source_does_not_claim_gpu_for_longlive():
    src = inspect.getsource(session.run_compose)
    assert "stop_comfy_for_longlive" not in src
    assert "release_gpu_for_moss_sfx" in src
    assert "restore_gpu_after_moss_sfx" in src

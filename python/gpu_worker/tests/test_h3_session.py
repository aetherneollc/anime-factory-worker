"""H3 session efficiency contracts: selective loads, prep overlap, flush."""

from __future__ import annotations

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
        lambda _path: {"width": 864, "height": 480, "duration": 8, "frames": 192, "size_bytes": 5000},
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
    events: list[str] = []
    result = session.run_anim("story-1", tmp_path, object(), router=object(), progress=events.append)
    assert result["generated"] == ["E01-01"]
    assert result["model_load_count"] == 1
    assert result["gpu_done"] is True
    assert "gpu_flushed" not in events
    assert any(key.endswith("v001.mp4") for key in uploads)
    assert "shot_uploaded:E01-01" in events
    assert "r2_uploaded:story.sqlite" in events


def test_flush_gpu_artifacts_emits_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(
        session,
        "_checkpoint_story",
        lambda *_args, **_kwargs: {"ok": True},
    )
    events: list[str] = []
    session.flush_gpu_artifacts("story-1", tmp_path, object(), progress=events.append)
    assert events == ["gpu_flushed"]

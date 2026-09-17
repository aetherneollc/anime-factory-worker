"""Regression tests for backend-aware deterministic + LongLive chain QC."""

from __future__ import annotations

from pathlib import Path

import pytest

from anime_factory.models import VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH
from anime_factory.qc import (
    LONGLIVE_NATIVE_HEIGHT,
    LONGLIVE_NATIVE_WIDTH,
    chain_qc,
    continuity_qc,
    deterministic_check,
    h3_grid_frames,
    h3_grid_step_seconds,
    identity_qc,
    incremental_qc_segment,
    longlive_duration_step_seconds,
    longlive_video_frames,
)
from gpu_worker import session


def _good_meta(*, width: int, height: int, frames: int, duration: float) -> dict:
    return {
        "width": width,
        "height": height,
        "frames": frames,
        "duration": duration,
        "size_bytes": 50_000,
        "black_frame_ratio": 0.0,
    }


def test_h3_deterministic_semantics_unchanged():
    duration = 8.0
    frames = h3_grid_frames(duration)
    meta = _good_meta(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, frames=frames, duration=duration)
    verdict, details = deterministic_check(meta, duration, used_mode="ref2va")
    assert verdict == "pass"
    assert details["issues"] == []

    meta_bad_res = {**meta, "width": LONGLIVE_NATIVE_WIDTH, "height": LONGLIVE_NATIVE_HEIGHT}
    verdict, details = deterministic_check(meta_bad_res, duration, used_mode="ref2va")
    assert verdict == "fail"
    assert "resolution" in details["issues"]

    meta_bad_frames = {**meta, "frames": frames + 20}
    verdict, details = deterministic_check(meta_bad_frames, duration, used_mode="fl2va_first")
    assert verdict == "fail"
    assert "frames" in details["issues"]

    tol = h3_grid_step_seconds() + 1e-6
    meta_bad_dur = {**meta, "duration": duration + tol + 0.01}
    verdict, details = deterministic_check(meta_bad_dur, duration, used_mode=None)
    assert verdict == "fail"
    assert "duration" in details["issues"]


def test_longlive_15s_native_passes():
    duration = 15.0
    frames = longlive_video_frames(duration)
    assert frames == 381
    actual_duration = frames / float(VIDEO_FPS)
    assert abs(actual_duration - 15.875) < 0.001

    meta = _good_meta(
        width=LONGLIVE_NATIVE_WIDTH,
        height=LONGLIVE_NATIVE_HEIGHT,
        frames=frames,
        duration=actual_duration,
    )
    verdict, details = deterministic_check(meta, duration, used_mode="i2v")
    assert verdict == "pass"
    assert details["issues"] == []

    verdict, details = deterministic_check(meta, duration, used_mode="t2v")
    assert verdict == "pass"


def test_longlive_wrong_resolution_or_frames_fails():
    duration = 15.0
    frames = longlive_video_frames(duration)
    good = _good_meta(
        width=LONGLIVE_NATIVE_WIDTH,
        height=LONGLIVE_NATIVE_HEIGHT,
        frames=frames,
        duration=15.875,
    )

    verdict, details = deterministic_check(
        {**good, "width": VIDEO_WIDTH, "height": VIDEO_HEIGHT},
        duration,
        used_mode="i2v",
    )
    assert verdict == "fail"
    assert "resolution" in details["issues"]

    verdict, details = deterministic_check({**good, "frames": frames + 40}, duration, used_mode="t2v")
    assert verdict == "fail"
    assert "frames" in details["issues"]

    tol = longlive_duration_step_seconds() + 1e-6
    verdict, details = deterministic_check({**good, "duration": duration + tol + 0.01}, duration, used_mode="i2v")
    assert verdict == "fail"
    assert "duration" in details["issues"]


def test_longlive_chain_tail_i2v_with_refs_passes(tmp_path: Path):
    last = tmp_path / "prev_last.png"
    last.write_bytes(b"\x89PNG" + b"x" * 64)
    chain_first = tmp_path / "next" / "chain_first.png"
    chain_first.parent.mkdir()
    chain_first.write_bytes(last.read_bytes())
    prev = {
        "chain_id": "scene-01",
        "last_frame_path": str(last),
    }
    segment = {
        "chain_id": "scene-01",
        "chain_index": 1,
        "character_id": "hero",
        "refs": ["char_hero_sheet.png"],
        "first_frame_path": str(chain_first),
    }

    id_verdict, id_details = identity_qc(segment, used_mode="i2v")
    assert id_verdict == "pass", id_details
    ch_verdict, ch_details = chain_qc(prev, segment)
    assert ch_verdict == "pass", ch_details
    cont_verdict, cont_details = continuity_qc(prev, segment, used_mode="i2v")
    assert cont_verdict == "pass", cont_details


def test_h3_chain_tail_still_requires_ref2va():
    segment = {
        "chain_id": "c1",
        "chain_index": 1,
        "character_id": "hero",
        "refs": ["char_hero_sheet.png"],
        "first_frame_path": "chain_first.png",
    }
    verdict, details = identity_qc(segment, used_mode="fl2va_first")
    assert verdict == "retry"
    assert "chain_dropped_refs" in details["issues"]

    verdict, details = identity_qc(segment, used_mode="ref2va")
    assert verdict == "pass"


def test_run_anim_longlive_passes_prev_segment_to_qc(tmp_path: Path, monkeypatch):
    frame = tmp_path / "f1.png"
    frame.write_bytes(b"\x89PNG" + b"x" * 64)
    shots = [
        {
            "id": "take-01",
            "take_id": "take-01",
            "duration": 8,
            "chain_id": "scene-a",
            "chain_index": 0,
            "scene_id": "scene-a",
            "first_frame_path": str(frame),
            "h3_prompt": "walk",
            "keyframe_source": "qc_keyframe",
        },
        {
            "id": "take-02",
            "take_id": "take-02",
            "duration": 8,
            "chain_id": "scene-a",
            "chain_index": 1,
            "scene_id": "scene-a",
            "character_id": "hero",
            "refs": ["char_hero.png"],
            "h3_prompt": "continue",
        },
    ]
    monkeypatch.setattr(session, "_board_shots", lambda _root: shots)
    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")

    def load(self, *, takes, workdir):
        if self.pipeline is None:
            self.pipeline = object()
            self.mode = "i2v"
            self.batch_dir = Path(workdir)
            self.model_load_count += 1
        return self.pipeline

    monkeypatch.setattr("gpu_worker.longlive_batch.LongLiveBatchRunner.load_model", load)
    monkeypatch.setattr(session, "stop_comfy_for_longlive", lambda: {"ok": True})
    monkeypatch.setattr(session, "_keep_native_strip_audio", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "mark_completed_passing", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "record_generation_result", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "checkpoint_and_upload", lambda _conn, dest: Path(dest))
    monkeypatch.setattr(session, "put_file", lambda *_a, **_k: {"ok": True})

    def fake_extract(_video, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x89PNG" + b"x" * 64)
        return dest

    monkeypatch.setattr(session, "extract_last_frame", fake_extract)

    qc_calls: list[dict] = []

    def capture_qc(conn, episode_code, segment_id, meta, duration, **kwargs):
        qc_calls.append(
            {
                "segment_id": segment_id,
                "prev": kwargs.get("prev_segment"),
                "used_mode": kwargs.get("used_mode"),
                "segment": kwargs.get("segment"),
            }
        )
        return "pass"

    monkeypatch.setattr(session, "incremental_qc_segment", capture_qc)
    monkeypatch.setattr(
        session,
        "_probe_video",
        lambda _path: {
            "width": LONGLIVE_NATIVE_WIDTH,
            "height": LONGLIVE_NATIVE_HEIGHT,
            "duration": 8,
            "frames": longlive_video_frames(8.0),
            "size_bytes": 80,
            "audio_streams": 0,
        },
    )

    def infer(**kwargs):
        dest = Path(kwargs["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 64 + b"ftyp")
        return dest

    monkeypatch.setattr(session, "_longlive_infer_hook", infer)

    out = session.run_anim("story-chain", tmp_path, object(), router=None)
    assert out["failed"] == []
    assert {call["segment_id"] for call in qc_calls} == {"take-01", "take-02"}

    first_call = next(c for c in qc_calls if c["segment_id"] == "take-01")
    second_call = next(c for c in qc_calls if c["segment_id"] == "take-02")
    assert first_call["prev"] is None
    assert second_call["prev"] is not None
    prev_last = second_call["prev"].get("last_frame_path") or ""
    assert prev_last.endswith("/take-01/last.png")
    assert second_call["used_mode"] == "i2v"
    chain_first = second_call["segment"].get("first_frame_path") or ""
    assert chain_first.endswith("/take-02/chain_first.png")
    assert Path(chain_first).read_bytes() == Path(prev_last).read_bytes()


def test_incremental_qc_longlive_deterministic_via_db(tmp_path: Path):
    from anime_factory.db import migrate, open_db

    conn = open_db(tmp_path / "qc.sqlite")
    migrate(conn)
    duration = 15.0
    frames = longlive_video_frames(duration)
    meta = _good_meta(
        width=LONGLIVE_NATIVE_WIDTH,
        height=LONGLIVE_NATIVE_HEIGHT,
        frames=frames,
        duration=frames / float(VIDEO_FPS),
    )
    verdict = incremental_qc_segment(
        conn,
        "EP001",
        "take-15",
        meta,
        duration,
        used_mode="i2v",
    )
    assert verdict == "pass"
    row = conn.execute(
        "SELECT verdict, details_json FROM qc_reports WHERE segment_id = ? AND gate = 'deterministic'",
        ("take-15",),
    ).fetchone()
    assert row["verdict"] == "pass"

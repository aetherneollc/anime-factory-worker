"""LongLive 2.0 image/backend contract, packing, one-load batch, and teardown."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anime_factory.compose import SCALE_PAD_FILTER
from anime_factory.models import VIDEO_HEIGHT, VIDEO_WIDTH
from anime_factory.longlive_workflow import (
    LongLiveWorkflowError,
    load_longlive_short_workflow,
    pack_short_takes,
    validate_longlive_short_workflow,
    validate_packed_takes,
)
from anime_factory.video_backend import (
    CAPABILITY_MISMATCH,
    VideoBackendLockError,
    lock_video_backend,
)
from gpu_worker.boot import assert_lease_capabilities
from gpu_worker.images import image_capabilities, image_can_lease
from gpu_worker.longlive import FAIL_CLOSED, LONGLIVE_NATIVE_HEIGHT, LONGLIVE_NATIVE_WIDTH, LONGLIVE_REF
from gpu_worker.longlive_batch import LongLiveBatchRunner, submit_longlive_batch
from gpu_worker.session import BatchBudget, CpuPostQueue, LeaseRuntime
from gpu_worker.weights import H3_FILES, ensure_h3_weights, h3_files_for_modes


def test_image_backend_capability_mismatch_fails_closed(monkeypatch):
    monkeypatch.setenv("AF_IMAGE_CAPABILITY", "h3")
    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    with pytest.raises(VideoBackendLockError, match=CAPABILITY_MISMATCH):
        lock_video_backend()


def test_longlive_image_refuses_h3_backend(monkeypatch):
    monkeypatch.setenv("AF_IMAGE_CAPABILITY", "longlive")
    monkeypatch.setenv("AF_VIDEO_BACKEND", "h3")
    with pytest.raises(VideoBackendLockError, match=CAPABILITY_MISMATCH):
        lock_video_backend()


def test_longlive_image_can_lease_with_h3_false():
    caps = image_capabilities("docker.io/aetherneo/anime-factory-gpu-longlive")
    assert caps["longlive"] is True
    assert caps["h3"] is False
    assert caps["comfy"] is True
    assert image_can_lease(caps) is True
    assert_lease_capabilities(caps)


def test_zero_h3_files_on_longlive_path(tmp_path, monkeypatch):
    monkeypatch.setenv("AF_IMAGE_CAPABILITY", "longlive")
    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    lock_video_backend()
    pulled = []
    monkeypatch.setattr("gpu_worker.weights._hf_file", lambda *a, **k: pulled.append(a))
    with pytest.raises(RuntimeError, match="refuse H3"):
        ensure_h3_weights(tmp_path)
    assert pulled == []
    assert not any("minimax_h3" in p.name for p in tmp_path.rglob("*"))
    assert not any("audio_vae" in item["dest"] for item in H3_FILES)


def test_h3_dits_are_lazy_by_board_mode():
    core = h3_files_for_modes(set())
    assert any("video_vae" in item["dest"] for item in core)
    assert not any("fl2va" in item["dest"] for item in core)
    assert not any("ref2va" in item["dest"] for item in core)
    fl = h3_files_for_modes({"fl2va_first"})
    assert any("fl2va" in item["dest"] for item in fl)
    assert not any("ref2va" in item["dest"] for item in fl)
    ref = h3_files_for_modes({"ref2va"})
    assert any("ref2va" in item["dest"] for item in ref)


def test_longlive_short_workflow_schema_and_silent_native():
    data = load_longlive_short_workflow()
    validate_longlive_short_workflow(data)
    assert data["silent"] is True
    assert data["audio"] is False
    assert data["native_width"] == LONGLIVE_NATIVE_WIDTH == 1280
    assert data["native_height"] == LONGLIVE_NATIVE_HEIGHT == 704
    assert data["compose"]["scale_pad_only"] is True
    assert data["compose"]["delivery_width"] == VIDEO_WIDTH
    assert data["compose"]["delivery_height"] == VIDEO_HEIGHT
    assert "force_original_aspect_ratio=decrease" in SCALE_PAD_FILTER
    assert f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}" in SCALE_PAD_FILTER


def test_longlive_pack_scene_boundaries_two_to_three_takes():
    shots = [
        {"id": "s1", "duration": 40, "scene_id": "cafe", "character_id": "hero", "h3_prompt": "enter"},
        {"id": "s2", "duration": 40, "scene_id": "cafe", "character_id": "hero", "h3_prompt": "talk"},
        {"id": "s3", "duration": 40, "scene_id": "street", "character_id": "hero", "h3_prompt": "leave"},
    ]
    takes = pack_short_takes(shots)
    assert 2 <= len(takes) <= 3
    assert all(float(t["duration"]) <= 60 for t in takes)
    assert takes[0]["keyframe_source"] == "qc_keyframe"
    cafe = [t for t in takes if t.get("scene_id") == "cafe"]
    if len(cafe) > 1:
        assert cafe[1]["keyframe_source"] == "prior_last_frame"
        assert cafe[1]["continue_from"] == cafe[0]["take_id"]
    street = [t for t in takes if t.get("scene_id") == "street"]
    assert street
    assert street[0]["keyframe_source"] == "new_keyframe_hard_cut"
    for take in takes:
        assert take["take_id"]
        assert take["end_frame"] > take["start_frame"]
        assert "cast" in take
        assert "prompt" in take


def test_scene_change_must_hard_cut():
    with pytest.raises(LongLiveWorkflowError, match="new_keyframe_hard_cut"):
        validate_packed_takes(
            [
                {
                    "take_id": "take-01",
                    "duration": 40,
                    "scene_id": "a",
                    "keyframe_source": "qc_keyframe",
                },
                {
                    "take_id": "take-02",
                    "duration": 40,
                    "scene_id": "b",
                    "keyframe_source": "prior_last_frame",
                },
            ]
        )


def test_one_model_load_maps_outputs_by_take_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        LongLiveBatchRunner,
        "load_model",
        lambda self: setattr(self, "pipeline", object())
        or setattr(self, "model_load_count", 1 if self.model_load_count == 0 else self.model_load_count)
        or self.pipeline,
    )
    loads = {"n": 0}

    def load(self):
        if self.pipeline is not None:
            return self.pipeline
        loads["n"] += 1
        self.pipeline = object()
        self.model_load_count += 1
        return self.pipeline

    monkeypatch.setattr(LongLiveBatchRunner, "load_model", load)

    def infer(*, take, dest, **_):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 64 + b"ftyp")
        return dest

    takes = [
        {"take_id": "take-01", "duration": 8, "h3_prompt": "one", "first_frame_path": str(tmp_path / "a.png")},
        {"take_id": "take-02", "duration": 8, "h3_prompt": "two", "first_frame_path": str(tmp_path / "b.png")},
    ]
    (tmp_path / "a.png").write_bytes(b"\x89PNG" + b"x" * 64)
    (tmp_path / "b.png").write_bytes(b"\x89PNG" + b"x" * 64)
    dests = {
        "take-01": tmp_path / "take-01.mp4",
        "take-02": tmp_path / "take-02.mp4",
    }
    out = submit_longlive_batch(takes, dests, infer=infer)
    assert out["model_load_count"] == 1
    assert loads["n"] == 1
    assert out["h3_downloads"] == 0
    assert out["native_audio"] is False
    assert out["outputs"]["take-01"].endswith("take-01.mp4")
    assert out["outputs"]["take-02"].endswith("take-02.mp4")
    marker = json.loads((tmp_path / "longlive_batch.json").read_text(encoding="utf-8"))
    assert marker["model_load_count"] == 1
    assert marker["h3_downloads"] == 0


def test_batch_fail_closed_without_placeholder(tmp_path, monkeypatch):
    monkeypatch.setattr(
        LongLiveBatchRunner,
        "load_model",
        lambda self: setattr(self, "pipeline", object())
        or setattr(self, "model_load_count", (self.model_load_count or 0) + (0 if self.pipeline else 0)),
    )

    def load(self):
        if self.pipeline is None:
            self.pipeline = object()
            self.model_load_count += 1
        return self.pipeline

    monkeypatch.setattr(LongLiveBatchRunner, "load_model", load)

    def infer(**_):
        return None

    dest = tmp_path / "missing.mp4"
    with pytest.raises(RuntimeError, match=FAIL_CLOSED):
        submit_longlive_batch(
            [{"take_id": "take-01", "duration": 8, "h3_prompt": "x"}],
            {"take-01": dest},
            infer=infer,
        )
    assert not dest.exists()


def test_run_anim_longlive_one_load_and_native_dims(tmp_path, monkeypatch):
    from gpu_worker import session

    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    monkeypatch.setenv("AF_IMAGE_CAPABILITY", "longlive")
    frame = tmp_path / "f1.png"
    frame.write_bytes(b"\x89PNG" + b"x" * 64)
    shots = [
        {
            "id": "take-01",
            "take_id": "take-01",
            "duration": 8,
            "scene_id": "cafe",
            "first_frame_path": str(frame),
            "h3_prompt": "walk",
            "keyframe_source": "qc_keyframe",
        },
        {
            "id": "take-02",
            "take_id": "take-02",
            "duration": 8,
            "scene_id": "street",
            "first_frame_path": str(frame),
            "h3_prompt": "cut",
            "keyframe_source": "new_keyframe_hard_cut",
        },
    ]
    monkeypatch.setattr(session, "_board_shots", lambda _root: shots)

    def load(self):
        if self.pipeline is None:
            self.pipeline = object()
            self.model_load_count += 1
        return self.pipeline

    monkeypatch.setattr("gpu_worker.longlive_batch.LongLiveBatchRunner.load_model", load)
    monkeypatch.setattr(session, "stop_comfy_for_longlive", lambda: {"ok": True})

    loads = {"n": 0}

    def infer(**kwargs):
        loads["n"] += 1
        dest = Path(kwargs["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 64 + b"ftyp")
        return dest

    monkeypatch.setattr(session, "_longlive_infer_hook", infer)
    monkeypatch.setattr(session, "_keep_native_strip_audio", lambda *_a, **_k: None)
    monkeypatch.setattr(
        session,
        "_probe_video",
        lambda _path: {
            "width": 1280,
            "height": 704,
            "duration": 8,
            "frames": 192,
            "size_bytes": 80,
            "audio_streams": 0,
        },
    )
    monkeypatch.setattr(session, "incremental_qc_segment", lambda *_a, **_k: "pass")
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
    scaled = []
    monkeypatch.setattr(
        session,
        "_normalize_h3_for_qc",
        lambda *a, **k: scaled.append(a) or (_ for _ in ()).throw(AssertionError("must not scale native")),
    )

    out = session.run_anim("story-ll", tmp_path, object(), router=None)
    assert out["failed"] == []
    assert set(out["generated"]) == {"take-01", "take-02"}
    assert out["model_load_count"] == 1
    assert out["native_audio"] is False
    assert out["resolution"]["width"] == 1280
    assert out["resolution"]["height"] == 704
    assert out["resolution"]["scale_pad_only"] is True
    assert loads["n"] == 2
    assert scaled == []
    assert out["mapped"]["take-01"]
    assert out["mapped"]["take-02"]


def test_post_checkpoint_idle_uses_seconds_and_queue(monkeypatch):
    monkeypatch.delenv("VAST_IDLE_SECONDS", raising=False)
    now = [0.0]
    runtime = LeaseRuntime(
        instance_id="1",
        started_at=0.0,
        idle_since=0.0,
        clock=lambda: now[0],
    )
    runtime.budget = BatchBudget(idle_minutes=15, post_checkpoint_idle_seconds=30)
    runtime.phase = "production"
    runtime.mark_idle()
    runtime.record_checkpoint()
    now[0] = 29.0
    assert not runtime.idle_expired(comfy_inflight=0, pending_jobs=0)
    now[0] = 30.0
    assert runtime.idle_expired(comfy_inflight=0, pending_jobs=0)
    assert not runtime.idle_expired(comfy_inflight=0, pending_jobs=1)
    assert not runtime.idle_expired(comfy_inflight=1, pending_jobs=0)


def test_cpu_post_queue_drains_after_gpu():
    order = []
    q = CpuPostQueue()
    q.submit(lambda: order.append("cpu") or "ok")
    order.append("gpu")
    assert q.drain() == ["ok"]
    assert order == ["gpu", "cpu"]


def test_clone_pins_nvlabs_commit():
    assert LONGLIVE_REF == "6b36d20ec6f7958d29d11a704dfa64611a9f2572"
    src = (Path(__file__).resolve().parents[1] / "python" / "gpu_worker" / "longlive.py").read_text(
        encoding="utf-8"
    )
    clone = src.split("def _clone_longlive_repo")[1].split("\ndef ")[0]
    assert "LONGLIVE_REF" in clone
    assert "--branch" not in clone
    assert "fetch" in clone
    assert "--depth" in clone


def test_dockerfile_longlive_contract():
    root = Path(__file__).resolve().parents[1]
    text = (root / "deploy" / "gpu-worker" / "Dockerfile.longlive").read_text(encoding="utf-8")
    assert "12.8.1-devel-ubuntu24.04" in text
    assert "12.8.1-runtime-ubuntu24.04" in text
    assert text.count("FROM nvidia/cuda") == 2
    assert "torch==2.7.0+cu128" in text
    assert "6b36d20ec6f7958d29d11a704dfa64611a9f2572" in text
    assert "AF_IMAGE_CAPABILITY=longlive" in text
    assert 'org.aetherneo.anime-factory.h3="false"' in text
    assert 'org.aetherneo.anime-factory.longlive="true"' in text
    assert "fouroversix" in text
    assert "flash_attn" in text
    assert "setup.py build_ext --inplace" in text
    assert "minimax_h3" not in text.lower()
    assert "huggingface-cli download" not in text
    assert "model_4o6.pt" in text  # path env, not a baked COPY
    assert "COPY --from=nvfp4" in text
    targets = json.loads((root / "deploy" / "docker-targets.json").read_text(encoding="utf-8"))
    ll = next(t for t in targets["targets"] if t["id"] == "longlive")
    assert ll["enabled"] is True
    assert "h3" not in ll["capabilities"]
    assert "longlive" in ll["capabilities"]
    h3 = next(t for t in targets["targets"] if t["id"] == "h3")
    assert "longlive" not in h3["capabilities"]


def test_incremental_r2_skips_existing(tmp_path, monkeypatch):
    from anime_factory import r2_client

    dest = tmp_path / "board.json"
    dest.write_bytes(b"abc")
    downloaded = []

    monkeypatch.setattr(
        r2_client,
        "list_prefix",
        lambda prefix, max_keys=2000: [
            {"key": "stories/s/board.json", "size": 3},
            {"key": "stories/s/episodes/EP002/board.json", "size": 4},
        ],
    )
    monkeypatch.setattr(
        r2_client,
        "download_file",
        lambda key, dest_path: downloaded.append(key) or dest_path.write_bytes(b"new") or True,
    )
    written = r2_client.download_prefix(
        "stories/s/",
        tmp_path,
        strip_prefix="stories/s/",
        skip_existing=True,
        key_filter=lambda key: "EP002" not in key,
    )
    assert "board.json" in written
    assert downloaded == []

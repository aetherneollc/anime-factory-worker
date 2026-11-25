"""fal video adapters, chain concurrency, and the CPU worker registration loop."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from anime_factory.db import migrate, open_db
from anime_factory.design import synthetic_still_png
from anime_factory.fal_models import build_i2v_payload, build_r2v_payload
from anime_factory.video_backend import lock_video_backend
from gpu_worker import session
from gpu_worker.__main__ import fal_worker_instance_id, maybe_notify_control_plane, suppress_fal_self_destroy
from gpu_worker.images import is_fal_worker_mode


def test_kling_r2v_maps_elements_images_and_markers():
    chars = [f"https://cdn.example/char{i}.png" for i in range(8)]
    scenes = ["https://cdn.example/plate.png", "https://cdn.example/start.png"]
    payload = build_r2v_payload(
        model="fal-ai/kling-video/o1/reference-to-video",
        prompt="<<<image_1>>> walks through <<<image_8>>>",
        char_urls=chars,
        scene_urls=scenes,
        duration_s=6.2,
    )
    assert len(payload["elements"]) == 7
    assert payload["elements"][0] == {
        "frontal_image_url": "https://cdn.example/char0.png",
        "reference_image_urls": ["https://cdn.example/char0.png"],
    }
    assert all(element["reference_image_urls"] for element in payload["elements"])
    assert "image_urls" not in payload
    assert "@Element1" in payload["prompt"]
    assert "<<<image_8>>>" in payload["prompt"]
    payload = build_r2v_payload(
        model="fal-ai/kling-video/o1/reference-to-video",
        prompt="<<<image_1>>> enters <<<image_2>>>",
        char_urls=["https://cdn.example/hero.png"],
        scene_urls=["https://cdn.example/plate.png"],
        duration_s=2.2,
    )
    assert payload["prompt"] == "@Element1 enters @Image1"
    assert payload["elements"][0]["reference_image_urls"] == ["https://cdn.example/hero.png"]
    assert payload["image_urls"] == ["https://cdn.example/plate.png"]
    assert payload["aspect_ratio"] == "16:9"
    assert payload["duration"] == "3"
    long = build_r2v_payload(
        model="fal-ai/kling-video/v3/elements",
        prompt="hold",
        char_urls=["https://cdn.example/hero.png"],
        scene_urls=[],
        duration_s=12,
    )
    assert long["duration"] == "10"
    assert long["elements"][0]["reference_image_urls"]
    assert "image_urls" not in long
    angles = build_r2v_payload(
        model="fal-ai/kling-video/o1/reference-to-video",
        prompt="turn",
        char_urls=[
            ("https://cdn.example/front.png", ["https://cdn.example/side.png"]),
            {
                "frontal_image_url": "https://cdn.example/bob.png",
                "reference_image_urls": ["https://cdn.example/bob-side.png"],
            },
        ],
        scene_urls=[],
    )
    assert angles["elements"][0] == {
        "frontal_image_url": "https://cdn.example/front.png",
        "reference_image_urls": ["https://cdn.example/side.png"],
    }
    assert angles["elements"][1]["reference_image_urls"] == ["https://cdn.example/bob-side.png"]
    assert "image_urls" not in angles
    plates = build_r2v_payload(
        model="fal-ai/kling-video/o1/reference-to-video",
        prompt="empty street",
        char_urls=[],
        scene_urls=["https://cdn.example/plate.png"],
    )
    assert "elements" not in plates
    assert plates["image_urls"] == ["https://cdn.example/plate.png"]
    generic = build_r2v_payload(
        model="fal-ai/wan/reference-to-video",
        prompt="<<<image_1>>> waves",
        char_urls=["https://cdn.example/hero.png"],
        scene_urls=["https://cdn.example/plate.png"],
        duration_s=5,
    )
    assert "elements" not in generic
    assert generic["image_urls"] == ["https://cdn.example/hero.png", "https://cdn.example/plate.png"]
    assert "<<<image_1>>>" in generic["prompt"]


def test_fl2v_and_i2v_adapters():
    hailuo = build_i2v_payload(
        model="fal-ai/minimax/hailuo-02/standard/image-to-video",
        prompt="pan right",
        image_url="https://cdn.example/first.png",
        end_image_url="https://cdn.example/last.png",
        duration_s=6.0,
    )
    assert hailuo["image_url"] == "https://cdn.example/first.png"
    assert hailuo["end_image_url"] == "https://cdn.example/last.png"
    assert hailuo["duration"] == "6"
    assert hailuo["resolution"] == "768P"
    assert hailuo["prompt_optimizer"] is False
    longer = build_i2v_payload(
        model="fal-ai/minimax/hailuo-02/standard/image-to-video",
        prompt="pan",
        image_url="https://cdn.example/first.png",
        duration_s=8,
    )
    assert longer["duration"] == "10"
    assert "end_image_url" not in longer
    kling = build_i2v_payload(
        model="fal-ai/kling-video/v3/pro/image-to-video",
        prompt="push in",
        image_url="https://cdn.example/first.png",
        end_image_url="https://cdn.example/last.png",
        duration_s=6.4,
        negative_prompt="text",
    )
    assert kling["start_image_url"] == "https://cdn.example/first.png"
    assert kling["end_image_url"] == "https://cdn.example/last.png"
    assert kling["duration"] == "6"
    assert kling["generate_audio"] is False
    assert kling["negative_prompt"] == "text"
    generic = build_i2v_payload(
        model="fal-ai/wan/v2.2/image-to-video",
        prompt="drift",
        image_url="https://cdn.example/first.png",
        end_image_url="https://cdn.example/last.png",
        duration_s=4,
    )
    assert generic["image_url"] == "https://cdn.example/first.png"
    assert generic["end_image_url"] == "https://cdn.example/last.png"
    assert "start_image_url" not in generic


def test_vast_h3_image_still_locks_to_h3(monkeypatch, tmp_path):
    monkeypatch.setenv("AF_IMAGE_CAPABILITY", "h3")
    monkeypatch.delenv("VIDEO_BACKEND", raising=False)
    monkeypatch.delenv("AF_VIDEO_BACKEND", raising=False)
    assert lock_video_backend(root=tmp_path) == "h3"


def test_fal_worker_mode_detection(monkeypatch):
    monkeypatch.setenv("AF_WORKER_MODE", "fal")
    monkeypatch.setenv("AF_START_COMFY", "1")
    assert is_fal_worker_mode() is True
    monkeypatch.delenv("AF_WORKER_MODE", raising=False)
    monkeypatch.setenv("VIDEO_BACKEND", "fal")
    monkeypatch.delenv("AF_START_COMFY", raising=False)
    assert is_fal_worker_mode() is True
    monkeypatch.setenv("AF_START_COMFY", "1")
    assert is_fal_worker_mode() is False
    monkeypatch.setenv("VIDEO_BACKEND", "h3")
    monkeypatch.setenv("AF_START_COMFY", "0")
    assert is_fal_worker_mode() is False


def test_fal_cpu_register_and_no_destroy(monkeypatch):
    monkeypatch.setenv("AF_WORKER_MODE", "fal")
    monkeypatch.setenv("AF_WORKER_ID", "box-7")
    monkeypatch.setenv("CONTROL_PLANE_URL", "https://control.example")
    monkeypatch.setenv("AF_START_COMFY", "0")
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    def open_control(req, timeout):
        seen.append((req.full_url, req.data))
        return Response()

    monkeypatch.setattr("gpu_worker.__main__.urllib.request.urlopen", open_control)
    events = maybe_notify_control_plane("ignored-vast-id", "idle", register=True)
    assert events == ["register", "heartbeat"]
    body = json.loads(seen[0][1].decode("utf-8"))
    assert body["vast_instance_id"] == "box-7"
    assert body["gpu_type"] == "CPU (fal.ai)"
    assert body["profile_id"] == "fal-api-cpu"
    caps = body["capabilities"]
    assert caps["fal"] is True
    assert caps["provider"] == "fal"
    assert caps["comfy"] is False
    assert caps["h3"] is False
    assert caps["image_gen"] is True
    assert caps["video_backend"] == "fal"
    assert caps["agent"] is True
    assert caps["nvenc"] is False
    assert caps["profile_id"] == "fal-api-cpu"
    heartbeat = json.loads(seen[1][1].decode("utf-8"))
    assert heartbeat["vast_instance_id"] == "box-7"

    monkeypatch.delenv("AF_WORKER_ID", raising=False)
    assert fal_worker_instance_id().startswith("fal-cpu-")

    pending, code = suppress_fal_self_destroy({"reason": "idle_ttl", "exit_code": 0}, once=False)
    assert pending is None and code is None
    pending, code = suppress_fal_self_destroy({"reason": "success", "exit_code": 0}, once=True)
    assert pending is None and code == 0
    monkeypatch.delenv("AF_WORKER_MODE", raising=False)
    monkeypatch.delenv("VIDEO_BACKEND", raising=False)
    monkeypatch.delenv("AF_VIDEO_BACKEND", raising=False)
    monkeypatch.setenv("AF_START_COMFY", "1")
    pending, code = suppress_fal_self_destroy({"reason": "idle_ttl", "exit_code": 0}, once=False)
    assert pending["reason"] == "idle_ttl"
    assert code is None


def _board(root: Path) -> None:
    ep = root / "episodes" / "EP001"
    ep.mkdir(parents=True)
    (ep / "board.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "a1",
                        "chain_id": "A",
                        "chain_index": 0,
                        "duration": 6,
                        "h3_mode": "fl2va_first",
                        "h3_prompt": "alpha-walk",
                    },
                    {
                        "id": "a2",
                        "chain_id": "A",
                        "chain_index": 1,
                        "duration": 6,
                        "h3_mode": "fl2va_first",
                        "h3_prompt": "alpha-continue",
                    },
                    {
                        "id": "b1",
                        "chain_id": "B",
                        "chain_index": 0,
                        "duration": 6,
                        "h3_mode": "fl2va_first",
                        "h3_prompt": "bravo-look",
                    },
                    {
                        "id": "s1",
                        "duration": 6,
                        "h3_mode": "fl2va_first",
                        "h3_prompt": "solo-hold",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_fal_chains_stay_ordered_while_other_chains_overlap(monkeypatch, tmp_path):
    monkeypatch.setenv("FAL_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("FAL_DRY_RUN", "1")
    _board(tmp_path)
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    records: list[dict] = []
    lock = threading.Lock()

    def fake_extract(video, dest):
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(synthetic_still_png(64, 64, tag=str(video), placeholder=False))
        return path

    def fake_generate(mode, prompt, **kwargs):
        start = time.monotonic()
        time.sleep(0.2)
        end = time.monotonic()
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32)
        with lock:
            records.append(
                {
                    "prompt": prompt,
                    "start": start,
                    "end": end,
                    "mode": mode,
                    "first": kwargs.get("first_frame_png"),
                }
            )
        return out

    def fake_qc(_root, _conn, shot, _dest, _prev, **_kwargs):
        last = Path(str(shot.get("last_frame_path") or ""))
        return "pass", shot.get("id"), last if last.is_file() else None

    monkeypatch.setattr(session, "extract_last_frame", fake_extract)
    monkeypatch.setattr("anime_factory.fal_models.generate_fal_video", fake_generate)
    monkeypatch.setattr(session, "_h3_normalize_and_qc", fake_qc)
    monkeypatch.setattr(session, "put_file", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(session, "_h3_commit_passing", lambda *_a, **_k: _a[3]["sid"])
    started = time.monotonic()
    result = session._run_anim_fal("story", tmp_path, conn)
    elapsed = time.monotonic() - started
    assert result["failed"] == []
    assert result["shots_done"] == 4
    assert result["video_backend"] == "fal"
    walk = next(row for row in records if "alpha-walk" in row["prompt"])
    cont = next(row for row in records if "alpha-continue" in row["prompt"])
    other = next(row for row in records if "bravo-look" in row["prompt"])
    assert cont["start"] >= walk["end"] - 0.02
    assert other["start"] < walk["end"]
    assert elapsed < 0.75
    assert cont["first"]
    assert cont["mode"] == "i2v"


def test_fal_retry_uses_plan_repair(monkeypatch, tmp_path):
    monkeypatch.setenv("FAL_MAX_CONCURRENCY", "1")
    ep = tmp_path / "episodes" / "EP001"
    ep.mkdir(parents=True)
    (ep / "board.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "s1",
                        "duration": 6,
                        "h3_mode": "fl2va_first",
                        "h3_prompt": "clerk turns",
                        "seed": 11,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    calls = {"n": 0, "seeds": []}

    def fake_extract(video, dest):
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(synthetic_still_png(64, 64, tag="last", placeholder=False))
        return path

    def fake_generate(_mode, _prompt, **kwargs):
        calls["n"] += 1
        calls["seeds"].append(kwargs.get("seed"))
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32)
        return out

    def fake_qc(_root, _conn, shot, _dest, _prev, **_kwargs):
        last = Path(str(shot.get("last_frame_path") or ""))
        if calls["n"] == 1:
            return "retry", "s1", last if last.is_file() else None
        return "pass", "s1", last if last.is_file() else None

    monkeypatch.setattr(session, "extract_last_frame", fake_extract)
    monkeypatch.setattr("anime_factory.fal_models.generate_fal_video", fake_generate)
    monkeypatch.setattr(session, "_h3_normalize_and_qc", fake_qc)
    monkeypatch.setattr(session, "put_file", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(session, "_h3_commit_passing", lambda *_a, **_k: _a[3]["sid"])
    result = session._run_anim_fal("story", tmp_path, conn)
    assert result["failed"] == []
    assert calls["n"] == 2
    assert calls["seeds"][1] != calls["seeds"][0]
    assert "s1" in result["repaired"]


def test_run_anim_dispatches_fal(monkeypatch, tmp_path):
    monkeypatch.setenv("AF_IMAGE_CAPABILITY", "fal")
    monkeypatch.setenv("VIDEO_BACKEND", "fal")
    monkeypatch.setenv("AF_VIDEO_BACKEND", "fal")
    seen = {}

    def fake(*_a, **kwargs):
        seen["cfg"] = kwargs.get("fal_config")
        return {"shots_total": 0, "shots_done": 0, "failed": [], "generated": [], "gpu_done": True}

    monkeypatch.setattr(session, "_run_anim_fal", fake)
    session.run_anim("story", tmp_path, None, None, fal_config={"video_i2v_model": "custom/i2v"})
    assert seen["cfg"]["video_i2v_model"] == "custom/i2v"

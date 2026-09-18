"""H3 1024×576 primary, 720p delivery, OOM downgrade, and mode scheduling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from anime_factory.models import (
    H3_GEN_HEIGHT,
    H3_GEN_WIDTH,
    H3_OOM_FALLBACK_HEIGHT,
    H3_OOM_FALLBACK_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from gpu_worker import session
from gpu_worker.h3 import (
    apply_chain_first_frame,
    h3_resolution_profile,
    is_h3_oom,
    native_h3_graph,
    ref_image_filenames,
)
from gpu_worker.h3_session import flatten_h3_group_indices, plan_h3_mode_groups
from gpu_worker.offers import MIN_CPU_RAM_MB, MIN_CUDA_VERSION, score_offer


def test_primary_resolution_is_1024x576():
    profile = h3_resolution_profile({})
    assert (profile["gen_width"], profile["gen_height"]) == (H3_GEN_WIDTH, H3_GEN_HEIGHT)
    assert (profile["delivery_width"], profile["delivery_height"]) == (VIDEO_WIDTH, VIDEO_HEIGHT)
    assert profile["tier"] == "gen_576p"
    assert profile["head_chunks"] == 16
    assert profile["chunks"] == 8
    assert profile["seq_threshold"] == 4096


def test_oom_fallback_resolution_is_864x480():
    profile = h3_resolution_profile({}, oom_fallback=True)
    assert (profile["gen_width"], profile["gen_height"]) == (
        H3_OOM_FALLBACK_WIDTH,
        H3_OOM_FALLBACK_HEIGHT,
    )
    assert profile["downgraded"] is True
    assert profile["tier"] == "oom_fallback_480p"


def test_native_graph_wires_low_vram_nodes_between_unet_and_sigma():
    graph = native_h3_graph({"id": "E01-01", "first_frame_path": "f1.png"}, "fl2va_first")
    assert graph["lv_attn"]["class_type"] == "MiniMaxLowVRAMAttention"
    assert graph["lv_attn"]["inputs"]["model"] == ["1", 0]
    assert graph["lv_attn"]["inputs"]["head_chunks"] == 16
    assert graph["lv_ff"]["class_type"] == "MiniMaxChunkFeedForward"
    assert graph["lv_ff"]["inputs"]["model"] == ["lv_attn", 0]
    assert graph["16"]["inputs"]["model"] == ["lv_ff", 0]
    assert graph["6"]["inputs"]["width"] == 1024
    assert graph["6"]["inputs"]["height"] == 576


def test_chain_last_frame_not_duplicated_in_ref_pack():
    segment = {
        "id": "E01-01-s02",
        "chain_index": 1,
        "h3_mode": "ref2va",
        "character_id": "ke",
        "refs": ["char_ke_sheet.png", "plate_store.png"],
        "first_frame_path": "prev_last.png",
        "chain_source_last_frame": "prev_last.png",
    }
    linked = apply_chain_first_frame(segment, "prev_last.png")
    assert linked.get("refs") == ["char_ke_sheet.png", "plate_store.png"]
    assert ref_image_filenames(linked) == ["char_ke_sheet.png", "plate_store.png"]


def test_is_h3_oom_classifies_execution_cuda_and_sigkill_text():
    assert is_h3_oom("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert is_h3_oom("execution_error: allocator failed")
    assert is_h3_oom("Process SIGKILL during sampling")
    assert not is_h3_oom("missing node MiniMaxH3ImageToVideo")


def test_normalize_h3_scales_to_1280x720(tmp_path, monkeypatch):
    shot = tmp_path / "v001.mp4"
    shot.write_bytes(b"native")
    seen = {}

    def transcode(command, progress=None, stage="compose"):
        seen["command"] = command
        Path(command[-1]).write_bytes(b"delivery")

    monkeypatch.setattr(session, "_run_checked", transcode)
    session._normalize_h3_for_qc(
        shot,
        h3_resolution_profile({}),
    )
    assert "scale=1280:720:flags=lanczos" in seen["command"]
    assert "-preset" in seen["command"] and "fast" in seen["command"]


def test_mode_groups_preserve_compose_order():
    shots = [
        {"id": "a", "h3_mode": "ref2va", "character_id": "ke", "refs": ["sheet"]},
        {"id": "b", "h3_mode": "ref2va", "character_id": "ke", "refs": ["sheet"]},
        {"id": "c", "h3_mode": "fl2va_first", "first_frame_path": "f1.png"},
        {"id": "d", "h3_mode": "ref2va", "character_id": "ke", "refs": ["sheet"]},
    ]
    groups = plan_h3_mode_groups(shots)
    assert [g["mode"] for g in groups] == ["ref2va", "fl2va_first", "ref2va"]
    assert flatten_h3_group_indices(groups) == [0, 1, 2, 3]


def test_cpu_post_queue_drains_background_upload(tmp_path, monkeypatch):
    import threading

    started = threading.Event()
    finished = threading.Event()

    def slow_upload(*_args, **_kwargs):
        started.set()
        finished.wait(timeout=2)
        return "ok"

    cpu = session.CpuPostQueue()
    cpu.submit(slow_upload)
    assert started.wait(timeout=1)
    finished.set()
    assert cpu.drain() == ["ok"]
    cpu.close()


def test_h3_oom_retries_once_then_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        session,
        "_board_shots",
        lambda _root: [{"id": "E01-01", "duration": 8, "h3_mode": "fl2va_first", "first_frame_path": "f1.png"}],
    )
    monkeypatch.setattr(session, "lock_video_backend", lambda **_k: "h3")
    monkeypatch.setattr(session, "existing_generation_file", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "next_generation_path", lambda *_a, **_k: (1, "episodes/EP001/shots/E01-01/v001.mp4"))
    monkeypatch.setattr(session, "H3PrepPool", lambda stage_fn: MagicMock(prime=stage_fn, schedule=lambda *_a: None, close=lambda: None))
    monkeypatch.setattr(session, "_upload_h3_inputs", lambda *_a, **_k: None)
    calls = {"n": 0}

    def boom(_router, shot, _root, dest, progress=None):
        calls["n"] += 1
        raise RuntimeError("CUDA out of memory")

    router = MagicMock()
    router.free = MagicMock(return_value={"ok": True})
    monkeypatch.setattr(session, "_submit_h3_gpu", boom)
    try:
        session.run_anim("story-1", tmp_path, object(), router=router, progress=None)
    except RuntimeError as exc:
        assert "h3_fail_closed" in str(exc)
        assert "oom_downshift_exhausted" in str(exc)
        assert "gen_480p" in str(exc)
        assert "fallback_tier" in str(exc)
    else:
        raise AssertionError("expected fail-closed after fallback OOM")
    assert calls["n"] == 2
    router.free.assert_called_once()


def test_offers_require_64gb_ram_and_cuda_13():
    offer = {
        "gpu_name": "RTX 5090",
        "gpu_ram": 32768,
        "cpu_ram": 32768,
        "cuda_max_good": 12.8,
        "duration": 4,
        "reliability": 0.995,
        "inet_down": 400,
        "inet_up": 200,
        "disk_space": 250,
        "geolocation": "US",
        "dph_total": 0.5,
    }
    low_ram = score_offer({**offer, "cpu_ram": MIN_CPU_RAM_MB - 1})
    assert low_ram["reject_reason"] == "cpu_ram"
    low_cuda = score_offer({**offer, "cpu_ram": MIN_CPU_RAM_MB, "cuda_max_good": MIN_CUDA_VERSION - 0.1})
    assert low_cuda["reject_reason"] == "cuda_version"
    ok = score_offer({**offer, "cpu_ram": MIN_CPU_RAM_MB, "cuda_max_good": MIN_CUDA_VERSION})
    assert ok.get("reject_reason") is None

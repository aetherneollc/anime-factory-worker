"""H3 1024×576 primary, 720p delivery, OOM downgrade, and mode scheduling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from anime_factory.models import (
    H3_GEN_HEIGHT,
    H3_GEN_WIDTH,
    H3_OOM_FALLBACK_HEIGHT,
    H3_OOM_FALLBACK_WIDTH,
    H3_OOM_SAFE_HEIGHT,
    H3_OOM_SAFE_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from gpu_worker import session
from gpu_worker.h3 import (
    H3ShotOomRefused,
    apply_chain_first_frame,
    choose_h3_start_tier,
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
    assert profile["oom_tier"] == 1


def test_oom_safe_resolution_is_640x352():
    profile = h3_resolution_profile({"h3_oom_tier": 2})
    assert (profile["gen_width"], profile["gen_height"]) == (
        H3_OOM_SAFE_WIDTH,
        H3_OOM_SAFE_HEIGHT,
    )
    assert profile["downgraded"] is True
    assert profile["tier"] == "oom_fallback_352p"
    assert profile["oom_tier"] == 2
    assert profile["head_chunks"] == 32
    assert profile["chunks"] == 16
    graph = native_h3_graph(
        {"id": "E01-02", "first_frame_path": "f1.png", "h3_oom_tier": 2},
        "fl2va_first",
    )
    assert graph["6"]["inputs"]["width"] == H3_OOM_SAFE_WIDTH
    assert graph["6"]["inputs"]["height"] == H3_OOM_SAFE_HEIGHT
    assert graph["lv_attn"]["inputs"]["head_chunks"] == 32


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


def test_choose_h3_start_tier_heads_stay_native_when_vram_is_full():
    head = {"id": "E01-01", "chain_index": 0}
    assert choose_h3_start_tier(head, used_mb=29_600, total_mb=32_768) == 0
    assert choose_h3_start_tier(head, used_mb=None, total_mb=None) == 0


def test_choose_h3_start_tier_chain_tails_start_at_480p_or_352p():
    tail = {"id": "E01-02", "chain_index": 1, "chain_source_last_frame": "prev.png"}
    assert choose_h3_start_tier(tail, used_mb=20_000, total_mb=32_768) == 0
    assert choose_h3_start_tier(tail, used_mb=25_768, total_mb=32_768) == 1
    assert choose_h3_start_tier(tail, used_mb=29_600, total_mb=32_768) == 2
    assert choose_h3_start_tier(tail, used_mb=None, total_mb=None) == 1
    latched = {**tail, "h3_oom_tier": 1}
    assert choose_h3_start_tier(latched, used_mb=29_600, total_mb=32_768) == 2


def test_sample_h3_probes_occupancy_before_free_and_starts_tail_at_352p(tmp_path, monkeypatch):
    order: list[object] = []

    def used():
        order.append("probe")
        return 29_600

    def total():
        return 32_768

    def reclaim(**_kw):
        order.append("free")
        return {"ok": True}

    def boom(_router, shot, _root, dest, progress=None):
        order.append(("sample", shot.get("h3_oom_tier"), shot.get("h3_oom_fallback")))
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(session, "gpu_vram_used_mb", used)
    monkeypatch.setattr(session, "gpu_vram_mb", total)
    monkeypatch.setattr(session, "_submit_h3_gpu", boom)
    router = MagicMock()
    router.free = MagicMock(side_effect=reclaim)
    shot = {
        "id": "E01-02",
        "duration": 8,
        "chain_index": 1,
        "chain_source_last_frame": "prev.png",
        "first_frame_path": "prev.png",
        "h3_mode": "fl2va_first",
    }
    try:
        session._sample_h3_with_oom_policy(
            router, shot, tmp_path, tmp_path / "v001.mp4", None, "fl2va_first", {}
        )
    except H3ShotOomRefused as exc:
        assert "h3_oom_shot_refused" in str(exc)
        assert "h3_fail_closed" not in str(exc)
        assert "640x352" in str(exc)
    else:
        raise AssertionError("expected H3ShotOomRefused after safe canvas")
    assert order[0] == "probe"
    assert "free" in order
    assert order.index("probe") < order.index("free")
    samples = [item for item in order if isinstance(item, tuple) and item[0] == "sample"]
    assert samples == [("sample", 2, True)]


def test_sample_h3_head_walks_tiers_then_refuses_without_fail_closed(tmp_path, monkeypatch):
    seen: list[int] = []

    def boom(_router, shot, _root, dest, progress=None):
        seen.append(int(shot.get("h3_oom_tier")))
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(session, "gpu_vram_used_mb", lambda: 10_000)
    monkeypatch.setattr(session, "gpu_vram_mb", lambda: 32_768)
    monkeypatch.setattr(session, "_submit_h3_gpu", boom)
    router = MagicMock()
    router.free = MagicMock(return_value={"ok": True})
    shot = {"id": "E01-01", "duration": 8, "h3_mode": "fl2va_first", "first_frame_path": "f1.png"}
    try:
        session._sample_h3_with_oom_policy(
            router, shot, tmp_path, tmp_path / "v001.mp4", None, "fl2va_first", {}
        )
    except H3ShotOomRefused as exc:
        assert "h3_oom_shot_refused" in str(exc)
        assert "h3_fail_closed" not in str(exc)
    else:
        raise AssertionError("expected H3ShotOomRefused after safe canvas")
    assert seen == [0, 1, 2]
    assert router.free.call_count == 3


def test_h3_oom_skips_shot_without_killing_episode(tmp_path, monkeypatch):
    shots = [
        {"id": "E01-01", "duration": 8, "h3_mode": "fl2va_first", "first_frame_path": "f1.png"},
        {"id": "E01-02", "duration": 8, "h3_mode": "fl2va_first", "first_frame_path": "f2.png"},
    ]
    monkeypatch.setattr(session, "_board_shots", lambda _root: shots)
    monkeypatch.setattr(session, "lock_video_backend", lambda **_k: "h3")
    monkeypatch.setattr(session, "existing_generation_file", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "select_passing_generation", lambda *_a, **_k: None)
    monkeypatch.setattr(
        session,
        "next_generation_path",
        lambda _root, sid: (1, f"episodes/EP001/shots/{sid}/v001.mp4"),
    )

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
    monkeypatch.setattr(session, "gpu_vram_used_mb", lambda: 10_000)
    monkeypatch.setattr(session, "gpu_vram_mb", lambda: 32_768)

    def submit(_router, shot, _root, dest, progress=None):
        if shot["id"] == "E01-01":
            raise RuntimeError("CUDA out of memory")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * 5000)
        last = tmp_path / "last.png"
        last.write_bytes(b"\x89PNG" + b"x" * 64)
        shot["last_frame_path"] = str(last)
        return {"shot": shot["id"]}

    monkeypatch.setattr(session, "_submit_h3_gpu", submit)
    monkeypatch.setattr(session, "_normalize_h3_for_qc", lambda *_a, **_k: None)
    monkeypatch.setattr(
        session,
        "_probe_video",
        lambda _path: {"width": 1280, "height": 720, "duration": 8, "frames": 192, "size_bytes": 5000},
    )
    monkeypatch.setattr(session, "incremental_qc_segment", lambda *_a, **_k: "pass")
    monkeypatch.setattr(session, "mark_completed_passing", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "record_generation_result", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "put_file", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(session, "_checkpoint_story", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(session, "_db_execute", lambda *_a, **_k: None)
    router = MagicMock()
    router.free = MagicMock(return_value={"ok": True})
    result = session.run_anim("story-1", tmp_path, object(), router=router, progress=None)
    assert result["generated"] == ["E01-02"]
    assert len(result["failed"]) == 1
    assert result["failed"][0]["id"] == "E01-01"
    assert "h3_oom_shot_refused" in str(result["failed"][0]["error"])
    assert "h3_fail_closed" not in str(result["failed"][0]["error"])
    assert result["gpu_done"] is False


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

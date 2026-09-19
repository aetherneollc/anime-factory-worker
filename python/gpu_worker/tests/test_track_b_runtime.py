from __future__ import annotations

import json
import os
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from gpu_worker import __main__ as worker_main
from gpu_worker import h3, poll, session, stack, weights
from gpu_worker.poll import AGENT_KEY_HEADER, control_headers, parse_batch


def test_control_headers_carry_shared_agent_key(monkeypatch):
    monkeypatch.setenv("STUDIO_AGENT_KEY", "box-secret")
    headers = control_headers(json_body=True)
    assert headers[AGENT_KEY_HEADER] == "box-secret"
    assert headers["Content-Type"] == "application/json"


def test_register_and_heartbeat_send_auth_and_tunnel(monkeypatch):
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    def open_control(req, timeout):
        seen.append(
            (
                req.full_url,
                {key.lower(): value for key, value in req.header_items()},
                req.data,
                timeout,
            )
        )
        return Response()

    monkeypatch.setenv("CONTROL_PLANE_URL", "https://control.example")
    monkeypatch.setenv("STUDIO_AGENT_KEY", "box-secret")
    monkeypatch.setattr(worker_main.urllib.request, "urlopen", open_control)
    fields = {
        "batch_id": "b-1",
        "spend_usd": 1.25,
        "idle_seconds": 0,
        "tunnel": {
            "connected": False,
            "hostname": None,
            "checked_at": "2026-09-07T00:00:00Z",
        },
    }
    events = worker_main.maybe_notify_control_plane(
        "123",
        "busy",
        heartbeat_fields=fields,
    )
    assert events == ["register", "heartbeat"]
    assert [row[0] for row in seen] == [
        "https://control.example/gpu/register",
        "https://control.example/gpu/heartbeat",
    ]
    assert all(row[1][AGENT_KEY_HEADER.lower()] == "box-secret" for row in seen)
    assert b'"batch_id": "b-1"' in seen[1][2]
    assert b'"tunnel":' in seen[1][2]


def test_heartbeat_loop_enforces_stop_when_main_thread_is_blocked(monkeypatch):
    runtime = session.LeaseRuntime(instance_id="123")
    runtime.stop_requested = session.ProgressStalled()
    stops = []
    loop = worker_main._HeartbeatLoop(
        "123",
        runtime,
        {},
        interval_s=30,
        on_stop=lambda stop: stops.append(stop) or {"ok": True},
    )

    class OneTick:
        calls = 0

        def wait(self, _timeout):
            self.calls += 1
            return self.calls > 1

    loop.stop_event = OneTick()
    monkeypatch.setattr(loop, "_notify", lambda: None)
    loop._run()
    assert len(stops) == 1
    assert isinstance(stops[0], session.ProgressStalled)
    assert runtime.monitored_teardown_result == {"ok": True}


def test_worker_once_without_comfy_remains_a_safe_dry_run(monkeypatch):
    monkeypatch.setenv("AF_ONCE", "1")
    monkeypatch.setenv("AF_START_COMFY", "0")
    monkeypatch.setenv("VAST_DRY_RUN", "1")
    monkeypatch.delenv("CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("AF_STORY_ID", raising=False)
    assert worker_main.main() == 0


def test_classified_comfy_startup_failure_destroys_without_waiting_for_probe(monkeypatch):
    monkeypatch.setenv("AF_ONCE", "1")
    monkeypatch.setenv("AF_START_COMFY", "1")
    monkeypatch.setenv("VAST_DRY_RUN", "1")
    monkeypatch.delenv("CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("AF_STORY_ID", raising=False)
    destroyed = []
    probed = []
    heartbeats = []

    monkeypatch.setattr(
        stack,
        "boot_gpu_stack",
        lambda **_k: {
            "router_ready": False,
            "comfy_error": "comfy_startup_failed:exit_1:Python.h missing",
            "handshake": {"preflight": {"ok": True}},
            "tunnel": {},
        },
    )
    monkeypatch.setattr(stack, "current_tunnel_status", lambda: {})
    monkeypatch.setattr(
        session,
        "destroy_self",
        lambda instance_id, reason, error=None: destroyed.append(
            {"instance_id": instance_id, "reason": reason, "error": error}
        )
        or {"ok": True, "destroyed": True},
    )

    def fake_probe(_base):
        probed.append(_base)
        return True

    monkeypatch.setattr("gpu_worker.boot.probe_comfy_system_stats", fake_probe)
    monkeypatch.setattr(
        worker_main,
        "maybe_notify_control_plane",
        lambda *args, **kwargs: heartbeats.append(kwargs.get("heartbeat_fields") or {}) or [],
    )
    monkeypatch.setattr(
        worker_main.time,
        "sleep",
        lambda _s: (_ for _ in ()).throw(RuntimeError("slept")),
    )
    assert worker_main.main() == 1
    assert probed == []
    assert destroyed
    assert "comfy_startup_failed" in str(destroyed[0]["error"])
    classified = [row for row in heartbeats if row.get("boot_stage") == "comfy_startup_failed"]
    assert classified


def test_classified_comfy_startup_uploads_boot_json_before_destroy(monkeypatch):
    monkeypatch.setenv("AF_ONCE", "1")
    monkeypatch.setenv("AF_START_COMFY", "1")
    monkeypatch.setenv("VAST_DRY_RUN", "1")
    monkeypatch.setenv("AF_STORY_ID", "story-canary-7516b66")
    monkeypatch.delenv("CONTROL_PLANE_URL", raising=False)
    order = []

    monkeypatch.setattr(
        stack,
        "boot_gpu_stack",
        lambda **_k: {
            "router_ready": False,
            "comfy_error": "comfy_startup_failed:exit_1:Python.h missing",
            "comfy_boot_log_tail": "traceback",
            "handshake": {"preflight": {"ok": True}},
            "tunnel": {},
        },
    )
    monkeypatch.setattr(stack, "current_tunnel_status", lambda: {})
    monkeypatch.setattr(
        session,
        "persist_boot_failure",
        lambda payload, story_id=None: order.append("persist") or {"ok": True},
    )
    monkeypatch.setattr(
        session,
        "destroy_self",
        lambda instance_id, reason, error=None: order.append("destroy")
        or {"ok": True, "destroyed": True},
    )
    monkeypatch.setattr(worker_main, "maybe_notify_control_plane", lambda *a, **k: [])
    monkeypatch.setattr(
        worker_main.time,
        "sleep",
        lambda _s: (_ for _ in ()).throw(RuntimeError("slept")),
    )
    assert worker_main.main() == 1
    assert order[:2] == ["persist", "destroy"]


def test_preflight_failure_uploads_boot_json_before_destroy(monkeypatch):
    from gpu_worker.preflight import PreflightFailure

    monkeypatch.setenv("AF_ONCE", "1")
    monkeypatch.setenv("AF_START_COMFY", "1")
    monkeypatch.setenv("VAST_DRY_RUN", "1")
    monkeypatch.setenv("AF_STORY_ID", "story-canary-7516b66")
    monkeypatch.delenv("CONTROL_PLANE_URL", raising=False)
    order = []

    def boom(**_k):
        raise PreflightFailure("preflight_failed", "disk_low", "disk 196.0GB below minimum")

    monkeypatch.setattr(stack, "boot_gpu_stack", boom)
    monkeypatch.setattr(stack, "current_tunnel_status", lambda: {})
    monkeypatch.setattr(
        session,
        "persist_boot_failure",
        lambda payload, story_id=None: order.append("persist")
        or {"ok": True},
    )
    monkeypatch.setattr(
        session,
        "destroy_self",
        lambda instance_id, reason, error=None: order.append("destroy")
        or {"ok": True, "destroyed": True},
    )
    monkeypatch.setattr(worker_main, "maybe_notify_control_plane", lambda *_a, **_k: [])
    assert worker_main.main() == 1
    assert order == ["persist", "destroy"]


def test_persist_boot_failure_skipped_without_story_id(monkeypatch):
    monkeypatch.delenv("AF_STORY_ID", raising=False)
    result = session.persist_boot_failure({"preflight_failed": {"code": "disk_low"}})
    assert result == {"ok": False, "skipped": True, "reason": "no_story_id"}


def test_abi_boot_exception_recycles_instead_of_install_failure(monkeypatch):
    monkeypatch.setenv("AF_ONCE", "1")
    monkeypatch.setenv("AF_START_COMFY", "1")
    monkeypatch.setenv("VAST_DRY_RUN", "1")
    monkeypatch.delenv("CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("AF_STORY_ID", raising=False)
    destroyed = []

    def boom(**_k):
        raise OSError("undefined symbol: cudaLaunchKernelExC")

    monkeypatch.setattr(stack, "boot_gpu_stack", boom)
    monkeypatch.setattr(stack, "current_tunnel_status", lambda: {})
    monkeypatch.setattr(
        session,
        "destroy_self",
        lambda instance_id, reason, error=None: destroyed.append(error)
        or {"ok": True, "destroyed": True},
    )
    monkeypatch.setattr(worker_main, "maybe_notify_control_plane", lambda *_a, **_k: [])
    assert worker_main.main() == 1
    assert destroyed
    assert "capability_mismatch" in str(destroyed[0])


def test_unclassified_install_failure_stays_fail_closed(monkeypatch):
    monkeypatch.setenv("AF_ONCE", "1")
    monkeypatch.setenv("AF_START_COMFY", "1")
    monkeypatch.setenv("VAST_DRY_RUN", "1")
    monkeypatch.delenv("CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("AF_STORY_ID", raising=False)
    destroyed = []

    def boom(**_k):
        raise RuntimeError("font_setup_failed: no package manager")

    monkeypatch.setattr(stack, "boot_gpu_stack", boom)
    monkeypatch.setattr(stack, "current_tunnel_status", lambda: {})
    monkeypatch.setattr(
        session,
        "destroy_self",
        lambda *_a, **_k: destroyed.append("called") or {"ok": True, "destroyed": True},
    )
    monkeypatch.setattr(worker_main, "maybe_notify_control_plane", lambda *_a, **_k: [])
    assert worker_main.main() == 1
    assert destroyed == []


def test_work_claim_and_report_send_auth(monkeypatch):
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true, "batch": null}'

    def open_control(req, timeout):
        seen.append(
            (
                req.full_url,
                {key.lower(): value for key, value in req.header_items()},
                timeout,
            )
        )
        return Response()

    monkeypatch.setenv("STUDIO_AGENT_KEY", "box-secret")
    monkeypatch.setattr(poll.urllib.request, "urlopen", open_control)
    poll.fetch_work("https://control.example")
    poll.claim_job("https://control.example", "123", "story-1", "EP001")
    poll.report_episode(
        "https://control.example",
        {
            "vast_instance_id": "123",
            "batch_id": "b-1",
            "episode_code": "EP001",
            "stage": "compose",
            "status": "done",
            "artifacts": {"final": "key", "shorts": []},
            "error": None,
        },
    )
    assert [row[0].rsplit("/", 1)[-1].split("?")[0] for row in seen] == ["work", "claim", "report"]
    assert all(row[1][AGENT_KEY_HEADER.lower()] == "box-secret" for row in seen)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://control.example/gpu/work", code, "err", None, None)


def test_fetch_work_403_fail_fast_no_retry(monkeypatch):
    hits = {"n": 0}

    def open_control(_req, timeout=20):
        hits["n"] += 1
        raise _http_error(403)

    monkeypatch.setattr(poll.urllib.request, "urlopen", open_control)
    first = poll.fetch_work("https://control.example")
    second = poll.fetch_work("https://control.example")
    assert first.get("fail_fast") is True
    assert first.get("failure_class") == "control_plane_403"
    assert first.get("http_status") == 403
    assert second.get("fail_fast") is True
    assert hits["n"] == 2


def test_fetch_work_502_stays_retryable(monkeypatch):
    def open_control(_req, timeout=20):
        raise _http_error(502)

    monkeypatch.setattr(poll.urllib.request, "urlopen", open_control)
    payload = poll.fetch_work("https://control.example")
    assert payload.get("fail_fast") is not True
    assert payload.get("ok") is False


def test_recycle_forbidden_host_rates_and_destroys(monkeypatch):
    rated = []
    destroyed = []

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def report_machine(self, machine_id, problem, message, rating=1):
            rated.append((machine_id, problem, rating, "403" in str(message)))
            return {"ok": True}

    monkeypatch.setenv("VAST_MACHINE_ID", "4242")
    monkeypatch.setattr(session, "VastClient", FakeClient)
    monkeypatch.setattr(
        session,
        "destroy_self",
        lambda instance_id, reason, error=None: destroyed.append((instance_id, reason, error))
        or {"ok": True, "destroyed": True},
    )
    out = session.recycle_forbidden_host("51603913", "control_plane_403:HTTP 403")
    assert rated == [("4242", "network", 1, True)]
    assert destroyed[0][0] == "51603913"
    assert out["teardown"]["destroyed"] is True


def test_main_fail_fast_on_control_plane_403_without_comfy(monkeypatch):
    destroyed = []
    booted = []

    monkeypatch.setenv("AF_ONCE", "1")
    monkeypatch.setenv("AF_START_COMFY", "1")
    monkeypatch.setenv("CONTROL_PLANE_URL", "https://control.example")
    monkeypatch.setenv("VAST_MACHINE_ID", "4242")
    monkeypatch.setenv("VAST_DRY_RUN", "1")
    monkeypatch.delenv("AF_STORY_ID", raising=False)

    def boom(**_k):
        booted.append("comfy")
        raise AssertionError("must not boot Comfy after 403")

    monkeypatch.setattr(stack, "boot_gpu_stack", boom)
    monkeypatch.setattr(stack, "current_tunnel_status", lambda: {})
    monkeypatch.setattr(worker_main, "maybe_notify_control_plane", lambda *_a, **_k: [])
    monkeypatch.setattr(
        poll,
        "fetch_work",
        lambda *_a, **_k: {
            "ok": False,
            "fail_fast": True,
            "error": "control_plane_403:HTTP 403",
            "failure_class": "control_plane_403",
            "batch": None,
            "jobs": [],
        },
    )
    monkeypatch.setattr(
        session,
        "recycle_forbidden_host",
        lambda instance_id, error: destroyed.append((instance_id, error))
        or {"rated": {"ok": True}, "teardown": {"ok": True, "destroyed": True}},
    )
    assert worker_main.main() == 1
    assert booted == []
    assert destroyed
    assert destroyed[0][1] == "control_plane_403:HTTP 403"


def test_parse_batch_preserves_multi_episode_contract():
    batch = {
        "batch_id": "b-1",
        "story_id": "story-1",
        "episodes": [{"episode_code": "EP001"}, {"episode_code": "EP002"}],
        "budget": {"max_minutes": 60, "max_usd": 3, "idle_minutes": 5},
    }
    assert parse_batch({"batch": batch, "jobs": []}) == batch
    assert parse_batch({"batch": None}) is None


def test_h3_resolution_primary_and_oom_fallback():
    native = h3.h3_resolution_profile({"width": 1344, "height": 768})
    assert (native["gen_width"], native["gen_height"]) == (1024, 576)
    assert native["downgraded"] is False
    assert native["tier"] == "gen_576p"
    low = h3.h3_resolution_profile({"width": 1344, "height": 768}, oom_fallback=True)
    assert (low["gen_width"], low["gen_height"]) == (864, 480)
    assert low["downgraded"] is True
    assert low["tier"] == "oom_fallback_480p"


def test_primary_canvas_ignores_legacy_h3_max_env(monkeypatch):
    monkeypatch.delenv("H3_MAX_WIDTH", raising=False)
    monkeypatch.delenv("H3_MAX_HEIGHT", raising=False)
    assert h3.h3_spatial_size({}) == (1024, 576)
    monkeypatch.setenv("H3_MAX_WIDTH", "512")
    monkeypatch.setenv("H3_MAX_HEIGHT", "288")
    assert h3.h3_spatial_size({}) == (1024, 576)
    graph = h3.native_h3_graph(
        {"id": "E01-01", "first_frame_path": "f.png"},
        "fl2va_first",
    )
    assert graph["6"]["inputs"]["width"] == 1024
    assert graph["6"]["inputs"]["height"] == 576


def test_h3_sampler_steps_default_remains_eight(monkeypatch):
    monkeypatch.delenv("H3_SAMPLER_STEPS", raising=False)
    monkeypatch.setattr(h3, "gpu_vram_mb", lambda: 32_640)
    graph = h3.native_h3_graph(
        {"id": "E01-01", "first_frame_path": "f.png"},
        "fl2va_first",
    )
    assert graph["9"]["inputs"]["steps"] == 8


def test_h3_dit_defaults_to_community_pruned_nvfp4():
    by_dest = {item["dest"]: item for item in weights.H3_AND_KOLORS_FILES}
    fl2va = by_dest["models/diffusion_models/minimax_h3_fl2va_pruned_nvfp4.safetensors"]
    ref2va = by_dest["models/diffusion_models/minimax_h3_ref2va_pruned_nvfp4.safetensors"]
    te = by_dest["models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"]
    video_vae = by_dest["models/vae/minimax_h3_video_vae_fp16.safetensors"]
    still = by_dest["models/checkpoints/animagine-xl-4.0.safetensors"]
    ipadapter = by_dest["models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors"]
    clip_vision = by_dest["models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"]
    visual_qc = by_dest[weights.VISUAL_QC_CLIP_WEIGHT_DEST]
    visual_cfg = by_dest[weights.VISUAL_QC_CLIP_CONFIG_DEST]
    assert fl2va["repo"] == "lilcheaty/MiniMax-H3-NVFP4"
    assert ref2va["repo"] == "lilcheaty/MiniMax-H3-NVFP4"
    assert fl2va["hf"] == "minimax_h3_fl2va_pruned_nvfp4.safetensors"
    assert ref2va["hf"] == "minimax_h3_ref2va_pruned_nvfp4.safetensors"
    assert te["repo"] == "Comfy-Org/MiniMax-H3"
    assert video_vae["repo"] == "Comfy-Org/MiniMax-H3"
    assert "models/vae/minimax_h3_audio_vae_fp32.safetensors" not in by_dest
    # Stills are anime-native SDXL now: real CFG, so FIXED_NEGATIVE is not dead code.
    assert still["repo"] == "cagliostrolab/animagine-xl-4.0"
    assert still["hf"] == "animagine-xl-4.0.safetensors"
    # IP-Adapter + ViT-H are what make the locked sheet bind to the shot.
    assert ipadapter["repo"] == "h94/IP-Adapter"
    assert ipadapter["hf"] == "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
    assert clip_vision["repo"] == "h94/IP-Adapter"
    assert clip_vision["hf"] == "models/image_encoder/model.safetensors"
    assert visual_qc["repo"] == weights.VISUAL_QC_CLIP_REPO
    assert visual_qc["hf"] == weights.VISUAL_QC_CLIP_WEIGHT_HF
    assert visual_cfg["hf"] == weights.VISUAL_QC_CLIP_CONFIG_HF
    assert "clip_vision" not in visual_qc["dest"]
    assert "visual_qc" not in clip_vision["dest"]
    assert "models/Kolors" not in by_dest
    assert not any("flux" in dest for dest in by_dest)
    assert not any(item.get("snapshot") == "1" for item in weights.H3_AND_KOLORS_FILES)
    assert not any("int8_convrot" in item["dest"] for item in weights.H3_AND_KOLORS_FILES)


def test_native_h3_graph_loads_nvfp4_dits(monkeypatch):
    monkeypatch.setattr(h3, "gpu_vram_mb", lambda: 32_640)
    fl2va = h3.native_h3_graph({"id": "E01-01", "first_frame_path": "f.png"}, "fl2va_first")
    ref2va = h3.native_h3_graph({"id": "E01-01", "first_frame_path": "f.png"}, "ref2va")
    assert fl2va["1"]["inputs"]["unet_name"] == "minimax_h3_fl2va_pruned_nvfp4.safetensors"
    assert ref2va["1"]["inputs"]["unet_name"] == "minimax_h3_ref2va_pruned_nvfp4.safetensors"
    assert fl2va["2"]["inputs"]["clip_name"] == "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    assert fl2va["3"]["inputs"]["vae_name"] == "minimax_h3_video_vae_fp16.safetensors"
    assert fl2va["9"]["inputs"]["steps"] == 8
    assert (fl2va["6"]["inputs"]["width"], fl2va["6"]["inputs"]["height"]) == (1024, 576)


def test_still_weights_do_not_block_on_h3(tmp_path, monkeypatch):
    pulled = []

    def fake_hf(repo, filename, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if filename == weights.VISUAL_QC_CLIP_WEIGHT_HF:
            dest.write_bytes(b"W" * weights.VISUAL_QC_CLIP_MIN_BYTES)
        elif filename == weights.VISUAL_QC_CLIP_CONFIG_HF:
            dest.write_bytes(b"{" + b"x" * weights.VISUAL_QC_CLIP_CONFIG_MIN_BYTES + b"}")
        else:
            dest.write_bytes(b"HF:" + filename.encode())
        pulled.append(filename)

    monkeypatch.setattr(weights, "_hf_file", fake_hf)
    weights.reset_h3_weight_job()
    still = weights.ensure_still_weights(tmp_path)
    assert still["kind"] == "stills"
    assert pulled == [
        "animagine-xl-4.0.safetensors",
        "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        "models/image_encoder/model.safetensors",
        weights.VISUAL_QC_CLIP_WEIGHT_HF,
        weights.VISUAL_QC_CLIP_CONFIG_HF,
    ]
    assert (tmp_path / "models/checkpoints/animagine-xl-4.0.safetensors").is_file()
    assert (tmp_path / "models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors").is_file()
    assert (tmp_path / "models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors").is_file()
    assert (tmp_path / weights.VISUAL_QC_CLIP_WEIGHT_DEST).is_file()
    assert (tmp_path / weights.VISUAL_QC_CLIP_CONFIG_DEST).is_file()
    assert not (tmp_path / "models/diffusion_models/minimax_h3_fl2va_pruned_nvfp4.safetensors").exists()

    def slow_h3(comfy_dir=None, progress=None, modes=None):
        (tmp_path / "models/diffusion_models").mkdir(parents=True, exist_ok=True)
        for item in weights.H3_FILES:
            dest = tmp_path / item["dest"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"H3")
        return {"kind": "h3", "downloaded": [i["dest"] for i in weights.H3_FILES]}

    monkeypatch.setattr(weights, "ensure_h3_weights", slow_h3)
    thread = weights.start_h3_weights_background(tmp_path)
    assert thread is not None
    joined = weights.join_h3_weights(tmp_path)
    assert joined["ok"] is True
    assert not weights.missing_h3_weight_labels(tmp_path)
    weights.reset_h3_weight_job()


def test_hf_download_local_dir_lands_root_nvfp4_on_dest():
    dest = Path(
        "/opt/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_nvfp4.safetensors"
    )
    assert weights.hf_download_local_dir(
        "minimax_h3_fl2va_pruned_nvfp4.safetensors", dest
    ) == "/opt/ComfyUI/models/diffusion_models"
    assert weights.hf_download_local_dir(
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        Path("/opt/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    ) == "/opt/ComfyUI/models"


def test_32gb_comfy_does_not_split_dit_during_sampling(monkeypatch, tmp_path):
    monkeypatch.setenv("AF_COMFY_LOWVRAM", "1")
    extra = tmp_path / "extra_model_paths.yaml"
    extra.write_text("anime-factory:\n", encoding="utf-8")
    native = stack.comfy_launch_args(
        main_py=tmp_path / "main.py",
        extra_paths=extra,
        vram_mb=32_640,
    )
    low = stack.comfy_launch_args(
        main_py=tmp_path / "main.py",
        extra_paths=extra,
        vram_mb=24_576,
    )
    assert "--lowvram" not in native
    assert "--highvram" not in native
    assert "--gpu-only" not in native
    assert "--lowvram" in low


def test_oom_fallback_output_is_upscaled_to_720p(tmp_path, monkeypatch):
    shot = tmp_path / "v001.mp4"
    shot.write_bytes(b"low-resolution")
    seen = {}

    def transcode(command, progress=None, stage="compose"):
        seen["command"] = command
        seen["stage"] = stage
        Path(command[-1]).write_bytes(b"delivery-resolution")

    monkeypatch.setattr(session, "_run_checked", transcode)
    session._normalize_h3_for_qc(
        shot,
        {
            "downgraded": True,
            "gen_width": 864,
            "gen_height": 480,
            "delivery_width": 1280,
            "delivery_height": 720,
        },
    )
    assert shot.read_bytes() == b"delivery-resolution"
    assert "scale=1280:720:flags=lanczos" in seen["command"]
    assert seen["stage"] == "anim"


def test_primary_output_is_upscaled_to_720p(tmp_path, monkeypatch):
    shot = tmp_path / "v001.mp4"
    shot.write_bytes(b"576p-generation")
    seen = {}

    def transcode(command, progress=None, stage="compose"):
        seen["command"] = command
        Path(command[-1]).write_bytes(b"720p-delivery")

    monkeypatch.setattr(session, "_run_checked", transcode)
    session._normalize_h3_for_qc(
        shot,
        {
            "downgraded": False,
            "gen_width": 1024,
            "gen_height": 576,
            "delivery_width": 1280,
            "delivery_height": 720,
            "tier": "gen_576p",
        },
    )
    assert shot.read_bytes() == b"720p-delivery"
    assert "scale=1280:720:flags=lanczos" in seen["command"]
    assert "-an" in seen["command"]
    assert "-crf" in seen["command"] and "18" in seen["command"]


def test_tunnel_status_reports_missing_token(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_TUNNEL_TOKEN", raising=False)
    monkeypatch.setattr(stack, "_TUNNEL_STATUS", None)
    status = stack.current_tunnel_status()
    assert status["connected"] is False
    assert status["hostname"] is None
    assert status["checked_at"].endswith("Z")


def test_tunnel_verification_requires_edge_metric(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_TUNNEL_TOKEN", "token")
    monkeypatch.setenv("CLOUDFLARE_TUNNEL_HOSTNAME", "box.example.com")
    monkeypatch.setattr(stack, "_tunnel_metrics_connected", lambda: True)
    status = stack.verify_tunnel_connection(timeout_s=0)
    assert status["connected"] is True
    assert status["hostname"] == "box.example.com"


def test_tunnel_probe_uses_cloudflared_ready_endpoint(monkeypatch):
    seen = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"status": 200, "readyConnections": 4}'

    def open_ready(req, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(stack.urllib.request, "urlopen", open_ready)
    assert stack._tunnel_metrics_connected()
    assert seen == {"url": "http://127.0.0.1:49312/ready", "timeout": 3}


def test_start_tunnel_skips_when_already_connected(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_TUNNEL_TOKEN", "token")
    monkeypatch.setattr(stack, "_tunnel_metrics_connected", lambda: True)
    popped = []
    monkeypatch.setattr(stack, "_popen", lambda args: popped.append(args) or SimpleNamespace(pid=1))
    assert stack.start_tunnel() is None
    assert popped == []
    assert stack.current_tunnel_status()["connected"] is True


def test_font_setup_installs_and_verifies_noto_cjk(tmp_path, monkeypatch):
    installed = [False]
    calls = []

    def which(name):
        if name == "apt-get":
            return "/usr/bin/apt-get"
        if name == "fc-cache" and installed[0]:
            return "/usr/bin/fc-cache"
        if name in {"fc-list", "fc-match"} and installed[0]:
            return f"/usr/bin/{name}"
        return None

    def check_call(command):
        calls.append(command)
        if "install" in command:
            installed[0] = True

    monkeypatch.delenv("ANIME_FACTORY_FONTS_DIR", raising=False)
    monkeypatch.setattr(stack.shutil, "which", which)
    monkeypatch.setattr(stack.subprocess, "check_call", check_call)
    monkeypatch.setattr(
        stack,
        "_noto_fonts_dir",
        lambda: tmp_path if installed[0] else None,
    )
    monkeypatch.setattr(
        stack,
        "_font_family_available",
        lambda _family: installed[0],
    )
    result = stack.ensure_noto_cjk_fonts()
    assert result["verified"] is True
    assert result["fonts_dir"] == str(tmp_path)
    assert os.environ["ANIME_FACTORY_FONTS_DIR"] == str(tmp_path)
    install = next(command for command in calls if "install" in command)
    assert "fontconfig" in install
    assert "fonts-noto-cjk" in install


def test_font_directory_points_at_actual_noto_files(tmp_path, monkeypatch):
    noto = tmp_path / "opentype" / "noto"
    noto.mkdir(parents=True)
    (noto / "NotoSansCJK-Regular.ttc").write_bytes(b"font")
    monkeypatch.setenv("ANIME_FACTORY_FONTS_DIR", str(tmp_path))
    assert stack._noto_fonts_dir() == noto


def test_font_setup_fails_explicitly_without_package_manager(monkeypatch):
    monkeypatch.delenv("ANIME_FACTORY_FONTS_DIR", raising=False)
    monkeypatch.setattr(stack, "_noto_fonts_dir", lambda: None)
    monkeypatch.setattr(stack, "_font_family_available", lambda _family: False)
    monkeypatch.setattr(stack.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="font_setup_failed"):
        stack.ensure_noto_cjk_fonts()


def test_startup_subphase_preflight_timeout_is_two_minutes():
    now = [0.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        clock=lambda: now[0],
    )
    runtime.begin_startup()
    runtime.set_startup_subphase("preflight")
    now[0] = 121.0
    stop = runtime.limit_reached()
    assert isinstance(stop, session.PreflightTimeout)


def test_torch_and_fonts_leave_two_minute_preflight_bucket():
    now = [0.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        clock=lambda: now[0],
    )
    runtime.begin_startup()
    runtime.observe_progress("startup_stage:preflight")
    now[0] = 30.0
    runtime.observe_progress("startup_stage:torch")
    now[0] = 121.0
    assert runtime.limit_reached() is None
    runtime.observe_progress("startup_stage:fonts")
    now[0] = 180.0
    assert runtime.limit_reached() is None
    assert runtime.startup_subphase == "weights"


def test_startup_subphase_comfy_timeout_is_ten_minutes():
    now = [0.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        clock=lambda: now[0],
    )
    runtime.begin_startup()
    runtime.set_startup_subphase("comfy")
    now[0] = 601.0
    stop = runtime.limit_reached()
    assert isinstance(stop, session.ComfyStartupTimeout)


def test_startup_subphase_weights_timeout_is_sixty_minutes():
    now = [0.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        clock=lambda: now[0],
        startup_timeout_seconds=150 * 60,
    )
    runtime.begin_startup()
    runtime.set_startup_subphase("weights")
    now[0] = 3599.0
    assert runtime.limit_reached() is None
    now[0] = 3600.0
    stop = runtime.limit_reached()
    assert isinstance(stop, session.WeightPullTimeout)
    assert type(stop) is session.WeightPullTimeout


def test_heartbeat_includes_handshake_fields():
    runtime = session.LeaseRuntime(instance_id="123")
    runtime.set_handshake(
        {
            "profile_id": "h3-comfy-cu130-sm120",
            "image_digest": "sha-0090c77",
            "host": {"sm": "sm_120"},
            "stack": {"torch": "2.13.0+cu130"},
            "resources": {"disk_free_gb": 180},
            "preflight": {"ok": True, "stage": "complete", "adaptations": []},
            "adaptations": [{"action": "fix_onstart", "before": {}, "after": {}, "attempt": 1, "result": "ok"}],
            "boot_log_tail": "ready",
        }
    )
    fields = runtime.heartbeat_fields({})
    assert fields["profile_id"] == "h3-comfy-cu130-sm120"
    assert fields["image_digest"] == "sha-0090c77"
    assert fields["preflight"]["ok"] is True
    assert fields["adaptations"][0]["action"] == "fix_onstart"
    assert fields["progress"]["boot_log_tail"] == "ready"


def test_stack_boot_reports_startup_stages_and_font_verification(monkeypatch):
    events = []
    monkeypatch.setenv("AF_VIDEO_BACKEND", "h3")
    monkeypatch.setattr(stack, "run_hardware_preflight", lambda **_k: {"profile_id": "h3-comfy-cu130-sm120", "preflight": {"ok": True}})
    monkeypatch.setattr(stack, "ensure_torch", lambda: None)
    monkeypatch.setattr(stack, "ensure_c_compiler", lambda: None)
    monkeypatch.setattr(stack, "ensure_comfy_reqs", lambda: None)
    monkeypatch.setattr(
        stack,
        "ensure_noto_cjk_fonts",
        lambda: {"verified": True, "fonts_dir": "/fonts"},
    )
    monkeypatch.setattr(stack, "ensure_infer_schema_sitecustomize", lambda: None)
    monkeypatch.setattr(stack, "patch_kitchen_triton_optional", lambda: False)
    monkeypatch.setattr(stack, "patch_comfy_kitchen_for_torch26", lambda: [])
    monkeypatch.setattr(stack, "probe_comfy_nodes", lambda *_a, **_k: {"nodes": ["IPAdapterAdvanced"]})
    monkeypatch.setattr(stack, "probe_comfy_torch_import", lambda **_k: {"ok": True})

    def stills(_root, progress=None):
        if progress:
            progress("startup_bytes:1024")
        return {"hits": [], "misses": ["flux"], "downloaded": ["flux"], "source": "huggingface", "kind": "stills"}

    monkeypatch.setattr(stack, "ensure_still_weights", stills)
    monkeypatch.setattr(stack, "start_h3_weights_background", lambda *a, **k: SimpleNamespace(name="h3-weights"))
    monkeypatch.setattr(stack, "runtime_weight_bytes", lambda _root: 1024)
    comfy_proc = SimpleNamespace(pid=11, poll=lambda: None)
    monkeypatch.setattr(stack, "start_comfy", lambda: comfy_proc)
    monkeypatch.setattr(stack, "start_router", lambda: SimpleNamespace(pid=12))
    monkeypatch.setattr(stack, "start_tunnel", lambda: None)
    monkeypatch.setattr(stack, "wait_comfy_process", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(stack, "comfy_process_exited", lambda _proc: False)
    monkeypatch.setattr(stack.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        stack,
        "wait_router_ready",
        lambda progress=None: progress("startup_health:ready") is None,
    )
    monkeypatch.setattr(
        stack,
        "verify_tunnel_connection",
        lambda timeout_s=None: {"connected": False},
    )
    monkeypatch.setattr(stack, "default_register_capabilities", lambda: {})
    result = stack.boot_gpu_stack(progress=events.append)
    assert result["router_ready"] is True
    assert result["fonts"]["verified"] is True
    assert result["startup_downloaded_bytes"] == 1024
    assert "startup_stage:preflight" in events
    assert "startup_stage:weights:setup" in events
    assert "startup_stage:fonts" in events
    assert "startup_stage:weights:stills" in events
    assert "startup_stage:comfy_torch_probe" in events
    assert "startup_stage:weights:h3_background" in events
    assert result["h3_weights_background"] is True
    assert "startup_bytes:1024" in events
    assert "startup_health:ready" in events


def test_stack_boot_longlive_starts_comfy_stills_skips_h3(monkeypatch):
    events = []
    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    monkeypatch.setenv("AF_IMAGE_CAPABILITY", "longlive")
    monkeypatch.setattr(stack, "run_hardware_preflight", lambda **_k: {"profile_id": "longlive-nvfp4-sm120", "preflight": {"ok": True}})
    monkeypatch.setattr(stack, "ensure_torch", lambda: None)
    monkeypatch.setattr(stack, "ensure_c_compiler", lambda: None)
    monkeypatch.setattr(stack, "ensure_comfy_reqs", lambda: None)
    monkeypatch.setattr(
        stack,
        "ensure_noto_cjk_fonts",
        lambda: {"verified": True, "fonts_dir": "/fonts"},
    )
    monkeypatch.setattr(stack, "ensure_infer_schema_sitecustomize", lambda: None)
    monkeypatch.setattr(stack, "patch_kitchen_triton_optional", lambda: False)
    monkeypatch.setattr(stack, "patch_comfy_kitchen_for_torch26", lambda: [])
    monkeypatch.setattr(stack, "probe_comfy_torch_import", lambda **_k: {"ok": True})

    def stills(_root, progress=None):
        if progress:
            progress("startup_bytes:2048")
        return {
            "hits": [],
            "misses": ["animagine"],
            "downloaded": ["animagine"],
            "source": "huggingface",
            "kind": "stills",
        }

    def boom(*_a, **_k):
        raise AssertionError("longlive boot must not download H3")

    monkeypatch.setattr(stack, "ensure_still_weights", stills)
    monkeypatch.setattr(stack, "start_h3_weights_background", boom)
    monkeypatch.setattr(stack, "runtime_weight_bytes", lambda _root: 2048)
    comfy_proc = SimpleNamespace(pid=21, poll=lambda: None)
    monkeypatch.setattr(stack, "start_comfy", lambda: comfy_proc)
    monkeypatch.setattr(stack, "start_router", lambda: SimpleNamespace(pid=22))
    monkeypatch.setattr(stack, "start_tunnel", lambda: None)
    monkeypatch.setattr(stack, "wait_comfy_process", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(stack.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        stack,
        "wait_router_ready",
        lambda progress=None: progress("startup_health:ready") is None,
    )
    monkeypatch.setattr(
        stack,
        "verify_tunnel_connection",
        lambda timeout_s=None: {"connected": False},
    )
    monkeypatch.setattr(stack, "default_register_capabilities", lambda: {"longlive": True, "h3": False})
    result = stack.boot_gpu_stack(progress=events.append)
    assert result["router_ready"] is True
    assert result["comfy_skipped_longlive"] is False
    assert result["h3_skipped_longlive"] is True
    assert result["comfy_pid"] == 21
    assert "startup_stage:weights:stills" in events
    assert "startup_stage:weights:h3_skipped_longlive" in events
    assert "startup_health:ready" in events


def test_runtime_enforces_time_cost_and_idle_budgets():
    now = [0.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        idle_since=0.0,
        clock=lambda: now[0],
    )
    runtime.set_hourly_rate(2.0)
    runtime.adopt_batch(
        {
            "batch_id": "b-1",
            "budget": {"max_minutes": 40, "max_usd": 1, "idle_minutes": 2},
        }
    )
    now[0] = 1800.0
    assert runtime.limit_reached().limit == "max_usd"
    runtime.mark_idle()
    now[0] += 120.0
    assert runtime.idle_expired(comfy_inflight=0)
    fields = runtime.heartbeat_fields(
        {"connected": False, "hostname": None, "checked_at": "2026-09-07T00:00:00Z"}
    )
    assert fields["batch_id"] == "b-1"
    assert fields["spend_usd"] > 1
    assert fields["idle_seconds"] == 120


def test_budget_defaults_and_global_usd_clamp(monkeypatch):
    for name in (
        "VAST_MAX_LEASE_MINUTES",
        "VAST_MAX_LEASE_USD",
        "VAST_IDLE_MINUTES",
        "VAST_WATCH_USD",
    ):
        monkeypatch.delenv(name, raising=False)
    defaults = session.BatchBudget.from_payload()
    assert defaults.max_minutes == 3600
    assert defaults.max_usd == 40
    assert defaults.idle_minutes == 15
    clamped = session.BatchBudget.from_payload(
        {"max_minutes": 7200, "max_usd": 999, "idle_minutes": 20}
    )
    assert clamped.max_usd == 100
    assert session.LeaseRuntime(instance_id="123").watch_usd == 20


def test_watch_usd_is_independently_configurable(monkeypatch):
    for name in (
        "VAST_MAX_LEASE_MINUTES",
        "VAST_MAX_LEASE_USD",
        "VAST_IDLE_MINUTES",
        "VAST_WATCH_USD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VAST_WATCH_USD", "27.5")
    runtime = session.LeaseRuntime(instance_id="123")
    assert runtime.watch_usd == 27.5
    assert runtime.budget.max_usd == 40


def test_global_usd_hard_stop_applies_when_configured_limit_is_disabled():
    now = [101 * 3600.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        clock=lambda: now[0],
        watch_usd=20,
    )
    runtime.set_hourly_rate(1.0)
    runtime.adopt_batch(
        {
            "batch_id": "b-1",
            "budget": {"max_minutes": 10_000, "max_usd": 0, "idle_minutes": 15},
        }
    )
    assert runtime.limit_reached().limit == "global_max_usd"


def test_progress_watchdog_stops_after_three_stagnant_checks():
    now = [21 * 3600.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        clock=lambda: now[0],
        watch_usd=20,
    )
    runtime.set_hourly_rate(1.0)
    runtime.adopt_batch(
        {
            "batch_id": "b-1",
            "budget": {"max_minutes": 3600, "max_usd": 100, "idle_minutes": 15},
        }
    )
    assert runtime.stop_condition() is None
    for _ in range(2):
        now[0] += 600
        assert runtime.stop_condition() is None
    now[0] += 600
    stop = runtime.stop_condition()
    assert isinstance(stop, session.ProgressStalled)
    assert runtime.watchdog_stale_checks == 3


def test_checkpoint_progress_resets_watchdog_stagnation():
    now = [21 * 3600.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        clock=lambda: now[0],
        watch_usd=20,
    )
    runtime.set_hourly_rate(1.0)
    runtime.adopt_batch(
        {
            "batch_id": "b-1",
            "budget": {"max_minutes": 3600, "max_usd": 100, "idle_minutes": 15},
        }
    )
    assert runtime.stop_condition() is None
    now[0] += 600
    assert runtime.stop_condition() is None
    assert runtime.watchdog_stale_checks == 1
    runtime.record_checkpoint()
    now[0] += 600
    assert runtime.stop_condition() is None
    assert runtime.watchdog_stale_checks == 0


def test_classified_startup_errors_recycle_immediately_without_150_minute_wait():
    now = [1.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        clock=lambda: now[0],
        startup_timeout_seconds=150 * 60,
    )
    runtime.begin_startup()
    cases = (
        ("preflight_failed:nvidia_smi", "preflight_failed"),
        ("capability_mismatch:torch_abi", "capability_mismatch"),
        ("comfy_startup_failed:exit_1", "comfy_startup_failed"),
        ("undefined symbol: cudaLaunchKernelExC", "capability_mismatch"),
    )
    for error, expected in cases:
        runtime.last_error = error
        runtime.stop_requested = None
        stop = runtime.limit_reached()
        assert isinstance(stop, session.ClassifiedStartupFailure), error
        assert stop.failure_class == expected
        assert stop.minutes == 0
        fields = runtime.heartbeat_fields({})
        assert fields["boot_stage"] == expected
        assert fields["failure_class"] == expected
        assert fields["progress"]["stage"] == expected


def test_unknown_startup_state_keeps_150_minute_fallback():
    now = [30 * 60.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        clock=lambda: now[0],
        watch_usd=1,
        startup_timeout_seconds=150 * 60,
    )
    runtime.set_hourly_rate(10.0)
    downloaded = [100]
    runtime.begin_startup(
        probe=lambda: {"downloaded_bytes": downloaded[0]}
    )
    runtime.set_startup_subphase("unknown")
    runtime.observe_progress("startup_stage:waiting_host")
    runtime.observe_progress("startup_health:waiting")
    assert runtime.stop_condition() is None
    for _ in range(3):
        now[0] += 600
        downloaded[0] += 1024
        runtime.sample_startup_progress()
        assert runtime.stop_condition() is None
    assert runtime.phase == "startup"
    now[0] = 149 * 60
    assert runtime.limit_reached() is None
    now[0] = 150 * 60
    stop = runtime.limit_reached()
    assert type(stop) is session.StartupTimeout
    assert stop.minutes == 150
    assert stop.limit == "startup_timeout"


def test_startup_does_not_use_three_check_production_stall_rule():
    now = [30 * 60.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        clock=lambda: now[0],
        watch_usd=1,
        startup_timeout_seconds=150 * 60,
    )
    runtime.set_hourly_rate(10.0)
    runtime.begin_startup()
    runtime.set_startup_subphase("unknown")
    assert runtime.stop_condition() is None
    for _ in range(3):
        now[0] += 600
        assert runtime.stop_condition() is None
    assert runtime.watchdog_stale_checks == 3
    runtime.observe_progress("production_started:E01-01")
    assert runtime.phase == "production"
    assert runtime.watchdog_stale_checks == 0


def test_weight_byte_probe_counts_incomplete_downloads(tmp_path):
    models = tmp_path / "models"
    (models / "done").mkdir(parents=True)
    (models / "done" / "model.safetensors").write_bytes(b"x" * 11)
    (models / ".cache").mkdir()
    (models / ".cache" / "model.incomplete").write_bytes(b"x" * 13)
    assert weights.runtime_weight_bytes(tmp_path) == 24


def test_run_anim_progress_requires_qc_and_successful_r2_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(
        session,
        "_board_shots",
        lambda _root: [{"id": "E01-01", "duration": 8}],
    )
    def submit(_router, _shot, _root, dest, progress=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * 5000)

    monkeypatch.setattr(session, "_submit_h3_gpu", submit)
    monkeypatch.setattr(session, "_normalize_h3_for_qc", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        session,
        "_probe_video",
        lambda _path: {
            "width": 1280,
            "height": 720,
            "duration": 8,
            "frames": 192,
            "size_bytes": 5000,
        },
    )
    monkeypatch.setattr(session, "incremental_qc_segment", lambda *_args, **_kwargs: "pass")
    monkeypatch.setattr(session, "mark_completed_passing", lambda *_args, **_kwargs: None)
    def fake_extract(_video, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x89PNG" + b"x" * 64)
        return dest

    monkeypatch.setattr(session, "extract_last_frame", fake_extract)
    monkeypatch.setattr(
        session,
        "put_file",
        lambda key, _path, _content_type: {"ok": True, "key": key},
    )
    monkeypatch.setattr(
        session,
        "checkpoint_and_upload",
        lambda _conn, dest: Path(dest),
    )
    events = []
    result = session.run_anim(
        "story-1",
        tmp_path,
        object(),
        router=object(),
        progress=events.append,
    )
    assert result["generated"] == ["E01-01"]
    assert "checkpoint_saved" in events
    assert "shot_uploaded:E01-01" in events


def test_best_effort_teardown_uploads_latest_stable_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "story.sqlite").write_bytes(b"checkpoint")
    seen = {}

    def upload(key, path, content_type):
        seen.update(key=key, path=path, content_type=content_type)
        return {"ok": True, "key": key}

    monkeypatch.setattr(session, "put_file", upload)
    result = session.best_effort_upload_checkpoint("story-1", tmp_path)
    assert result["ok"] is True
    assert seen["key"] == "stories/story-1/story.sqlite"
    assert seen["path"] == tmp_path / "story.sqlite"


def test_idle_requires_no_job_and_empty_comfy_queue():
    now = [900.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        idle_since=0.0,
        clock=lambda: now[0],
    )
    runtime.budget = session.BatchBudget(idle_minutes=15)
    runtime.job_active = True
    assert not runtime.idle_expired(comfy_inflight=0)
    runtime.job_active = False
    assert not runtime.idle_expired(comfy_inflight=1)
    assert not runtime.idle_expired(comfy_inflight=None)
    assert runtime.idle_expired(comfy_inflight=0)


def test_comfy_inflight_counts_running_and_pending_prompts():
    router = session.ComfyRouter(
        "http://127.0.0.1:8199",
        opener=lambda _req: {
            "queue_running": [["running"]],
            "queue_pending": [["one"], ["two"]],
        },
    )
    assert router.inflight_count() == 3
    malformed = session.ComfyRouter(
        "http://127.0.0.1:8199",
        opener=lambda _req: {"queue_running": None, "queue_pending": "unknown"},
    )
    with pytest.raises(RuntimeError):
        malformed.inflight_count()


def test_lease_age_includes_pre_agent_boot_time():
    now = datetime(2026, 9, 7, 14, 0, tzinfo=timezone.utc)
    age = session.lease_age_seconds_from_work(
        {
            "workers": [
                {
                    "vast_instance_id": "123",
                    "started_at": "2026-09-07T13:30:00Z",
                }
            ]
        },
        "123",
        now=now,
    )
    assert age == 1800


def test_new_batch_episode_rows_do_not_reset_completed_resume(tmp_path, monkeypatch):
    for code, shot_id in (("EP001", "E01-01"), ("EP002", "E02-01")):
        board = tmp_path / "episodes" / code / "board.json"
        board.parent.mkdir(parents=True, exist_ok=True)
        board.write_text(
            f'{{"shots": [{{"id": "{shot_id}", "duration": 8}}]}}',
            encoding="utf-8",
        )
    monkeypatch.setattr(session, "EP", "EP001")
    conn = session.ensure_story_db("story-1", tmp_path)
    conn.execute("UPDATE segments SET status = 'completed' WHERE id = 'E01-01'")
    conn.commit()
    monkeypatch.setattr(session, "EP", "EP002")
    session.ensure_episode_rows("story-1", tmp_path, conn)
    rows = conn.execute(
        "SELECT id, episode_code, status FROM segments ORDER BY id"
    ).fetchall()
    assert [(row["id"], row["episode_code"], row["status"]) for row in rows] == [
        ("E01-01", "EP001", "completed"),
        ("E02-01", "EP002", "prepared"),
    ]


def test_batch_runs_every_episode_and_reports(monkeypatch):
    calls = []

    def run_episode(story_id, **kwargs):
        code = kwargs["episode_code"]
        calls.append(
            (story_id, code, tuple(kwargs["langs"]), kwargs["finish_story"])
        )
        return {
            "remaining": 0,
            "anim": {
                "resolution": {
                    "width": 1024,
                    "height": 576,
                    "downgraded": False,
                    "reason": "primary_1024x576",
                }
            },
            "compose": {
                "final_keys": [
                    f"stories/story-1/episodes/{code}/final/{code}.zh.mp4"
                ],
                "subtitle_keys": [
                    f"stories/story-1/episodes/{code}/final/{code}.zh.srt"
                ],
                "shorts": {"keys": [], "errors": []},
            },
        }

    monkeypatch.setattr(session, "run_gpu_episode", run_episode)
    runtime = session.LeaseRuntime(instance_id="123")
    reports = []
    summary = session.run_gpu_batch(
        {
            "batch_id": "b-1",
            "story_id": "story-1",
            "episodes": [
                {"episode_code": "EP001", "langs": ["zh"]},
                {"episode_code": "EP002", "langs": ["en", "ja"]},
            ],
            "budget": {"max_minutes": 60, "max_usd": 5, "idle_minutes": 2},
        },
        runtime,
        claim=lambda story, episode: {"ok": True, "story_id": story, "episode": episode},
        report=lambda body: reports.append(body) or {"ok": True},
    )
    assert calls == [
        ("story-1", "EP001", ("zh",), False),
        ("story-1", "EP002", ("en", "ja"), False),
    ]
    assert [body["episode_code"] for body in reports] == ["EP001", "EP002"]
    assert all(body["status"] == "done" for body in reports)
    assert reports[0]["artifacts"]["h3_resolution"]["width"] == 1024
    assert reports[0]["artifacts"]["final"] == (
        "stories/story-1/episodes/EP001/final/EP001.zh.mp4"
    )
    assert reports[0]["artifacts"]["subtitles"] == [
        "stories/story-1/episodes/EP001/final/EP001.zh.srt"
    ]
    assert summary["episodes_done"] == 2
    assert summary["destroy_reason"] == "success"
    assert runtime.status == "idle"


def test_batch_fail_closes_on_longlive_fouroversix(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("longlive_fail_closed: fouroversix CUDA extension missing and nvcc is not on this image")

    monkeypatch.setattr(session, "run_gpu_episode", boom)
    runtime = session.LeaseRuntime(instance_id="123")
    reports = []
    summary = session.run_gpu_batch(
        {
            "batch_id": "b-1",
            "story_id": "story-1",
            "episodes": [{"episode_code": "EP001"}, {"episode_code": "EP002"}],
            "budget": {"max_minutes": 60, "max_usd": 5, "idle_minutes": 2},
        },
        runtime,
        report=lambda body: reports.append(body) or {"ok": True},
    )
    assert summary["destroy_reason"] == "explicit_abort"
    assert summary["destroy_error"] is None
    assert reports[0]["status"] == "failed"
    assert "longlive_fail_closed" in str(reports[0]["error"])
    assert len(reports) == 1


def test_batch_fail_closes_on_nvfp4_dtype(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError(
            "h3 s1 execution_error: AttributeError: module 'torch' has no attribute "
            "'float4_e2m1fn_x2' node=SamplerCustomAdvanced:11"
        )

    monkeypatch.setattr(session, "run_gpu_episode", boom)
    checkpoints = []
    monkeypatch.setattr(
        session,
        "best_effort_upload_checkpoint",
        lambda story_id, root: checkpoints.append((story_id, root)) or {"ok": True},
    )
    runtime = session.LeaseRuntime(instance_id="123")
    reports = []
    summary = session.run_gpu_batch(
        {
            "batch_id": "b-1",
            "story_id": "story-1",
            "episodes": [{"episode_code": "EP001"}, {"episode_code": "EP002"}],
            "budget": {"max_minutes": 60, "max_usd": 5, "idle_minutes": 2},
        },
        runtime,
        report=lambda body: reports.append(body) or {"ok": True},
    )
    assert summary["destroy_reason"] == "explicit_abort"
    assert summary["limit"] == "capability_mismatch"
    assert checkpoints
    assert len(reports) == 1
    assert "float4_e2m1fn_x2" in str(reports[0]["error"])


def test_extract_execution_error_skips_cached_prefix():
    from gpu_worker.comfy import extract_execution_error, format_execution_error
    from gpu_worker.preflight import recycle_failure_class

    hist = {
        "pid-1": {
            "status": {
                "status_str": "error",
                "completed": True,
                "messages": [
                    ["execution_start", {"prompt_id": "pid-1"}],
                    ["execution_cached", {"nodes": [str(i) for i in range(80)], "prompt_id": "pid-1"}],
                    [
                        "execution_error",
                        {
                            "exception_type": "AttributeError",
                            "exception_message": "module 'torch' has no attribute 'float4_e2m1fn_x2'",
                            "node_id": "11",
                            "node_type": "SamplerCustomAdvanced",
                            "traceback": ["x" * 4000],
                            "current_inputs": {"noise": "x" * 4000},
                        },
                    ],
                ],
            }
        }
    }
    err = extract_execution_error(hist, "pid-1")
    assert err is not None
    assert err["exception_type"] == "AttributeError"
    assert "float4_e2m1fn_x2" in err["exception_message"]
    assert err["node_type"] == "SamplerCustomAdvanced"
    assert err["node_id"] == "11"
    dumped = json.dumps(hist["pid-1"]["status"])[:300]
    assert "float4" not in dumped
    formatted = format_execution_error(err, "E01-01")
    assert "float4_e2m1fn_x2" in formatted
    assert recycle_failure_class(formatted) == "capability_mismatch"


def test_batch_preserves_box_on_comfy_install_failure(monkeypatch):
    monkeypatch.setattr(
        session,
        "run_gpu_episode",
        lambda *_args, **_kwargs: {
            "remaining": 1,
            "anim": {"failed": [{"id": "s1", "error": "no_comfy_router"}]},
            "compose": None,
        },
    )
    runtime = session.LeaseRuntime(instance_id="123")
    summary = session.run_gpu_batch(
        {
            "batch_id": "b-1",
            "story_id": "story-1",
            "episodes": [{"episode_code": "EP001"}],
            "budget": {"max_minutes": 60, "max_usd": 5, "idle_minutes": 2},
        },
        runtime,
        report=lambda _body: {"ok": True},
    )
    assert summary["destroy_reason"] is None
    assert summary["destroy_error"] == "install_failure"
    assert runtime.status == "busy"
    assert runtime.idle_seconds() == 0
    refused = session.destroy_self("123", "explicit_abort", summary["destroy_error"])
    assert refused["refused"] is True


def test_stalled_batch_attempts_checkpoint_before_teardown(tmp_path, monkeypatch):
    monkeypatch.setattr(
        session,
        "run_gpu_episode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(session.ProgressStalled()),
    )
    checkpoints = []
    monkeypatch.setattr(
        session,
        "best_effort_upload_checkpoint",
        lambda story_id, root: checkpoints.append((story_id, root))
        or {"ok": True},
    )
    runtime = session.LeaseRuntime(instance_id="123")
    summary = session.run_gpu_batch(
        {
            "batch_id": "b-1",
            "story_id": "story-1",
            "episodes": [{"episode_code": "EP001"}],
            "budget": {"max_minutes": 3600, "max_usd": 100, "idle_minutes": 15},
        },
        runtime,
        root=tmp_path,
        report=lambda _body: {"ok": True},
    )
    assert checkpoints == [("story-1", tmp_path)]
    assert summary["limit"] == "progress_stalled"
    assert summary["stop_checkpoint"] == {"ok": True}
    assert summary["destroy_reason"] == "explicit_abort"


def test_startup_idle_seconds_stay_zero_during_weight_download():
    now = [0.0]
    runtime = session.LeaseRuntime(
        instance_id="123",
        started_at=0.0,
        clock=lambda: now[0],
    )
    runtime.begin_startup()
    now[0] = 20 * 60
    assert runtime.status == "booting"
    assert runtime.phase == "startup"
    assert runtime.idle_seconds() == 0
    assert not runtime.idle_timer_expired()
    assert not runtime.idle_expired(comfy_inflight=0)
    fields = runtime.heartbeat_fields(
        {"connected": False, "hostname": None, "checked_at": "2026-09-08T00:00:00Z"}
    )
    assert fields["idle_seconds"] == 0
    assert "connected" not in fields["tunnel"]


def test_compose_gate_blocks_zero_shots():
    from gpu_worker.session import compose_gate

    remaining, allow = compose_gate({"shots_total": 15, "shots_done": 0, "failed": [{"id": "s001"}]})
    assert remaining == 15
    assert allow is False
    remaining, allow = compose_gate({"shots_total": 0, "shots_done": 0, "failed": []})
    assert remaining >= 1
    assert allow is False
    remaining, allow = compose_gate({"shots_total": 15, "shots_done": 15, "failed": []})
    assert remaining == 0
    assert allow is True


def test_remaining_shots_keep_lease_busy_and_do_not_destroy(monkeypatch):
    monkeypatch.setattr(
        session,
        "run_gpu_episode",
        lambda *_args, **_kwargs: {
            "remaining": 15,
            "anim": {"failed": [{"id": "s001", "error": "comfy /prompt HTTP 400"}]},
            "compose": None,
        },
    )
    runtime = session.LeaseRuntime(instance_id="123")
    summary = session.run_gpu_batch(
        {
            "batch_id": "b-1",
            "story_id": "story-1",
            "episodes": [{"episode_code": "EP001"}],
            "budget": {"max_minutes": 3600, "max_usd": 40, "idle_minutes": 15},
        },
        runtime,
        report=lambda _body: {"ok": True},
    )
    assert summary["destroy_reason"] is None
    assert summary["results"][0]["status"] == "running"
    assert "remaining=15" in str(summary["results"][0]["error"])
    assert runtime.status == "busy"
    assert runtime.idle_seconds() == 0
    assert not runtime.idle_expired(comfy_inflight=0)


def test_gpu_cycle_idle_reason_archived_remaining_and_until_stage():
    assert session.gpu_cycle_idle_reason({"status": "archived"}) == "archived"
    assert session.gpu_cycle_idle_reason({"episode_status": "archived"}) == "archived"
    assert session.gpu_cycle_idle_reason({"remaining": 0}) == "remaining=0"
    assert session.gpu_cycle_idle_reason({"until_stage": "archive"}) == "until_stage"
    assert session.gpu_cycle_idle_reason({"gpu_done": True}) == "until_stage"
    assert session.gpu_cycle_idle_reason({"until_stage": "compose", "stage": "anim"}) is None
    assert session.gpu_cycle_idle_reason({"remaining": 15}) is None


def test_archived_or_remaining_zero_marks_idle_without_skip_cycle(monkeypatch):
    calls = []

    def run_episode(*_args, **_kwargs):
        calls.append(_kwargs.get("episode_code"))
        return {"remaining": 0, "compose": {"final_keys": ["stories/s/episodes/EP001/final/EP001.zh.mp4"]}}

    monkeypatch.setattr(session, "run_gpu_episode", run_episode)
    now = [100.0]
    runtime = session.LeaseRuntime(instance_id="50319899", started_at=0.0, clock=lambda: now[0])
    batch = {
        "batch_id": "b-1",
        "story_id": "story-tiangou",
        "episodes": [{"episode_code": "EP001", "langs": ["zh", "en", "ja"]}],
        "budget": {"max_minutes": 3600, "max_usd": 40, "idle_minutes": 15},
    }
    first = session.run_gpu_batch(batch, runtime, report=lambda _body: {"ok": True})
    assert calls == ["EP001"]
    assert first["skip_cycle"] is False
    assert first["destroy_reason"] == "success"
    assert runtime.status == "idle"
    idle_since = runtime.idle_since
    now[0] = 160.0
    assert runtime.idle_seconds() == 60

    second = session.run_gpu_batch(batch, runtime, report=lambda _body: {"ok": True})
    assert calls == ["EP001"]
    assert second["skip_cycle"] is True
    assert second["destroy_reason"] == "success"
    assert runtime.status == "idle"
    assert runtime.idle_since == idle_since
    now[0] = 220.0
    assert runtime.idle_seconds() == 120
    assert runtime.heartbeat_fields({})["idle_seconds"] == 120

    archived_calls = []
    monkeypatch.setattr(
        session,
        "run_gpu_episode",
        lambda *_args, **_kwargs: archived_calls.append(1) or {"remaining": 1},
    )
    archived = session.run_gpu_batch(
        {
            "batch_id": "b-2",
            "story_id": "story-archived",
            "episodes": [{"episode_code": "EP001", "status": "archived", "until_stage": "archive"}],
            "budget": {"max_minutes": 3600, "max_usd": 40, "idle_minutes": 15},
        },
        runtime,
        report=lambda _body: {"ok": True},
    )
    assert archived_calls == []
    assert archived["skip_cycle"] is True
    assert runtime.status == "idle"


def test_run_gpu_episode_skips_design_anim_qc_compose_when_finals_exist(tmp_path, monkeypatch):
    final = tmp_path / "episodes" / "EP001" / "final"
    final.mkdir(parents=True)
    for lang in ("zh", "en", "ja"):
        (final / f"EP001.{lang}.mp4").write_bytes(b"video")
    pulled = []
    monkeypatch.setattr(session, "pull_story", lambda *_args, **_kwargs: pulled.append("pull") or [])
    monkeypatch.setattr(
        session,
        "generate_missing_stills",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("skip-cycle stills")),
    )
    monkeypatch.setattr(
        session,
        "run_anim",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("skip-cycle anim")),
    )
    monkeypatch.setattr(
        session,
        "run_compose",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("skip-cycle compose")),
    )
    out = session.run_gpu_episode("story-tiangou", root=tmp_path, episode_code="EP001")
    assert pulled == []
    assert out["remaining"] == 0
    assert out["skipped_cycle"] is True
    assert out["skip_reason"] == "remaining=0"
    archived = session.run_gpu_episode(
        "story-tiangou",
        root=tmp_path,
        episode_code="EP001",
        story_status="archived",
        until_stage="archive",
    )
    assert archived["skip_reason"] == "archived"
    assert archived["remaining"] == 0


def test_run_gpu_episode_compose_error_keeps_remaining_and_does_not_finish(tmp_path, monkeypatch):
    jobs: list[tuple[str, str, str | None]] = []

    class Control:
        def job(self, _story, stage, status, error=None, episode_code=None, **_kwargs):
            jobs.append((stage, status, error))

        def action(self, *_args, **_kwargs):
            raise AssertionError("finish must not run after compose failure")

    monkeypatch.setattr(session, "gpu_cycle_idle_reason", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(session, "episode_finals_ready", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(session, "pull_story", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(session, "run_pre_gpu_if_needed", lambda *_args, **_kwargs: {"skipped": True})
    monkeypatch.setattr(session, "ensure_story_db", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(session, "ComfyRouter", lambda: (_ for _ in ()).throw(RuntimeError("no comfy")))
    monkeypatch.setattr(session, "generate_missing_stills", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(session, "join_h3_weights", lambda: None)
    monkeypatch.setattr(session, "ensure_h3_dits_for_shots", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(session, "unload_still_models", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        session,
        "run_anim",
        lambda *_args, **_kwargs: {"shots_total": 15, "shots_done": 15, "failed": []},
    )
    monkeypatch.setattr(
        session,
        "run_compose",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("DialogueMissingError:compose: 5 zh line wavs on disk, need >= 15")
        ),
    )
    monkeypatch.setattr(session, "_checkpoint_story", lambda *_args, **_kwargs: None)
    out = session.run_gpu_episode(
        "story-tiangou",
        root=tmp_path,
        episode_code="EP001",
        control=Control(),
        finish_story=True,
        skip_shorts=True,
    )
    assert out["remaining"] == 1
    assert out["compose"]["ok"] is False
    assert "5 zh line wavs" in str(out["compose"]["error"])
    assert ("compose", "failed", out["compose"]["error"]) in jobs
    assert not any(stage == "compose" and status == "succeeded" for stage, status, _err in jobs)


def test_reuse_scene_plate_as_fl2va_keyframe(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "EP", "EP001")
    plate = tmp_path / "assets" / "scenes" / "loc_dorm" / "plate_base.png"
    plate.parent.mkdir(parents=True)
    plate.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 12_000)
    shot = {"id": "s001", "location_id": "loc_dorm", "h3_mode": "fl2va_first"}
    assert session._reuse_scene_plate_as_fl2va_keyframe(tmp_path, shot) is True
    kf = tmp_path / "episodes" / "EP001" / "keyframes" / "s001" / "f1.png"
    assert kf.is_file()
    assert kf.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_heartbeat_reports_tunnel_down_only_after_startup():
    runtime = session.LeaseRuntime(instance_id="123")
    runtime.begin_startup()
    booting = runtime.heartbeat_fields({"connected": False})
    assert "connected" not in booting["tunnel"]
    runtime.complete_startup(True)
    ready = runtime.heartbeat_fields({"connected": False})
    assert ready["tunnel"]["connected"] is False


def test_compose_uploads_video_and_subtitles_to_episode_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "EP", "EP001")
    final = tmp_path / "episodes" / "EP001" / "final"
    final.mkdir(parents=True)
    shot = tmp_path / "episodes" / "EP001" / "shots" / "s001" / "v001.mp4"
    shot.parent.mkdir(parents=True)
    shot.write_bytes(b"x" * 5000)
    for lang in ("zh", "ja"):
        (final / f"EP001.{lang}.mp4").write_bytes(b"video")
        (final / f"EP001.{lang}.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nline\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(session, "collect_shot_paths", lambda *_args: [("s001", str(shot))])
    monkeypatch.setattr(session, "_board_shots", lambda _root: [])
    monkeypatch.setattr(
        session,
        "prepare_episode_sfx",
        lambda *_args, **_kwargs: {
            "sfx_clips": [],
            "ambience_clips": [],
            "sfx_cues": [],
            "results": {},
        },
    )
    monkeypatch.setattr(
        session,
        "compose_episode_audio",
        lambda *_args, **_kwargs: ({}, SimpleNamespace(sfx=(), episode_code="EP001")),
    )
    monkeypatch.setattr(
        session,
        "build_compose_plan",
        lambda *_args, **_kwargs: SimpleNamespace(
            encode=[],
            muxes=[],
            srt_outputs=[],
        ),
    )
    monkeypatch.setattr(session, "_run_checked", lambda *_args, **_kwargs: None)
    uploaded = []

    def upload(key, _path, content_type):
        uploaded.append((key, content_type))
        return {"ok": True, "key": key}

    monkeypatch.setattr(session, "put_file", upload)
    result = session.run_compose(
        "story-1",
        tmp_path,
        object(),
        langs=("zh", "ja"),
    )
    assert result["final_keys"] == [
        "stories/story-1/episodes/EP001/final/EP001.zh.mp4",
        "stories/story-1/episodes/EP001/final/EP001.ja.mp4",
    ]
    assert result["subtitle_keys"] == [
        "stories/story-1/episodes/EP001/final/EP001.zh.srt",
        "stories/story-1/episodes/EP001/final/EP001.ja.srt",
    ]
    assert uploaded[:4] == [
        (
            "stories/story-1/episodes/EP001/final/EP001.zh.mp4",
            "video/mp4",
        ),
        (
            "stories/story-1/episodes/EP001/final/EP001.zh.srt",
            "application/x-subrip",
        ),
        (
            "stories/story-1/episodes/EP001/final/EP001.ja.mp4",
            "video/mp4",
        ),
        (
            "stories/story-1/episodes/EP001/final/EP001.ja.srt",
            "application/x-subrip",
        ),
    ]
    assert (
        "stories/story-1/episodes/EP001/audio/sfx_evidence.json",
        "application/json",
    ) in uploaded
    assert (
        "stories/story-1/episodes/EP001/audio/sfx_credits.json",
        "application/json",
    ) in uploaded
    assert "stories/story-1/episodes/EP001/audio/sfx_evidence.json" in (
        result.get("evidence_keys") or []
    )


def _compose_harness(tmp_path, monkeypatch, pairs, shots):
    """Stub out ffmpeg/ffprobe/R2 and capture what run_compose hands the mixer."""
    monkeypatch.setattr(session, "EP", "EP001")
    final = tmp_path / "episodes" / "EP001" / "final"
    final.mkdir(parents=True)
    for suffix in ("mp4", "srt"):
        (final / f"EP001.zh.{suffix}").write_bytes(b"x")
    monkeypatch.setattr(session, "collect_shot_paths", lambda *_args: pairs)
    monkeypatch.setattr(session, "_board_shots", lambda _root: shots)
    monkeypatch.setattr(session, "_probe_video", lambda path: {"duration": 8.0})
    monkeypatch.setattr(
        session, "_probe_audio", lambda path: {"has_audio_stream": False, "audio_rms": 0.0}
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        session,
        "_run_checked",
        lambda command, **_kwargs: commands.append(list(command)),
    )
    captured: dict[str, object] = {}
    prepare_calls: list[tuple] = []

    def prepare_stub(passed_shots, passed_root, episode_code, **kwargs):
        prepare_calls.append((passed_shots, passed_root, episode_code, kwargs))
        captured["prepare_root"] = passed_root
        return {
            "sfx_clips": [],
            "ambience_clips": [],
            "sfx_cues": [],
            "results": {},
        }

    def compose_stub(_work, _ep, mixed_shots, **kwargs):
        captured["shots"] = [dict(s) for s in mixed_shots]
        captured["clip_durations"] = dict(kwargs.get("clip_durations") or {})
        captured["video_paths"] = list(kwargs.get("video_paths") or [])
        captured["sfx_clips"] = list(kwargs.get("sfx_clips") or [])
        captured["ambience_clips"] = list(kwargs.get("ambience_clips") or [])
        captured["sfx_cues"] = list(kwargs.get("sfx_cues") or [])
        return ({}, SimpleNamespace(sfx=(), episode_code=_ep))

    monkeypatch.setattr(session, "prepare_episode_sfx", prepare_stub)
    monkeypatch.setattr(session, "compose_episode_audio", compose_stub)
    captured["prepare_calls"] = prepare_calls
    monkeypatch.setattr(
        session,
        "build_compose_plan",
        lambda *_args, **_kwargs: SimpleNamespace(encode=[], muxes=[], srt_outputs=[]),
    )
    monkeypatch.setattr(
        session, "put_file", lambda key, _path, _content_type: {"ok": True, "key": key}
    )
    return captured, commands


def test_chain_trim_decrements_shot_duration_before_mixing(tmp_path, monkeypatch):
    """Dropping the duplicated join frame without shortening the timeline drifts 41.7ms per join."""
    shots = [
        {"id": "s001", "chain_id": "c1", "duration": 8.0},
        {"id": "s002", "chain_id": "c1", "duration": 8.0},
        {"id": "s003", "chain_id": "c1", "duration": 8.0},
        {"id": "s004", "chain_id": "c2", "duration": 8.0},
    ]
    clips = []
    for shot in shots:
        clip = tmp_path / "episodes" / "EP001" / "shots" / shot["id"] / "generation-001.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"x" * 5000)
        clips.append(clip)
    pairs = [
        (shot["id"], f"episodes/EP001/shots/{shot['id']}/generation-001.mp4") for shot in shots
    ]
    captured, commands = _compose_harness(tmp_path, monkeypatch, pairs, shots)
    session.run_compose("story-1", tmp_path, object(), langs=("zh",))

    frame = 1.0 / session.VIDEO_FPS
    durations = {s["id"]: s["duration"] for s in captured["shots"]}
    # First clip of each chain keeps its length; later joins lose exactly one frame.
    assert durations["s001"] == 8.0
    assert durations["s004"] == 8.0
    assert abs(durations["s002"] - (8.0 - frame)) < 1e-9
    assert abs(durations["s003"] - (8.0 - frame)) < 1e-9
    assert captured["clip_durations"] == durations
    trims = [c for c in commands if any("select=gte" in str(x) for x in c)]
    assert len(trims) == 2
    # The trimmed files, not the originals, are what gets concatenated.
    assert [Path(p).parent.name for p in captured["video_paths"]] == [
        "s001",
        "compose_trim",
        "compose_trim",
        "s004",
    ]


def test_compose_pairs_shots_to_clips_by_id_not_position(tmp_path, monkeypatch):
    """COMPOSE_ALLOW_MISSING_SHOTS drops a shot; zip() then misaligned every later clip."""
    shots = [
        {"id": "s001", "duration": 8.0, "line": {"zh": "一"}},
        {"id": "s002", "duration": 8.0, "line": {"zh": "二"}},
        {"id": "s003", "duration": 8.0, "line": {"zh": "三"}},
    ]
    for sid in ("s001", "s003"):
        clip = tmp_path / "episodes" / "EP001" / "shots" / sid / "generation-001.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"x" * 5000)
    pairs = [
        ("s001", "episodes/EP001/shots/s001/generation-001.mp4"),
        ("s003", "episodes/EP001/shots/s003/generation-001.mp4"),
    ]
    captured, _commands = _compose_harness(tmp_path, monkeypatch, pairs, shots)
    session.run_compose("story-1", tmp_path, object(), langs=("zh",))

    # s002 has no picture, so it leaves the timeline entirely instead of shifting
    # s003's dialogue onto s001's clip.
    assert [s["id"] for s in captured["shots"]] == ["s001", "s003"]
    assert [Path(p).parent.name for p in captured["video_paths"]] == ["s001", "s003"]


def test_run_compose_integrates_prepare_episode_sfx_into_compose_audio(tmp_path, monkeypatch):
    """Board SFX prep uses story root; compose receives clips/cues; cue-less boards still compose."""
    monkeypatch.setattr(session, "EP", "EP001")
    shots = [{"id": "s001", "duration": 8.0}]
    clip = tmp_path / "episodes" / "EP001" / "shots" / "s001" / "generation-001.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"x" * 5000)
    pairs = [("s001", "episodes/EP001/shots/s001/generation-001.mp4")]
    final = tmp_path / "episodes" / "EP001" / "final"
    final.mkdir(parents=True)
    (final / "EP001.zh.mp4").write_bytes(b"video")
    (final / "EP001.zh.srt").write_bytes(b"srt")

    monkeypatch.setattr(session, "collect_shot_paths", lambda *_args: pairs)
    monkeypatch.setattr(session, "_board_shots", lambda _root: shots)
    monkeypatch.setattr(session, "_probe_video", lambda _path: {"duration": 8.0})
    monkeypatch.setattr(
        session, "_probe_audio", lambda _path: {"has_audio_stream": False, "audio_rms": 0.0}
    )
    monkeypatch.setattr(
        session,
        "fit_clip_durations_to_speech",
        lambda _work, _shots, _langs, durations: dict(durations),
    )
    monkeypatch.setattr(session, "_run_checked", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        session,
        "build_compose_plan",
        lambda *_args, **_kwargs: SimpleNamespace(encode=[], muxes=[], srt_outputs=[]),
    )
    monkeypatch.setattr(
        session, "put_file", lambda key, _path, _content_type: {"ok": True, "key": key}
    )

    fake_sfx = {
        "sfx_clips": [(1.0, b"RIFFfake")],
        "ambience_clips": [(0.0, b"RIFFbed")],
        "sfx_cues": [
            {
                "cue_key": "door1",
                "bus": "sfx",
                "onset_s": 1.0,
                "duration_s": 0.5,
                "shared": False,
            }
        ],
        "results": {
            "door1": SimpleNamespace(
                status="resolved",
                provenance=SimpleNamespace(source="freesound", cached=True),
            )
        },
    }
    prepare_args: list[tuple] = []

    def prepare_stub(passed_shots, passed_root, episode_code, **kwargs):
        prepare_args.append((passed_shots, passed_root, episode_code, kwargs))
        before = kwargs.get("before_moss")
        if callable(before):
            before()
        return fake_sfx

    compose_calls: list[dict] = []

    def compose_stub(work, ep, _mixed_shots, **kwargs):
        compose_calls.append({"work": work, "ep": ep, "kwargs": dict(kwargs)})
        timeline = SimpleNamespace(
            sfx=(SimpleNamespace(cue_key="door1"),),
            episode_code=ep,
        )
        return ({"zh": str(work / "audio" / "master.zh.wav")}, timeline)

    handoff_calls: list[dict] = []
    restore_calls: list[dict] = []
    monkeypatch.setattr(session, "select_video_backend", lambda **_k: "h3")
    monkeypatch.setattr(
        session,
        "release_gpu_for_moss_sfx",
        lambda **kwargs: handoff_calls.append(kwargs) or {"ok": True},
    )
    monkeypatch.setattr(
        session,
        "restore_gpu_after_moss_sfx",
        lambda **kwargs: restore_calls.append(kwargs) or {"ok": True, "restored": False},
    )
    monkeypatch.setattr(session, "prepare_episode_sfx", prepare_stub)
    monkeypatch.setattr(session, "compose_episode_audio", compose_stub)

    result = session.run_compose("story-1", tmp_path, object(), langs=("zh",))

    assert len(prepare_args) == 1
    _shots, passed_root, ep_code, kw = prepare_args[0]
    assert passed_root == tmp_path
    assert passed_root != tmp_path / "episodes" / "EP001"
    assert ep_code == "EP001"
    assert kw.get("story_id") == "story-1"
    assert "clip_durations" in kw
    assert callable(kw.get("before_moss"))
    # H3 compose must not hand MOSS the LongLive claim path.
    assert kw.get("before_moss") is not session.stop_comfy_for_longlive
    assert handoff_calls == [{"backend": "h3"}]
    assert restore_calls == [{"backend": "h3"}]

    assert len(compose_calls) == 1
    ck = compose_calls[0]["kwargs"]
    assert ck["sfx_clips"] == fake_sfx["sfx_clips"]
    assert ck["ambience_clips"] == fake_sfx["ambience_clips"]
    assert ck["sfx_cues"] == fake_sfx["sfx_cues"]
    assert compose_calls[0]["work"] == tmp_path / "episodes" / "EP001"

    assert result["sfx"]["sfx_clip_count"] == 1
    assert result["sfx"]["ambience_clip_count"] == 1
    assert result["sfx"]["sfx_event_count"] == 1
    assert result["sfx"]["timeline_path"] == "audio/timeline.json"
    assert result["sfx"]["resolved_cues"] == [
        {"cue_key": "door1", "source": "freesound", "cached": True}
    ]

    # Cue-less boards: empty orchestrator output still composes.
    prepare_args.clear()
    compose_calls.clear()

    def empty_prepare_stub(passed_shots, passed_root, episode_code, **kwargs):
        prepare_args.append((passed_shots, passed_root, episode_code, kwargs))
        return {
            "sfx_clips": [],
            "ambience_clips": [],
            "sfx_cues": [],
            "results": {},
        }

    def empty_compose_stub(work, ep, _mixed_shots, **kwargs):
        compose_calls.append(dict(kwargs))
        return ({}, SimpleNamespace(sfx=(), episode_code=ep))

    monkeypatch.setattr(session, "prepare_episode_sfx", empty_prepare_stub)
    monkeypatch.setattr(session, "compose_episode_audio", empty_compose_stub)

    no_cue = session.run_compose("story-1", tmp_path, object(), langs=("zh",))
    assert compose_calls[0]["sfx_clips"] == []
    assert compose_calls[0]["ambience_clips"] == []
    assert compose_calls[0]["sfx_cues"] == []
    assert no_cue["sfx"]["sfx_clip_count"] == 0
    assert no_cue["sfx"]["resolved_cues"] == []


def test_compose_blocks_when_no_clip_matches_a_board_segment(tmp_path, monkeypatch):
    from anime_factory.compose import MissingShotError

    shots = [{"id": "s001", "duration": 8.0}]
    clip = tmp_path / "episodes" / "EP001" / "shots" / "other" / "generation-001.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"x" * 5000)
    pairs = [("zzz", "episodes/EP001/shots/other/generation-001.mp4")]
    _compose_harness(tmp_path, monkeypatch, pairs, shots)
    with pytest.raises(MissingShotError, match="board segment id"):
        session.run_compose("story-1", tmp_path, object(), langs=("zh",))


def test_compose_subtitle_fallback_uses_matching_language_dialogue(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "EP", "EP001")
    zh = tmp_path / "EP001.zh.srt"
    ja = tmp_path / "EP001.ja.srt"
    plan = SimpleNamespace(srt_outputs=[str(zh), str(ja)])
    session._ensure_compose_subtitles(
        plan,
        [
            {
                "duration": 8,
                "line": {"zh": "真实中文对白", "ja": "実際の日本語台詞"},
            }
        ],
        ("zh", "ja"),
    )
    assert "真实中文对白" in zh.read_text(encoding="utf-8")
    assert "実際の日本語台詞" in ja.read_text(encoding="utf-8")
    assert "真实中文对白" not in ja.read_text(encoding="utf-8")


def test_compose_subtitle_fallback_refuses_cross_language_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "EP", "EP001")
    plan = SimpleNamespace(srt_outputs=[str(tmp_path / "EP001.ja.srt")])
    with pytest.raises(RuntimeError, match="language: ja"):
        session._ensure_compose_subtitles(
            plan,
            [{"duration": 8, "line": {"zh": "只有中文"}}],
            ("ja",),
        )


def test_compose_refuses_silent_missing_subtitle(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "EP", "EP001")
    final = tmp_path / "episodes" / "EP001" / "final"
    final.mkdir(parents=True)
    shot = tmp_path / "episodes" / "EP001" / "shots" / "s001" / "v001.mp4"
    shot.parent.mkdir(parents=True)
    shot.write_bytes(b"x" * 5000)
    (final / "EP001.zh.mp4").write_bytes(b"video")
    monkeypatch.setattr(session, "collect_shot_paths", lambda *_args: [("s001", str(shot))])
    monkeypatch.setattr(session, "_board_shots", lambda _root: [])
    monkeypatch.setattr(
        session,
        "prepare_episode_sfx",
        lambda *_args, **_kwargs: {
            "sfx_clips": [],
            "ambience_clips": [],
            "sfx_cues": [],
            "results": {},
        },
    )
    monkeypatch.setattr(
        session,
        "compose_episode_audio",
        lambda *_args, **_kwargs: ({}, SimpleNamespace(sfx=(), episode_code="EP001")),
    )
    monkeypatch.setattr(
        session,
        "build_compose_plan",
        lambda *_args, **_kwargs: SimpleNamespace(
            encode=[],
            muxes=[],
            srt_outputs=[],
        ),
    )
    monkeypatch.setattr(session, "_run_checked", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        session,
        "put_file",
        lambda key, _path, _content_type: {"ok": True, "key": key},
    )
    with pytest.raises(RuntimeError, match=r"EP001\.zh\.srt"):
        session.run_compose(
            "story-1",
            tmp_path,
            object(),
            langs=("zh",),
        )


def test_compose_refuses_empty_mp4s(tmp_path, monkeypatch):
    from anime_factory.compose import MissingShotError

    monkeypatch.setattr(session, "EP", "EP001")
    monkeypatch.setattr(session, "collect_shot_paths", lambda *_args: [])
    monkeypatch.setattr(session, "_board_shots", lambda _root: [])
    with pytest.raises(MissingShotError, match="0 shot"):
        session.run_compose("story-1", tmp_path, object(), langs=("zh",))


def test_shorts_library_outputs_use_episode_r2_prefix(tmp_path, monkeypatch):
    fake = ModuleType("anime_factory.shorts")
    fake.ShortPlan = object
    fake.plan_shorts = lambda board, count, min_s, max_s: [{"index": 1}]

    def render_short(_plan, ep_root, lang, youtube_url=None):
        assert youtube_url is None
        out = ep_root / "shorts" / f"01.{lang}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"video")
        out.with_suffix(".jpg").write_bytes(b"thumb")
        return Path("shorts") / out.name

    fake.render_short = render_short
    monkeypatch.setitem(sys.modules, "anime_factory.shorts", fake)
    monkeypatch.setattr(session, "EP", "EP001")
    board = tmp_path / "episodes" / "EP001" / "board.json"
    board.parent.mkdir(parents=True)
    board.write_text('{"shots": []}', encoding="utf-8")
    uploaded = []
    monkeypatch.setattr(
        session,
        "put_file",
        lambda key, path, content_type: uploaded.append((key, content_type)) or {
            "ok": True,
            "key": key,
        },
    )
    result = session.render_episode_shorts("story-1", tmp_path, ("zh",))
    assert result["keys"] == [
        "stories/story-1/episodes/EP001/shorts/01.zh.mp4"
    ]
    assert uploaded == [
        ("stories/story-1/episodes/EP001/shorts/01.zh.mp4", "video/mp4"),
        ("stories/story-1/episodes/EP001/shorts/01.zh.jpg", "image/jpeg"),
    ]


def test_destroy_self_obeys_historical_safety_gate(monkeypatch):
    monkeypatch.setenv("VAST_DRY_RUN", "1")
    monkeypatch.setenv("ANIME_FACTORY_LIVE_VAST", "0")
    monkeypatch.delenv("CONTAINER_API_KEY", raising=False)
    allowed = session.destroy_self("123", "success")
    assert allowed["ok"] is True
    assert allowed["dry_run"] is True
    refused = session.destroy_self("123", "explicit_abort", error="tts_404")
    assert refused["ok"] is False
    assert refused["refused"] is True


def test_destroy_self_uses_vast_instance_key(monkeypatch):
    seen = {}

    class Client:
        def __init__(self, api_key, dry_run):
            seen["api_key"] = api_key
            seen["dry_run"] = dry_run

    monkeypatch.setenv("VAST_DRY_RUN", "0")
    monkeypatch.setenv("CONTAINER_API_KEY", "instance-key")
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.setattr(session, "VastClient", Client)
    monkeypatch.setattr(
        session,
        "destroy_if_allowed",
        lambda client, lease, reason, ids, error=None: {
            "success": True,
            "id": lease.instance_id,
        },
    )
    result = session.destroy_self("123", "success")
    assert result["ok"] is True
    assert result["destroyed"] is True
    assert seen == {"api_key": "instance-key", "dry_run": False}


def test_destroy_self_live_vast_overrides_image_dry_run(monkeypatch):
    seen = {}

    class Client:
        def __init__(self, api_key, dry_run):
            seen["api_key"] = api_key
            seen["dry_run"] = dry_run

    monkeypatch.setenv("VAST_DRY_RUN", "1")
    monkeypatch.setenv("ANIME_FACTORY_LIVE_VAST", "1")
    monkeypatch.setenv("VAST_API_KEY", "lease-key")
    monkeypatch.delenv("CONTAINER_API_KEY", raising=False)
    monkeypatch.setattr(session, "VastClient", Client)
    monkeypatch.setattr(
        session,
        "destroy_if_allowed",
        lambda client, lease, reason, ids, error=None: {
            "success": True,
            "id": lease.instance_id,
        },
    )
    result = session.destroy_self("51548736", "success")
    assert result["ok"] is True
    assert result["destroyed"] is True
    assert result["dry_run"] is False
    assert seen == {"api_key": "lease-key", "dry_run": False}


def test_self_destroy_dry_run_false_on_container_key(monkeypatch):
    monkeypatch.setenv("VAST_DRY_RUN", "1")
    monkeypatch.setenv("ANIME_FACTORY_LIVE_VAST", "0")
    monkeypatch.setenv("CONTAINER_API_KEY", "instance-key")
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    assert session.self_destroy_dry_run() is False


def test_filtered_comfy_requirements_skip_torch_family(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text(
        "torch\n"
        "torchvision>=0.1\n"
        "torchaudio\n"
        "einops\n"
        "comfy-kitchen==0.2.33\n",
        encoding="utf-8",
    )
    filtered = stack.filtered_comfy_requirements(req)
    text = filtered.read_text(encoding="utf-8")
    assert "torch" not in text.split()
    assert "einops" in text
    assert "comfy-kitchen==0.2.33" in text


def test_stop_comfy_for_longlive_waits_until_vram_drops(monkeypatch):
    calls = []
    used = {"mb": 9000}

    def fake_run(args, **_kwargs):
        calls.append(list(args))
        if args[:1] == ["pkill"]:
            used["mb"] = 400
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    monkeypatch.setattr(stack, "_gpu_memory_used_mb", lambda: used["mb"])
    monkeypatch.setattr(stack.time, "sleep", lambda _s: None)
    out = stack.stop_comfy_for_longlive()
    assert out["ok"] is True
    assert out["vram_used_mb"] == 400
    assert any(row[:2] == ["pkill", "-TERM"] for row in calls)
    assert any(row[:2] == ["pkill", "-KILL"] for row in calls)
    assert any("ComfyUI/main.py" in row for row in calls)
    assert stack.longlive_owns_gpu() is True
    stack.release_gpu_from_longlive()
    assert stack.longlive_owns_gpu() is False


def test_release_gpu_for_moss_sfx_h3_does_not_claim_longlive(monkeypatch):
    calls = []
    monkeypatch.setattr(stack.subprocess, "run", lambda args, **_k: calls.append(list(args)) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(stack, "_gpu_memory_used_mb", lambda: 200)
    monkeypatch.setattr(stack.time, "sleep", lambda _s: None)
    stack.release_gpu_from_longlive()
    out = stack.release_gpu_for_moss_sfx(backend="h3")
    assert out["ok"] is True
    assert out["claimed_longlive"] is False
    assert out["backend"] == "h3"
    assert stack.longlive_owns_gpu() is False
    assert any("ComfyUI/main.py" in row for row in calls)


def test_release_gpu_for_moss_sfx_longlive_clears_ownership(monkeypatch):
    monkeypatch.setattr(stack.subprocess, "run", lambda args, **_k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(stack, "_gpu_memory_used_mb", lambda: 100)
    monkeypatch.setattr(stack.time, "sleep", lambda _s: None)
    stack.claim_gpu_for_longlive()
    assert stack.longlive_owns_gpu() is True
    out = stack.release_gpu_for_moss_sfx(backend="longlive")
    assert out["ok"] is True
    assert stack.longlive_owns_gpu() is False
    assert out["claimed_longlive"] is False


def test_restore_gpu_after_moss_sfx_h3_allows_comfy_restart(monkeypatch):
    started = []
    monkeypatch.setattr(stack, "_port_open", lambda *_a, **_k: False)
    monkeypatch.setattr(stack, "start_comfy", lambda: started.append("comfy") or SimpleNamespace())
    stack.claim_gpu_for_longlive()  # simulate a bad prior handoff
    out = stack.restore_gpu_after_moss_sfx(backend="h3")
    assert out["ok"] is True
    assert out["restored"] is True
    assert started == ["comfy"]
    assert stack.longlive_owns_gpu() is False


def test_start_comfy_refuses_while_longlive_owns_gpu(monkeypatch):
    monkeypatch.setattr(stack, "_port_open", lambda *_a, **_k: False)
    started = []
    monkeypatch.setattr(stack, "_start_comfy_logged", lambda *_a, **_k: started.append(True))
    stack.claim_gpu_for_longlive()
    try:
        assert stack.start_comfy() is None
        assert started == []
    finally:
        stack.release_gpu_from_longlive()


def test_ensure_c_compiler_installs_python_dev_when_gcc_present(monkeypatch):
    calls = []
    monkeypatch.setattr(stack.shutil, "which", lambda name: "/usr/bin/gcc" if name == "gcc" else None)
    monkeypatch.setattr(stack, "_python_headers_available", lambda: False)
    monkeypatch.setattr(stack.shutil, "which", lambda name: "/usr/bin/apt-get" if name == "apt-get" else "/usr/bin/gcc")
    monkeypatch.setattr(stack.subprocess, "call", lambda args: calls.append(args) or 0)
    stack.ensure_c_compiler()
    assert calls[1][-1] == "python3-dev"


def test_stack_boot_surfaces_comfy_exit_before_router_wait(monkeypatch):
    events = []
    monkeypatch.setenv("AF_VIDEO_BACKEND", "h3")
    monkeypatch.setattr(stack, "run_hardware_preflight", lambda **_k: {"profile_id": "h3-comfy-cu130-sm120", "preflight": {"ok": True}})
    monkeypatch.setattr(stack, "ensure_torch", lambda: None)
    monkeypatch.setattr(stack, "ensure_c_compiler", lambda: None)
    monkeypatch.setattr(stack, "ensure_comfy_reqs", lambda: None)
    monkeypatch.setattr(stack, "ensure_noto_cjk_fonts", lambda: {"verified": True})
    monkeypatch.setattr(stack, "ensure_infer_schema_sitecustomize", lambda: None)
    monkeypatch.setattr(stack, "patch_kitchen_triton_optional", lambda: False)
    monkeypatch.setattr(stack, "patch_comfy_kitchen_for_torch26", lambda: [])
    monkeypatch.setattr(stack, "probe_comfy_torch_import", lambda **_k: {"ok": True})
    monkeypatch.setattr(stack, "ensure_still_weights", lambda *_a, **_k: {"kind": "stills"})
    monkeypatch.setattr(stack, "start_h3_weights_background", lambda *_a, **_k: None)
    monkeypatch.setattr(stack, "start_comfy", lambda: SimpleNamespace(pid=99, poll=lambda: 1))
    monkeypatch.setattr(
        stack,
        "wait_comfy_process",
        lambda *_a, **_k: {"ok": False, "exit_code": 1, "tail": "Python.h: No such file"},
    )
    monkeypatch.setattr(stack, "start_router", lambda: SimpleNamespace(pid=12))
    monkeypatch.setattr(stack, "start_tunnel", lambda: None)
    monkeypatch.setattr(stack.time, "sleep", lambda _s: None)
    monkeypatch.setattr(stack, "wait_router_ready", lambda **_k: True)
    monkeypatch.setattr(stack, "verify_tunnel_connection", lambda **_k: {"connected": True})
    monkeypatch.setattr(stack, "default_register_capabilities", lambda: {})
    result = stack.boot_gpu_stack(progress=events.append)
    assert result["router_ready"] is False
    assert "comfy_startup_failed" in str(result.get("comfy_error"))

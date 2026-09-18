from __future__ import annotations

import json

import pytest

from gpu_worker.images import (
    CAPABILITY_PROFILES,
    IMAGE_DIGEST_CONFIG_FIELDS,
    expected_image_digest,
    parse_content_digest,
    reported_image_digest,
    resolve_capability_profile,
)
from gpu_worker.preflight import (
    EXPECTED_ONSTART,
    MIN_DISK_GB,
    PreflightFailure,
    handshake_payload,
    probe_comfy_nodes,
    probe_nvfp4_runtime,
    production_stack_locked,
    recycle_failure_class,
    run_hardware_preflight,
    select_profile_id,
    validate_container_contract,
    validate_image_digest,
    validate_profile,
)
from gpu_worker.stack import StackAdapter
from gpu_worker import __main__ as worker_main


DIGEST_A = "sha256:" + ("a" * 64)
DIGEST_B = "sha256:" + ("b" * 64)


def _host(**overrides):
    host = {
        "arch": "x86_64",
        "sm": "sm_120",
        "vram_mb": 32_640,
        "driver_version": "580.65.06",
        "cuda_driver_ok": True,
        "disk_total_gb": 200,
        "disk_free_gb": 150,
        "mem_available_gb": 64,
    }
    host.update(overrides)
    return host


def _stack(**overrides):
    stack = {
        "torch": "2.13.0+cu130",
        "torchvision": "0.28.0+cu130",
        "torchaudio": "2.11.0+cu130",
    }
    stack.update(overrides)
    return stack


def test_capability_profiles_default_5090_and_longlive():
    assert "h3-comfy-cu130-sm120" in CAPABILITY_PROFILES
    assert "longlive-nvfp4-sm120" in CAPABILITY_PROFILES
    assert "sm_89" not in str(CAPABILITY_PROFILES)
    h3 = resolve_capability_profile("h3-comfy-cu130-sm120")
    assert h3.supported_sm == ("sm_120",)
    assert h3.min_disk_gb == 200
    assert h3.min_mem_gb == 64.0
    assert h3.require_flash_attn is False
    assert h3.expected_torch == "2.13.0+cu130"
    assert h3.expected_torchvision == "0.28.0+cu130"
    assert h3.expected_torchaudio == "2.11.0+cu130"
    ll = resolve_capability_profile("longlive-nvfp4-sm120")
    assert ll.require_flash_attn is True
    assert ll.require_fouroversix is True
    assert ll.expected_torch == "2.10.0+cu128"
    assert ll.expected_torchvision == "0.25.0+cu128"
    assert ll.expected_torchaudio == "2.10.0+cu128"
    assert ll.expected_flash_attn == "2.8.3"
    assert MIN_DISK_GB == 200
    assert "h3-comfy-cu128-sm89" not in CAPABILITY_PROFILES
    assert "4090_48" not in CAPABILITY_PROFILES


def test_validate_profile_longlive_stack_versions(monkeypatch):
    monkeypatch.setattr("gpu_worker.preflight._fouroversix_ok", lambda: True)
    stack = {
        "torch": "2.10.0+cu128",
        "torchvision": "0.25.0+cu128",
        "torchaudio": "2.10.0+cu128",
        "flash_attn": "2.8.3",
    }
    validate_profile("longlive-nvfp4-sm120", _host(), stack)


def test_validate_profile_longlive_rejects_stale_torch():
    with pytest.raises(PreflightFailure) as exc:
        validate_profile(
            "longlive-nvfp4-sm120",
            _host(),
            _stack(flash_attn="2.8.3"),
        )
    assert exc.value.code == "torch_version"


def test_validate_profile_longlive_rejects_wrong_flash_attn(monkeypatch):
    monkeypatch.setattr("gpu_worker.preflight._fouroversix_ok", lambda: True)
    stack = {
        "torch": "2.10.0+cu128",
        "torchvision": "0.25.0+cu128",
        "torchaudio": "2.10.0+cu128",
        "flash_attn": "2.7.4.post1",
    }
    with pytest.raises(PreflightFailure) as exc:
        validate_profile("longlive-nvfp4-sm120", _host(), stack)
    assert exc.value.code == "flash_attn_missing"


def test_select_profile_id_respects_video_backend(monkeypatch):
    monkeypatch.delenv("AF_GPU_PROFILE", raising=False)
    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    assert select_profile_id() == "longlive-nvfp4-sm120"
    monkeypatch.setenv("AF_VIDEO_BACKEND", "h3")
    assert select_profile_id() == "h3-comfy-cu130-sm120"


def test_expected_image_digest_ignores_tags_and_reads_env(monkeypatch):
    monkeypatch.delenv("AF_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("AF_EXPECTED_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("AF_REPORTED_IMAGE_DIGEST", raising=False)
    monkeypatch.setenv("VAST_GPU_IMAGE", "docker.io/aetherneo/anime-factory-gpu:sha-0090c77")
    monkeypatch.setenv("GPU_IMAGE", "docker.io/aetherneo/anime-factory-gpu:main")
    assert expected_image_digest() is None
    assert reported_image_digest() is None
    assert parse_content_digest("sha-0090c77") is None
    assert parse_content_digest("main") is None
    monkeypatch.setenv("AF_EXPECTED_IMAGE_DIGEST", DIGEST_A)
    assert expected_image_digest() == DIGEST_A


def test_validate_profile_rejects_low_vram():
    with pytest.raises(PreflightFailure) as exc:
        validate_profile("h3-comfy-cu130-sm120", _host(vram_mb=24_000), _stack())
    assert exc.value.failure_class == "capability_mismatch"
    assert exc.value.code == "vram_below_minimum"


def test_validate_profile_allows_formatted_200gb_volume():
    validate_profile(
        "h3-comfy-cu130-sm120",
        _host(disk_total_gb=196, disk_free_gb=170),
        _stack(),
    )


def test_validate_profile_rejects_truly_small_disk():
    with pytest.raises(PreflightFailure) as exc:
        validate_profile(
            "h3-comfy-cu130-sm120",
            _host(disk_total_gb=150, disk_free_gb=100),
            _stack(),
        )
    assert exc.value.code == "disk_low"


def test_run_hardware_preflight_smoke(monkeypatch):
    monkeypatch.setenv("AF_GPU_PROFILE", "h3-comfy-cu130-sm120")
    monkeypatch.delenv("AF_PRODUCTION_STACK", raising=False)
    monkeypatch.delenv("AF_COMFY_REQS_BAKED", raising=False)
    monkeypatch.delenv("AF_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("AF_EXPECTED_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("AF_REPORTED_IMAGE_DIGEST", raising=False)
    monkeypatch.setattr("gpu_worker.preflight.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("gpu_worker.preflight.sys.version_info", (3, 12, 3, "final", 0))
    monkeypatch.setattr(
        "gpu_worker.preflight._nvidia_smi_query",
        lambda: {
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "vram_mb": 32640,
            "driver_version": "580.65.06",
            "sm": "sm_120",
        },
    )
    monkeypatch.setattr(
        "gpu_worker.preflight.collect_stack_versions",
        lambda: {
            "python": "3.12.3",
            "torch": "2.13.0+cu130",
            "torchvision": "0.28.0+cu130",
            "torchaudio": "2.11.0+cu130",
        },
    )
    monkeypatch.setattr(
        "gpu_worker.preflight.cuda_smoke_test",
        lambda: {"ok": True, "device": "NVIDIA GeForce RTX 5090"},
    )
    monkeypatch.setattr(
        "gpu_worker.preflight.probe_nvfp4_runtime",
        lambda: {"ok": True, "torch": "2.13.0+cu130", "dtype": "float4_e2m1fn_x2", "scaled_mm": True},
    )
    monkeypatch.setattr(
        "gpu_worker.preflight.collect_host_resources",
        lambda: {
            "arch": "x86_64",
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "sm": "sm_120",
            "vram_mb": 32640,
            "driver_version": "580.65.06",
            "cuda_driver_ok": True,
            "disk_total_gb": 200,
            "disk_free_gb": 150,
            "mem_available_gb": 64,
            "comfy_dir": "/opt/ComfyUI",
        },
    )
    monkeypatch.setattr("gpu_worker.preflight.validate_container_contract", lambda: {"onstart": EXPECTED_ONSTART})
    out = run_hardware_preflight()
    assert out["profile_id"] == "h3-comfy-cu130-sm120"
    assert out["preflight"]["ok"] is True
    assert out["host"]["sm"] == "sm_120"
    assert "adaptations" in out


def test_probe_comfy_nodes_requires_ipadapter_advanced(monkeypatch):
    monkeypatch.setattr(
        "gpu_worker.preflight.fetch_comfy_json",
        lambda path, **_k: (
            {"system": {"comfyui_version": "0.3.0"}}
            if path == "/system_stats"
            else {
                "KSampler": {},
                "IPAdapterAdvanced": {},
                "MiniMaxH3ImageToVideo": {},
                "MiniMaxH3ReferenceToVideo": {},
                "MiniMaxH3SigmaShift": {},
                "MiniMaxH3AddGuide": {},
                "MiniMaxLowVRAMAttention": {},
                "MiniMaxChunkFeedForward": {},
            }
        ),
    )
    nodes = probe_comfy_nodes()
    assert "IPAdapterAdvanced" in nodes["nodes"]
    assert "MiniMaxH3ImageToVideo" in nodes["required_nodes"]


def test_probe_comfy_nodes_missing_node_fails(monkeypatch):
    monkeypatch.setattr(
        "gpu_worker.preflight.fetch_comfy_json",
        lambda path, **_k: (
            {"system": {}} if path == "/system_stats" else {"KSampler": {}}
        ),
    )
    with pytest.raises(PreflightFailure) as exc:
        probe_comfy_nodes()
    assert exc.value.failure_class == "comfy_startup_failed"
    assert exc.value.code == "missing_nodes"


def test_recycle_failure_class_maps_stable_codes():
    assert recycle_failure_class("capability_mismatch:torch_abi") == "capability_mismatch"
    assert recycle_failure_class("comfy_startup_failed:exit_1") == "comfy_startup_failed"
    assert recycle_failure_class(PreflightFailure("preflight_failed", "nvidia_smi", "no gpu")) == (
        "preflight_failed"
    )
    assert recycle_failure_class("OSError: undefined symbol: cudaLaunchKernelExC") == (
        "capability_mismatch"
    )
    assert recycle_failure_class("cannot open shared object file: libcudart.so.12") == (
        "capability_mismatch"
    )
    assert recycle_failure_class("module 'torch' has no attribute 'float4_e2m1fn_x2'") == (
        "capability_mismatch"
    )
    assert recycle_failure_class("install_failure: font_setup_failed") is None


def test_probe_nvfp4_runtime_missing_and_present(monkeypatch):
    import sys
    import types

    missing = types.ModuleType("torch")
    missing.__version__ = "2.7.0+cu128"
    missing.cuda = types.SimpleNamespace(is_available=lambda: False)
    missing.ops = types.SimpleNamespace(aten=None)
    monkeypatch.setitem(sys.modules, "torch", missing)
    out = probe_nvfp4_runtime()
    assert out["ok"] is False
    assert "float4_e2m1fn_x2" in str(out["error"])

    present = types.ModuleType("torch")
    present.__version__ = "2.8.0+cu128"
    present.float4_e2m1fn_x2 = object()
    present.cuda = types.SimpleNamespace(is_available=lambda: False)
    present.ops = types.SimpleNamespace(aten=types.SimpleNamespace(_scaled_mm=object()))
    monkeypatch.setitem(sys.modules, "torch", present)
    ok = probe_nvfp4_runtime()
    assert ok["ok"] is True
    assert ok["dtype"] == "float4_e2m1fn_x2"


def test_run_hardware_preflight_rejects_missing_nvfp4(monkeypatch):
    monkeypatch.setenv("AF_GPU_PROFILE", "h3-comfy-cu130-sm120")
    monkeypatch.delenv("AF_PRODUCTION_STACK", raising=False)
    monkeypatch.delenv("AF_COMFY_REQS_BAKED", raising=False)
    monkeypatch.delenv("AF_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("AF_EXPECTED_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("AF_REPORTED_IMAGE_DIGEST", raising=False)
    monkeypatch.setattr("gpu_worker.preflight.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("gpu_worker.preflight.sys.version_info", (3, 12, 3, "final", 0))
    monkeypatch.setattr(
        "gpu_worker.preflight.collect_host_resources",
        lambda: {
            "arch": "x86_64",
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "sm": "sm_120",
            "vram_mb": 32640,
            "driver_version": "580.65.06",
            "cuda_driver_ok": True,
            "disk_total_gb": 200,
            "disk_free_gb": 150,
            "mem_available_gb": 64,
            "comfy_dir": "/opt/ComfyUI",
        },
    )
    monkeypatch.setattr("gpu_worker.preflight.collect_stack_versions", lambda: _stack())
    monkeypatch.setattr("gpu_worker.preflight.cuda_smoke_test", lambda: {"ok": True})
    monkeypatch.setattr(
        "gpu_worker.preflight.probe_nvfp4_runtime",
        lambda: {"ok": False, "error": "module 'torch' has no attribute 'float4_e2m1fn_x2'"},
    )
    with pytest.raises(PreflightFailure) as exc:
        run_hardware_preflight()
    assert exc.value.failure_class == "capability_mismatch"
    assert exc.value.code == "nvfp4_dtype"


def test_handshake_payload_json_serializable():
    payload = handshake_payload(
        profile_id="h3-comfy-cu130-sm120",
        host=_host(),
        stack=_stack(),
        adaptations=[{"action": "fix_workdir", "before": {}, "after": {}, "attempt": 1, "result": "ok"}],
    )
    encoded = json.dumps(payload)
    assert "adaptations" in encoded
    assert payload["adaptations"][0]["action"] == "fix_workdir"


def test_production_stack_locked(monkeypatch):
    monkeypatch.setenv("AF_PRODUCTION_STACK", "1")
    assert production_stack_locked() is True
    monkeypatch.delenv("AF_PRODUCTION_STACK", raising=False)
    monkeypatch.setenv("AF_COMFY_REQS_BAKED", "1")
    assert production_stack_locked() is True


def test_onstart_and_paths_contract(tmp_path, monkeypatch):
    onstart = tmp_path / "af-start"
    onstart.write_text("#!/bin/sh\n", encoding="utf-8")
    onstart.chmod(0o755)
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    (comfy / "main.py").write_text("# comfy\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("AF_EXPECTED_ONSTART", str(onstart))
    monkeypatch.setenv("AF_ONSTART_PATH", str(onstart))
    monkeypatch.setenv("AF_ONSTART", str(onstart))
    monkeypatch.setenv("COMFYUI_DIR", str(comfy))
    monkeypatch.setenv("AF_WORK_DIR", str(work))
    out = validate_container_contract()
    assert out["onstart"] == str(onstart)
    assert out["paths"]["work_dir"] == str(work)

    monkeypatch.setenv("AF_ONSTART", "/bin/false")
    with pytest.raises(PreflightFailure) as exc:
        validate_container_contract()
    assert exc.value.code == "onstart_mismatch"

    monkeypatch.setenv("AF_ONSTART", str(onstart))
    (comfy / "main.py").unlink()
    with pytest.raises(PreflightFailure) as exc:
        validate_container_contract()
    assert exc.value.code == "container_paths"
    assert str(comfy / "main.py") in exc.value.details["missing"]


def test_image_digest_missing_mismatch_and_match(monkeypatch):
    monkeypatch.setenv("AF_PRODUCTION_STACK", "1")
    for key in (
        "AF_IMAGE_DIGEST",
        "AF_EXPECTED_IMAGE_DIGEST",
        "AF_REPORTED_IMAGE_DIGEST",
        "IMAGE_DIGEST",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(PreflightFailure) as missing:
        validate_image_digest(production=True)
    assert missing.value.code == "image_digest_missing"
    assert missing.value.failure_class == "capability_mismatch"
    for field in IMAGE_DIGEST_CONFIG_FIELDS:
        assert field in missing.value.details["config_fields"]

    monkeypatch.setenv("AF_EXPECTED_IMAGE_DIGEST", DIGEST_A)
    monkeypatch.setenv("AF_REPORTED_IMAGE_DIGEST", DIGEST_B)
    with pytest.raises(PreflightFailure) as mismatch:
        validate_image_digest(production=True)
    assert mismatch.value.code == "image_digest_mismatch"
    assert mismatch.value.details["expected"] == DIGEST_A
    assert mismatch.value.details["reported"] == DIGEST_B

    monkeypatch.setenv("AF_REPORTED_IMAGE_DIGEST", DIGEST_A)
    assert validate_image_digest(production=True) == DIGEST_A

    monkeypatch.delenv("AF_EXPECTED_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("AF_REPORTED_IMAGE_DIGEST", raising=False)
    monkeypatch.setenv("AF_IMAGE_DIGEST", DIGEST_A)
    assert validate_image_digest(production=True) == DIGEST_A


def test_whitelist_adapt_once_then_reject(tmp_path):
    adapter = StackAdapter()
    work = tmp_path / "work"
    rec = adapter.fix_workdir(work)
    assert rec["action"] == "fix_workdir"
    assert rec["attempt"] == 1
    assert rec["result"] == "ok"
    assert rec["before"]["exists"] is False
    assert rec["after"]["exists"] is True
    assert work.is_dir()
    second = adapter.fix_workdir(work)
    assert second["attempt"] == 2
    assert second["result"] == "rejected"
    assert adapter.as_list()[0]["action"] == "fix_workdir"


def test_abi_unfixable_does_not_pip_upgrade(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "gpu_worker.stack.subprocess.check_call",
        lambda args: calls.append(args) or 0,
    )
    adapter = StackAdapter()
    with pytest.raises(PreflightFailure) as exc:
        adapter.apply("torch_abi", before={"sm": "sm_89"})
    assert exc.value.failure_class == "capability_mismatch"
    assert "unfixable_torch_abi" in exc.value.code
    assert calls == []
    assert adapter.as_list()[0]["result"] == "unfixable"
    from gpu_worker import stack as stack_mod

    monkeypatch.setattr(stack_mod, "torch_supports_device", lambda: False)
    with pytest.raises(PreflightFailure) as torch_exc:
        stack_mod.ensure_torch()
    assert torch_exc.value.code == "torch_abi_mismatch"
    assert calls == []


def test_ensure_torchaudio_accepts_baked_211(monkeypatch):
    import sys
    import types

    from gpu_worker import stack as stack_mod

    fake = types.ModuleType("torchaudio")
    fake.__version__ = "2.11.0+cu130"
    monkeypatch.setitem(sys.modules, "torchaudio", fake)
    stack_mod.ensure_torchaudio()


def test_ensure_torchaudio_accepts_longlive_210(monkeypatch):
    import sys
    import types

    from gpu_worker import stack as stack_mod

    fake = types.ModuleType("torchaudio")
    fake.__version__ = "2.10.0+cu128"
    monkeypatch.setitem(sys.modules, "torchaudio", fake)
    monkeypatch.setenv("AF_GPU_PROFILE", "longlive-nvfp4-sm120")
    stack_mod.ensure_torchaudio()


def test_ensure_torchaudio_rejects_stale_27(monkeypatch):
    import sys
    import types

    from gpu_worker import stack as stack_mod

    fake = types.ModuleType("torchaudio")
    fake.__version__ = "2.7.0+cu128"
    monkeypatch.setitem(sys.modules, "torchaudio", fake)
    with pytest.raises(PreflightFailure) as exc:
        stack_mod.ensure_torchaudio()
    assert exc.value.code == "torchaudio_missing"


def test_handshake_register_and_heartbeat_include_audit(monkeypatch):
    from gpu_worker import session

    digest = DIGEST_A
    adaptations = [
        {
            "action": "fix_workdir",
            "before": {"exists": False},
            "after": {"exists": True},
            "attempt": 1,
            "result": "ok",
        }
    ]
    runtime = session.LeaseRuntime(instance_id="123")
    runtime.set_handshake(
        {
            "profile_id": "h3-comfy-cu130-sm120",
            "image_digest": digest,
            "host": {"sm": "sm_120", "onstart": EXPECTED_ONSTART},
            "stack": {"torch": "2.8.0+cu128"},
            "resources": {"disk_total_gb": 200},
            "preflight": {"ok": True, "stage": "complete", "adaptations": adaptations},
            "adaptations": adaptations,
        }
    )
    fields = runtime.heartbeat_fields({})
    assert fields["profile_id"] == "h3-comfy-cu130-sm120"
    assert fields["image_digest"] == digest
    assert fields["adaptations"][0]["action"] == "fix_workdir"
    assert fields["preflight"]["ok"] is True

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

    monkeypatch.setenv("CONTROL_PLANE_URL", "https://control.example")
    monkeypatch.setenv("STUDIO_AGENT_KEY", "box-secret")
    monkeypatch.setattr(worker_main.urllib.request, "urlopen", open_control)
    events = worker_main.maybe_notify_control_plane(
        "123",
        "idle",
        handshake=runtime.handshake,
        heartbeat_fields=fields,
    )
    assert events == ["register", "heartbeat"]
    register_body = json.loads(seen[0][1].decode("utf-8"))
    assert register_body["capabilities"]["image_digest"] == digest
    assert register_body["adaptations"][0]["result"] == "ok"
    assert register_body["preflight"]["ok"] is True


def test_register_body_includes_longlive_video_backend(monkeypatch):
    from gpu_worker import __main__ as worker_main

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

    monkeypatch.setenv("CONTROL_PLANE_URL", "https://control.example")
    monkeypatch.setenv("AF_IMAGE_CAPABILITY", "longlive")
    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    monkeypatch.setattr(worker_main.urllib.request, "urlopen", open_control)
    events = worker_main.maybe_notify_control_plane("5090-1", "idle", register=True)
    assert events == ["register", "heartbeat"]
    register_body = json.loads(seen[0][1].decode("utf-8"))
    caps = register_body["capabilities"]
    assert caps["longlive"] is True
    assert caps["h3"] is False
    assert caps["video_backend"] == "longlive"

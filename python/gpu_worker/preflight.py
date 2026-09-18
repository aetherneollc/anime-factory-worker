"""GPU capability preflight before large weight downloads.

Runs on-box checks (arch, driver, SM/VRAM, disk/RAM, baked stack versions, CUDA
smoke test) and post-Comfy node inventory. Deterministic mismatches raise
``PreflightFailure`` with a stable ``failure_class`` for control-plane recycle.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from gpu_worker.images import (
    IMAGE_DIGEST_CONFIG_FIELDS,
    expected_image_digest,
    handshake_image_digest,
    parse_content_digest,
    reported_image_digest,
    resolve_capability_profile,
)

PREFLIGHT_TIMEOUT_S = 120.0
COMFY_STARTUP_TIMEOUT_S = 600.0
WEIGHT_PULL_TIMEOUT_S = 3600.0

REQUIRED_COMFY_NODES = frozenset(
    {
        "IPAdapterAdvanced",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3ReferenceToVideo",
        "MiniMaxH3SigmaShift",
        "MiniMaxH3AddGuide",
        "MiniMaxLowVRAMAttention",
        "MiniMaxChunkFeedForward",
    }
)

# Control-plane recycle triggers (packages/cf-control seasons / gpu_scheduler).
RECYCLE_FAILURE_CLASSES = frozenset(
    {
        "preflight_failed",
        "capability_mismatch",
        "comfy_startup_failed",
        "weight_pull_timeout",
    }
)

# Deterministic wheel/driver ABI failures that must recycle, not wait or pip-upgrade.
ABI_RECYCLE_MARKERS = (
    "undefined_symbol",
    "cannot_open_shared_object",
    "libcudart",
    "glibcxx_",
    "cxxabi_",
    "abi_mismatch",
    "torch_abi",
    "no_kernel_image",
    "invalid_device_function",
    "float4_e2m1fn",
    "float4_e2m1fn_x2",
)

EXPECTED_PYTHON = (3, 12)
EXPECTED_TORCH = "2.13.0+cu130"
EXPECTED_TORCHVISION = "0.28.0+cu130"
EXPECTED_TORCHAUDIO = "2.11.0+cu130"
NVFP4_DTYPES = ("float4_e2m1fn_x2",)
EXPECTED_FLASH_ATTN_PREFIX = "2.7.4"
MIN_DRIVER_FOR_CUDA_128 = (570, 0)
MIN_DRIVER_FOR_CUDA_130 = (580, 0)
MIN_VRAM_MB = 32_000
MIN_DISK_GB = 200.0
DISK_FORMAT_SLACK_GB = 24.0
MIN_MEM_GB = 16.0
EXPECTED_ONSTART = "/usr/local/bin/af-start"

COMFY_DIR = Path(os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI")
COMFY_ROUTER_URL = os.environ.get("COMFYUI_BASE_URL") or "http://127.0.0.1:8199"
COMFY_BOOT_LOG = Path(os.environ.get("AF_COMFY_BOOT_LOG") or "/tmp/af-comfy-boot.log")
SOURCE_MANIFEST = COMFY_DIR / ".af-source-manifest.json"


@dataclass
class PreflightFailure(RuntimeError):
    """Deterministic incompatibility — recycle the instance, do not retry install."""

    failure_class: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.failure_class}:{self.code}:{self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "failure_class": self.failure_class,
            "code": self.code,
            "error": self.message,
            "details": self.details,
        }


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def production_stack_locked() -> bool:
    raw = os.environ.get("AF_PRODUCTION_STACK", "")
    if raw.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return os.environ.get("AF_COMFY_REQS_BAKED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def recycle_failure_class(error: BaseException | str | None) -> str | None:
    if error is None:
        return None
    if isinstance(error, PreflightFailure):
        return error.failure_class
    for attr in ("failure_class", "limit"):
        classified = getattr(error, attr, None)
        if isinstance(classified, str) and classified in RECYCLE_FAILURE_CLASSES:
            return classified
    text = str(error).lower().replace("-", "_").replace(" ", "_")
    for cls in RECYCLE_FAILURE_CLASSES:
        if cls in text:
            return cls
    if any(marker in text for marker in ABI_RECYCLE_MARKERS):
        return "capability_mismatch"
    return None


def boot_log_tail(max_chars: int = 4096) -> str:
    if not COMFY_BOOT_LOG.is_file():
        return ""
    try:
        raw = COMFY_BOOT_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return raw[-max(1, max_chars) :]


def _nvidia_smi_query() -> dict[str, Any] | None:
    smi = shutil.which("nvidia-smi") or "nvidia-smi"
    try:
        raw = subprocess.check_output(
            [
                smi,
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            timeout=15,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "replace")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None
    name, mem, driver, cap = parts[0], parts[1], parts[2], parts[3]
    vram_mb = None
    mem_match = re.search(r"(\d+)", mem.replace(",", ""))
    if mem_match:
        vram_mb = int(mem_match.group(1))
        if "mib" in mem.lower() or "miB" in mem:
            pass
        elif "gib" in mem.lower():
            vram_mb *= 1024
    sm = None
    if cap and cap[0].isdigit():
        major, _, minor = cap.partition(".")
        if major.isdigit() and minor.isdigit():
            sm = f"sm_{major}{minor}"
    return {
        "gpu_name": name,
        "vram_mb": vram_mb,
        "driver_version": driver,
        "compute_cap": cap,
        "sm": sm,
    }


def _driver_supports_cuda(driver_version: str | None, minimum: tuple[int, int]) -> bool:
    if not driver_version:
        return False
    nums = re.findall(r"\d+", driver_version)
    if not nums:
        return False
    major = int(nums[0])
    minor = int(nums[1]) if len(nums) > 1 else 0
    return (major, minor) >= minimum


def _driver_supports_cuda_128(driver_version: str | None) -> bool:
    return _driver_supports_cuda(driver_version, MIN_DRIVER_FOR_CUDA_128)


def _driver_supports_cuda_130(driver_version: str | None) -> bool:
    return _driver_supports_cuda(driver_version, MIN_DRIVER_FOR_CUDA_130)


def _profile_driver_ok(profile_id: str, driver_version: str | None) -> bool:
    if str(profile_id or "").startswith("h3-comfy-cu130"):
        return _driver_supports_cuda_130(driver_version)
    if str(profile_id or "").startswith("longlive"):
        return _driver_supports_cuda_128(driver_version)
    return _driver_supports_cuda_128(driver_version)


def expected_onstart() -> str:
    return (os.environ.get("AF_EXPECTED_ONSTART") or EXPECTED_ONSTART).strip() or EXPECTED_ONSTART


def reported_onstart() -> str | None:
    for key in ("AF_ONSTART", "VAST_ONSTART"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return raw
    return None


def container_contract_paths() -> dict[str, Path]:
    onstart = Path(os.environ.get("AF_ONSTART_PATH") or expected_onstart())
    comfy = Path(os.environ.get("COMFYUI_DIR") or str(COMFY_DIR))
    work = Path(os.environ.get("AF_WORK_DIR") or "/work")
    return {
        "onstart": onstart,
        "comfy_dir": comfy,
        "comfy_main": comfy / "main.py",
        "work_dir": work,
    }


def validate_container_contract() -> dict[str, Any]:
    """onstart/entrypoint plus Comfy and work paths must exist before weight pull."""
    expected = expected_onstart()
    reported = reported_onstart()
    if reported and reported != expected:
        raise PreflightFailure(
            "preflight_failed",
            "onstart_mismatch",
            f"onstart {reported!r} does not match entrypoint contract {expected!r}",
            {"expected_onstart": expected, "reported_onstart": reported},
        )
    paths = container_contract_paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise PreflightFailure(
            "preflight_failed",
            "container_paths",
            f"missing container paths: {missing}",
            {"missing": missing, "expected_onstart": expected},
        )
    onstart = paths["onstart"]
    if onstart.is_file() and not os.access(onstart, os.X_OK):
        raise PreflightFailure(
            "preflight_failed",
            "onstart_not_executable",
            f"{onstart} exists but is not executable",
            {"onstart": str(onstart)},
        )
    return {
        "onstart": expected,
        "reported_onstart": reported or expected,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def validate_image_digest(*, production: bool | None = None, profile_id: str | None = None) -> str | None:
    """Match injected expected vs reported sha256. Production fails closed if unknown."""
    production = production_stack_locked() if production is None else production
    profile = resolve_capability_profile(profile_id)
    expected = expected_image_digest(profile_id)
    pinned = parse_content_digest(profile.expected_image_digest)
    if pinned:
        expected = pinned if expected is None else expected
        if expected != pinned:
            raise PreflightFailure(
                "capability_mismatch",
                "image_digest_mismatch",
                f"env digest {expected} does not match profile pin {pinned}",
                {
                    "expected": expected,
                    "profile_digest": pinned,
                    "config_fields": list(IMAGE_DIGEST_CONFIG_FIELDS),
                },
            )
    reported = reported_image_digest()
    require = production or bool(expected) or bool(pinned)
    if require and not reported:
        raise PreflightFailure(
            "capability_mismatch",
            "image_digest_missing",
            "production/profile requires a reported content digest; set AF_IMAGE_DIGEST "
            "or AF_REPORTED_IMAGE_DIGEST to sha256:<64 hex>. Do not use floating tags "
            "(:main/:latest/:sha-…). Current deploy has no baked sha256 — configure "
            "AF_EXPECTED_IMAGE_DIGEST / CapabilityProfile.expected_image_digest / GpuImage.digest.",
            {
                "reported": reported,
                "expected": expected,
                "config_fields": list(IMAGE_DIGEST_CONFIG_FIELDS),
            },
        )
    if require and not expected:
        raise PreflightFailure(
            "capability_mismatch",
            "image_digest_missing",
            "no known sha256 to match against; configure AF_EXPECTED_IMAGE_DIGEST, "
            "CapabilityProfile.expected_image_digest, or GpuImage.digest. Do not invent a digest.",
            {
                "reported": reported,
                "expected": expected,
                "config_fields": list(IMAGE_DIGEST_CONFIG_FIELDS),
            },
        )
    if expected and reported and expected != reported:
        raise PreflightFailure(
            "capability_mismatch",
            "image_digest_mismatch",
            f"image digest {reported} does not match expected {expected}",
            {
                "reported": reported,
                "expected": expected,
                "config_fields": list(IMAGE_DIGEST_CONFIG_FIELDS),
            },
        )
    return reported or expected


def _disk_stats(path: Path) -> dict[str, float | None]:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return {"disk_free_gb": None, "disk_total_gb": None}
    return {
        "disk_free_gb": usage.free / (1024**3),
        "disk_total_gb": usage.total / (1024**3),
    }


def _disk_free_gb(path: Path) -> float | None:
    return _disk_stats(path).get("disk_free_gb")


def _mem_available_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            lines = {k: v for k, v in (ln.split(":", 1) for ln in fh if ":" in ln)}
        avail = lines.get("MemAvailable") or lines.get("MemFree")
        if not avail:
            return None
        kb = int(re.search(r"\d+", avail).group(0))  # type: ignore[union-attr]
        return kb / (1024**2)
    except (OSError, ValueError, AttributeError):
        return None


def _package_version(import_name: str, attr: str = "__version__") -> str | None:
    try:
        mod = __import__(import_name)
    except ImportError:
        return None
    return str(getattr(mod, attr, "") or "") or None


def _read_source_manifest() -> dict[str, str]:
    if not SOURCE_MANIFEST.is_file():
        return {}
    try:
        data = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): str(v) for k, v in data.items() if v}


def _git_head_sha(repo: Path) -> str | None:
    git = shutil.which("git")
    if not git or not (repo / ".git").exists():
        return None
    try:
        out = subprocess.check_output(
            [git, "-C", str(repo), "rev-parse", "HEAD"],
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "replace").strip()
        return out or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def collect_stack_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "torch": _package_version("torch"),
        "torchvision": _package_version("torchvision"),
        "torchaudio": _package_version("torchaudio"),
        "flash_attn": _package_version("flash_attn"),
        "fouroversix": _package_version("fouroversix"),
    }
    manifest = _read_source_manifest()
    comfy_sha = manifest.get("comfyui_sha") or _git_head_sha(COMFY_DIR)
    ip_sha = manifest.get("ipadapter_sha")
    ip_dir = COMFY_DIR / "custom_nodes" / "ComfyUI_IPAdapter_plus"
    if not ip_sha:
        ip_sha = _git_head_sha(ip_dir)
    versions["comfyui_sha"] = comfy_sha
    versions["ipadapter_sha"] = ip_sha
    versions["comfyui_ref"] = manifest.get("comfyui_ref")
    versions["ipadapter_ref"] = manifest.get("ipadapter_ref")
    return versions


def probe_nvfp4_runtime() -> dict[str, Any]:
    """Detect torch.float4_e2m1fn_x2 before pulling ~63GB of NVFP4 H3 weights.

    Do not import comfy_kitchen here — that can trigger Triton JIT on boot.
    """
    try:
        import torch
    except ImportError as exc:
        return {"ok": False, "error": f"torch_import:{exc}"}
    missing = [name for name in NVFP4_DTYPES if not hasattr(torch, name)]
    if missing:
        return {
            "ok": False,
            "error": f"module 'torch' has no attribute '{missing[0]}'",
            "missing_dtypes": missing,
            "torch": getattr(torch, "__version__", None),
        }
    aten = getattr(getattr(torch, "ops", None), "aten", None)
    op_ok = aten is not None and hasattr(aten, "_scaled_mm")
    result: dict[str, Any] = {
        "ok": True,
        "torch": getattr(torch, "__version__", None),
        "scaled_mm": bool(op_ok),
        "dtype": NVFP4_DTYPES[0],
    }
    if not torch.cuda.is_available():
        return result
    try:
        buf = torch.empty(4, dtype=torch.uint8, device="cuda")
        buf.view(torch.float4_e2m1fn_x2)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 — capability probe
        return {
            "ok": False,
            "error": f"{type(exc).__name__}:{exc}",
            "torch": getattr(torch, "__version__", None),
        }
    return result


def cuda_smoke_test() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"ok": False, "error": f"torch_import:{exc}"}
    if not torch.cuda.is_available():
        return {"ok": False, "error": "cuda_unavailable"}
    try:
        device = torch.device("cuda:0")
        x = torch.zeros(1, device=device)
        y = (x + 1).item()
        torch.cuda.synchronize()
        return {"ok": int(y) == 1, "device": torch.cuda.get_device_name(0)}
    except Exception as exc:  # noqa: BLE001 — smoke probe
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}"}


def _version_matches(actual: str | None, expected: str) -> bool:
    if not actual:
        return False
    return actual.startswith(expected.split("+")[0])


def _flash_attn_ok(version: str | None, expected_prefix: str | None = None) -> bool:
    if not version:
        return False
    prefix = expected_prefix or EXPECTED_FLASH_ATTN_PREFIX
    return version.startswith(prefix)


def _fouroversix_ok() -> bool:
    try:
        import fouroversix  # noqa: F401
        from fouroversix import _C  # noqa: F401
    except ImportError:
        return False
    return True


def collect_host_resources() -> dict[str, Any]:
    gpu = _nvidia_smi_query() or {}
    from gpu_worker.stack import cuda_sm

    sm = cuda_sm() or gpu.get("sm")
    disk_root = COMFY_DIR if COMFY_DIR.is_dir() else Path("/")
    disk = _disk_stats(disk_root)
    paths = container_contract_paths()
    return {
        "arch": platform.machine(),
        "gpu_name": gpu.get("gpu_name"),
        "sm": sm,
        "vram_mb": gpu.get("vram_mb"),
        "driver_version": gpu.get("driver_version"),
        "cuda_driver_ok": _driver_supports_cuda_130(gpu.get("driver_version")),
        "disk_free_gb": disk.get("disk_free_gb"),
        "disk_total_gb": disk.get("disk_total_gb"),
        "mem_available_gb": _mem_available_gb(),
        "comfy_dir": str(paths["comfy_dir"]),
        "work_dir": str(paths["work_dir"]),
        "onstart": reported_onstart() or expected_onstart(),
        "expected_onstart": expected_onstart(),
    }


def validate_profile(profile_id: str, host: dict[str, Any], stack: dict[str, Any]) -> None:
    profile = resolve_capability_profile(profile_id)
    arch = str(host.get("arch") or "")
    if arch not in profile.supported_arch:
        raise PreflightFailure(
            "capability_mismatch",
            "arch_unsupported",
            f"architecture {arch!r} not in {profile.supported_arch}",
            {"profile_id": profile_id, "arch": arch},
        )
    sm = str(host.get("sm") or "")
    if sm and sm not in profile.supported_sm and not any(
        sm.startswith(prefix) for prefix in profile.supported_sm
    ):
        raise PreflightFailure(
            "capability_mismatch",
            "sm_unsupported",
            f"GPU {sm!r} not supported by profile {profile_id}",
            {"profile_id": profile_id, "sm": sm, "supported_sm": list(profile.supported_sm)},
        )
    vram = host.get("vram_mb")
    if vram is not None and int(vram) < profile.min_vram_mb:
        raise PreflightFailure(
            "capability_mismatch",
            "vram_below_minimum",
            f"VRAM {vram}MB below profile minimum {profile.min_vram_mb}MB",
            {"profile_id": profile_id, "vram_mb": vram},
        )
    driver_ok = _profile_driver_ok(profile_id, host.get("driver_version"))
    if not driver_ok:
        min_driver = MIN_DRIVER_FOR_CUDA_130 if str(profile_id).startswith("h3-comfy-cu130") else MIN_DRIVER_FOR_CUDA_128
        raise PreflightFailure(
            "preflight_failed",
            "driver_cuda",
            f"NVIDIA driver {host.get('driver_version')} does not meet CUDA minimum "
            f"{min_driver[0]}.{min_driver[1]}+ for profile {profile_id}",
            {"profile_id": profile_id, "min_driver": f"{min_driver[0]}.{min_driver[1]}"},
        )
    disk = host.get("disk_total_gb")
    if disk is None:
        disk = host.get("disk_free_gb")
    threshold = profile.min_disk_gb - DISK_FORMAT_SLACK_GB
    if disk is not None and float(disk) < threshold:
        raise PreflightFailure(
            "preflight_failed",
            "disk_low",
            f"disk {float(disk):.1f}GB below minimum {threshold:.1f}GB "
            f"(lease {profile.min_disk_gb:g}GB minus format slack)",
            {
                "profile_id": profile_id,
                "disk_total_gb": host.get("disk_total_gb"),
                "disk_free_gb": host.get("disk_free_gb"),
                "threshold_gb": threshold,
            },
        )
    mem = host.get("mem_available_gb")
    if mem is not None and float(mem) < profile.min_mem_gb:
        raise PreflightFailure(
            "preflight_failed",
            "mem_low",
            f"available memory {mem:.1f}GB below minimum {profile.min_mem_gb}GB",
            {"profile_id": profile_id},
        )
    if profile.require_torch:
        torch_expected = profile.expected_torch or EXPECTED_TORCH
        vision_expected = profile.expected_torchvision or EXPECTED_TORCHVISION
        audio_expected = profile.expected_torchaudio or EXPECTED_TORCHAUDIO
        for key, expected in (
            ("torch", torch_expected),
            ("torchvision", vision_expected),
            ("torchaudio", audio_expected),
        ):
            if not _version_matches(stack.get(key), expected):
                raise PreflightFailure(
                    "capability_mismatch",
                    f"{key}_version",
                    f"expected {key} {expected}, got {stack.get(key)}",
                    {"profile_id": profile_id, key: stack.get(key)},
                )
    if profile.require_flash_attn:
        flash_expected = profile.expected_flash_attn or EXPECTED_FLASH_ATTN_PREFIX
        if not _flash_attn_ok(stack.get("flash_attn"), flash_expected):
            raise PreflightFailure(
                "capability_mismatch",
                "flash_attn_missing",
                f"flash-attn {flash_expected}+ required, got {stack.get('flash_attn')}",
                {"profile_id": profile_id, "expected_flash_attn": flash_expected},
            )
    if profile.require_fouroversix and not _fouroversix_ok():
        raise PreflightFailure(
            "capability_mismatch",
            "fouroversix_missing",
            "fouroversix CUDA extension not importable",
            {"profile_id": profile_id},
        )
    validate_image_digest(profile_id=profile_id)


def run_hardware_preflight(
    *,
    profile_id: str | None = None,
    deadline_s: float | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Two-minute bounded preflight before weight download."""
    clock = clock or time.monotonic
    deadline = clock() + (deadline_s if deadline_s is not None else PREFLIGHT_TIMEOUT_S)
    profile_id = profile_id or os.environ.get("AF_GPU_PROFILE") or resolve_capability_profile().profile_id

    if platform.machine() not in {"x86_64", "AMD64"}:
        raise PreflightFailure(
            "preflight_failed",
            "arch",
            f"unsupported architecture {platform.machine()}",
        )

    if clock() > deadline:
        raise PreflightFailure("preflight_failed", "timeout", "preflight deadline exceeded")

    host = collect_host_resources()
    if not host.get("gpu_name"):
        raise PreflightFailure(
            "preflight_failed",
            "nvidia_smi",
            "nvidia-smi did not report a GPU",
            host,
        )

    stack = collect_stack_versions()
    py = sys.version_info[:2]
    if py != EXPECTED_PYTHON:
        raise PreflightFailure(
            "capability_mismatch",
            "python_version",
            f"expected Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}, got {py[0]}.{py[1]}",
        )

    smoke = cuda_smoke_test()
    if not smoke.get("ok"):
        raise PreflightFailure(
            "preflight_failed",
            "cuda_smoke",
            f"CUDA smoke test failed: {smoke.get('error')}",
            smoke,
        )

    if str(profile_id or "").startswith("h3-comfy"):
        nvfp4 = probe_nvfp4_runtime()
        if not nvfp4.get("ok"):
            raise PreflightFailure(
                "capability_mismatch",
                "nvfp4_dtype",
                nvfp4.get("error") or "nvfp4 runtime missing",
                nvfp4,
            )
        if not nvfp4.get("scaled_mm"):
            raise PreflightFailure(
                "capability_mismatch",
                "nvfp4_kernel",
                "NVFP4 scaled_mm kernel unavailable on this torch build",
                nvfp4,
            )

    validate_container_contract()
    validate_profile(profile_id, host, stack)

    return handshake_payload(
        profile_id=profile_id,
        host=host,
        stack=stack,
        preflight_ok=True,
        preflight_stage="hardware_complete",
    )


def fetch_comfy_json(path: str, base_url: str = COMFY_ROUTER_URL, timeout_s: float = 10.0) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", "replace")
            return json.loads(body or "{}")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise PreflightFailure(
            "comfy_startup_failed",
            "http_probe",
            f"{path} probe failed: {type(exc).__name__}:{exc}",
        ) from exc


def probe_comfy_nodes(
    base_url: str = COMFY_ROUTER_URL,
    required: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Require /system_stats and /object_info plus workflow node inventory."""
    required = required or REQUIRED_COMFY_NODES
    stats = fetch_comfy_json("/system_stats", base_url=base_url)
    object_info = fetch_comfy_json("/object_info", base_url=base_url)
    nodes = sorted(str(k) for k in object_info.keys()) if isinstance(object_info, dict) else []
    missing = sorted(n for n in required if n not in object_info)
    if missing:
        raise PreflightFailure(
            "comfy_startup_failed",
            "missing_nodes",
            f"Comfy missing required nodes: {missing}",
            {"missing": missing, "node_count": len(nodes)},
        )
    return {
        "system_stats_ok": bool(stats),
        "object_info_ok": bool(object_info),
        "nodes": nodes,
        "required_nodes": sorted(required),
    }


def handshake_payload(
    *,
    profile_id: str,
    host: dict[str, Any] | None = None,
    stack: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    preflight_ok: bool = True,
    preflight_stage: str = "complete",
    preflight_error: str | None = None,
    failure_class: str | None = None,
    comfy_nodes: dict[str, Any] | None = None,
    adaptations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    host = host or collect_host_resources()
    stack = stack or collect_stack_versions()
    adaptations = list(adaptations or [])
    resources = resources or {
        "disk_free_gb": host.get("disk_free_gb"),
        "disk_total_gb": host.get("disk_total_gb"),
        "mem_available_gb": host.get("mem_available_gb"),
        "comfy_dir": host.get("comfy_dir"),
        "work_dir": host.get("work_dir"),
        "onstart": host.get("onstart"),
        "weight_pull_timeout_s": WEIGHT_PULL_TIMEOUT_S,
        "comfy_startup_timeout_s": COMFY_STARTUP_TIMEOUT_S,
        "preflight_timeout_s": PREFLIGHT_TIMEOUT_S,
    }
    return {
        "profile_id": profile_id,
        "image_digest": handshake_image_digest(),
        "host": host,
        "stack": stack,
        "resources": resources,
        "adaptations": adaptations,
        "preflight": {
            "ok": bool(preflight_ok),
            "stage": preflight_stage,
            "error": preflight_error,
            "failure_class": failure_class,
            "checked_at": _utc_iso(),
            "adaptations": adaptations,
        },
        "comfy_nodes": comfy_nodes,
        "boot_log_tail": boot_log_tail() or None,
    }


def select_profile_id() -> str:
    from anime_factory.video_backend import select_video_backend

    explicit = (os.environ.get("AF_GPU_PROFILE") or "").strip()
    if explicit:
        return explicit
    if select_video_backend() == "longlive":
        return "longlive-nvfp4-sm120"
    return "h3-comfy-cu130-sm120"


__all__ = [
    "ABI_RECYCLE_MARKERS",
    "COMFY_STARTUP_TIMEOUT_S",
    "EXPECTED_ONSTART",
    "MIN_DISK_GB",
    "PREFLIGHT_TIMEOUT_S",
    "RECYCLE_FAILURE_CLASSES",
    "REQUIRED_COMFY_NODES",
    "WEIGHT_PULL_TIMEOUT_S",
    "PreflightFailure",
    "boot_log_tail",
    "collect_host_resources",
    "collect_stack_versions",
    "container_contract_paths",
    "cuda_smoke_test",
    "handshake_payload",
    "probe_nvfp4_runtime",
    "probe_comfy_nodes",
    "production_stack_locked",
    "recycle_failure_class",
    "run_hardware_preflight",
    "select_profile_id",
    "validate_container_contract",
    "validate_image_digest",
    "validate_profile",
]

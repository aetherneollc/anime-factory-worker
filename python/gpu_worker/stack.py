"""Start ComfyUI :8188, router :8199, optional cloudflared, then the agent.

No bash `set -e`. Health wait is Python. Bind Comfy/router to 127.0.0.1 only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from gpu_worker.boot import health_wait_loop, probe_comfy_system_stats
from gpu_worker.images import default_register_capabilities, resolve_capability_profile
from gpu_worker.preflight import (
    COMFY_STARTUP_TIMEOUT_S,
    EXPECTED_ONSTART,
    EXPECTED_TORCHAUDIO,
    PreflightFailure,
    handshake_payload,
    probe_comfy_nodes,
    run_hardware_preflight,
    select_profile_id,
)
from anime_factory.video_backend import VideoBackendLockError, lock_video_backend
from gpu_worker.weights import (
    ensure_still_weights,
    runtime_weight_bytes,
    start_h3_weights_background,
)

COMFY_DIR = Path(os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI")
ROUTER_URL = os.environ.get("COMFYUI_BASE_URL") or "http://127.0.0.1:8199"
TORCH_CU124 = "https://download.pytorch.org/whl/cu124"
TORCH_CU128 = "https://download.pytorch.org/whl/cu128"
TUNNEL_METRICS_URL = "http://127.0.0.1:49312/metrics"
DEFAULT_TUNNEL_HOSTNAME = ""
COMFY_BOOT_LOG = Path(os.environ.get("AF_COMFY_BOOT_LOG") or "/tmp/af-comfy-boot.log")
_TORCH_REQ_PACKAGES = frozenset({"torch", "torchvision", "torchaudio"})
NOTO_CJK_FAMILIES = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK KR",
)

_TUNNEL_STATUS: dict | None = None

ADAPT_MAX_ATTEMPTS = 1
WHITELIST_ADAPT_ACTIONS = frozenset(
    {
        "bind_ports",
        "fix_onstart",
        "fix_workdir",
        "comfy_lowvram",
        "install_baked_wheel",
        "ensure_fonts",
        "ensure_headers",
        "restart_comfy",
        "restart_router",
        "restart_cloudflared",
    }
)
UNFIXABLE_ADAPT_ACTIONS = frozenset(
    {
        "driver",
        "sm",
        "torch_abi",
        "core_wheel",
        "node_api",
        "upgrade_torch",
        "upgrade_cuda",
        "pip_upgrade",
    }
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    pid = getattr(value, "pid", None)
    poll = getattr(value, "poll", None)
    if pid is not None or callable(poll):
        return {"pid": pid, "poll": None if poll is None else poll()}
    return str(value)


@dataclass
class AdaptRecord:
    action: str
    before: Any
    after: Any
    attempt: int
    result: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "before": _json_safe(self.before),
            "after": _json_safe(self.after),
            "attempt": self.attempt,
            "result": self.result,
        }


class StackAdapter:
    """Bounded whitelist repairs. ABI/driver/SM/core wheels fail closed — no pip upgrade."""

    def __init__(self, max_attempts: int = ADAPT_MAX_ATTEMPTS) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.attempts: dict[str, int] = {}
        self.audit: list[AdaptRecord] = []

    def as_list(self) -> list[dict[str, Any]]:
        return [row.as_dict() for row in self.audit]

    def record_unfixable(self, action: str, *, before: Any = None, error: str | None = None) -> dict[str, Any]:
        rec = AdaptRecord(
            action=action,
            before=before,
            after=before,
            attempt=self.attempts.get(action, 0) + 1,
            result="unfixable",
        )
        rec_dict = rec.as_dict()
        if error:
            rec_dict["error"] = error
            rec.after = {"error": error}
            rec_dict["after"] = rec.after
        self.audit.append(rec)
        return rec_dict

    def apply(
        self,
        action: str,
        *,
        before: Any = None,
        repair: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        if action in UNFIXABLE_ADAPT_ACTIONS or action not in WHITELIST_ADAPT_ACTIONS:
            rec = self.record_unfixable(action, before=before)
            raise PreflightFailure(
                "capability_mismatch",
                f"unfixable_{action}",
                f"{action} is not on the adapt whitelist; refuse live pip upgrade",
                rec,
            )
        n = self.attempts.get(action, 0) + 1
        self.attempts[action] = n
        if n > self.max_attempts:
            rec = AdaptRecord(action=action, before=before, after=before, attempt=n, result="rejected")
            self.audit.append(rec)
            return rec.as_dict()
        after = before
        result = "ok"
        try:
            if repair is not None:
                after = repair()
        except PreflightFailure:
            rec = AdaptRecord(action=action, before=before, after=before, attempt=n, result="unfixable")
            self.audit.append(rec)
            raise
        except Exception as exc:  # noqa: BLE001 — audit the failed repair
            rec = AdaptRecord(
                action=action,
                before=before,
                after={"error": f"{type(exc).__name__}:{exc}"},
                attempt=n,
                result="failed",
            )
            self.audit.append(rec)
            raise
        rec = AdaptRecord(action=action, before=before, after=after, attempt=n, result=result)
        self.audit.append(rec)
        return rec.as_dict()

    def fix_workdir(self, work_dir: Path | None = None) -> dict[str, Any]:
        work = work_dir or Path(os.environ.get("AF_WORK_DIR") or "/work")
        before = {"path": str(work), "exists": work.is_dir()}
        if work.is_dir():
            return self.apply("fix_workdir", before=before, repair=lambda: before)

        def repair() -> dict[str, Any]:
            work.mkdir(parents=True, exist_ok=True)
            return {"path": str(work), "exists": work.is_dir()}

        return self.apply("fix_workdir", before=before, repair=repair)

    def fix_onstart(self, onstart: Path | None = None) -> dict[str, Any]:
        path = onstart or Path(os.environ.get("AF_ONSTART_PATH") or EXPECTED_ONSTART)
        before = {"path": str(path), "executable": bool(path.is_file() and os.access(path, os.X_OK))}
        if before["executable"]:
            return self.apply("fix_onstart", before=before, repair=lambda: before)

        def repair() -> dict[str, Any]:
            if path.is_file():
                path.chmod(path.stat().st_mode | 0o111)
            return {
                "path": str(path),
                "executable": bool(path.is_file() and os.access(path, os.X_OK)),
            }

        return self.apply("fix_onstart", before=before, repair=repair)

    def install_baked_wheel(self, wheel: Path) -> dict[str, Any]:
        root = Path(os.environ.get("LONGLIVE_WHEELS") or "/opt/longlive-wheels").resolve()
        resolved = wheel.resolve()
        before = {"wheel": str(resolved), "baked_root": str(root)}

        def repair() -> dict[str, Any]:
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise PreflightFailure(
                    "capability_mismatch",
                    "core_wheel",
                    f"wheel {resolved} is not inside image bake dir {root}",
                    {"wheel": str(resolved), "baked_root": str(root)},
                ) from exc
            if resolved.suffix != ".whl" or not resolved.is_file():
                raise PreflightFailure(
                    "capability_mismatch",
                    "core_wheel",
                    f"not a baked wheel file: {resolved}",
                    {"wheel": str(resolved)},
                )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(resolved)]
            )
            return {"wheel": str(resolved), "installed": True}

        return self.apply("install_baked_wheel", before=before, repair=repair)


def _popen(args: list[str], extra_env: dict[str, str] | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(args, env=env, stdout=sys.stdout, stderr=sys.stderr)


def _nvidia_smi_sm() -> str | None:
    """Fall back when torch cannot see the device (cu124 on a 5090 returns nothing)."""
    smi = shutil.which("nvidia-smi") or "nvidia-smi"
    try:
        raw = subprocess.check_output(
            [smi, "--query-gpu=compute_cap,name", "--format=csv,noheader"],
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — probe only
        return None
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    if not line:
        return None
    cap, _, name = line.partition(",")
    cap = cap.strip()
    name = name.strip().lower()
    if cap and cap[0].isdigit():
        major, _, minor = cap.partition(".")
        if major.isdigit() and minor.isdigit():
            return f"sm_{major}{minor}"
    if any(tok in name for tok in ("5090", "5080", "5070", "blackwell")):
        return "sm_120"
    return None


def cuda_sm() -> str | None:
    """Return sm_XY for the visible GPU, or None if CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    else:
        try:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                return f"sm_{major}{minor}"
        except Exception:  # noqa: BLE001 — probe only
            pass
    return _nvidia_smi_sm()


def torch_index_url(sm: str | None = None) -> str:
    """Blackwell (sm_12x / 50-series) has no kernels in cu124 wheels."""
    explicit = os.environ.get("TORCH_INDEX_URL")
    if explicit:
        return explicit
    sm = sm or cuda_sm()
    if sm and sm.startswith("sm_12"):
        return TORCH_CU128
    return TORCH_CU124


def torch_supports_device() -> bool:
    try:
        import torch
    except ImportError:
        return False
    sm = cuda_sm()
    if not sm:
        # Unknown device: do not claim success; nvidia-smi fallback already ran.
        return bool(torch.cuda.is_available())
    try:
        archs = list(torch.cuda.get_arch_list() or [])
    except Exception:  # noqa: BLE001
        archs = []
    if sm not in archs and not any(str(a).startswith("sm_12") for a in archs):
        return False
    try:
        x = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        return int((x + 1).item()) == 1
    except Exception:  # noqa: BLE001
        return False


def ensure_torch() -> None:
    if torch_supports_device():
        ensure_torchaudio()
        return
    raise PreflightFailure(
        "capability_mismatch",
        "torch_abi_mismatch",
        "baked torch wheel does not support this GPU; live pip upgrade is not on the adapt whitelist",
        {"sm": cuda_sm(), "index": torch_index_url()},
    )


def ensure_torchaudio() -> None:
    """Comfy imports torchaudio at startup; a wrong wheel is ABI, not a live pip repair."""
    try:
        import torchaudio as ta
    except ImportError as exc:
        raise PreflightFailure(
            "capability_mismatch",
            "torchaudio_missing",
            "baked torchaudio missing; live pip upgrade is not on the adapt whitelist",
            {"error": str(exc)},
        ) from exc
    except OSError as exc:
        raise PreflightFailure(
            "capability_mismatch",
            "torchaudio_missing",
            "baked torchaudio ABI mismatch; live pip upgrade is not on the adapt whitelist",
            {"error": str(exc)},
        ) from exc
    profile = resolve_capability_profile(select_profile_id())
    expected_audio = profile.expected_torchaudio or EXPECTED_TORCHAUDIO
    if not str(getattr(ta, "__version__", "") or "").startswith(expected_audio.split("+")[0]):
        raise PreflightFailure(
            "capability_mismatch",
            "torchaudio_missing",
            "baked torchaudio missing or wrong ABI; live pip upgrade is not on the adapt whitelist",
            {"version": getattr(ta, "__version__", None), "expected": expected_audio},
        )


def _python_headers_available() -> bool:
    major, minor = sys.version_info.major, sys.version_info.minor
    return any(
        path.is_file()
        for path in (
            Path(f"/usr/include/python{major}.{minor}/Python.h"),
            Path("/usr/include/python3/Python.h"),
        )
    )


def ensure_c_compiler() -> None:
    """Triton/comfy_kitchen JIT needs gcc **and** python3-dev (Python.h)."""
    need_gcc = not (shutil.which("gcc") or shutil.which("cc"))
    need_pydev = not _python_headers_available()
    if not need_gcc and not need_pydev:
        return
    apt = shutil.which("apt-get")
    if not apt:
        print(
            json.dumps(
                {
                    "build_deps": "missing",
                    "apt-get": False,
                    "need_gcc": need_gcc,
                    "need_python_dev": need_pydev,
                }
            ),
            flush=True,
        )
        return
    packages: list[str] = []
    if need_gcc:
        packages.extend(["gcc", "g++"])
    if need_pydev:
        packages.append("python3-dev")
    subprocess.call([apt, "update", "-qq"])
    subprocess.call([apt, "install", "-y", "-qq", *packages])


def _font_family_available(family: str) -> bool:
    fc_match = shutil.which("fc-match")
    if not fc_match:
        return False
    try:
        matched = subprocess.check_output(
            [fc_match, "-f", "%{family}\n", family],
            timeout=15,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — verified explicitly by ensure_noto_cjk_fonts
        return False
    return family.lower() in matched.lower()


def _noto_font_parent(path: Path) -> Path | None:
    try:
        for item in path.rglob("*"):
            name = item.name.lower()
            if (
                item.is_file()
                and item.suffix.lower() in {".otf", ".ttf", ".ttc"}
                and "noto" in name
                and any(
                    token in name
                    for token in ("cjk", "sanssc", "sansjp", "sanskr")
                )
            ):
                return item.parent
    except OSError:
        pass
    return None


def _noto_fonts_dir() -> Path | None:
    explicit = (os.environ.get("ANIME_FACTORY_FONTS_DIR") or "").strip()
    candidates = (
        [Path(explicit)]
        if explicit
        else [
            Path("/usr/share/fonts/opentype/noto"),
            Path("/usr/share/fonts/truetype/noto"),
            Path("/usr/share/fonts"),
        ]
    )
    for path in candidates:
        if path.is_dir():
            parent = _noto_font_parent(path)
            if parent is not None:
                return parent
    return None


def ensure_noto_cjk_fonts() -> dict:
    """Install and verify fontconfig-visible SC/JP/KR fonts; never render tofu silently."""
    fonts_dir = _noto_fonts_dir()
    missing = [family for family in NOTO_CJK_FAMILIES if not _font_family_available(family)]
    if not shutil.which("fc-list") or missing or fonts_dir is None:
        apt = shutil.which("apt-get")
        if not apt:
            raise RuntimeError(
                "font_setup_failed: apt-get unavailable; need fontconfig and fonts-noto-cjk"
            )
        try:
            subprocess.check_call([apt, "update", "-qq"])
            subprocess.check_call(
                [apt, "install", "-y", "-qq", "fontconfig", "fonts-noto-cjk"]
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"font_setup_failed: package installation failed: {exc}"
            ) from exc
        fc_cache = shutil.which("fc-cache")
        if not fc_cache:
            raise RuntimeError("font_setup_failed: fc-cache missing after package install")
        try:
            subprocess.check_call([fc_cache, "-f"])
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"font_setup_failed: fc-cache failed: {exc}") from exc
        fonts_dir = _noto_fonts_dir()
        missing = [
            family for family in NOTO_CJK_FAMILIES if not _font_family_available(family)
        ]
    if fonts_dir is None:
        raise RuntimeError("font_setup_failed: Noto CJK font directory not found")
    if not shutil.which("fc-list"):
        raise RuntimeError("font_setup_failed: fc-list missing after package install")
    if missing:
        raise RuntimeError(f"font_setup_failed: fontconfig families missing: {missing}")
    os.environ["ANIME_FACTORY_FONTS_DIR"] = str(fonts_dir)
    return {
        "fonts_dir": str(fonts_dir),
        "families": list(NOTO_CJK_FAMILIES),
        "verified": True,
    }


def _comfy_requirement_name(line: str) -> str:
    pkg = line.strip().split("#", 1)[0].strip()
    if not pkg:
        return ""
    return pkg.split("[", 1)[0].split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip().lower()


def filtered_comfy_requirements(req_path: Path) -> Path:
    """ComfyUI requirements pin `torch` without a CUDA index — never replace the baked cu128 wheel."""
    filtered = req_path.parent / ".af-comfy-requirements.filtered.txt"
    kept: list[str] = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        name = _comfy_requirement_name(line)
        if not name or name in _TORCH_REQ_PACKAGES:
            continue
        kept.append(line)
    filtered.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return filtered


def ensure_comfy_reqs() -> None:
    if os.environ.get("AF_COMFY_REQS_BAKED", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    req = COMFY_DIR / "requirements.txt"
    if not req.is_file():
        return
    filtered = filtered_comfy_requirements(req)
    if not filtered.read_text(encoding="utf-8").strip():
        return
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(filtered),
            "--extra-index-url",
            torch_index_url(),
        ]
    )


def _kitchen_patch_files(roots: list[Path] | None = None) -> list[Path]:
    files: list[Path] = []
    if roots is None:
        roots = []
        try:
            import site

            for base in site.getsitepackages():
                kitchen = Path(base) / "comfy_kitchen"
                if kitchen.is_dir():
                    roots.append(kitchen)
        except Exception:  # noqa: BLE001 — probe only
            pass
        fallback = (
            Path("/usr/local/lib")
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "comfy_kitchen"
        )
        if fallback.is_dir() and fallback not in roots:
            roots.append(fallback)
        quant = COMFY_DIR / "comfy" / "quant_ops.py"
        if quant.is_file():
            roots.append(quant)
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("*.py"))
    return files


_SITECUSTOMIZE_INFER_SCHEMA = '''"""Patch torch.library.infer_schema after torch loads. Do not import torch at site startup."""
from __future__ import annotations

import sys
import types
import typing

_af_patched = False


def _rewrite(ann):
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is list:
        inner = _rewrite(args[0]) if args else typing.Any
        return typing.List[inner]
    if origin is types.UnionType or origin is typing.Union:
        parts = [_rewrite(a) for a in args]
        non_none = [a for a in parts if a is not type(None)]
        if type(None) in parts and len(non_none) == 1:
            return typing.Optional[non_none[0]]
        return typing.Union[tuple(parts)]
    return ann


def _install() -> None:
    global _af_patched
    if _af_patched:
        return
    import torch.library

    orig = torch.library.infer_schema

    def infer_schema(fn, *a, **kw):
        anns = getattr(fn, "__annotations__", None)
        if anns:
            try:
                fn.__annotations__ = {k: _rewrite(v) for k, v in anns.items()}
            except Exception:
                pass
        return orig(fn, *a, **kw)

    torch.library.infer_schema = infer_schema
    try:
        import torch._library.infer_schema as mod

        mod.infer_schema = infer_schema
    except Exception:
        pass
    _af_patched = True


class _AfTorchPatchFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname != "torch":
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            orig = spec.loader

            class _Loader:
                def create_module(self, spec):
                    cm = getattr(orig, "create_module", None)
                    return cm(spec) if cm else None

                def exec_module(self, module):
                    orig.exec_module(module)
                    try:
                        _install()
                    except Exception:
                        pass

            spec.loader = _Loader()
            return spec
        return None


sys.meta_path.insert(0, _AfTorchPatchFinder())
'''


def ensure_infer_schema_sitecustomize() -> Path | None:
    """torch 2.6 infer_schema rejects PEP 585/604; patch after torch loads, never at site import."""
    dest = (
        Path("/usr/local/lib")
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "sitecustomize.py"
    )
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        existing = dest.read_text(encoding="utf-8") if dest.is_file() else ""
        if "_AfTorchPatchFinder" in existing and "import torch" not in existing.split("def _install", 1)[0]:
            return dest
        dest.write_text(_SITECUSTOMIZE_INFER_SCHEMA, encoding="utf-8")
        return dest
    except OSError:
        return None


def patch_kitchen_triton_optional() -> bool:
    """Hub image may lack gcc; triton backend compile must not block Comfy import."""
    path = (
        Path("/usr/local/lib")
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "comfy_kitchen"
        / "__init__.py"
    )
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    old = "from .backends import triton as _triton_backend  # noqa: F401"
    if old not in text or "triton backend skipped" in text:
        return False
    new = (
        "try:\n"
        "    from .backends import triton as _triton_backend  # noqa: F401\n"
        "except Exception as _e:\n"
        "    print('comfy_kitchen: triton backend skipped', _e)\n"
        "    _triton_backend = None"
    )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_comfy_kitchen_for_torch26(roots: list[Path] | None = None) -> list[str]:
    """comfy_kitchen uses list[int]; torch 2.6 infer_schema only accepts typing.List[int]."""
    hits: list[str] = []
    for path in _kitchen_patch_files(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        orig = text
        if "list[int]" not in text and "list[bool]" not in text:
            continue
        if "import typing" not in text and "from typing import" not in text:
            lines = text.splitlines(keepends=True)
            out: list[str] = []
            inserted = False
            for i, line in enumerate(lines):
                out.append(line)
                if line.strip().startswith("from __future__ import"):
                    nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    if not nxt.startswith("from __future__ import"):
                        out.append("import typing\n")
                        inserted = True
            text = "".join(out) if inserted else "import typing\n" + text
        text = text.replace("list[int]", "typing.List[int]").replace("list[bool]", "typing.List[bool]")
        if text != orig:
            path.write_text(text, encoding="utf-8")
            hits.append(str(path))
    return hits


def _port_open(host: str, port: int) -> bool:
    import socket

    sock = socket.socket()
    sock.settimeout(0.4)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def want_comfy_lowvram(vram_mb: int | None = None) -> bool:
    """Split UNet only on sub-32GB cards. 32GB NVFP4 sampling must stay GPU-resident."""
    from gpu_worker.h3 import H3_NATIVE_VRAM_MB, gpu_vram_mb

    probed = gpu_vram_mb() if vram_mb is None else vram_mb
    if probed is not None and probed >= H3_NATIVE_VRAM_MB:
        return False
    return os.environ.get("AF_COMFY_LOWVRAM", "").strip().lower() in {"1", "true", "yes", "on"}


def comfy_launch_args(
    main_py: Path | None = None,
    extra_paths: Path | None = None,
    vram_mb: int | None = None,
) -> list[str]:
    """Comfy argv. Default VRAM mode unloads TE after encode; do not --highvram."""
    args = [
        sys.executable,
        str(main_py or (COMFY_DIR / "main.py")),
        "--listen",
        "127.0.0.1",
        "--port",
        "8188",
        "--disable-auto-launch",
    ]
    if want_comfy_lowvram(vram_mb):
        args.append("--lowvram")
    extra = extra_paths if extra_paths is not None else (COMFY_DIR / "extra_model_paths.yaml")
    if extra.is_file():
        args.extend(["--extra-model-paths-config", str(extra)])
    return args


def comfy_boot_log_tail(max_lines: int = 40) -> str:
    if not COMFY_BOOT_LOG.is_file():
        return ""
    try:
        lines = COMFY_BOOT_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max(1, max_lines) :])


def _start_comfy_logged(args: list[str]) -> subprocess.Popen:
    COMFY_BOOT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with COMFY_BOOT_LOG.open("ab") as logf:
        logf.write(f"\n--- comfy start {_utc_iso()} ---\n".encode())
    log_handle = COMFY_BOOT_LOG.open("ab", buffering=0)
    env = os.environ.copy()
    return subprocess.Popen(args, env=env, stdout=log_handle, stderr=subprocess.STDOUT)


def comfy_process_exited(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is not None


def wait_comfy_process(
    proc: subprocess.Popen | None,
    *,
    timeout_s: float = 120.0,
    interval_s: float = 1.0,
) -> dict:
    """Fail fast when Comfy exits before :8188 is ready."""
    if proc is None:
        return {"ok": True, "skipped": True}
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            tail = comfy_boot_log_tail()
            return {"ok": False, "exit_code": code, "tail": tail[-1200:] if tail else ""}
        if _port_open("127.0.0.1", 8188):
            return {"ok": True, "exit_code": None}
        time.sleep(min(interval_s, max(0.0, deadline - time.monotonic())))
    if proc.poll() is not None:
        tail = comfy_boot_log_tail()
        return {"ok": False, "exit_code": proc.poll(), "tail": tail[-1200:] if tail else ""}
    return {"ok": True, "exit_code": None, "waiting": True}


def start_comfy() -> subprocess.Popen | None:
    main_py = COMFY_DIR / "main.py"
    if not main_py.is_file():
        print(json.dumps({"comfy": "missing", "dir": str(COMFY_DIR)}), flush=True)
        return None
    if _port_open("127.0.0.1", 8188):
        print(json.dumps({"comfy": "already_up", "port": 8188}), flush=True)
        return None
    try:
        COMFY_BOOT_LOG.write_text("", encoding="utf-8")
    except OSError:
        pass
    return _start_comfy_logged(comfy_launch_args(main_py))


_COMFY_LONGLIVE_PATTERNS = ("ComfyUI/main.py", "gpu_worker.router")
# Driver + empty CUDA context on a 5090 is well under this; Animagine/Comfy is not.
_VRAM_VACATE_MB = 2048
_VRAM_VACATE_S = 30.0


def _gpu_memory_used_mb() -> int | None:
    smi = shutil.which("nvidia-smi") or "nvidia-smi"
    try:
        raw = subprocess.check_output(
            [smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — vacate can still proceed
        return None
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    try:
        return int(float(line.strip()))
    except ValueError:
        return None


def _wait_gpu_memory_below(limit_mb: int, timeout_s: float) -> int | None:
    deadline = time.monotonic() + timeout_s
    last = _gpu_memory_used_mb()
    while True:
        if last is None or last <= limit_mb:
            return last
        if time.monotonic() >= deadline:
            return last
        time.sleep(0.5)
        last = _gpu_memory_used_mb()


def stop_comfy_for_longlive() -> dict:
    """Vacate Comfy/router so LongLive can own the 32GB card.

    Stills (Animagine via Comfy) run first; video is the last heavy visual GPU
    step. Unloading Flux VRAM is not enough: leftover Comfy plus UMT5 used to
    SIGKILL a 32GB cgroup, and a fire-and-forget pkill left the NVFP4 sampler
    competing for the same 32GB. MOSS SFX still needs GPU after LongLive
    ``release_model``.
    """
    killed: list[str] = []
    for sig in ("-TERM", "-KILL"):
        for pattern in _COMFY_LONGLIVE_PATTERNS:
            try:
                proc = subprocess.run(["pkill", sig, "-f", pattern], check=False, timeout=15)
                killed.append(f"{sig}:{pattern}:{proc.returncode}")
            except Exception as exc:  # noqa: BLE001 — inference can still try
                killed.append(f"{sig}:{pattern}:err:{type(exc).__name__}")
        time.sleep(0.4 if sig == "-TERM" else 0.1)
    used = _wait_gpu_memory_below(_VRAM_VACATE_MB, _VRAM_VACATE_S)
    try:
        import torch

        cuda = getattr(torch, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
            cuda.empty_cache()
    except Exception:  # noqa: BLE001 — LongLive can still try on a CPU test box
        pass
    print(
        json.dumps({"comfy": "stopped_for_longlive", "killed": killed, "vram_used_mb": used}),
        flush=True,
    )
    return {"ok": True, "killed": killed, "vram_used_mb": used}


def start_router() -> subprocess.Popen | None:
    if _port_open("127.0.0.1", 8199):
        print(json.dumps({"router": "already_up", "port": 8199}), flush=True)
        return None
    return _popen([sys.executable, "-m", "gpu_worker.router"])


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _tunnel_hostname() -> str:
    raw = (
        os.environ.get("CLOUDFLARE_TUNNEL_HOSTNAME")
        or os.environ.get("GPU_TUNNEL_URL")
        or DEFAULT_TUNNEL_HOSTNAME
    ).strip()
    parsed = urllib.parse.urlparse(raw if "://" in raw else f"https://{raw}")
    return parsed.hostname or raw


def _set_tunnel_status(connected: bool, *, configured: bool) -> dict:
    global _TUNNEL_STATUS
    _TUNNEL_STATUS = {
        "connected": bool(connected),
        "hostname": _tunnel_hostname() if configured else None,
        "checked_at": _utc_iso(),
    }
    return dict(_TUNNEL_STATUS)


def current_tunnel_status() -> dict:
    """Heartbeat-safe status, including an explicit unconfigured state."""
    if _TUNNEL_STATUS is None:
        configured = bool((os.environ.get("CLOUDFLARE_TUNNEL_TOKEN") or "").strip())
        return _set_tunnel_status(False, configured=configured)
    return dict(_TUNNEL_STATUS)


def _tunnel_metrics_connected(metrics_url: str = TUNNEL_METRICS_URL) -> bool:
    """cloudflared /ready is 200 only with an active connection to the edge."""
    parsed = urllib.parse.urlparse(metrics_url)
    ready_url = urllib.parse.urlunparse(parsed._replace(path="/ready", query="", fragment=""))
    req = urllib.request.Request(ready_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            status = getattr(resp, "status", 200)
            body = json.loads(resp.read().decode("utf-8", "replace") or "{}")
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    except json.JSONDecodeError:
        return False
    try:
        ready_connections = int(body.get("readyConnections") or 0)
    except (AttributeError, TypeError, ValueError):
        return False
    return status == 200 and ready_connections > 0


def verify_tunnel_connection(timeout_s: float | None = None, interval_s: float = 1.0) -> dict:
    """Wait for cloudflared's edge-connection metric, not merely a live PID."""
    configured = bool((os.environ.get("CLOUDFLARE_TUNNEL_TOKEN") or "").strip())
    if not configured:
        return _set_tunnel_status(False, configured=False)
    timeout = (
        float(os.environ.get("CLOUDFLARE_TUNNEL_CHECK_TIMEOUT_S") or 30)
        if timeout_s is None
        else max(0.0, float(timeout_s))
    )
    deadline = time.monotonic() + timeout
    while True:
        if _tunnel_metrics_connected():
            return _set_tunnel_status(True, configured=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _set_tunnel_status(False, configured=True)
        time.sleep(min(interval_s, remaining))


def start_tunnel() -> subprocess.Popen | None:
    token = (os.environ.get("CLOUDFLARE_TUNNEL_TOKEN") or "").strip()
    if not token:
        _set_tunnel_status(False, configured=False)
        print(json.dumps({"tunnel": "not_configured"}), flush=True)
        return None
    if _tunnel_metrics_connected():
        _set_tunnel_status(True, configured=True)
        print(json.dumps({"tunnel": "already_up"}), flush=True)
        return None
    cf = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
    if not Path(cf).is_file() and not shutil.which("cloudflared"):
        _set_tunnel_status(False, configured=True)
        print(json.dumps({"tunnel": "cloudflared_missing"}), flush=True)
        return None
    _set_tunnel_status(False, configured=True)
    return _popen(
        [
            cf,
            "tunnel",
            "--no-autoupdate",
            "--metrics",
            urllib.parse.urlparse(TUNNEL_METRICS_URL).netloc,
            "run",
            "--token",
            token,
        ]
    )


def wait_router_ready(
    timeout_s: float = 900.0,
    progress: Callable[[str], None] | None = None,
) -> bool:
    def probe():
        from gpu_worker.boot import ReadyProbe

        ok = probe_comfy_system_stats(ROUTER_URL)
        if progress:
            progress(f"startup_health:{'ready' if ok else 'waiting'}")
        return ReadyProbe(ssh_echo_ok=True, nvidia_smi_ok=True, comfy_required=True, comfy_system_stats_ok=ok)

    last = health_wait_loop(probe, timeout_s=timeout_s, interval_s=5.0, production=False)
    return last.ready


def _attach_adaptations(handshake: dict | None, adapter: StackAdapter, **extra: Any) -> dict:
    payload = dict(handshake or {})
    records = adapter.as_list()
    payload["adaptations"] = records
    preflight = dict(payload.get("preflight") or {})
    preflight["adaptations"] = records
    payload["preflight"] = preflight
    payload.update(extra)
    return payload


def boot_gpu_stack(progress: Callable[[str], None] | None = None) -> dict:
    profile_id = select_profile_id()
    handshake: dict | None = None
    adapter = StackAdapter()
    try:
        adapter.fix_workdir()
    except OSError:
        pass
    try:
        adapter.fix_onstart()
    except OSError:
        pass
    if progress:
        progress("startup_stage:preflight")
    try:
        handshake = run_hardware_preflight(profile_id=profile_id)
    except PreflightFailure as exc:
        if progress:
            progress("startup_health:failed")
        payload = handshake_payload(
            profile_id=profile_id,
            preflight_ok=False,
            preflight_stage="hardware",
            preflight_error=str(exc),
            failure_class=exc.failure_class,
            adaptations=adapter.as_list(),
        )
        print(json.dumps({"preflight_failed": exc.as_dict(), "handshake": payload}, ensure_ascii=False), flush=True)
        try:
            from gpu_worker.session import persist_boot_failure

            persist_boot_failure(
                {
                    "preflight_failed": exc.as_dict(),
                    "handshake": payload,
                    "failure_class": exc.failure_class,
                }
            )
        except Exception:
            pass
        raise
    if progress:
        progress("startup_stage:torch")
    try:
        ensure_torch()
    except PreflightFailure as exc:
        adapter.record_unfixable("torch_abi", before={"sm": cuda_sm()}, error=str(exc))
        raise
    if progress:
        progress("startup_stage:compiler")
    adapter.apply("ensure_headers", before={"need": True}, repair=lambda: ensure_c_compiler() or {"ok": True})
    if progress:
        progress("startup_stage:comfy_requirements")
    ensure_comfy_reqs()
    if progress:
        progress("startup_stage:fonts")
    fonts = adapter.apply("ensure_fonts", before={"verified": False}, repair=ensure_noto_cjk_fonts).get("after") or ensure_noto_cjk_fonts()
    if progress:
        progress("startup_stage:compatibility_patches")
    ensure_infer_schema_sitecustomize()
    patch_kitchen_triton_optional()
    patch_comfy_kitchen_for_torch26()
    try:
        backend = lock_video_backend()
    except VideoBackendLockError as exc:
        raise PreflightFailure(
            "capability_mismatch",
            "video_backend",
            str(exc),
            {"requested": os.environ.get("AF_VIDEO_BACKEND"), "image": os.environ.get("AF_IMAGE_CAPABILITY")},
        ) from exc
    longlive = backend == "longlive"
    h3_thread = None
    if longlive:
        if progress:
            progress("startup_stage:weights:h3_skipped_longlive")
            progress("startup_stage:weights:stills")
        weights = ensure_still_weights(COMFY_DIR, progress=progress)
        if progress:
            progress("startup_stage:start_comfy")
        comfy = start_comfy()
        comfy_wait = wait_comfy_process(
            comfy,
            timeout_s=float(os.environ.get("AF_COMFY_BOOT_WAIT_S") or COMFY_STARTUP_TIMEOUT_S),
        )
        comfy_error: str | None = None
        if not comfy_wait.get("ok"):
            restarted_box: dict[str, Any] = {}

            def _restart_ll() -> dict:
                restarted = start_comfy()
                waited = wait_comfy_process(
                    restarted,
                    timeout_s=float(os.environ.get("AF_COMFY_BOOT_WAIT_S") or COMFY_STARTUP_TIMEOUT_S),
                )
                restarted_box["proc"] = restarted
                restarted_box["wait"] = waited
                return {"wait_ok": bool(waited.get("ok")), "exit_code": waited.get("exit_code")}

            rec = adapter.apply("restart_comfy", before={"exit_code": comfy_wait.get("exit_code")}, repair=_restart_ll)
            waited = restarted_box.get("wait") or comfy_wait
            if rec.get("result") == "ok" and isinstance(waited, dict) and waited.get("ok"):
                comfy = restarted_box.get("proc") or comfy
                comfy_wait = waited
            else:
                code = comfy_wait.get("exit_code")
                tail = str(comfy_wait.get("tail") or "").strip()
                comfy_error = f"comfy_startup_failed:exit_{code}"
                if tail:
                    comfy_error = f"{comfy_error}:{tail[-400:]}"
        if progress:
            progress("startup_stage:start_router")
        router = start_router()
        time.sleep(2)
        if progress:
            progress("startup_stage:start_tunnel")
        tunnel = start_tunnel()
        if progress:
            progress("startup_stage:wait_comfy")
        ready = False if comfy_error else wait_router_ready(progress=progress)
        caps = default_register_capabilities()
        if handshake is None:
            handshake = handshake_payload(profile_id=profile_id, adaptations=adapter.as_list())
        handshake = _attach_adaptations(handshake, adapter)
        if progress:
            progress(f"startup_health:{'ready' if ready else 'failed'}")
        tunnel_status = verify_tunnel_connection(timeout_s=0 if tunnel is None else None)
        return {
            "comfy_pid": getattr(comfy, "pid", None),
            "router_pid": getattr(router, "pid", None),
            "tunnel_pid": getattr(tunnel, "pid", None),
            "tunnel": tunnel_status,
            "router_ready": ready,
            "comfy_error": comfy_error,
            "comfy_skipped_longlive": False,
            "h3_skipped_longlive": True,
            "fonts": fonts,
            "capabilities": caps,
            "handshake": handshake,
            "adaptations": adapter.as_list(),
            "profile_id": profile_id,
            "startup_downloaded_bytes": runtime_weight_bytes(COMFY_DIR),
            "h3_weights_background": False,
            "video_backend": backend,
            "weights": {
                k: weights.get(k)
                for k in ("hits", "misses", "downloaded", "source", "kind")
                if k in weights
            },
        }
    if progress:
        progress("startup_stage:weights:h3_background")
    h3_thread = start_h3_weights_background(COMFY_DIR, progress=progress)
    if progress:
        progress("startup_stage:weights:stills")
    weights = ensure_still_weights(COMFY_DIR, progress=progress)
    if progress:
        progress("startup_stage:start_comfy")
    comfy = start_comfy()
    comfy_wait = wait_comfy_process(
        comfy,
        timeout_s=float(os.environ.get("AF_COMFY_BOOT_WAIT_S") or COMFY_STARTUP_TIMEOUT_S),
    )
    comfy_error: str | None = None
    if not comfy_wait.get("ok"):
        restarted_box: dict[str, Any] = {}

        def _restart() -> dict:
            restarted = start_comfy()
            waited = wait_comfy_process(
                restarted,
                timeout_s=float(os.environ.get("AF_COMFY_BOOT_WAIT_S") or COMFY_STARTUP_TIMEOUT_S),
            )
            restarted_box["proc"] = restarted
            restarted_box["wait"] = waited
            return {"wait_ok": bool(waited.get("ok")), "exit_code": waited.get("exit_code")}

        rec = adapter.apply("restart_comfy", before={"exit_code": comfy_wait.get("exit_code")}, repair=_restart)
        waited = restarted_box.get("wait") or comfy_wait
        if rec.get("result") == "ok" and isinstance(waited, dict) and waited.get("ok"):
            comfy = restarted_box.get("proc") or comfy
            comfy_wait = waited
        else:
            code = comfy_wait.get("exit_code")
            tail = str(comfy_wait.get("tail") or "").strip()
            comfy_error = f"comfy_startup_failed:exit_{code}"
            if tail:
                comfy_error = f"{comfy_error}:{tail[-400:]}"
            print(
                json.dumps(
                    {
                        "comfy": "exited_before_ready",
                        "exit_code": code,
                        "tail": tail[-800:] if tail else "",
                        "adaptations": adapter.as_list(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if progress:
        progress("startup_stage:start_router")
    router = start_router()
    time.sleep(2)
    if progress:
        progress("startup_stage:start_tunnel")
    tunnel = start_tunnel()
    if progress:
        progress("startup_stage:wait_comfy")
    ready = False if comfy_error else wait_router_ready(progress=progress)
    if not comfy_error and comfy_process_exited(comfy):
        restarted_box: dict[str, Any] = {}
        rec = adapter.apply(
            "restart_comfy",
            before={"exited": True},
            repair=lambda: restarted_box.update(proc=start_comfy()) or {"pid": getattr(restarted_box.get("proc"), "pid", None)},
        )
        if rec.get("result") == "ok":
            comfy = restarted_box.get("proc") or comfy
        if comfy_process_exited(comfy):
            tail = comfy_boot_log_tail()
            comfy_error = f"comfy_startup_failed:exit_{comfy.poll() if comfy else 'unknown'}"
            if tail:
                comfy_error = f"{comfy_error}:{tail[-400:]}"
            ready = False
    if not ready and not comfy_error:
        adapter.apply("restart_router", before={"ready": False}, repair=lambda: start_router())
        ready = wait_router_ready(progress=progress)
    comfy_nodes: dict | None = None
    if ready and not comfy_error:
        if progress:
            progress("startup_stage:comfy_probe")
        try:
            comfy_nodes = probe_comfy_nodes(ROUTER_URL)
        except PreflightFailure as exc:
            adapter.record_unfixable("node_api", before={"error": str(exc)}, error=str(exc))
            comfy_error = str(exc)
            ready = False
    if progress:
        progress(f"startup_health:{'ready' if ready else 'failed'}")
    tunnel_status = verify_tunnel_connection(timeout_s=0 if tunnel is None else None)
    token = (os.environ.get("CLOUDFLARE_TUNNEL_TOKEN") or "").strip()
    if token and not tunnel_status.get("connected"):
        adapter.apply(
            "restart_cloudflared",
            before={"connected": False},
            repair=lambda: start_tunnel(),
        )
        tunnel_status = verify_tunnel_connection(timeout_s=0)
    caps = default_register_capabilities()
    if handshake is None:
        handshake = handshake_payload(
            profile_id=profile_id,
            comfy_nodes=comfy_nodes,
            adaptations=adapter.as_list(),
        )
    elif comfy_nodes:
        handshake = handshake_payload(
            profile_id=profile_id,
            host=handshake.get("host"),
            stack=handshake.get("stack"),
            comfy_nodes=comfy_nodes,
            adaptations=adapter.as_list(),
        )
    handshake = _attach_adaptations(handshake, adapter)
    return {
        "comfy_pid": getattr(comfy, "pid", None),
        "router_pid": getattr(router, "pid", None),
        "tunnel_pid": getattr(tunnel, "pid", None),
        "tunnel": tunnel_status,
        "router_ready": ready,
        "comfy_error": comfy_error,
        "comfy_boot_log": str(COMFY_BOOT_LOG),
        "comfy_boot_log_tail": comfy_boot_log_tail()[-1200:] or None,
        "fonts": fonts,
        "capabilities": caps,
        "handshake": handshake,
        "adaptations": adapter.as_list(),
        "profile_id": profile_id,
        "startup_downloaded_bytes": runtime_weight_bytes(COMFY_DIR),
        "h3_weights_background": h3_thread is not None,
        "weights": {
            k: weights.get(k)
            for k in ("hits", "misses", "downloaded", "source", "kind")
            if k in weights
        },
    }

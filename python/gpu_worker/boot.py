"""Vast GPU boot/lease state machine.

Why the spec talks about 4 cards (ANIME_FACTORY.md §12.5–12.6)
----------------------------------------------------------------
A 600s episode is ~9.65 GPU-hours on one 5090. Three episodes plus the
H3 NVFP4 + anime SDXL cold start plan at 30–42 lease-hours, inside the corrected
`VAST_MAX_LEASE_MINUTES=3600` default. Scene-modulo sharding across N cards
is an optional throughput tool. Default remains **one card**. That is **not**
a reason to lease during bible / TTS. TTS stays SiliconFlow CosyVoice2.
GPU session = 生图 (anime SDXL/Comfy, missing/regen stills) + H3 anim + qc + compose.
Lease only after script / tts / board exist. Do not lease a comfy:false image.

Four historical failures this module encodes
--------------------------------------------
1. **TTS-before-lease** — CosyVoice 404 (wrong upload URL
   `/v1/audio/voice/upload` instead of `/v1/uploads/audio/voice`) while a
   Vast box was already burning money; a naive `finally: destroy` then
   killed the instance.
2. **running-but-no-ssh** — Vast `cur_state=running` with
   `actual_status is None` and empty ports (KR 5090 class). SSH / nvidia-smi
   / comfy never came up. Running ≠ ready. Cap the wait; do **not** flip SKU.
3. **set-e health kill** — bash `set -e` + missing `seq` aborted the health
   script after ~12s; the orchestrator destroyed the box while Comfy was
   still installing. Health wait is a Python loop (10–20 min).
4. **agent-image-without-comfy** — never PUT /asks/ for an image with
   `comfy/h3/kolors` false. Hub `aetherneo/anime-factory-gpu` is now
   Comfy+H3+SDXL stills; probe `:8199` on that image. Legacy agent-only names
   still skip :8199 and must not be leased.

Destroy policy
--------------
`VAST_ALLOW_REPLACE=0`. One lease at a time. Install failure → retry on the
**same** instance. `finally: destroy` only after success, idle TTL, or
explicit abort — never on TTS 404 or health-script bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from anime_factory.config import load_settings
from gpu_worker.images import image_can_lease, image_capabilities, should_probe_comfy
from gpu_worker.vast_client import VastClient, VastSafetyError

# Hosted (SiliconFlow / DeepSeek) must finish before PUT /asks/.
# timing folds into the tts box. design/keyframe stills MAY run on GPU.
HOSTED_STAGES = (
    "bible",
    "canon",
    "script",
    "tts",
    "board",
)
PRE_LEASE_STAGES = (
    "bible",
    "canon",
    "script",
    "tts",
    "timing",
    "board",
)
GPU_STILL_STAGES = ("design", "keyframe")
GPU_SESSION_STAGES = ("anim", "qc", "compose")
GPU_ALLOWED_STAGES = GPU_STILL_STAGES + GPU_SESSION_STAGES

# SiliconFlow CosyVoice2 — the working upload path (MVP 404'd on the other one).
VOICE_UPLOAD_PATH = "/v1/uploads/audio/voice"
WRONG_VOICE_UPLOAD_PATH = "/v1/audio/voice/upload"

DESTROY_REASONS = frozenset({"success", "idle_ttl", "explicit_abort"})
NEVER_DESTROY_ERRORS = frozenset(
    {
        "tts_404",
        "voice_upload_404",
        "health_script_bug",
        "install_failure",
        "set_e_health_kill",
    }
)

# Deterministic boot/capability failures — control plane should recycle quickly.
RECYCLE_FAILURE_CLASSES = frozenset(
    {
        "preflight_failed",
        "capability_mismatch",
        "comfy_startup_failed",
        "weight_pull_timeout",
    }
)

HEALTH_WAIT_MIN_S = 10 * 60
HEALTH_WAIT_MAX_S = 20 * 60
DEFAULT_HEALTH_WAIT_S = 15 * 60
READY_WAIT_CAP_S = 20 * 60
COMFY_ROUTER_PORT = 8199
COMFY_STATS_PATH = "/system_stats"

# Why N>1 exists — encoded so orchestrators do not "just lease 4" at t=0.
WHY_MULTI_CARD = (
    "One 5090 takes ~9.65 h per 600s episode; three episodes plus cold boot "
    "plan at 30–42 h inside VAST_MAX_LEASE_MINUTES=3600. "
    "N cards = N Comfy backends and optional scene-modulo throughput, "
    "after each boot machine reaches ready. Default VAST_LEASE_COUNT=1 until "
    "SSH+nvidia-smi+comfy succeed first-try. One card runs 生图 (anime SDXL) + H3; "
    "TTS stays SiliconFlow. Do not lease comfy:false / h3:false / kolors:false."
)

STATES = (
    "idle",
    "preflight",
    "leasing",
    "leased",
    "waiting_ssh",
    "waiting_gpu",
    "waiting_comfy",
    "ready",
    "running",
    "install_retry",
    "wait_capped",
    "destroying",
    "destroyed",
    "aborted",
)


class PreLeaseError(RuntimeError):
    """GPU lease refused — hosted script/TTS/board work is not done."""


class ImageCapabilityError(RuntimeError):
    """GPU lease refused — Hub image cannot run Comfy + H3 + 生图."""


class OneLeaseError(RuntimeError):
    """VAST_ALLOW_REPLACE=0: only one live instance."""


class SkuFlipError(RuntimeError):
    """Wait capped; do not search a new SKU / replace the offer."""


class DestroyRefused(RuntimeError):
    """finally: destroy is not allowed for this error class."""


@dataclass
class ReadyProbe:
    ssh_echo_ok: bool = False
    nvidia_smi_ok: bool = False
    comfy_system_stats_ok: bool = False
    comfy_required: bool = True

    @property
    def ready(self) -> bool:
        if not (self.ssh_echo_ok and self.nvidia_smi_ok):
            return False
        if self.comfy_required and not self.comfy_system_stats_ok:
            return False
        return True

    def missing(self) -> list[str]:
        out = []
        if not self.ssh_echo_ok:
            out.append("ssh_echo")
        if not self.nvidia_smi_ok:
            out.append("nvidia_smi")
        if self.comfy_required and not self.comfy_system_stats_ok:
            out.append("comfy_system_stats")
        return out


@dataclass
class LeaseSession:
    """One Vast instance. Install failures retry here; they do not replace."""

    offer_id: str | None = None
    instance_id: str | None = None
    image: str = "docker.io/aetherneo/anime-factory-gpu:main"
    sku: str | None = None
    state: str = "idle"
    capabilities: dict = field(default_factory=dict)
    destroy_reason: str | None = None
    last_error: str | None = None
    install_attempts: int = 0
    probes: ReadyProbe = field(default_factory=ReadyProbe)
    lease_env: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.capabilities:
            self.capabilities = image_capabilities(self.image)
        self.probes.comfy_required = bool(self.capabilities.get("comfy"))


def assert_pre_lease(flags: dict[str, str] | None) -> None:
    """Never lease GPU before bible/canon/script/tts/timing/board. TTS is SiliconFlow."""
    if not flags:
        raise PreLeaseError("GPU lease refused; missing stage flags (TTS/board not proven)")
    missing = [s for s in PRE_LEASE_STAGES if flags.get(s) != "passed"]
    if missing:
        raise PreLeaseError(f"GPU session refused; pre-lease stages not passed: {missing}")


def assert_lease_capabilities(capabilities: dict | None) -> None:
    """Refuse PUT /asks/ unless the image reports comfy + 生图 + one video line."""
    if image_can_lease(capabilities):
        return
    raise ImageCapabilityError(
        "refuse lease: image must have comfy:true, kolors/image_gen:true, and "
        "exactly one of h3:true or longlive:true; do not PUT /asks/ for a card "
        "that cannot 生图 + video"
    )


def assert_gpu_session_only(stage: str) -> None:
    if stage not in GPU_ALLOWED_STAGES:
        raise PreLeaseError(
            f"GPU session is 生图 (design/keyframe remaining) + anim+qc+compose; "
            f"refused stage {stage!r} (TTS = SiliconFlow, no Vast)"
        )


def voice_upload_url_ok(url: str) -> bool:
    if WRONG_VOICE_UPLOAD_PATH in url:
        return False
    return VOICE_UPLOAD_PATH in url


def vast_payload_not_ready_reason(payload: dict | None) -> str | None:
    """`cur_state=running` is NOT ready. KR 5090 class: actual_status None + empty ports."""
    if not payload:
        return "empty_payload"
    actual = payload.get("actual_status", payload.get("actualStatus"))
    ports = payload.get("ports") or payload.get("public_ipaddr") or payload.get("ssh_host")
    empty_ports = ports in (None, "", {}, [])
    if actual is None and empty_ports:
        return "actual_status_none_empty_ports"
    cur = payload.get("cur_state") or payload.get("status") or payload.get("intended_status")
    ssh_ok = bool(payload.get("ssh_host") or payload.get("ssh_idx") or payload.get("ssh_port"))
    if cur == "running" and not ssh_ok and empty_ports:
        return "running_but_no_ssh"
    return None


def is_vast_api_running_enough(payload: dict | None, probes: ReadyProbe) -> bool:
    """Ready = SSH echo OK AND nvidia-smi AND (if comfy) :8199 /system_stats 200."""
    if vast_payload_not_ready_reason(payload):
        return False
    return probes.ready


def bash_health_script_is_unsafe(script: str) -> bool:
    """The 12s killer: `set -e` plus `seq` (often missing on slim images)."""
    text = script or ""
    has_set_e = "set -e" in text or "set -o errexit" in text
    uses_seq = " seq " in f" {text} " or text.strip().startswith("seq ") or "$(seq" in text
    return has_set_e and uses_seq


def production_health_wait_seconds(seconds: int | None = None) -> int:
    """Clamp production Comfy-install waits to 10–20 minutes."""
    raw = DEFAULT_HEALTH_WAIT_S if seconds is None else int(seconds)
    return max(HEALTH_WAIT_MIN_S, min(raw, HEALTH_WAIT_MAX_S))


def health_wait_loop(
    probe: Callable[[], ReadyProbe],
    *,
    timeout_s: float | None = None,
    interval_s: float = 10,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    production: bool = False,
) -> ReadyProbe:
    """Python wait while Comfy installs. Never bash `set -e`.

    Production callers pass production=True so the window is 10–20 minutes.
    Tests pass a short timeout_s without production=True.
    """
    import time as _time

    sleep = sleep or _time.sleep
    clock = clock or _time.monotonic
    if production:
        timeout = float(production_health_wait_seconds(None if timeout_s is None else int(timeout_s)))
    else:
        timeout = float(DEFAULT_HEALTH_WAIT_S if timeout_s is None else timeout_s)
    deadline = clock() + timeout
    last = ReadyProbe()
    while clock() < deadline:
        last = probe()
        if last.ready:
            return last
        remaining = deadline - clock()
        if remaining <= 0:
            break
        sleep(min(interval_s, remaining))
    return last


def probe_comfy_system_stats(
    base_url: str,
    opener: Callable[[Request], tuple[int, dict]] | None = None,
) -> bool:
    url = base_url.rstrip("/") + COMFY_STATS_PATH
    req = Request(url, method="GET")
    if opener:
        status, _body = opener(req)
        return status == 200
    try:
        with urlopen(req, timeout=10) as resp:
            return getattr(resp, "status", 200) == 200
    except (URLError, TimeoutError, OSError):
        return False


def decide_probes(image: str | None = None, capabilities: dict | None = None) -> ReadyProbe:
    comfy = should_probe_comfy(image, capabilities)
    return ReadyProbe(comfy_required=comfy)


def may_destroy(reason: str | None, error: str | None = None) -> bool:
    if error and error in NEVER_DESTROY_ERRORS:
        return False
    return reason in DESTROY_REASONS


def request_lease(
    client: VastClient,
    offer_id: str,
    session: LeaseSession,
    stage_flags: dict[str, str] | None,
) -> dict:
    assert_pre_lease(stage_flags)
    settings = load_settings()
    if settings.vast_allow_replace != 0:
        raise VastSafetyError("VAST_ALLOW_REPLACE must be 0")
    if session.instance_id and session.state not in {"destroyed", "aborted", "idle"}:
        raise OneLeaseError(
            f"one lease at a time; retry install on {session.instance_id} "
            "(VAST_ALLOW_REPLACE=0, do not flip SKU)"
        )
    if not session.capabilities:
        session.capabilities = image_capabilities(session.image)
    assert_lease_capabilities(session.capabilities)
    session.state = "leasing"
    session.offer_id = offer_id
    image = session.image or "docker.io/aetherneo/anime-factory-gpu:main"
    if hasattr(client, "lease_body"):
        leased = client.lease(offer_id, client.lease_body(image=image, env=session.lease_env or None))
    else:
        leased = client.lease(offer_id)
    if leased.get("skipped") or leased.get("dry_run"):
        session.state = "idle"
        return {"action": "dry_run_skip_lease", "leased": leased, "put_asks": False}
    session.instance_id = str(
        leased.get("new_contract") or leased.get("id") or leased.get("instance_id") or offer_id
    )
    session.state = "leased"
    return {"action": "lease", "leased": leased, "put_asks": True, "instance_id": session.instance_id}


def retry_install_same_instance(session: LeaseSession) -> LeaseSession:
    """Failure during install: retry ON THE SAME instance. Do not replace."""
    if not session.instance_id:
        raise OneLeaseError("no instance to retry; do not search a new SKU")
    session.install_attempts += 1
    session.state = "install_retry"
    session.last_error = "install_failure"
    return session


def cap_wait_no_sku_flip(session: LeaseSession) -> LeaseSession:
    session.state = "wait_capped"
    session.last_error = "ready_wait_capped"
    # Explicitly not a destroy and not a replace.
    return session


def destroy_if_allowed(
    client: VastClient,
    session: LeaseSession,
    reason: str,
    registered_ids: set[str],
    error: str | None = None,
) -> dict:
    if not may_destroy(reason, error or session.last_error):
        raise DestroyRefused(
            f"refuse destroy reason={reason!r} error={error or session.last_error!r} "
            "(TTS 404 / health-script bugs / install failure must not kill the box)"
        )
    if not session.instance_id:
        session.state = "destroyed"
        session.destroy_reason = reason
        return {"skipped": True, "reason": reason}
    out = client.destroy(session.instance_id, registered_ids)
    session.state = "destroyed"
    session.destroy_reason = reason
    return out


def evaluate_readiness(
    vast_payload: dict | None,
    probes: ReadyProbe,
    image: str | None = None,
    capabilities: dict | None = None,
) -> tuple[bool, str | None]:
    caps = capabilities or image_capabilities(image)
    needed = decide_probes(image, caps)
    probes.comfy_required = needed.comfy_required
    if not needed.comfy_required:
        # Agent-only image: never require :8199.
        probes.comfy_system_stats_ok = False
    reason = vast_payload_not_ready_reason(vast_payload)
    if reason:
        return False, reason
    if probes.ready:
        return True, None
    return False, "probes_incomplete:" + ",".join(probes.missing())


# Hosted stages (TTS/script/board) must be in PRE_LEASE. design/keyframe may be GPU 生图.
assert set(HOSTED_STAGES).issubset(PRE_LEASE_STAGES)

__all__ = [
    "HOSTED_STAGES",
    "PRE_LEASE_STAGES",
    "GPU_STILL_STAGES",
    "GPU_SESSION_STAGES",
    "GPU_ALLOWED_STAGES",
    "VOICE_UPLOAD_PATH",
    "WRONG_VOICE_UPLOAD_PATH",
    "DESTROY_REASONS",
    "NEVER_DESTROY_ERRORS",
    "WHY_MULTI_CARD",
    "PreLeaseError",
    "ImageCapabilityError",
    "OneLeaseError",
    "SkuFlipError",
    "DestroyRefused",
    "ReadyProbe",
    "LeaseSession",
    "assert_pre_lease",
    "assert_lease_capabilities",
    "assert_gpu_session_only",
    "voice_upload_url_ok",
    "vast_payload_not_ready_reason",
    "is_vast_api_running_enough",
    "bash_health_script_is_unsafe",
    "health_wait_loop",
    "probe_comfy_system_stats",
    "decide_probes",
    "may_destroy",
    "request_lease",
    "retry_install_same_instance",
    "cap_wait_no_sku_flip",
    "destroy_if_allowed",
    "evaluate_readiness",
    "RECYCLE_FAILURE_CLASSES",
]

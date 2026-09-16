"""python -m gpu_worker — Comfy+H3+Flux stills agent. Dry-run by default (no Vast lease)."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from gpu_worker.entrypoint import main_dry_run
from gpu_worker.images import default_register_capabilities
from gpu_worker.poll import control_headers
from gpu_worker.registry import GpuRegistry
from gpu_worker.vast_client import VastClient


def work_queue_depth(payload: dict[str, Any] | None) -> int:
    """Control-plane items still waiting. Empty queue is required for idle teardown."""
    if not isinstance(payload, dict):
        return 0
    n = 0
    if payload.get("batch"):
        n += 1
    jobs = payload.get("jobs")
    if isinstance(jobs, list):
        n += len(jobs)
    hitch = payload.get("hitchhikers") or payload.get("hitchhiker")
    if isinstance(hitch, list):
        n += len(hitch)
    return n


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers=control_headers(json_body=True),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def maybe_notify_control_plane(
    instance_id: str,
    status: str = "idle",
    *,
    heartbeat_fields: dict[str, Any] | None = None,
    register: bool = True,
    handshake: dict[str, Any] | None = None,
) -> list[str]:
    """POST register/heartbeat when CONTROL_PLANE_URL is set. Never talks to Vast lease APIs."""
    base = os.environ.get("CONTROL_PLANE_URL", "").rstrip("/")
    events: list[str] = []
    if not base:
        return events
    caps = default_register_capabilities()
    hs = handshake or {}
    try:
        if register:
            register_body: dict[str, Any] = {
                "vast_instance_id": instance_id,
                "gpu_type": os.environ.get("GPU_TYPE") or "RTX",
                "capabilities": {
                    "comfy": bool(caps.get("comfy")),
                    "h3": bool(caps.get("h3")),
                    "kolors": bool(caps.get("kolors")),
                    "image_gen": bool(caps.get("image_gen")),
                    "agent": True,
                    "nvenc": True,
                    "profile_id": hs.get("profile_id") or caps.get("profile_id"),
                    "image_digest": hs.get("image_digest") or caps.get("image_digest"),
                },
            }
            for key in ("host", "stack", "resources", "preflight", "adaptations"):
                if hs.get(key) is not None:
                    register_body[key] = hs[key]
            _post_json(f"{base}/gpu/register", register_body)
            events.append("register")
        heartbeat = {"vast_instance_id": instance_id, "status": status}
        if heartbeat_fields:
            heartbeat.update(heartbeat_fields)
        _post_json(
            f"{base}/gpu/heartbeat",
            heartbeat,
        )
        events.append("heartbeat")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        events.append(f"control_plane_error:{type(exc).__name__}")
    return events


class _HeartbeatLoop:
    def __init__(
        self,
        instance_id: str,
        runtime: Any,
        tunnel: dict[str, Any],
        interval_s: int,
        on_stop: Any = None,
    ):
        self.instance_id = instance_id
        self.runtime = runtime
        self.tunnel = tunnel
        self.interval_s = max(5, min(interval_s, 30))
        self.on_stop = on_stop
        self.stop_event = threading.Event()
        self.teardown_started = threading.Event()
        self.teardown_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="gpu-heartbeat", daemon=True)

    def _notify(self) -> None:
        events = maybe_notify_control_plane(
            self.instance_id,
            self.runtime.status,
            heartbeat_fields=self.runtime.heartbeat_fields(self.tunnel),
            register=False,
        )
        errors = [event for event in events if event.startswith("control_plane_error:")]
        if errors:
            print(json.dumps({"heartbeat": errors}), flush=True)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self.runtime.sample_startup_progress()
            stop = self.runtime.stop_condition()
            if stop is not None:
                self.trigger_stop(stop)
            self._notify()

    def trigger_stop(self, stop: BaseException) -> dict[str, Any] | None:
        if not self.on_stop:
            return None
        with self.teardown_lock:
            if self.teardown_started.is_set():
                return self.runtime.monitored_teardown_result
            self.teardown_started.set()
            try:
                result = self.on_stop(stop)
            except Exception as exc:  # noqa: BLE001 — keep heartbeat thread observable
                result = {
                    "ok": False,
                    "destroyed": False,
                    "error": f"{type(exc).__name__}:{exc}",
                }
            self.runtime.monitored_teardown_result = result
            return result

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)


def _instance_id() -> str:
    label = (os.environ.get("VAST_CONTAINERLABEL") or "").strip()
    label_id = label.split(".", 1)[1] if label.startswith("C.") else label
    return (
        os.environ.get("VAST_INSTANCE_ID")
        or os.environ.get("CONTAINER_ID")
        or label_id
        or os.environ.get("HOSTNAME")
        or "local-dry"
    )


def maybe_pm_control() -> object | None:
    base = os.environ.get("CONTROL_PLANE_URL", "").rstrip("/")
    password = os.environ.get("STUDIO_PASSWORD") or ""
    if not base or not password:
        return None
    from anime_factory.produce import ControlPlane

    control = ControlPlane(base, os.environ.get("STUDIO_USER") or "studio", password)
    try:
        control.login()
    except Exception:  # noqa: BLE001
        return None
    return control


def main() -> int:
    lease_started = time.monotonic()
    instance_id = _instance_id()
    start_comfy_enabled = _env_bool("AF_START_COMFY", False)
    stack_info: dict = {}
    from gpu_worker.stack import COMFY_DIR, boot_gpu_stack, current_tunnel_status
    from gpu_worker.weights import runtime_weight_bytes
    from anime_factory.models import is_short_kind
    from gpu_worker.preflight import PreflightFailure, recycle_failure_class
    from gpu_worker.session import (
        BudgetExceeded,
        ClassifiedStartupFailure,
        ComfyStartupTimeout,
        LeaseRuntime,
        PreflightTimeout,
        ProgressStalled,
        StartupTimeout,
        WeightPullTimeout,
        best_effort_upload_checkpoint,
        destroy_self,
        persist_boot_failure,
        gpu_cycle_idle_reason,
        hourly_rate_from_work,
        lease_age_seconds_from_work,
        never_destroy_error,
        run_gpu_batch,
        run_gpu_episode,
        run_pre_gpu_if_needed,
    )
    from gpu_worker.poll import (
        claim_job,
        fetch_work,
        parse_batch,
        parse_hitchhikers,
        parse_pre_gpu,
        parse_work_payload,
        pick_job,
        report_episode,
    )

    tunnel = current_tunnel_status()
    runtime = LeaseRuntime(instance_id=instance_id, started_at=lease_started)
    if start_comfy_enabled:
        runtime.begin_startup(
            probe=lambda: {
                "downloaded_bytes": runtime_weight_bytes(COMFY_DIR),
            }
        )
    client = VastClient(dry_run=True)
    registry = GpuRegistry(vast=client)
    result = main_dry_run(registry, instance_id)
    result["control_plane"] = maybe_notify_control_plane(
        instance_id,
        runtime.status,
        heartbeat_fields=runtime.heartbeat_fields(tunnel),
        register=True,
        handshake=runtime.handshake,
    )

    control_base = os.environ.get("CONTROL_PLANE_URL", "").rstrip("/")
    heartbeat_s = max(int(os.environ.get("HEARTBEAT_SECONDS", "30")), 5)
    startup_payload = (
        fetch_work(control_base, instance_id)
        if control_base
        else {"batch": None, "jobs": [], "workers": []}
    )
    runtime.set_hourly_rate(hourly_rate_from_work(startup_payload, instance_id))
    runtime.account_for_lease_age(
        lease_age_seconds_from_work(startup_payload, instance_id)
    )
    startup_batch = parse_batch(startup_payload)
    if startup_batch:
        runtime.adopt_batch(startup_batch)

    def monitored_teardown(stop: BaseException) -> dict[str, Any]:
        runtime.status = "draining"
        checkpoint_box: dict[str, Any] = {}

        def upload_checkpoint() -> None:
            checkpoint_box["result"] = best_effort_upload_checkpoint(
                runtime.active_story_id,
                runtime.active_root,
            )

        checkpoint_thread = threading.Thread(
            target=upload_checkpoint,
            name="gpu-final-checkpoint",
            daemon=True,
        )
        checkpoint_thread.start()
        checkpoint_thread.join(
            timeout=max(
                0.0,
                float(os.environ.get("WATCHDOG_CHECKPOINT_TIMEOUT_S") or 30),
            )
        )
        checkpoint = checkpoint_box.get("result") or {
            "ok": False,
            "error": "checkpoint_upload_timeout",
        }
        maybe_notify_control_plane(
            instance_id,
            "draining",
            heartbeat_fields=runtime.heartbeat_fields(tunnel),
            register=False,
        )
        try:
            persist_boot_failure(
                {
                    "watchdog_teardown": True,
                    "error": runtime.last_error,
                    "trigger": getattr(stop, "limit", type(stop).__name__),
                    "instance_id": instance_id,
                    "handshake": runtime.handshake,
                }
            )
        except Exception:  # noqa: BLE001 — never skip destroy
            pass
        teardown = destroy_self(
            instance_id,
            "explicit_abort",
            error=runtime.last_error,
        )
        result = {
            **teardown,
            "trigger": getattr(stop, "limit", type(stop).__name__),
            "checkpoint": checkpoint,
        }
        print(json.dumps({"watchdog_teardown": result}, ensure_ascii=False), flush=True)
        return result

    heartbeat = _HeartbeatLoop(
        instance_id,
        runtime,
        tunnel,
        heartbeat_s,
        on_stop=monitored_teardown,
    )
    heartbeat.start()
    if start_comfy_enabled:
        try:
            stack_info = boot_gpu_stack(progress=runtime.observe_progress)
        except PreflightFailure as exc:
            from gpu_worker.preflight import handshake_payload, select_profile_id

            runtime.set_handshake(
                handshake_payload(
                    profile_id=str(exc.details.get("profile_id") or select_profile_id()),
                    preflight_ok=False,
                    preflight_stage="hardware",
                    preflight_error=str(exc),
                    failure_class=exc.failure_class,
                )
            )
            runtime.last_error = str(exc)
            runtime.status = "booting"
            runtime.record_startup_health("failed")
            runtime.record_startup_stage(exc.failure_class)
            maybe_notify_control_plane(
                instance_id,
                runtime.status,
                heartbeat_fields=runtime.heartbeat_fields(tunnel),
                register=False,
                handshake=runtime.handshake,
            )
            boot_upload = {"ok": False, "skipped": True}
            try:
                boot_upload = persist_boot_failure(
                    {
                        "preflight_failed": exc.as_dict(),
                        "handshake": runtime.handshake,
                        "failure_class": exc.failure_class,
                        "instance_id": instance_id,
                    }
                )
            except Exception as upload_exc:  # noqa: BLE001 — never skip destroy
                boot_upload = {"ok": False, "error": f"{type(upload_exc).__name__}:{upload_exc}"}
            teardown = destroy_self(instance_id, "explicit_abort", error=str(exc))
            print(
                json.dumps(
                    {
                        "preflight_failed": exc.as_dict(),
                        "self_destroy": teardown,
                        "failure_class": exc.failure_class,
                        "boot_upload": boot_upload,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            heartbeat.stop()
            return 1
        except (
            BudgetExceeded,
            ProgressStalled,
            StartupTimeout,
            ClassifiedStartupFailure,
            PreflightTimeout,
            ComfyStartupTimeout,
            WeightPullTimeout,
        ) as exc:
            failure = recycle_failure_class(exc) or getattr(exc, "limit", type(exc).__name__)
            runtime.last_error = f"{failure}:{exc}"
            runtime.record_startup_health("failed")
            runtime.record_startup_stage(str(failure))
            maybe_notify_control_plane(
                instance_id,
                runtime.status,
                heartbeat_fields=runtime.heartbeat_fields(tunnel),
                register=False,
                handshake=runtime.handshake,
            )
            boot_upload = {"ok": False, "skipped": True}
            try:
                boot_upload = persist_boot_failure(
                    {
                        "startup_stopped": str(exc),
                        "failure_class": failure,
                        "instance_id": instance_id,
                        "handshake": runtime.handshake,
                    }
                )
            except Exception as upload_exc:  # noqa: BLE001 — never skip destroy
                boot_upload = {"ok": False, "error": f"{type(upload_exc).__name__}:{upload_exc}"}
            teardown = heartbeat.trigger_stop(exc)
            print(
                json.dumps(
                    {
                        "startup_stopped": str(exc),
                        "failure_class": failure,
                        "self_destroy": teardown,
                        "boot_upload": boot_upload,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            heartbeat.stop()
            return 1
        except Exception as exc:  # noqa: BLE001 — unclassified install stays on this box
            classified = recycle_failure_class(exc)
            if classified:
                runtime.last_error = f"{classified}:{exc}"
                runtime.status = "booting"
                runtime.record_startup_health("failed")
                runtime.record_startup_stage(classified)
                maybe_notify_control_plane(
                    instance_id,
                    runtime.status,
                    heartbeat_fields=runtime.heartbeat_fields(tunnel),
                    register=False,
                    handshake=runtime.handshake,
                )
                boot_upload = {"ok": False, "skipped": True}
                try:
                    boot_upload = persist_boot_failure(
                        {
                            "stack_failed": f"{type(exc).__name__}:{exc}",
                            "failure_class": classified,
                            "instance_id": instance_id,
                            "handshake": runtime.handshake,
                        }
                    )
                except Exception as upload_exc:  # noqa: BLE001 — never skip destroy
                    boot_upload = {
                        "ok": False,
                        "error": f"{type(upload_exc).__name__}:{upload_exc}",
                    }
                teardown = destroy_self(
                    instance_id,
                    "explicit_abort",
                    error=runtime.last_error,
                )
                print(
                    json.dumps(
                        {
                            "stack_failed": f"{type(exc).__name__}:{exc}",
                            "failure_class": classified,
                            "self_destroy": teardown,
                            "boot_upload": boot_upload,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                heartbeat.stop()
                return 1
            runtime.last_error = never_destroy_error(exc) or "install_failure"
            runtime.status = "booting"
            try:
                persist_boot_failure(
                    {
                        "stack_failed": f"{type(exc).__name__}:{exc}",
                        "failure_class": runtime.last_error,
                        "instance_id": instance_id,
                        "handshake": runtime.handshake,
                    }
                )
            except Exception:  # noqa: BLE001 — stay fail-closed on this box
                pass
            print(
                json.dumps(
                    {
                        "stack_failed": f"{type(exc).__name__}:{exc}",
                        "destroy_blocked_by": runtime.last_error,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            maybe_notify_control_plane(
                instance_id,
                runtime.status,
                heartbeat_fields=runtime.heartbeat_fields(tunnel),
                register=False,
            )
            heartbeat.stop()
            return 1
        os.environ.setdefault("ANIME_FACTORY_GPU_STILLS", "1")
        tunnel = stack_info.get("tunnel") or current_tunnel_status()
        heartbeat.tunnel = tunnel
        runtime.set_handshake(stack_info.get("handshake"))
        runtime.complete_startup(bool(stack_info.get("router_ready")))
    else:
        runtime.mark_idle()

    stack_failed = bool(stack_info and not stack_info.get("router_ready"))
    if stack_failed:
        runtime.status = "booting"
        runtime.idle_since = None
        runtime.last_error = str(stack_info.get("comfy_error") or "comfy_startup_failed")
        if runtime.handshake is None and stack_info.get("handshake"):
            runtime.set_handshake(stack_info.get("handshake"))
        recycle_now = recycle_failure_class(runtime.last_error)
        if recycle_now:
            runtime.record_startup_health("failed")
            runtime.record_startup_stage(recycle_now)
            maybe_notify_control_plane(
                instance_id,
                runtime.status,
                heartbeat_fields=runtime.heartbeat_fields(tunnel),
                register=False,
                handshake=runtime.handshake,
            )
            try:
                persist_boot_failure(
                    {
                        "comfy_error": runtime.last_error,
                        "comfy_boot_log_tail": (stack_info or {}).get("comfy_boot_log_tail"),
                        "failure_class": recycle_now,
                        "instance_id": instance_id,
                        "handshake": runtime.handshake,
                    }
                )
            except Exception:  # noqa: BLE001 — never skip destroy
                pass

    result["vast_dry_run"] = True
    result["put_asks"] = False
    result["capabilities"] = default_register_capabilities()
    result["stack"] = {
        k: stack_info.get(k)
        for k in (
            "router_ready",
            "comfy_pid",
            "comfy_error",
            "comfy_boot_log",
            "comfy_boot_log_tail",
            "weights",
            "fonts",
            "tunnel",
            "startup_downloaded_bytes",
        )
        if stack_info
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    from gpu_worker.comfy import ComfyRouter

    idle_comfy = ComfyRouter() if start_comfy_enabled and not stack_info.get("comfy_skipped_longlive") else None
    control = maybe_pm_control()
    once = _env_bool("AF_ONCE", False)
    can_run = start_comfy_enabled and not stack_failed
    wanted_story = (os.environ.get("AF_STORY_ID") or "").strip()
    processed_legacy: set[tuple[str, str]] = set()
    pending_reports: dict[str, Any] | None = None
    pending_destroy: dict[str, Any] | None = None
    last_queue_depth = 0
    recycle_error = recycle_failure_class(runtime.last_error)
    destroy_blocked = stack_failed and (
        str(runtime.last_error or "").startswith("install_failure") and not recycle_error
    )
    if stack_failed and not destroy_blocked:
        pending_destroy = {
            "reason": "explicit_abort",
            "error": runtime.last_error,
            "exit_code": 1,
            "last_attempt": 0.0,
        }
    did_work = False

    try:
        while True:
            if stack_failed and not recycle_error:
                from gpu_worker.boot import probe_comfy_system_stats

                base = os.environ.get("COMFYUI_BASE_URL") or "http://127.0.0.1:8199"
                if probe_comfy_system_stats(base):
                    stack_failed = False
                    can_run = True
                    destroy_blocked = False
                    runtime.last_error = None
                    runtime.complete_startup(True)

            monitored = runtime.monitored_teardown_result
            if monitored is not None:
                runtime.monitored_teardown_result = None
                if monitored.get("ok"):
                    return 1
                if monitored.get("refused"):
                    destroy_blocked = True

            if pending_reports:
                still_failed = []
                for item in pending_reports.get("reports") or []:
                    response = item.get("response")
                    if isinstance(response, dict) and response.get("ok", True) is not False:
                        continue
                    retried = report_episode(control_base, item["payload"])
                    item["response"] = retried
                    if retried.get("ok", True) is False:
                        still_failed.append(str(retried.get("error") or "report failed"))
                if not still_failed:
                    all_done = all(
                        row.get("status") == "done"
                        for row in (pending_reports.get("results") or [])
                    )
                    pending_destroy = {
                        "reason": "success" if all_done else "explicit_abort",
                        "error": pending_reports.get("destroy_error"),
                        "exit_code": 0 if all_done else 1,
                        "last_attempt": 0.0,
                    }
                    pending_reports = None

            if pending_destroy and not destroy_blocked:
                now = time.monotonic()
                if now - float(pending_destroy.get("last_attempt") or 0) >= 60 or not pending_destroy.get(
                    "last_attempt"
                ):
                    runtime.status = "draining"
                    maybe_notify_control_plane(
                        instance_id,
                        "draining",
                        heartbeat_fields=runtime.heartbeat_fields(tunnel),
                        register=False,
                    )
                    teardown = destroy_self(
                        instance_id,
                        str(pending_destroy["reason"]),
                        error=pending_destroy.get("error"),
                    )
                    print(json.dumps({"self_destroy": teardown}, ensure_ascii=False), flush=True)
                    pending_destroy["last_attempt"] = now
                    if teardown.get("ok"):
                        return int(pending_destroy.get("exit_code") or 0)
                    runtime.mark_idle()
                    if teardown.get("refused"):
                        destroy_blocked = True
                        pending_destroy = None

            if (
                can_run
                and not pending_reports
                and not pending_destroy
                and not destroy_blocked
            ):
                try:
                    payload = fetch_work(control_base, instance_id) if control_base else {"batch": None, "jobs": []}
                    last_queue_depth = work_queue_depth(payload)
                    runtime.set_hourly_rate(hourly_rate_from_work(payload, instance_id))
                    runtime.account_for_lease_age(
                        lease_age_seconds_from_work(payload, instance_id)
                    )
                    for item in parse_pre_gpu(payload):
                        sid = str(item.get("story_id") or "").strip()
                        ep_code = str(item.get("episode_code") or "EP001")
                        if not sid:
                            continue
                        did_work = True
                        pre_root = Path(os.environ.get("AF_STORY_ROOT") or f"/work/{sid}")
                        runtime.set_active_work(sid, pre_root)
                        runtime.mark_busy()
                        pre = run_pre_gpu_if_needed(
                            sid,
                            pre_root,
                            ep_code,
                            langs=item.get("langs") if isinstance(item.get("langs"), list) else None,
                            control=control,
                            title=str(item.get("title") or "") or None,
                            logline=str(item.get("logline") or "") or None,
                        )
                        runtime.mark_idle()
                        print(json.dumps({"pre_gpu": pre}, ensure_ascii=False, default=str), flush=True)
                    batch = parse_batch(payload)
                    last_batch_summary: dict[str, Any] | None = None
                    if batch:
                        summary = run_gpu_batch(
                            batch,
                            runtime,
                            control=control,
                            claim=lambda story, episode: claim_job(
                                control_base,
                                instance_id,
                                story,
                                episode,
                            ),
                            report=lambda body: report_episode(control_base, body),
                        )
                        last_batch_summary = summary
                        if not summary.get("skip_cycle"):
                            did_work = True
                        print(json.dumps({"batch": summary}, ensure_ascii=False, default=str), flush=True)
                        if summary.get("report_errors"):
                            pending_reports = summary
                    hitch = parse_hitchhikers(payload)
                    if control_base:
                        after = fetch_work(control_base, instance_id)
                        extra = parse_hitchhikers(after)
                        if extra:
                            hitch = extra
                        last_queue_depth = work_queue_depth(after)
                    live_hitch: list[dict[str, Any]] = []
                    for item in hitch:
                        sid = str(item.get("story_id") or "").strip()
                        ep_code = str(item.get("episode_code") or "EP001")
                        if not sid:
                            continue
                        hitch_root = Path(os.environ.get("AF_STORY_ROOT") or f"/work/{sid}")
                        idle = gpu_cycle_idle_reason(
                            item,
                            root=hitch_root,
                            episode_code=ep_code,
                        )
                        if idle or runtime.already_completed(sid, ep_code):
                            runtime.remember_completed(sid, ep_code)
                            runtime.mark_idle()
                            continue
                        live_hitch.append(item)
                    hitch = live_hitch
                    for item in hitch:
                        sid = str(item.get("story_id") or "").strip()
                        ep_code = str(item.get("episode_code") or "EP001")
                        if not sid:
                            continue
                        did_work = True
                        hitch_root = Path(os.environ.get("AF_STORY_ROOT") or f"/work/{sid}")
                        runtime.set_active_work(sid, hitch_root)
                        runtime.mark_busy()
                        if control_base:
                            claim_job(control_base, instance_id, sid, ep_code)
                        episode = run_gpu_episode(
                            sid,
                            root=hitch_root,
                            control=control,
                            episode_code=ep_code,
                            langs=item.get("langs") if isinstance(item.get("langs"), list) else None,
                            progress=runtime.observe_progress,
                            finish_story=True,
                            skip_shorts=is_short_kind(item.get("kind")),
                            story_status=str(item.get("story_status") or item.get("status") or "") or None,
                            until_stage=str(item.get("until_stage") or "") or None,
                            episode_status=str(item.get("episode_status") or "") or None,
                        )
                        if episode.get("remaining"):
                            runtime.mark_busy()
                        else:
                            runtime.remember_completed(sid, ep_code)
                            runtime.mark_idle()
                        print(json.dumps({"hitchhiker": episode}, ensure_ascii=False), flush=True)
                        if control_base and not episode.get("skipped_cycle"):
                            report_episode(
                                control_base,
                                {
                                    "vast_instance_id": instance_id,
                                    "batch_id": item.get("batch_id"),
                                    "episode_code": ep_code,
                                    "stage": "compose" if not episode.get("remaining") else "anim",
                                    "status": "done" if not episode.get("remaining") else "running",
                                    "artifacts": episode.get("compose") or {},
                                    "error": None if not episode.get("remaining") else "anim incomplete",
                                },
                            )
                    if last_batch_summary and not hitch and not pending_reports:
                        destroy_reason = last_batch_summary.get("destroy_reason")
                        remaining = int(last_batch_summary.get("episodes_done") or 0) < int(
                            last_batch_summary.get("episodes_total") or 0
                        ) or any(
                            int((item.get("result") or {}).get("remaining") or 0) > 0
                            for item in (last_batch_summary.get("results") or [])
                            if isinstance(item, dict)
                        )
                        if destroy_reason:
                            pending_destroy = {
                                "reason": destroy_reason,
                                "error": last_batch_summary.get("destroy_error"),
                                "exit_code": 0 if not remaining else 1,
                                "last_attempt": 0.0,
                            }
                        elif remaining:
                            runtime.mark_busy()
                        else:
                            runtime.mark_idle()
                    elif not batch:
                        job = pick_job(parse_work_payload(payload), wanted_story)
                        work_fetch_ok = payload.get("ok", True) is not False
                        if not work_fetch_ok:
                            print(
                                json.dumps(
                                    {"work_poll": payload.get("error") or "failed"},
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                        if (
                            not job
                            and wanted_story
                            and not processed_legacy
                            and not runtime.already_completed(
                                wanted_story, os.environ.get("AF_EPISODE") or "EP001"
                            )
                            and (work_fetch_ok or not control_base)
                        ):
                            job = {
                                "story_id": wanted_story,
                                "episode_code": os.environ.get("AF_EPISODE") or "EP001",
                            }
                        if job and str(job.get("story_id") or "").strip():
                            story_id = str(job["story_id"])
                            episode_code = str(job.get("episode_code") or "EP001")
                            legacy_key = (story_id, episode_code)
                            legacy_root = Path(
                                os.environ.get("AF_STORY_ROOT")
                                or f"/work/{story_id}"
                            )
                            idle = gpu_cycle_idle_reason(
                                job,
                                root=legacy_root,
                                episode_code=episode_code,
                            )
                            if (
                                idle
                                or runtime.already_completed(story_id, episode_code)
                                or legacy_key in processed_legacy
                            ):
                                runtime.remember_completed(story_id, episode_code)
                                processed_legacy.add(legacy_key)
                                runtime.mark_idle()
                            else:
                                did_work = True
                                runtime.set_active_work(story_id, legacy_root)
                                runtime.mark_busy()
                                if control_base:
                                    claim_job(control_base, instance_id, story_id, episode_code)
                                episode = run_gpu_episode(
                                    story_id,
                                    root=legacy_root,
                                    control=control,
                                    episode_code=episode_code,
                                    langs=job.get("langs") if isinstance(job.get("langs"), list) else None,
                                    progress=runtime.observe_progress,
                                    story_status=str(job.get("story_status") or job.get("status") or "")
                                    or None,
                                    until_stage=str(job.get("until_stage") or "") or None,
                                    episode_status=str(job.get("episode_status") or "") or None,
                                )
                                processed_legacy.add(legacy_key)
                                if episode.get("remaining"):
                                    runtime.mark_busy()
                                else:
                                    runtime.remember_completed(story_id, episode_code)
                                    runtime.mark_idle()
                                print(json.dumps({"episode": episode}, ensure_ascii=False), flush=True)
                                if once:
                                    return 0 if not episode.get("remaining") else 1

                except (BudgetExceeded, ProgressStalled, StartupTimeout) as exc:
                    pending_destroy = {
                        "reason": "explicit_abort",
                        "error": str(exc),
                        "exit_code": 1,
                        "last_attempt": 0.0,
                    }
                    runtime.last_error = str(exc)
                    runtime.mark_idle()
                except Exception as exc:  # noqa: BLE001 — keep heartbeat thread alive
                    print(
                        json.dumps(
                            {"work_tick_error": f"{type(exc).__name__}:{exc}"},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    runtime.mark_idle()

            registry.heartbeat(instance_id, runtime.status if runtime.status in {"busy", "idle"} else "idle")
            if not pending_destroy and not destroy_blocked:
                limit = runtime.limit_reached()
                if limit:
                    pending_destroy = {
                        "reason": "explicit_abort",
                        "error": runtime.last_error,
                        "exit_code": 1,
                        "last_attempt": 0.0,
                    }
                elif runtime.idle_timer_expired():
                    try:
                        comfy_inflight = (
                            idle_comfy.inflight_count()
                            if idle_comfy is not None
                            else 0
                        )
                    except Exception as exc:  # noqa: BLE001 — unknown queue must fail closed
                        longlive = (
                            str(os.environ.get("AF_IMAGE_CAPABILITY") or "").strip().lower()
                            == "longlive"
                            or str(os.environ.get("AF_VIDEO_BACKEND") or "").strip().lower()
                            == "longlive"
                            or bool(stack_info.get("h3_skipped_longlive"))
                        )
                        comfy_inflight = 0 if longlive else None
                        print(
                            json.dumps(
                                {
                                    "idle_probe": "comfy_queue_unknown",
                                    "error": f"{type(exc).__name__}:{exc}",
                                    "longlive_treat_empty": longlive,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    if runtime.idle_expired(comfy_inflight, pending_jobs=last_queue_depth):
                        pending_destroy = {
                            "reason": "idle_ttl",
                            "error": runtime.last_error,
                            "exit_code": 0,
                            "last_attempt": 0.0,
                        }
            if once and not did_work:
                return 1 if stack_failed else 0
            time.sleep(min(heartbeat_s, 30))
    finally:
        heartbeat.stop()


if __name__ == "__main__":
    raise SystemExit(main())

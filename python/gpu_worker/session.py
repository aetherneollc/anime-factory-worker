"""GPU episode session: missing 生图 on this card, then H3 anim, qc, compose.

If bible→board is missing (laptop was off), this session runs produce_episode
first via DeepSeek / SiliconFlow HTTP. TTS stays CosyVoice2, not Vast.
One card. Do not silent-truncate 600s. Partial shots stay on R2 if the lease TTL hits.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from anime_factory.board import expand_shots_to_segments, persist_board
from anime_factory.canon import export_geo, hydrate_geo_interiors, write_canon_dir
from anime_factory.compose import (
    ComposePlan,
    MIN_SPEECH_RMS,
    MissingShotError,
    assert_no_h3_dialogue,
    build_compose_plan,
    chain_overlap_frames,
    collect_shot_paths,
    collect_shots,
    compose_episode_audio,
    drop_first_frames_cmd,
    extend_clip_cmd,
    fit_clip_durations_to_speech,
    write_srt,
)
from anime_factory.sfx_orchestrate import prepare_episode_sfx
from anime_factory.db import checkpoint_and_upload, migrate, open_db, utcnow
from anime_factory.design import KolorsClient
from anime_factory.directors.common import hydrate_shot_identity, is_chain_head, needs_first_frame_still
from anime_factory.keyframe import assert_keyframe_files, ensure_keyframe
from anime_factory.models import H3_MAX_RETRIES, VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH, is_short_kind
from anime_factory.produce import DEFAULT_EPISODE, ControlPlane, produce_episode
from anime_factory.qc import (
    continuity_qc,
    existing_generation_file,
    incremental_qc_segment,
    mark_completed_passing,
    next_generation_path,
    open_repair_task,
    plan_repair,
    record_generation_result,
    select_passing_generation,
    shot_version_path,
)
from anime_factory.r2_client import download_prefix, put_file, upload_tree
from anime_factory.r2_paths import join_story, story_prefix
from gpu_worker.boot import (
    NEVER_DESTROY_ERRORS,
    DestroyRefused,
    LeaseSession,
    destroy_if_allowed,
    may_destroy,
)
from gpu_worker.comfy import ComfyRouter, extract_execution_error, format_execution_error
from gpu_worker.h3 import (
    apply_chain_first_frame,
    extract_last_frame,
    h3_resolution_profile,
    is_directory_like,
    gpu_vram_mb,
    is_h3_oom,
    prepare_workflow,
    ref_image_filenames,
    select_mode,
)
from gpu_worker.longlive import (
    FAIL_CLOSED,
    LONGLIVE_NATIVE_HEIGHT,
    LONGLIVE_NATIVE_WIDTH,
    ensure_longlive,
    is_longlive_fatal_error,
    submit_longlive,
)
from gpu_worker.longlive_batch import submit_longlive_batch
from gpu_worker.preflight import RECYCLE_FAILURE_CLASSES, recycle_failure_class
from gpu_worker.stack import (
    release_gpu_for_moss_sfx,
    restore_gpu_after_moss_sfx,
    stop_comfy_for_longlive,
)
from gpu_worker.stills import generate_still, unload_still_models
from gpu_worker.vast_client import VastClient
from gpu_worker.h3_session import (
    H3PrepPool,
    H3SessionTracker,
    can_prefetch_staging,
    instrument_h3_session,
    model_load_count_for_modes,
    modes_for_shots,
    plan_h3_mode_groups,
)
from gpu_worker.weights import ensure_h3_dits_for_shots, join_h3_weights
from anime_factory.video_backend import (
    VideoBackendLockError,
    lock_video_backend,
    max_seconds_for_backend,
    select_video_backend,
)

# Default keeps existing EP001 artifacts; override via AF_EPISODE / run_gpu_episode(episode_code=).
EP = DEFAULT_EPISODE
ProgressCallback = Callable[[str], None]
ReportCallback = Callable[[dict[str, Any]], dict[str, Any]]
ClaimCallback = Callable[[str, str], dict[str, Any]]
_DONE_STORY_STATUSES = frozenset({"archived", "finished"})
_GPU_UNTIL_DONE = frozenset({"archive", "finished"})
_STAGE_ORDER = (
    "bible",
    "canon",
    "script",
    "tts",
    "board",
    "design",
    "keyframe",
    "anim",
    "qc",
    "compose",
    "archive",
)


def _normalize_episode_code(episode_code: str | None = None) -> str:
    ep = (episode_code or os.environ.get("AF_EPISODE") or DEFAULT_EPISODE).strip().upper()
    if not ep.startswith("EP"):
        ep = f"EP{int(ep):03d}" if ep.isdigit() else DEFAULT_EPISODE
    return ep


def _norm_status(value: Any) -> str:
    return str(value or "").strip().lower()


def episode_finals_ready(
    root: Path | None,
    episode_code: str | None = None,
    langs: tuple[str, ...] | list[str] | None = None,
) -> bool:
    if root is None:
        return False
    ep = _normalize_episode_code(episode_code)
    wanted = tuple(
        str(lang).strip().lower()
        for lang in (langs or ("zh", "en", "ja"))
        if str(lang).strip()
    ) or ("zh", "en", "ja")
    final = Path(root) / "episodes" / ep / "final"
    return all(
        (final / f"{ep}.{lang}.mp4").is_file() and (final / f"{ep}.{lang}.mp4").stat().st_size > 0
        for lang in wanted
    )


def gpu_cycle_idle_reason(
    item: dict[str, Any] | None,
    *,
    remaining: int | None = None,
    compose: Any = None,
    root: Path | None = None,
    episode_code: str | None = None,
    langs: tuple[str, ...] | list[str] | None = None,
) -> str | None:
    """Why this episode must sit idle instead of skip-existing design→anim→qc→compose."""
    payload = item if isinstance(item, dict) else {}
    if payload.get("archived") is True or payload.get("gpu_done") is True:
        return "archived" if payload.get("archived") is True else "until_stage"
    for key in ("status", "story_status", "episode_status"):
        if _norm_status(payload.get(key)) in _DONE_STORY_STATUSES:
            return "archived"
    until = _norm_status(payload.get("until_stage"))
    gate = _norm_status(
        payload.get("gate") or payload.get("next_stage") or payload.get("current_stage")
    )
    if until in _GPU_UNTIL_DONE or gate in _GPU_UNTIL_DONE:
        return "until_stage"
    if payload.get("until_stage_reached") is True:
        return "until_stage"
    if until and gate and until in _STAGE_ORDER and gate in _STAGE_ORDER:
        if _STAGE_ORDER.index(gate) > _STAGE_ORDER.index(until):
            return "until_stage"
    value = remaining if remaining is not None else payload.get("remaining")
    if value is not None:
        try:
            if int(value) == 0:
                return "remaining=0"
        except (TypeError, ValueError):
            pass
    compose = compose if compose is not None else payload.get("compose")
    if isinstance(compose, dict) and (compose.get("final_keys") or compose.get("skipped_existing")):
        return "remaining=0"
    code = episode_code or str(payload.get("episode_code") or "") or None
    if episode_finals_ready(root, code, langs or payload.get("langs")):
        return "remaining=0"
    return None
DEFAULT_MAX_LEASE_MINUTES = 3600.0
DEFAULT_MAX_LEASE_USD = 40.0
GLOBAL_MAX_LEASE_USD = 100.0
DEFAULT_IDLE_MINUTES = 15.0
DEFAULT_POST_CHECKPOINT_IDLE_SECONDS = 30.0
DEFAULT_WATCH_USD = 20.0
DEFAULT_STARTUP_TIMEOUT_MINUTES = 150.0
PREFLIGHT_TIMEOUT_SECONDS = 120.0
COMFY_STARTUP_TIMEOUT_SECONDS = 600.0
WEIGHT_PULL_TIMEOUT_SECONDS = 3600.0
# Hardware preflight is 2 minutes. These stages run after nvidia-smi and include
# pip/fonts — they must not inherit PreflightTimeout or the box self-destroys.
STARTUP_SETUP_STAGES = frozenset(
    {
        "torch",
        "compiler",
        "comfy_requirements",
        "fonts",
        "compatibility_patches",
        "comfy_torch_probe",
    }
)
WATCHDOG_INTERVAL_SECONDS = 10 * 60.0
WATCHDOG_STALE_CHECKS = 3
STARTUP_WATCHDOG_STALE_CHECKS = 15


def _env_number(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name) or default))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def self_destroy_dry_run() -> bool:
    """Image default is VAST_DRY_RUN=1 / LIVE_VAST=0. A billed Vast box must still DELETE."""
    if _env_flag("ANIME_FACTORY_LIVE_VAST"):
        return False
    if (os.environ.get("CONTAINER_API_KEY") or "").strip():
        return False
    return (os.environ.get("VAST_DRY_RUN") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


class BudgetExceeded(RuntimeError):
    def __init__(self, limit: str, value: float, maximum: float):
        self.limit = limit
        self.value = value
        self.maximum = maximum
        super().__init__(f"{limit} exceeded: {value:.2f} >= {maximum:.2f}")


class ProgressStalled(RuntimeError):
    limit = "progress_stalled"

    def __init__(
        self,
        checks: int = WATCHDOG_STALE_CHECKS,
        phase: str = "production",
    ):
        self.checks = checks
        self.phase = phase
        super().__init__(
            f"no {phase} progress for {checks} watchdog checks"
        )


class StartupTimeout(RuntimeError):
    limit = "startup_timeout"

    def __init__(self, minutes: float, phase: str = "startup"):
        self.minutes = minutes
        self.phase = phase
        super().__init__(f"{phase} exceeded {minutes:.1f} minutes")


class PreflightTimeout(StartupTimeout):
    limit = "preflight_failed"

    def __init__(self) -> None:
        super().__init__(PREFLIGHT_TIMEOUT_SECONDS / 60.0, phase="preflight")


class ComfyStartupTimeout(StartupTimeout):
    limit = "comfy_startup_failed"

    def __init__(self) -> None:
        super().__init__(COMFY_STARTUP_TIMEOUT_SECONDS / 60.0, phase="comfy")


class WeightPullTimeout(StartupTimeout):
    limit = "weight_pull_timeout"

    def __init__(self) -> None:
        super().__init__(WEIGHT_PULL_TIMEOUT_SECONDS / 60.0, phase="weights")


class ClassifiedStartupFailure(StartupTimeout):
    """Already-classified boot failure — recycle now, do not wait 150 minutes."""

    def __init__(self, failure_class: str, detail: str = ""):
        self.limit = failure_class
        self.failure_class = failure_class
        self.minutes = 0.0
        self.phase = failure_class
        RuntimeError.__init__(self, detail or failure_class)


@dataclass(frozen=True)
class BatchBudget:
    max_minutes: float = DEFAULT_MAX_LEASE_MINUTES
    max_usd: float = DEFAULT_MAX_LEASE_USD
    idle_minutes: float = DEFAULT_IDLE_MINUTES
    idle_seconds: float | None = None
    post_checkpoint_idle_seconds: float = DEFAULT_POST_CHECKPOINT_IDLE_SECONDS

    def idle_ttl_seconds(self) -> float:
        if self.idle_seconds is not None:
            return max(0.0, float(self.idle_seconds))
        return max(0.0, float(self.idle_minutes) * 60.0)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None = None) -> "BatchBudget":
        payload = payload or {}

        def number(key: str, env_name: str, default: float) -> float:
            raw = payload.get(key)
            if raw in (None, ""):
                raw = os.environ.get(env_name)
            try:
                return max(0.0, float(default if raw in (None, "") else raw))
            except (TypeError, ValueError):
                return default

        idle_seconds_raw = payload.get("idle_seconds")
        if idle_seconds_raw in (None, ""):
            idle_seconds_raw = os.environ.get("VAST_IDLE_SECONDS")
        idle_seconds: float | None = None
        if idle_seconds_raw not in (None, ""):
            try:
                idle_seconds = max(0.0, float(idle_seconds_raw))
            except (TypeError, ValueError):
                idle_seconds = None

        return cls(
            max_minutes=number(
                "max_minutes",
                "VAST_MAX_LEASE_MINUTES",
                DEFAULT_MAX_LEASE_MINUTES,
            ),
            max_usd=min(
                GLOBAL_MAX_LEASE_USD,
                number("max_usd", "VAST_MAX_LEASE_USD", DEFAULT_MAX_LEASE_USD),
            ),
            idle_minutes=number(
                "idle_minutes",
                "VAST_IDLE_MINUTES",
                DEFAULT_IDLE_MINUTES,
            ),
            idle_seconds=idle_seconds,
            post_checkpoint_idle_seconds=number(
                "post_checkpoint_idle_seconds",
                "VAST_POST_CHECKPOINT_IDLE_SECONDS",
                DEFAULT_POST_CHECKPOINT_IDLE_SECONDS,
            ),
        )


@dataclass
class LeaseRuntime:
    instance_id: str
    started_at: float = field(default_factory=time.monotonic)
    hourly_usd: float = 0.0
    batch_id: str | None = None
    budget: BatchBudget = field(default_factory=BatchBudget.from_payload)
    status: str = "idle"
    idle_since: float | None = field(default_factory=time.monotonic)
    last_error: str | None = None
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    job_active: bool = False
    uploaded_shot_ids: set[str] = field(default_factory=set)
    last_checkpoint_at: float | None = None
    last_r2_upload_at: float | None = None
    watchdog_next_check_at: float | None = None
    watchdog_last_signature: tuple[Any, ...] | None = None
    watchdog_stale_checks: int = 0
    stop_requested: BaseException | None = field(default=None, repr=False)
    active_story_id: str | None = None
    active_root: str | None = None
    monitored_teardown_result: dict[str, Any] | None = field(default=None, repr=False)
    watch_usd: float = field(
        default_factory=lambda: min(
            GLOBAL_MAX_LEASE_USD,
            _env_number("VAST_WATCH_USD", DEFAULT_WATCH_USD),
        )
    )
    startup_timeout_seconds: float = field(
        default_factory=lambda: (
            _env_number(
                "VAST_STARTUP_TIMEOUT_MINUTES",
                DEFAULT_STARTUP_TIMEOUT_MINUTES,
            )
            * 60.0
        )
    )
    phase: str = "production"
    startup_started_at: float | None = None
    startup_ready: bool = False
    startup_downloaded_bytes: int = 0
    startup_stage: str | None = None
    startup_stage_revision: int = 0
    startup_health: str | None = None
    startup_health_revision: int = 0
    startup_probe: Callable[[], dict[str, Any]] | None = field(
        default=None,
        repr=False,
    )
    startup_subphase: str = "preflight"
    startup_subphase_started_at: float | None = None
    handshake: dict[str, Any] | None = None
    completed_episodes: set[tuple[str, str]] = field(default_factory=set)

    def elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    def spend_usd(self) -> float:
        return self.elapsed_seconds() * max(0.0, self.hourly_usd) / 3600.0

    def idle_seconds(self) -> float:
        if self.phase == "startup" or self.status != "idle" or self.idle_since is None:
            return 0.0
        return max(0.0, self.clock() - self.idle_since)

    def set_hourly_rate(self, hourly_usd: float | None) -> None:
        if hourly_usd is not None and hourly_usd > 0:
            self.hourly_usd = float(hourly_usd)

    def account_for_lease_age(self, age_seconds: float | None) -> None:
        if age_seconds is not None and age_seconds > 0:
            self.started_at = min(self.started_at, self.clock() - age_seconds)
            if self.phase == "startup":
                self.startup_started_at = min(
                    self.startup_started_at or self.started_at,
                    self.started_at,
                )

    def _reset_watchdog(self) -> None:
        self.watchdog_next_check_at = None
        self.watchdog_last_signature = None
        self.watchdog_stale_checks = 0
        self.stop_requested = None

    def begin_startup(
        self,
        probe: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.phase = "startup"
        self.startup_started_at = self.started_at
        self.startup_ready = False
        self.startup_probe = probe
        self.startup_subphase = "preflight"
        self.startup_subphase_started_at = self.clock()
        self.status = "booting"
        self.job_active = True
        self.idle_since = None
        self._reset_watchdog()

    def set_handshake(self, payload: dict[str, Any] | None) -> None:
        self.handshake = payload

    def set_startup_subphase(self, subphase: str) -> None:
        if subphase and subphase != self.startup_subphase:
            self.startup_subphase = subphase
            self.startup_subphase_started_at = self.clock()

    def complete_startup(self, ready: bool) -> None:
        self.startup_ready = bool(ready)
        if ready:
            self.phase = "ready"
            self.record_startup_health("ready")
            self._reset_watchdog()
            self.mark_idle()

    def begin_production(self) -> None:
        if self.phase == "production":
            return
        self.phase = "production"
        self._reset_watchdog()
        self.mark_busy()

    def completed_key(self, story_id: str, episode_code: str | None = None) -> tuple[str, str]:
        return str(story_id), _normalize_episode_code(episode_code)

    def remember_completed(self, story_id: str, episode_code: str | None = None) -> None:
        sid = str(story_id or "").strip()
        if sid:
            self.completed_episodes.add(self.completed_key(sid, episode_code))

    def already_completed(self, story_id: str, episode_code: str | None = None) -> bool:
        sid = str(story_id or "").strip()
        return bool(sid) and self.completed_key(sid, episode_code) in self.completed_episodes

    def adopt_batch(self, batch: dict[str, Any], *, busy: bool = True) -> None:
        self.batch_id = str(batch.get("batch_id") or "") or None
        self.budget = BatchBudget.from_payload(
            batch.get("budget") if isinstance(batch.get("budget"), dict) else None
        )
        if self.phase == "ready":
            self.phase = "startup"
        self.uploaded_shot_ids.clear()
        self.last_checkpoint_at = None
        self.last_r2_upload_at = None
        self._reset_watchdog()
        if busy:
            self.mark_busy()

    def mark_busy(self) -> None:
        if self.phase == "ready":
            self.phase = "startup"
        self.status = "busy"
        self.job_active = True
        self.idle_since = None

    def mark_idle(self) -> None:
        if self.status != "idle" or self.idle_since is None:
            self.idle_since = self.clock()
        self.status = "idle"
        self.job_active = False

    def set_active_work(self, story_id: str, root: Path) -> None:
        self.active_story_id = story_id
        self.active_root = str(root)

    def record_startup_bytes(self, downloaded_bytes: int) -> None:
        self.startup_downloaded_bytes = max(
            self.startup_downloaded_bytes,
            max(0, int(downloaded_bytes)),
        )

    def record_startup_stage(self, stage: str) -> None:
        if stage and stage != self.startup_stage:
            self.startup_stage = stage
            self.startup_stage_revision += 1

    def record_startup_health(self, health: str) -> None:
        if health and health != self.startup_health:
            self.startup_health = health
            self.startup_health_revision += 1

    def sample_startup_progress(self) -> None:
        if (
            self.phase == "production"
            or (self.phase == "ready" and not self.job_active)
            or self.startup_probe is None
        ):
            return
        try:
            snapshot = self.startup_probe()
        except Exception:  # noqa: BLE001 — a failed sample is no progress, not boot failure
            return
        try:
            self.record_startup_bytes(int(snapshot.get("downloaded_bytes") or 0))
        except (AttributeError, TypeError, ValueError):
            pass
        if isinstance(snapshot, dict) and snapshot.get("health"):
            self.record_startup_health(str(snapshot["health"]))

    def record_uploaded_shot(self, shot_id: str) -> None:
        if shot_id:
            self.uploaded_shot_ids.add(shot_id)
        self.last_r2_upload_at = self.clock()

    def record_checkpoint(self) -> None:
        self.last_checkpoint_at = self.clock()

    def record_r2_upload(self) -> None:
        self.last_r2_upload_at = self.clock()

    def progress_signature(self) -> tuple[int, float]:
        latest = max(self.last_checkpoint_at or 0.0, self.last_r2_upload_at or 0.0)
        return len(self.uploaded_shot_ids), latest

    def startup_progress_signature(self) -> tuple[int, int, int]:
        return (
            self.startup_downloaded_bytes,
            self.startup_stage_revision,
            self.startup_health_revision,
        )

    def observe_progress(self, event: str) -> None:
        if event.startswith("production_started:"):
            self.begin_production()
        elif event.startswith("startup_bytes:"):
            try:
                self.record_startup_bytes(int(event.split(":", 1)[1]))
            except ValueError:
                pass
        elif event.startswith("startup_stage:"):
            stage = event.split(":", 1)[1]
            self.record_startup_stage(stage)
            if stage == "preflight":
                self.set_startup_subphase("preflight")
            elif stage.startswith("weights") or stage in STARTUP_SETUP_STAGES:
                # torch/fonts/pip after hardware preflight are not the 2-minute nvidia-smi window.
                self.set_startup_subphase("weights")
            elif stage.startswith("start_comfy") or stage.startswith("wait_comfy") or stage == "comfy_probe":
                self.set_startup_subphase("comfy")
        elif event.startswith("startup_health:"):
            self.record_startup_health(event.split(":", 1)[1])
        elif event.startswith("shot_uploaded:"):
            if self.phase != "production":
                self.begin_production()
            self.record_uploaded_shot(event.split(":", 1)[1])
        elif event == "checkpoint_saved":
            if self.phase == "production":
                self.record_checkpoint()
            else:
                self.record_startup_stage("checkpoint")
        elif event.startswith("r2_uploaded:"):
            if self.phase == "production":
                self.record_r2_upload()
            else:
                self.record_startup_stage(event)
        elif self.phase != "production" and event:
            self.record_startup_stage(event)
        self.assert_within_budget()

    def classified_recycle_class(self) -> str | None:
        classified = recycle_failure_class(self.last_error)
        if classified:
            return classified
        hs = self.handshake if isinstance(self.handshake, dict) else {}
        pf = hs.get("preflight") if isinstance(hs.get("preflight"), dict) else {}
        if pf.get("ok") is False:
            return (
                recycle_failure_class(pf.get("failure_class") or pf.get("error"))
                or "preflight_failed"
            )
        return recycle_failure_class(getattr(self.stop_requested, "limit", None))

    def limit_reached(self) -> BudgetExceeded | StartupTimeout | None:
        minutes = self.elapsed_seconds() / 60.0
        if self.budget.max_minutes > 0 and minutes >= self.budget.max_minutes:
            return BudgetExceeded("max_minutes", minutes, self.budget.max_minutes)
        spend = self.spend_usd()
        if self.hourly_usd > 0 and spend >= GLOBAL_MAX_LEASE_USD:
            return BudgetExceeded(
                "global_max_usd",
                spend,
                GLOBAL_MAX_LEASE_USD,
            )
        if self.budget.max_usd > 0 and self.hourly_usd > 0 and spend >= self.budget.max_usd:
            return BudgetExceeded("max_usd", spend, self.budget.max_usd)
        if self.phase == "startup" and not self.startup_ready:
            classified = self.classified_recycle_class()
            if classified:
                return ClassifiedStartupFailure(
                    classified,
                    str(self.last_error or classified),
                )
            sub_started = self.startup_subphase_started_at or self.startup_started_at
            if sub_started is not None:
                elapsed = self.clock() - sub_started
                if self.startup_subphase == "preflight" and elapsed >= PREFLIGHT_TIMEOUT_SECONDS:
                    return PreflightTimeout()
                if self.startup_subphase == "comfy" and elapsed >= COMFY_STARTUP_TIMEOUT_SECONDS:
                    return ComfyStartupTimeout()
                if self.startup_subphase == "weights" and elapsed >= WEIGHT_PULL_TIMEOUT_SECONDS:
                    return WeightPullTimeout()
            if (
                self.startup_started_at is not None
                and self.startup_timeout_seconds > 0
                and self.clock() - self.startup_started_at >= self.startup_timeout_seconds
            ):
                return StartupTimeout(self.startup_timeout_seconds / 60.0)
        return None

    def watchdog_reached(self) -> ProgressStalled | None:
        if not self.job_active or self.spend_usd() <= self.watch_usd:
            return None
        now = self.clock()
        production = self.phase == "production"
        signature = (
            self.progress_signature()
            if production
            else self.startup_progress_signature()
        )
        stale_limit = (
            WATCHDOG_STALE_CHECKS
            if production
            else (
                max(
                    WATCHDOG_STALE_CHECKS,
                    int(self.startup_timeout_seconds / WATCHDOG_INTERVAL_SECONDS),
                )
                if self.startup_timeout_seconds > 0
                else STARTUP_WATCHDOG_STALE_CHECKS
            )
        )
        if self.watchdog_next_check_at is None:
            self.watchdog_last_signature = signature
            self.watchdog_next_check_at = now + WATCHDOG_INTERVAL_SECONDS
            self.watchdog_stale_checks = 0
            return None
        if now < self.watchdog_next_check_at:
            return None
        if signature == self.watchdog_last_signature:
            self.watchdog_stale_checks += 1
        else:
            self.watchdog_stale_checks = 0
            self.watchdog_last_signature = signature
        self.watchdog_next_check_at = now + WATCHDOG_INTERVAL_SECONDS
        if self.watchdog_stale_checks >= stale_limit:
            return ProgressStalled(self.watchdog_stale_checks, phase=self.phase)
        return None

    def stop_condition(self) -> BaseException | None:
        if self.stop_requested is not None:
            return self.stop_requested
        stop = self.limit_reached() or self.watchdog_reached()
        if stop is not None:
            self.stop_requested = stop
        return stop

    def assert_within_budget(self) -> None:
        reached = self.stop_condition()
        if reached:
            raise reached

    def idle_wait_seconds(self) -> float:
        configured = self.budget.idle_ttl_seconds()
        if configured <= 0:
            return 0.0
        if self.last_checkpoint_at is not None:
            post = float(self.budget.post_checkpoint_idle_seconds or 0.0)
            if post > 0:
                return min(configured, post)
        return configured

    def idle_timer_expired(self) -> bool:
        wait = self.idle_wait_seconds()
        return wait > 0 and self.idle_seconds() >= wait

    def idle_expired(self, comfy_inflight: int | None, pending_jobs: int | None = 0) -> bool:
        if self.phase == "startup":
            return False
        try:
            pending = int(pending_jobs or 0)
        except (TypeError, ValueError):
            pending = 0
        if pending > 0:
            return False
        return (
            not self.job_active
            and self.status == "idle"
            and self.idle_timer_expired()
            and comfy_inflight == 0
        )

    def heartbeat_fields(self, tunnel: dict[str, Any]) -> dict[str, Any]:
        if self.phase == "production":
            progress = {
                "uploaded_shots": len(self.uploaded_shot_ids),
                "watchdog_stale_checks": self.watchdog_stale_checks,
            }
        else:
            progress = {
                "downloaded_bytes": self.startup_downloaded_bytes,
                "stage": self.startup_stage,
                "comfy_health": self.startup_health,
                "boot_error": self.last_error,
                "watchdog_stale_checks": self.watchdog_stale_checks,
            }
        tunnel_payload: dict[str, Any] = {
            "hostname": tunnel.get("hostname"),
            "checked_at": tunnel.get("checked_at"),
        }
        # Explicit connected=false during HF pull makes cron treat the box as tunnel-down.
        # Omit it until Comfy is up; a live heartbeat still proves the process is alive.
        if self.phase != "startup" or tunnel.get("connected"):
            tunnel_payload["connected"] = bool(tunnel.get("connected"))
        payload: dict[str, Any] = {
            "batch_id": self.batch_id,
            "spend_usd": round(self.spend_usd(), 4),
            "idle_seconds": int(self.idle_seconds()),
            "phase": self.phase,
            "progress": progress,
            "tunnel": tunnel_payload,
        }
        if self.handshake:
            payload["profile_id"] = self.handshake.get("profile_id")
            payload["image_digest"] = self.handshake.get("image_digest")
            payload["host"] = self.handshake.get("host")
            payload["stack"] = self.handshake.get("stack")
            payload["resources"] = self.handshake.get("resources")
            payload["preflight"] = self.handshake.get("preflight")
            payload["adaptations"] = self.handshake.get("adaptations") or []
            if self.handshake.get("boot_log_tail"):
                progress["boot_log_tail"] = self.handshake.get("boot_log_tail")
        classified = self.classified_recycle_class()
        if classified:
            progress["stage"] = classified
            progress["boot_error"] = self.last_error or classified
            payload["boot_stage"] = classified
            payload["boot_error"] = self.last_error or classified
            payload["failure_class"] = classified
        return payload


def _matching_work_workers(payload: dict[str, Any], instance_id: str) -> list[dict[str, Any]]:
    workers = [w for w in (payload.get("workers") or []) if isinstance(w, dict)]
    matches = [w for w in workers if str(w.get("vast_instance_id") or "") == str(instance_id)]
    if not matches and len(workers) == 1:
        matches = workers
    return matches


def hourly_rate_from_work(payload: dict[str, Any], instance_id: str) -> float | None:
    """Find this lease's dph in /gpu/work workers, with env as a fallback."""
    env_rate = (
        os.environ.get("VAST_LEASE_HOURLY_USD")
        or os.environ.get("VAST_DPH_TOTAL")
        or os.environ.get("VAST_DPH")
        or ""
    )
    matches = _matching_work_workers(payload, instance_id)
    for worker in matches:
        raw = worker.get("lease_hourly_usd") or worker.get("dph_total") or worker.get("dph")
        if raw in (None, ""):
            caps = worker.get("capabilities")
            try:
                caps = json.loads(caps) if isinstance(caps, str) else caps
            except json.JSONDecodeError:
                caps = {}
            if isinstance(caps, dict):
                raw = caps.get("dph")
        try:
            rate = float(raw)
            if rate > 0:
                return rate
        except (TypeError, ValueError):
            continue
    try:
        rate = float(env_rate)
        return rate if rate > 0 else None
    except ValueError:
        return None


def lease_age_seconds_from_work(
    payload: dict[str, Any],
    instance_id: str,
    now: datetime | None = None,
) -> float | None:
    """Account for image-pull time before the Python process began."""
    now = now or datetime.now(timezone.utc)
    for worker in _matching_work_workers(payload, instance_id):
        raw = worker.get("started_at") or worker.get("start_date")
        if raw in (None, ""):
            continue
        try:
            timestamp = float(raw)
            if timestamp > 10_000_000_000:
                timestamp /= 1000.0
            return max(0.0, now.timestamp() - timestamp)
        except (TypeError, ValueError):
            pass
        try:
            started = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            return max(0.0, (now - started.astimezone(timezone.utc)).total_seconds())
        except ValueError:
            continue
    return None


def never_destroy_error(error: BaseException | str | None) -> str | None:
    text = str(error or "").lower().replace("-", "_").replace(" ", "_")
    for code in sorted(NEVER_DESTROY_ERRORS):
        if code in text:
            return code
    if "404" in text and "voice" in text and "upload" in text:
        return "voice_upload_404"
    if "404" in text and ("tts" in text or "cosyvoice" in text):
        return "tts_404"
    return None


def destroy_self(instance_id: str, reason: str, error: str | None = None) -> dict[str, Any]:
    """Destroy only this registered instance through boot.py's historical safety gate."""
    dry_run = self_destroy_dry_run()
    api_key = os.environ.get("CONTAINER_API_KEY") or os.environ.get("VAST_API_KEY") or ""
    if not instance_id:
        return {"ok": False, "destroyed": False, "error": "VAST_INSTANCE_ID missing"}
    if not may_destroy(reason, error):
        return {
            "ok": False,
            "destroyed": False,
            "refused": True,
            "reason": reason,
            "error": f"destroy refused by boot safety gate: reason={reason!r} error={error!r}",
        }
    if not dry_run and not api_key:
        return {
            "ok": False,
            "destroyed": False,
            "error": "CONTAINER_API_KEY and VAST_API_KEY missing",
        }
    client = VastClient(api_key=api_key, dry_run=dry_run)
    lease = LeaseSession(instance_id=instance_id, state="running", last_error=error)
    try:
        result = destroy_if_allowed(client, lease, reason, {instance_id}, error=error)
    except DestroyRefused as exc:
        return {
            "ok": False,
            "destroyed": False,
            "refused": True,
            "reason": reason,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 — preserve the worker so it can retry/report
        return {
            "ok": False,
            "destroyed": False,
            "reason": reason,
            "error": f"{type(exc).__name__}:{exc}",
        }
    if isinstance(result, dict) and (
        result.get("success") is False or result.get("ok") is False
    ):
        return {
            "ok": False,
            "destroyed": False,
            "reason": reason,
            "error": str(result.get("error") or result.get("msg") or result),
            "result": result,
        }
    return {
        "ok": True,
        "destroyed": not dry_run,
        "dry_run": dry_run,
        "reason": reason,
        "result": result,
    }


def persist_boot_failure(payload: dict[str, Any], story_id: str | None = None) -> dict[str, Any]:
    """Upload boot/preflight failure JSON to R2 before instance teardown."""
    story_id = (story_id or os.environ.get("AF_STORY_ID") or "").strip()
    if not story_id:
        return {"ok": False, "skipped": True, "reason": "no_story_id"}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preflight_failed.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            result = put_file(
                join_story(story_id, "gpu/boot/preflight_failed.json"),
                path,
                "application/json",
            )
    except Exception as exc:  # noqa: BLE001 — teardown must continue
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}"}
    return result if isinstance(result, dict) else {"ok": False, "result": result}


def _probe_audio(path: Path) -> dict:
    """Whether a rendered clip still carries H3/Hailuo speech that must not be muxed."""
    meta = {"has_audio_stream": False, "audio_rms": 0.0}
    if not path.is_file():
        return meta
    try:
        raw = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type,duration",
                "-of",
                "json",
                str(path),
            ],
            timeout=30,
        )
        data = json.loads(raw.decode())
        streams = data.get("streams") or []
        if streams:
            meta["has_audio_stream"] = True
    except Exception:  # noqa: BLE001
        return meta
    if meta["has_audio_stream"]:
        wav = path.with_suffix(".af_probe.wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                check=True,
                capture_output=True,
                timeout=30,
            )
            from anime_factory.compose import wav_rms

            if wav.is_file():
                meta["audio_rms"] = float(wav_rms(wav))
        except Exception:  # noqa: BLE001
            meta["audio_rms"] = MIN_SPEECH_RMS if meta["has_audio_stream"] else 0.0
        finally:
            wav.unlink(missing_ok=True)
    return meta


def _probe_video(path: Path) -> dict:
    meta = {"width": VIDEO_WIDTH, "height": VIDEO_HEIGHT, "duration": 8.0, "frames": int(8 * VIDEO_FPS), "size_bytes": path.stat().st_size if path.is_file() else 0}
    if not path.is_file():
        return meta
    try:
        raw = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            timeout=30,
        )
        data = json.loads(raw.decode())
        stream = (data.get("streams") or [{}])[0]
        meta["width"] = int(stream.get("width") or VIDEO_WIDTH)
        meta["height"] = int(stream.get("height") or VIDEO_HEIGHT)
        dur = float(stream.get("duration") or 0) or meta["duration"]
        meta["duration"] = dur
        frames = stream.get("nb_frames")
        meta["frames"] = int(frames) if frames and str(frames).isdigit() else int(round(dur * VIDEO_FPS))
    except Exception:  # noqa: BLE001
        pass
    return meta


def pull_story(
    story_id: str,
    root: Path,
    episode_code: str | None = None,
    *,
    skip_existing: bool = True,
) -> list[str]:
    """Incremental R2 sync. Reuses local files and, when episode-scoped, skips other episodes."""
    root.mkdir(parents=True, exist_ok=True)
    prefix = story_prefix(story_id)
    ep = ""
    if episode_code:
        ep = _normalize_episode_code(episode_code)

    def key_filter(key: str) -> bool:
        if not ep:
            return True
        rel = key[len(prefix) :] if key.startswith(prefix) else key
        if rel.startswith("episodes/"):
            return rel.startswith(f"episodes/{ep}/")
        return True

    return download_prefix(
        prefix,
        root,
        strip_prefix=prefix,
        skip_existing=skip_existing,
        key_filter=key_filter if ep else None,
    )


def _upload_ok(result: Any) -> bool:
    return isinstance(result, dict) and result.get("ok") is True


def best_effort_upload_checkpoint(story_id: str | None, root: Path | str | None) -> dict[str, Any]:
    """Upload the latest stable sqlite checkpoint without touching a live connection."""
    if not story_id or not root:
        return {"ok": False, "skipped": True, "reason": "active_story_missing"}
    db_path = Path(root) / "story.sqlite"
    if not db_path.is_file():
        return {"ok": False, "skipped": True, "reason": "checkpoint_missing"}
    try:
        result = put_file(
            join_story(story_id, "story.sqlite"),
            db_path,
            "application/octet-stream",
        )
    except Exception as exc:  # noqa: BLE001 — teardown must continue
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}"}
    return result if isinstance(result, dict) else {"ok": False, "result": result}


def _checkpoint_story(
    conn,
    story_id: str,
    root: Path,
    progress: ProgressCallback | None = None,
    *,
    upload: bool = False,
) -> dict[str, Any]:
    db_path = checkpoint_and_upload(conn, root / "story.sqlite")
    stop: BudgetExceeded | ProgressStalled | StartupTimeout | None = None
    try:
        if progress:
            progress("checkpoint_saved")
    except (BudgetExceeded, ProgressStalled, StartupTimeout) as exc:
        stop = exc
    if not upload:
        if stop:
            raise stop
        return {"ok": True, "path": str(db_path)}
    try:
        result = put_file(
            join_story(story_id, "story.sqlite"),
            db_path,
            "application/octet-stream",
        )
    except Exception:
        if stop:
            raise stop
        raise
    if stop:
        raise stop
    if _upload_ok(result) and progress:
        progress("r2_uploaded:story.sqlite")
    return result if isinstance(result, dict) else {"ok": False, "result": result}


def _cast_map_from_shots(shots: list[dict]) -> dict[str, dict]:
    cast: dict[str, dict] = {}
    for shot in shots:
        cid = str(shot.get("character_id") or "").strip()
        if cid:
            cast[cid] = {}
        for cut in shot.get("cuts") or []:
            for other in cut.get("characters") or []:
                if str(other).strip():
                    cast[str(other).strip()] = {}
    return cast


def _hydrate_board_shots(shots: list[dict]) -> list[dict]:
    cast = _cast_map_from_shots(shots)
    return [hydrate_shot_identity(shot, cast=cast) for shot in shots]


def _board_shots(root: Path) -> list[dict]:
    """Video units: prefer expanded segments. LongLive keeps scene-length takes; H3 still splits at 8s."""
    board = root / "episodes" / EP / "board.json"
    if board.is_file():
        data = json.loads(board.read_text(encoding="utf-8"))
        segs = list(data.get("segments") or [])
        if segs:
            return _hydrate_board_shots(segs)
        shots = list(data.get("shots") or [])
        if shots:
            limit = max_seconds_for_backend(select_video_backend(root=root))
            return _hydrate_board_shots(expand_shots_to_segments(shots, max_s=limit))
    return []


def _scene_plate_file(root: Path, shot: dict) -> Path | None:
    lid = str(shot.get("location_id") or shot.get("scene_id") or "").strip()
    if not lid:
        return None
    for name in ("plate_base.png", "plate.png"):
        path = root / "assets" / "scenes" / lid / name
        if path.is_file():
            return path
    return None


def _reuse_scene_plate_as_fl2va_keyframe(root: Path, shot: dict) -> bool:
    """Establishing fl2va heads should use the locked loc plate, not a new Flux still."""
    from anime_factory.design import still_file_ok
    from anime_factory.keyframe import keyframe_file_ok

    sid = str(shot.get("id") or "").strip()
    if not sid:
        return False
    kf = root / "episodes" / EP / "keyframes" / sid / "f1.png"
    if keyframe_file_ok(kf):
        return True
    plate = _scene_plate_file(root, shot)
    if plate is None or not still_file_ok(plate):
        return False
    kf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plate, kf)
    return keyframe_file_ok(kf)


def generate_missing_stills(
    story_id: str,
    root: Path,
    conn,
    router=None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Anime SDXL on this GPU for missing sheets/plates/keyframes. Skip files already on disk/R2.

    chain_index > 0 never mints stills. ref2va heads bind character sheets. fl2va heads
    must actually write f1.png or this stage fails — never mark keyframe succeeded with 0 files.
    """
    os.environ.setdefault("ANIME_FACTORY_GPU_STILLS", "1")
    client = KolorsClient(["gpu"], live=True, gpu_generate=lambda p: generate_still(p, router=router))
    created: list[str] = []
    errors: list[str] = []
    from anime_factory.design import library_from_db

    lib = library_from_db(conn, story_id, client, root, skip_existing=True)
    if lib.get("needs_human"):
        raise RuntimeError(f"design visual QC needs_human: {lib['needs_human']}; refusing H3")
    created.extend(lib.get("created") or [])
    assets_index = {"items": [{"id": k, **v} for k, v in (lib.get("specs") or {}).items()]}
    geo = export_geo(conn, story_id, invented=True, sources=[])
    geo_path = root / "canon" / "geo.json"
    if geo_path.is_file():
        try:
            file_geo = json.loads(geo_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            file_geo = {}
        if isinstance(file_geo, dict):
            geo = {**geo, **file_geo}
    geo = hydrate_geo_interiors(conn, geo)
    if geo.get("interiors"):
        timeline = {"story_id": story_id, "events": []}
        continuity = {"story_id": story_id}
        tl_path = root / "canon" / "timeline.json"
        cont_path = root / "canon" / "continuity.json"
        if tl_path.is_file():
            try:
                timeline = json.loads(tl_path.read_text(encoding="utf-8")) or timeline
            except json.JSONDecodeError:
                pass
        if cont_path.is_file():
            try:
                continuity = json.loads(cont_path.read_text(encoding="utf-8")) or continuity
            except json.JSONDecodeError:
                pass
        write_canon_dir(root, geo, timeline, continuity)
    shots = _board_shots(root)
    board_path = root / "episodes" / EP / "board.json"
    if board_path.is_file() and shots:
        try:
            payload = json.loads(board_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            if payload.get("segments"):
                payload["segments"] = shots
            else:
                payload["shots"] = shots
            board_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for shot in shots:
        sid = str(shot.get("id") or "")
        if not sid:
            continue
        refs_json = json.dumps(list(shot.get("refs") or []), ensure_ascii=False)
        mode = str(shot.get("h3_mode") or "")
        _db_execute(
            conn,
            "UPDATE segments SET refs_json = ?, h3_mode = ? WHERE id = ?",
            (refs_json, mode, sid),
        )
        _db_execute(
            conn,
            "UPDATE shots SET refs_json = ?, h3_mode = ? WHERE id = ?",
            (refs_json, mode, sid),
        )
    for shot in shots:
        if not is_chain_head(shot):
            continue
        if needs_first_frame_still(shot):
            from anime_factory.keyframe import keyframe_file_ok

            kf = root / "episodes" / EP / "keyframes" / shot["id"] / "f1.png"
            # A 3-byte f1.png used to satisfy `>= 1` and skip the redraw.
            if keyframe_file_ok(kf) or _reuse_scene_plate_as_fl2va_keyframe(root, shot):
                continue
        try:
            ensure_keyframe(conn, story_id, EP, shot, geo, assets_index, lib.get("specs") or {}, client, None, "fiction", root)
            created.append(shot["id"])
        except Exception as exc:  # noqa: BLE001 — collect, then fail the stage closed
            errors.append(f"keyframe_error:{shot['id']}:{exc}")
    uploads = [
        *upload_tree(story_id, root, "assets"),
        *upload_tree(story_id, root, "episodes"),
        *upload_tree(story_id, root, "canon"),
    ]
    failed_puts = [
        str(item.get("key") or item.get("error") or "upload")
        for item in uploads
        if isinstance(item, dict) and item.get("ok") is False and not item.get("skipped")
    ]
    if failed_puts:
        errors.append("r2_upload_failed:" + ",".join(failed_puts[:8]))
    if progress:
        for upload in uploads:
            if isinstance(upload, dict) and upload.get("ok"):
                progress(f"r2_uploaded:{upload.get('key') or 'stills'}")
    try:
        assert_keyframe_files(root, EP, shots, assets_index)
    except RuntimeError as exc:
        errors.append(str(exc))
    if errors:
        raise RuntimeError("keyframe stage failed: " + "; ".join(errors[:8]))
    return {"created": created, "count": len(created), "backend": "gpu_anime_sdxl", "ok": True}


def _comfy_input_dir() -> Path:
    return Path(os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI") / "input"


def _stage_named_image(src: Path, name: str) -> str | None:
    if not src.is_file():
        return None
    input_dir = _comfy_input_dir()
    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        dest = input_dir / name
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)
        return name
    except OSError:
        return name if src.is_file() else None


def _comfy_image_name(src: Path) -> str:
    """Unique Comfy input/ name so two characters' sheet_front.png do not clobber."""
    name = src.name if src.suffix else f"{src.name}.png"
    parts = src.parts
    if "characters" in parts:
        idx = parts.index("characters")
        if idx + 1 < len(parts):
            cid = parts[idx + 1]
            if cid and not name.startswith(f"{cid}_"):
                return f"{cid}_{name}"
    if "scenes" in parts:
        idx = parts.index("scenes")
        if idx + 1 < len(parts):
            lid = parts[idx + 1]
            if lid and not name.startswith(f"{lid}_"):
                return f"{lid}_{name}"
    return name


def _read_asset_lock_index(root: Path) -> dict[str, Any]:
    path = Path(root) / "assets" / "index.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _path_if_file(path: Path | None) -> Path | None:
    """Locked reference PNGs only. A 1-byte file is not a character sheet."""
    from anime_factory.keyframe import keyframe_file_ok

    if path is None:
        return None
    return path if keyframe_file_ok(path) else None


def _coerce_lock_path(root: Path, kind: str, aid: str, hit: Any) -> Path | None:
    if hit is None:
        return None
    if isinstance(hit, Path):
        return _path_if_file(hit) or _path_if_file(root / "assets" / kind / aid / hit.name)
    text = str(hit).strip()
    if not text:
        return None
    cand = Path(text)
    if cand.is_file():
        return _path_if_file(cand)
    return (
        _path_if_file(root / text)
        or _path_if_file(root / "assets" / kind / aid / Path(text).name)
        or _path_if_file(root / "assets" / kind / aid / text)
    )


def _locked_selected_from_index(root: Path, kind: str, aid: str) -> Path | None:
    data = _read_asset_lock_index(root)
    bucket = data.get(kind)
    row = bucket.get(aid) if isinstance(bucket, dict) else None
    selected = str((row or {}).get("selected") or "").strip() if isinstance(row, dict) else ""
    if selected:
        hit = _coerce_lock_path(root, kind, aid, selected)
        if hit is not None:
            return hit
    return None


def _locked_character_file(root: Path, cid: str) -> Path | None:
    aid = str(cid or "").strip()
    if not aid:
        return None
    try:
        from anime_factory.asset_lock import is_qc_locked, locked_character_file

        if not is_qc_locked(root, character_id=aid):
            return None
        hit = _coerce_lock_path(root, "characters", aid, locked_character_file(root, aid))
        if hit is not None:
            return hit
    except (ImportError, AttributeError, TypeError):
        pass
    return None


def _locked_scene_file(root: Path, lid: str) -> Path | None:
    aid = str(lid or "").strip()
    if not aid:
        return None
    try:
        from anime_factory.asset_lock import is_qc_locked, locked_scene_file

        if not is_qc_locked(root, scene_id=aid):
            return None
        hit = _coerce_lock_path(root, "scenes", aid, locked_scene_file(root, aid))
        if hit is not None:
            return hit
    except (ImportError, AttributeError, TypeError):
        pass
    return None


def _resolve_ref_file(root: Path, ref: str) -> Path | None:
    text = str(ref or "").strip()
    if not text:
        return None
    name = Path(text).name
    rid = text
    if rid.startswith("char_") and rid.endswith("_sheet"):
        locked = _locked_character_file(root, rid[len("char_") : -len("_sheet")])
        if locked is not None:
            return locked
        return None
    stem = Path(text).stem
    if stem.startswith("char_") and stem.endswith("_sheet"):
        locked = _locked_character_file(root, stem[len("char_") : -len("_sheet")])
        if locked is not None:
            return locked
        return None
    if rid.startswith("plate_"):
        locked = _locked_scene_file(root, rid[len("plate_") :])
        if locked is not None:
            return locked
        return None
    parts = Path(text).parts
    if "characters" in parts:
        idx = parts.index("characters")
        if idx + 1 < len(parts):
            locked = _locked_character_file(root, parts[idx + 1])
            if locked is not None:
                return locked
            return None
    if "scenes" in parts:
        idx = parts.index("scenes")
        if idx + 1 < len(parts):
            locked = _locked_scene_file(root, parts[idx + 1])
            if locked is not None:
                return locked
            return None
    candidates = [
        Path(text),
        root / text,
        root / "assets" / text,
        _comfy_input_dir() / name,
        _comfy_input_dir() / (name if "." in name else f"{name}.png"),
    ]
    for path in candidates:
        found = _path_if_file(path)
        if found is not None:
            return found
    return None


def _character_sheet_files(root: Path, ref: str) -> list[Path]:
    """One locked sheet or plate per ref. Do not dump every sheet_*.png in the folder."""
    src = _resolve_ref_file(root, ref)
    return [src] if src is not None else []


def _stage_first_frame(shot: dict, root: Path) -> dict:
    """Copy last-frame guide + the locked character-sheet/scene-plate pack into Comfy input/."""
    shot = dict(shot)
    mode = select_mode(shot)
    chain_tail = int(shot.get("chain_index") or 0) > 0 or bool(shot.get("chain_source_last_frame"))
    first = Path(str(shot.get("first_frame_path") or ""))
    if not first.is_file() and chain_tail:
        chained = root / "episodes" / EP / "keyframes" / shot["id"] / "chain_first.png"
        if chained.is_file() and chained.stat().st_size >= 1:
            first = chained
    if not first.is_file() and not chain_tail:
        first = root / "episodes" / EP / "keyframes" / shot["id"] / "f1.png"
    if first.is_file() and first.stat().st_size >= 1:
        name = first.name if first.name.endswith(".png") else f"{shot['id']}_f1.png"
        staged = _stage_named_image(first, name)
        if staged:
            shot["first_frame_path"] = staged
    else:
        # Do not leave an R2 key / dummy f1.png for LoadImage on ref2va heads.
        shot["first_frame_path"] = None
    last = Path(str(shot.get("last_frame_path") or ""))
    if last.is_file() and last.stat().st_size >= 1:
        staged_last = _stage_named_image(last, last.name)
        if staged_last:
            shot["last_frame_path"] = staged_last
    staged_refs: list[str] = []
    seen_names: set[str] = set()
    pack_refs = list(shot.get("refs") or [])
    plate_id = str(shot.get("plate_id") or "").strip()
    if plate_id and plate_id not in pack_refs:
        pack_refs.append(plate_id)
    for ref in pack_refs:
        files = _character_sheet_files(root, str(ref))
        if not files:
            continue
        for src in files:
            name = _comfy_image_name(src)
            if name in seen_names:
                continue
            staged = _stage_named_image(src, name) or name
            staged_refs.append(staged)
            seen_names.add(name)
    if mode == "ref2va" and not staged_refs:
        cid = str(shot.get("character_id") or "").strip()
        if cid:
            locked = _locked_character_file(root, cid)
            files = [locked] if locked is not None else _character_sheet_files(root, f"char_{cid}_sheet")
            for src in files:
                name = _comfy_image_name(src)
                if name in seen_names:
                    continue
                staged = _stage_named_image(src, name) or name
                staged_refs.append(staged)
                seen_names.add(name)
    if staged_refs:
        shot["refs"] = staged_refs
    elif mode == "ref2va":
        shot["refs"] = []
    return shot


def _upload_h3_inputs(router: ComfyRouter, shot: dict, root: Path) -> None:
    """Stage first frame and refs in Comfy before expensive sampling. Fail closed on upload errors."""
    sid = str(shot.get("id") or "?")
    first_name = shot.get("first_frame_path")
    first_src = None
    if first_name:
        candidate = Path(str(first_name))
        if candidate.is_file():
            first_src = candidate
        else:
            kf = root / "episodes" / EP / "keyframes" / sid / "f1.png"
            if kf.is_file():
                first_src = kf
            chained = root / "episodes" / EP / "keyframes" / sid / "chain_first.png"
            if chained.is_file():
                first_src = chained
            input_hit = _comfy_input_dir() / Path(str(first_name)).name
            if first_src is None and input_hit.is_file():
                first_src = input_hit
    if first_src and first_src.is_file():
        router.upload_image(Path(str(first_name or first_src.name)).name, first_src.read_bytes())
    for ref in list(shot.get("refs") or []):
        src = _resolve_ref_file(root, str(ref))
        if src is None:
            staged = _comfy_input_dir() / Path(str(ref)).name
            if staged.is_file():
                src = staged
        if src is None or not src.is_file():
            raise RuntimeError(f"h3 upload missing ref for {sid}: {ref}")
        router.upload_image(Path(str(ref)).name, src.read_bytes())
    last = shot.get("last_frame_path")
    if last and not is_directory_like(last):
        last_path = Path(str(last))
        if not last_path.is_file():
            staged_last = _comfy_input_dir() / last_path.name
            if staged_last.is_file():
                last_path = staged_last
        if last_path.is_file():
            router.upload_image(last_path.name, last_path.read_bytes())


def _submit_h3_gpu(
    router: ComfyRouter,
    shot: dict,
    root: Path,
    dest: Path,
    progress: ProgressCallback | None = None,
) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    chain_tail = int(shot.get("chain_index") or 0) > 0 or bool(shot.get("chain_source_last_frame"))
    if chain_tail and is_directory_like(shot.get("first_frame_path")):
        chained = root / "episodes" / EP / "keyframes" / shot["id"] / "chain_first.png"
        if chained.is_file():
            shot = dict(shot, first_frame_path=str(chained))
        else:
            raise RuntimeError(f"chain tail {shot.get('id')} refuses submit without last.png")
    shot = _stage_first_frame(shot, root)
    mode = select_mode(shot)
    if chain_tail and is_directory_like(shot.get("first_frame_path")):
        raise RuntimeError(f"chain tail {shot.get('id')} refuses submit without last.png")
    if mode == "ref2va":
        names = ref_image_filenames(shot)
        if not names:
            raise RuntimeError(f"ref2va {shot.get('id')} has no staged character/scene refs")
    elif is_directory_like(shot.get("first_frame_path")):
        raise RuntimeError(f"{mode} {shot.get('id')} missing first frame on disk")
    _upload_h3_inputs(router, shot, root)
    graph = prepare_workflow(shot)
    prompt_id = router.prompt(graph.get("prompt", graph), client_id=shot["id"])
    deadline = time.time() + float(os.environ.get("H3_SHOT_TIMEOUT_S") or 1200)
    next_progress = 0.0
    history: dict = {}
    while time.time() < deadline:
        now = time.monotonic()
        if progress and now >= next_progress:
            progress(f"anim:{shot['id']}")
            next_progress = now + max(5.0, float(os.environ.get("HEARTBEAT_SECONDS") or 30))
        history = router.history(prompt_id) or {}
        exec_err = extract_execution_error(history, prompt_id)
        if exec_err:
            raise RuntimeError(format_execution_error(exec_err, shot["id"]))
        body = history.get(prompt_id) if prompt_id in history else history
        if isinstance(body, dict):
            st_obj = body.get("status") if isinstance(body.get("status"), dict) else {}
            st = (st_obj or {}).get("status_str") or body.get("status_str")
            completed = bool((st_obj or {}).get("completed") or body.get("completed"))
            if st in {"error", "failed", "cancelled"}:
                raise RuntimeError(f"h3 {shot['id']} {st}")
            outputs = body.get("outputs")
            if outputs:
                break
            if completed:
                raise RuntimeError(f"h3 {shot['id']} completed with no output")
        time.sleep(5)
    else:
        raise TimeoutError(f"h3 timeout {shot['id']}")
    body = history.get(prompt_id) if prompt_id in history else history
    outputs = (body or {}).get("outputs") or {}
    filename = None
    sub = ""
    for node_out in outputs.values():
        videos = (node_out or {}).get("gifs") or (node_out or {}).get("videos") or (node_out or {}).get("images") or []
        if videos:
            filename = videos[0].get("filename")
            sub = videos[0].get("subfolder") or ""
            break
    if not filename:
        raise RuntimeError(f"h3 no output {shot['id']}")
    blob = router.view(filename, subfolder=sub, type_="output")
    dest.write_bytes(blob)
    last_rel = f"episodes/{EP}/keyframes/{shot['id']}/last.png"
    last_path = root / last_rel
    extract_last_frame(dest, last_path)
    shot["last_frame_path"] = str(last_path)
    router.upload_image(f"{shot['id']}_last.png", last_path.read_bytes())
    return {"shot": shot["id"], "path": str(dest), "bytes": len(blob), "last_frame": shot.get("last_frame_path")}


def _submit_h3(
    router: ComfyRouter,
    shot: dict,
    root: Path,
    dest: Path,
    progress: ProgressCallback | None = None,
) -> dict:
    """GPU sampling only. Tests and legacy callers still patch this symbol."""
    return _submit_h3_gpu(router, shot, root, dest, progress=progress)


def _h3_normalize_and_qc(
    root: Path,
    conn,
    shot: dict,
    dest: Path,
    prev: dict | None,
    *,
    version: int,
    rel: str,
    story_id: str,
    progress: ProgressCallback | None = None,
) -> tuple[str, str, Path | None]:
    sid = str(shot.get("id") or "")
    used_mode = select_mode(shot)
    _normalize_h3_for_qc(
        dest,
        h3_resolution_profile(shot, oom_fallback=bool(shot.get("h3_oom_fallback"))),
        progress=progress,
    )
    print({"h3_done": sid, "bytes": dest.stat().st_size if dest.is_file() else 0}, flush=True)
    last_path = Path(str(shot.get("last_frame_path") or ""))
    if not last_path.is_file():
        last_path = _ensure_last_frame(root, shot, dest)
    meta = _probe_video(dest)
    verdict = incremental_qc_segment(
        conn,
        EP,
        sid,
        meta,
        float(shot.get("duration") or 8.0),
        segment=shot,
        prev_segment=prev,
        used_mode=used_mode,
        video_path=dest,
        story_root=root,
    )
    try:
        record_generation_result(
            conn,
            sid,
            version,
            join_story(story_id, rel),
            shot.get("seed"),
            used_mode,
            "completed" if verdict == "pass" else "failed",
            verdict,
        )
    except Exception:
        pass
    return verdict, sid, last_path


def _h3_upload_files(
    story_id: str,
    root: Path,
    sid: str,
    dest: Path,
    rel: str,
    last_path: Path | None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Background-safe R2/file I/O only. Caller commits sqlite on the main thread."""
    upload = put_file(join_story(story_id, rel), dest, "video/mp4")
    if not _upload_ok(upload):
        raise RuntimeError(f"shot R2 upload failed: {sid}")
    if last_path is None or not last_path.is_file() or last_path.stat().st_size < 32:
        raise RuntimeError(f"shot last.png missing: {sid}")
    last_upload = put_file(
        join_story(story_id, f"episodes/{EP}/keyframes/{sid}/last.png"),
        last_path,
        "image/png",
    )
    if not _upload_ok(last_upload):
        raise RuntimeError(f"shot last.png R2 upload failed: {sid}")
    if progress:
        progress(f"shot_uploaded:{sid}")
    return {"sid": sid, "rel": rel, "story_id": story_id}


def _h3_commit_passing(
    conn,
    story_id: str,
    root: Path,
    upload_result: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> str:
    """Main-thread sqlite + checkpoint after background upload succeeds."""
    sid = str(upload_result.get("sid") or "")
    rel = str(upload_result.get("rel") or "")
    mark_completed_passing(conn, sid, join_story(story_id, rel))
    _checkpoint_story(conn, story_id, root, progress=progress, upload=True)
    return sid


def _h3_upload_passing(
    story_id: str,
    root: Path,
    conn,
    sid: str,
    dest: Path,
    rel: str,
    last_path: Path | None,
    progress: ProgressCallback | None = None,
) -> str:
    """Compat wrapper: files then main-thread-style commit (avoid from worker threads)."""
    uploaded = _h3_upload_files(story_id, root, sid, dest, rel, last_path, progress)
    return _h3_commit_passing(conn, story_id, root, uploaded, progress=progress)


def flush_gpu_artifacts(
    story_id: str,
    root: Path,
    conn,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Push the latest sqlite + per-shot uploads before compose/idle teardown."""
    result = _checkpoint_story(conn, story_id, root, progress=progress, upload=True)
    if progress:
        progress("gpu_flushed")
    return result if isinstance(result, dict) else {"ok": True}


def _stage_longlive_first_frame(shot: dict, root: Path, src: Path) -> dict:
    """Copy the locked still to the canonical keyframe path so anim/R2 share one png."""
    shot = dict(shot)
    sid = str(shot.get("id") or "")
    if not sid or not src.is_file() or src.stat().st_size < 1:
        shot["first_frame_path"] = str(src)
        return shot
    staged = root / "episodes" / EP / "keyframes" / sid / "f1.png"
    if src.resolve() != staged.resolve():
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, staged)
    shot["first_frame_path"] = str(staged if staged.is_file() else src)
    return shot


def _prepare_longlive_shot(shot: dict, root: Path) -> dict:
    """Keep absolute first-frame paths. LongLive is first-frame I2V, not an H3 sheet pack."""
    shot = dict(shot)
    first = Path(str(shot.get("first_frame_path") or ""))
    if first.is_file():
        return _stage_longlive_first_frame(shot, root, first)
    sid = str(shot.get("id") or "")
    chained = root / "episodes" / EP / "keyframes" / sid / "chain_first.png"
    if chained.is_file():
        return _stage_longlive_first_frame(shot, root, chained)
    kf = root / "episodes" / EP / "keyframes" / sid / "f1.png"
    if kf.is_file():
        return _stage_longlive_first_frame(shot, root, kf)
    cid = str(shot.get("character_id") or "").strip()
    locked = _locked_character_file(root, cid) if cid else None
    if locked is not None and locked.is_file():
        return _stage_longlive_first_frame(shot, root, locked)
    plate_id = str(shot.get("plate_id") or "").strip()
    plate = _resolve_ref_file(root, plate_id) if plate_id else None
    if plate is not None and plate.is_file():
        return _stage_longlive_first_frame(shot, root, plate)
    return shot


def _submit_video(
    router: ComfyRouter | None,
    shot: dict,
    root: Path,
    dest: Path,
    progress: ProgressCallback | None,
    backend: str,
) -> dict:
    if backend == "longlive":
        shot = _prepare_longlive_shot(shot, root)
        return submit_longlive(shot, dest, root=root, progress=progress)
    if router is None:
        raise RuntimeError("no_comfy_router")
    return _submit_h3(router, shot, root, dest, progress=progress)


def _normalize_h3_for_qc(
    path: Path,
    resolution: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> None:
    """Lanczos upscale gen canvas to 1280×720 delivery and strip Hailuo audio.

    H3 samples at 1024×576 (or 864×480 OOM fallback). Always strip H3 audio (`-an`)
    so Hailuo speech cannot leak into the CosyVoice2 mix.
    """
    gen_w = int(resolution.get("gen_width") or resolution.get("width") or 0)
    gen_h = int(resolution.get("gen_height") or resolution.get("height") or 0)
    del_w = int(resolution.get("delivery_width") or VIDEO_WIDTH)
    del_h = int(resolution.get("delivery_height") or VIDEO_HEIGHT)
    normalized = path.with_name(f"{path.stem}.delivery{path.suffix}")
    vf = "null"
    if gen_w != del_w or gen_h != del_h:
        vf = f"scale={del_w}:{del_h}:flags=lanczos"
    try:
        _run_checked(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-vf",
                vf,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(normalized),
            ],
            progress=progress,
            stage="anim",
        )
        normalized.replace(path)
    finally:
        normalized.unlink(missing_ok=True)


def _keep_native_strip_audio(
    path: Path,
    progress: ProgressCallback | None = None,
) -> None:
    """LongLive native 1280×704 stays until compose scale/pad. Strip audio only."""
    silent = path.with_name(f"{path.stem}.silent{path.suffix}")
    try:
        _run_checked(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-an",
                str(silent),
            ],
            progress=progress,
            stage="anim",
        )
        silent.replace(path)
    except Exception:
        silent.unlink(missing_ok=True)
        return
    finally:
        silent.unlink(missing_ok=True)


class CpuPostQueue:
    """Single-thread background upload/QC while the next GPU shot samples."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-cpu-post")
        self._futures: list[Future[Any]] = []

    def submit(self, fn, *args, **kwargs) -> None:
        self._futures.append(self._executor.submit(fn, *args, **kwargs))

    def drain(self) -> list[Any]:
        out: list[Any] = []
        errors: list[BaseException] = []
        for future in self._futures:
            try:
                out.append(future.result())
            except BaseException as exc:  # noqa: BLE001 — collect then fail closed
                errors.append(exc)
        self._futures.clear()
        if errors:
            raise errors[0]
        return out

    def close(self) -> None:
        self.drain()
        self._executor.shutdown(wait=True)


def _db_execute(conn, sql: str, params: tuple = ()) -> None:
    execute = getattr(conn, "execute", None)
    if not callable(execute):
        return
    try:
        execute(sql, params)
        commit = getattr(conn, "commit", None)
        if callable(commit):
            commit()
    except Exception:
        return


def _link_next_chain(shots: list[dict], index: int, last_path: Path) -> None:
    if index + 1 >= len(shots) or not last_path.is_file():
        return
    cur = shots[index]
    nxt = shots[index + 1]
    if nxt.get("chain_id") != cur.get("chain_id"):
        return
    if int(nxt.get("chain_index") or 0) != int(cur.get("chain_index") or 0) + 1:
        return
    dest = last_path.parent.parent / nxt["id"] / "chain_first.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != last_path.resolve():
        shutil.copy2(last_path, dest)
    shots[index + 1] = apply_chain_first_frame(nxt, dest)


def _link_longlive_continuation(shots: list[dict], index: int, last_path: Path) -> None:
    """Same-scene takes continue from prior last frame; scene changes stay hard cuts."""
    if index + 1 >= len(shots) or not last_path.is_file():
        return
    nxt = dict(shots[index + 1])
    source = str(nxt.get("keyframe_source") or "")
    if source == "new_keyframe_hard_cut":
        return
    cur = shots[index]
    same_scene = str(nxt.get("scene_id") or nxt.get("chain_id") or "") == str(
        cur.get("scene_id") or cur.get("chain_id") or ""
    )
    if source == "prior_last_frame" or (same_scene and source != "qc_keyframe"):
        sid = str(nxt.get("id") or nxt.get("take_id") or "next")
        dest = last_path.parent.parent / sid / "chain_first.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() != last_path.resolve():
            shutil.copy2(last_path, dest)
        shots[index + 1] = apply_chain_first_frame(nxt, dest)
        return
    _link_next_chain(shots, index, last_path)


def _ensure_last_frame(root: Path, shot: dict, video: Path) -> Path | None:
    sid = str(shot.get("id") or shot.get("take_id") or "")
    last_path = root / "episodes" / EP / "keyframes" / sid / "last.png"
    if last_path.is_file() and last_path.stat().st_size > 32:
        shot["last_frame_path"] = str(last_path)
        return last_path
    if not video.is_file():
        return None
    try:
        extract_last_frame(video, last_path)
        shot["last_frame_path"] = str(last_path)
        return last_path
    except Exception:
        return None


def _longlive_infer_hook(**kwargs: Any) -> Path:
    """Seam for tests. Production samples with the one loaded NVFP4 pipeline."""
    from gpu_worker.longlive_batch import sample_with_loaded_pipeline

    return sample_with_loaded_pipeline(**kwargs)


def _run_anim_longlive(
    story_id: str,
    root: Path,
    conn,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    shots = _board_shots(root)
    done: list[str] = []
    failed: list[Any] = []
    skipped: list[str] = []
    repaired: list[str] = []
    resolution = {
        "width": LONGLIVE_NATIVE_WIDTH,
        "height": LONGLIVE_NATIVE_HEIGHT,
        "delivery_width": VIDEO_WIDTH,
        "delivery_height": VIDEO_HEIGHT,
        "video_backend": "longlive",
        "scale_pad_only": True,
        "native_audio": False,
    }
    pending: list[dict[str, Any]] = []
    dests: dict[str, Path] = {}
    cpu = CpuPostQueue()
    stop_comfy_for_longlive()

    for index, shot in enumerate(shots):
        sid = str(shot.get("id") or shot.get("take_id") or f"take-{index + 1:02d}")
        shot = dict(shot)
        shot["id"] = sid
        shot["take_id"] = str(shot.get("take_id") or sid)
        shots[index] = shot
        try:
            if progress:
                if index == 0:
                    progress(f"production_started:{sid}")
            prev = shots[index - 1] if index else None
            if int(shot.get("chain_index") or 0) > 0:
                prev_last = prev.get("last_frame_path") if prev else None
                if prev and prev.get("chain_id") == shot.get("chain_id") and prev_last and not is_directory_like(prev_last):
                    shot = apply_chain_first_frame(shot, prev_last)
                    shots[index] = shot
                elif shot.get("keyframe_source") == "prior_last_frame":
                    failed.append({"id": sid, "error": "pending_chain_missing_last_frame"})
                    continue
            selected = select_passing_generation(conn, root, sid)
            if selected is not None:
                existing, _version, _how = selected
                skipped.append(sid)
                rel = existing.relative_to(root).as_posix()
                mark_completed_passing(conn, sid, join_story(story_id, rel))
                last_path = _ensure_last_frame(root, shot, existing)
                if last_path is not None:
                    _link_longlive_continuation(shots, index, last_path)
                    try:
                        put_file(join_story(story_id, f"episodes/{EP}/keyframes/{sid}/last.png"), last_path, "image/png")
                    except Exception:
                        pass
                try:
                    _upload_ok(put_file(join_story(story_id, rel), existing, "video/mp4"))
                except Exception:
                    pass
                continue
            # Do not auto-green a non-passing on-disk mp4 via existing_generation_file.
            orphan = existing_generation_file(root, sid)
            if orphan is not None:
                print(
                    {
                        "resume_skip_orphan": sid,
                        "path": str(orphan),
                        "reason": "no_qc_pass_generation",
                    },
                    flush=True,
                )
            shot = _prepare_longlive_shot(shot, root)
            shots[index] = shot
            first = Path(str(shot.get("first_frame_path") or ""))
            if first.is_file():
                try:
                    put_file(
                        join_story(story_id, f"episodes/{EP}/keyframes/{sid}/f1.png"),
                        first,
                        "image/png",
                    )
                except Exception:
                    pass
            version, rel = next_generation_path(root, sid)
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            take_id = str(shot.get("take_id") or sid)
            shot["_index"] = index
            shot["_rel"] = rel
            shot["_version"] = version
            pending.append(shot)
            dests[take_id] = dest
        except (BudgetExceeded, ProgressStalled, StartupTimeout):
            raise
        except Exception as exc:  # noqa: BLE001
            failed.append({"id": sid, "error": str(exc)[:800]})

    def infer(**kwargs: Any) -> Path:
        take = dict(kwargs["take"])
        dest = Path(kwargs["dest"])
        if progress:
            progress(f"anim:{take.get('take_id') or take.get('id')}")
        produced = _longlive_infer_hook(**kwargs)
        src = Path(produced) if produced else dest
        last_path = _ensure_last_frame(root, take, src)
        idx = take.get("_index")
        if last_path is not None and idx is not None:
            shots[int(idx)]["last_frame_path"] = str(last_path)
            _link_longlive_continuation(shots, int(idx), last_path)
            nxt_idx = int(idx) + 1
            if nxt_idx < len(shots):
                nxt = shots[nxt_idx]
                nxt_tid = str(nxt.get("take_id") or nxt.get("id") or "")
                for item in pending:
                    if str(item.get("take_id") or item.get("id") or "") == nxt_tid:
                        item["first_frame_path"] = nxt.get("first_frame_path")
                        break
        prev_segment: dict[str, Any] | None = None
        if idx is not None and int(idx) > 0:
            prev_segment = dict(shots[int(idx) - 1])
        cpu.submit(
            _longlive_cpu_post,
            story_id,
            root,
            take,
            src,
            last_path,
            prev_segment,
            progress,
        )
        return src

    mapped: dict[str, Any] = {}
    if pending:
        batch = submit_longlive_batch(pending, dests, progress=progress, infer=infer)
        mapped = batch.get("mapped") or {}
        if int(batch.get("model_load_count") or 0) != 1:
            raise RuntimeError(
                f"{FAIL_CLOSED}: expected model_load_count==1, got {batch.get('model_load_count')}"
            )
        if len(mapped) != len(pending):
            raise RuntimeError(
                f"{FAIL_CLOSED}: batch mapped {len(mapped)} of {len(pending)} takes"
            )
        if int(batch.get("h3_downloads") or 0) != 0:
            raise RuntimeError(f"{FAIL_CLOSED}: LongLive path downloaded H3 weights")

    posted: list[str] = []
    try:
        packages = cpu.drain()
        for package in packages:
            if not isinstance(package, dict):
                continue
            posted.append(_longlive_commit_post(conn, story_id, root, package, progress=progress))
    except (BudgetExceeded, ProgressStalled, StartupTimeout):
        raise
    except Exception as exc:  # noqa: BLE001
        failed.append({"id": "cpu_post", "error": str(exc)[:800]})

    done.extend(posted)
    for take in pending:
        sid = str(take.get("id") or take.get("take_id"))
        if sid in posted or sid in skipped:
            continue
        dest = dests.get(str(take.get("take_id") or sid))
        if dest is None or not dest.is_file() or dest.stat().st_size < 32:
            if not any(isinstance(row, dict) and row.get("id") == sid for row in failed):
                failed.append({"id": sid, "error": "longlive produced no video"})

    return {
        "shots_total": len(shots),
        "shots_done": len(done) + len(skipped),
        "generated": done,
        "skipped_existing": skipped,
        "repaired": repaired,
        "failed": failed,
        "resolution": resolution,
        "mapped": {tid: row.get("path") for tid, row in mapped.items()},
        "model_load_count": 1 if pending else 0,
        "native_audio": False,
    }


def _longlive_cpu_post(
    story_id: str,
    root: Path,
    take: dict[str, Any],
    dest: Path,
    last_path: Path | None,
    prev_segment: dict[str, Any] | None,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    """Background file/normalize work only. Main thread commits sqlite after drain."""
    sid = str(take.get("id") or take.get("take_id") or "")
    rel = str(take.get("_rel") or dest.relative_to(root).as_posix())
    version = take.get("_version") or 1
    _keep_native_strip_audio(dest, progress=progress)
    meta = _probe_video(dest)
    from gpu_worker.longlive import select_longlive_mode as _ll_mode

    used_mode = _ll_mode(take)
    upload = put_file(join_story(story_id, rel), dest, "video/mp4")
    if not _upload_ok(upload):
        raise RuntimeError(f"shot R2 upload failed: {sid}")
    if last_path is not None and last_path.is_file() and last_path.stat().st_size >= 32:
        last_upload = put_file(
            join_story(story_id, f"episodes/{EP}/keyframes/{sid}/last.png"),
            last_path,
            "image/png",
        )
        if not _upload_ok(last_upload):
            raise RuntimeError(f"shot last.png R2 upload failed: {sid}")
    if progress:
        progress(f"shot_uploaded:{sid}")
    return {
        "sid": sid,
        "rel": rel,
        "version": version,
        "meta": meta,
        "used_mode": used_mode,
        "take": take,
        "prev_segment": prev_segment,
        "dest": str(dest),
        "story_id": story_id,
    }


def _longlive_commit_post(
    conn,
    story_id: str,
    root: Path,
    package: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> str:
    sid = str(package.get("sid") or "")
    take = dict(package.get("take") or {})
    verdict = incremental_qc_segment(
        conn,
        EP,
        sid,
        dict(package.get("meta") or {}),
        float(take.get("duration") or 8.0),
        segment=take,
        prev_segment=package.get("prev_segment"),
        used_mode=str(package.get("used_mode") or ""),
        video_path=package.get("dest"),
        story_root=root,
    )
    try:
        record_generation_result(
            conn,
            sid,
            int(package.get("version") or 1),
            join_story(story_id, str(package.get("rel") or "")),
            take.get("seed"),
            package.get("used_mode"),
            "completed" if verdict == "pass" else "failed",
            verdict,
        )
    except Exception:
        pass
    if verdict != "pass":
        raise RuntimeError(f"{FAIL_CLOSED}: longlive qc {sid} verdict={verdict}")
    mark_completed_passing(conn, sid, join_story(story_id, str(package.get("rel") or "")))
    _checkpoint_story(conn, story_id, root, progress=progress)
    return sid


def _run_anim_h3(
    story_id: str,
    root: Path,
    conn,
    router: ComfyRouter | None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    shots = _board_shots(root)
    done: list[str] = []
    failed: list[Any] = []
    skipped: list[str] = []
    repaired: list[str] = []
    resolution = {
        **h3_resolution_profile(shots[0] if shots else {}),
        "delivery_width": VIDEO_WIDTH,
        "delivery_height": VIDEO_HEIGHT,
        "video_backend": "h3",
    }
    mode_groups = plan_h3_mode_groups(shots)
    mode_fallback_latched: dict[str, bool] = {}
    shot_metrics: list[dict[str, Any]] = []
    session_tracker = H3SessionTracker()
    prep_pool = H3PrepPool(lambda shot: _stage_first_frame(shot, root))
    cpu = CpuPostQueue()
    try:
        for index, shot in enumerate(shots):
            sid = str(shot.get("id") or "")
            try:
                if progress:
                    if index == 0:
                        progress(f"production_started:{sid}")
                    progress(f"anim:{sid}")
                prev = shots[index - 1] if index else None
                if int(shot.get("chain_index") or 0) > 0:
                    prev_last = prev.get("last_frame_path") if prev else None
                    if (
                        prev
                        and prev.get("chain_id") == shot.get("chain_id")
                        and prev_last
                        and not is_directory_like(prev_last)
                    ):
                        shot = apply_chain_first_frame(shot, prev_last)
                        shots[index] = shot
                    else:
                        failed.append({"id": sid, "error": "pending_chain_missing_last_frame"})
                        continue
                selected = select_passing_generation(conn, root, sid)
                if selected is not None:
                    existing, _version, _how = selected
                    skipped.append(sid)
                    rel = existing.relative_to(root).as_posix()
                    mark_completed_passing(conn, sid, join_story(story_id, rel))
                    last_path = _ensure_last_frame(root, shot, existing)
                    if last_path is not None:
                        _link_next_chain(shots, index, last_path)
                        try:
                            put_file(
                                join_story(story_id, f"episodes/{EP}/keyframes/{sid}/last.png"),
                                last_path,
                                "image/png",
                            )
                        except Exception:
                            pass
                    try:
                        _upload_ok(put_file(join_story(story_id, rel), existing, "video/mp4"))
                    except Exception:
                        pass
                    next_shot = shots[index + 1] if index + 1 < len(shots) else None
                    if next_shot is not None and can_prefetch_staging(next_shot):
                        prep_pool.schedule(next_shot)
                    continue
                orphan = existing_generation_file(root, sid)
                if orphan is not None:
                    print(
                        {
                            "resume_skip_orphan": sid,
                            "path": str(orphan),
                            "reason": "no_qc_pass_generation",
                        },
                        flush=True,
                    )
                if router is None:
                    failed.append({"id": sid, "error": "no_comfy_router"})
                    continue
                shot = prep_pool.prime(shot)
                shots[index] = shot
                next_shot = shots[index + 1] if index + 1 < len(shots) else None
                if next_shot is not None and can_prefetch_staging(next_shot):
                    prep_pool.schedule(next_shot)
                version, rel = next_generation_path(root, sid)
                dest = root / rel
                used_mode = select_mode(shot)
                session_tracker.note_mode(used_mode)
                use_fallback = bool(mode_fallback_latched.get(used_mode) or shot.get("h3_oom_fallback"))
                if use_fallback:
                    shot = dict(shot, h3_oom_fallback=True)
                    shots[index] = shot
                profile = h3_resolution_profile(shot, oom_fallback=use_fallback)
                print(
                    {
                        "h3_submit": sid,
                        "mode": used_mode,
                        "backend": "h3",
                        "chain_index": shot.get("chain_index", 0),
                        "gen_width": profile.get("gen_width"),
                        "gen_height": profile.get("gen_height"),
                        "downgraded": profile.get("downgraded"),
                    },
                    flush=True,
                )
                submit_error: BaseException | None = None
                oom_retried = False
                sample_s = 0.0
                for submit_attempt in range(2):
                    try:
                        sample_t0 = time.monotonic()
                        _submit_h3_gpu(router, shot, root, dest, progress)
                        sample_s = time.monotonic() - sample_t0
                        submit_error = None
                        break
                    except (BudgetExceeded, ProgressStalled, StartupTimeout):
                        raise
                    except Exception as exc:  # noqa: BLE001 — OOM downgrade or fail closed
                        submit_error = exc
                        if recycle_failure_class(exc):
                            break
                        if is_h3_oom(exc) and not oom_retried:
                            router.free()
                            oom_retried = True
                            mode_fallback_latched[used_mode] = True
                            shot = dict(
                                shot,
                                h3_oom_fallback=True,
                                h3_downgrade_reason="oom_primary",
                            )
                            shots[index] = shot
                            print(
                                {
                                    "h3_oom_downgrade": sid,
                                    "mode": used_mode,
                                    "attempt": submit_attempt + 1,
                                    "backend": "h3",
                                },
                                flush=True,
                            )
                            continue
                        if is_h3_oom(exc):
                            raise RuntimeError(
                                f"h3_fail_closed: oom_downshift_exhausted gen_480p fallback_tier 864x480 shot={sid}"
                            ) from exc
                        break
                if submit_error is not None:
                    classified = recycle_failure_class(submit_error)
                    if classified:
                        raise RuntimeError(f"{classified}:{submit_error}") from submit_error
                    raise submit_error
                final_profile = h3_resolution_profile(
                    shot,
                    oom_fallback=bool(shot.get("h3_oom_fallback")),
                )
                shot_metrics.append(
                    {
                        "id": sid,
                        "mode": used_mode,
                        "refs": list(shot.get("refs") or []),
                        "gen_width": final_profile.get("gen_width"),
                        "gen_height": final_profile.get("gen_height"),
                        "downgraded": final_profile.get("downgraded"),
                        "downgrade_reason": final_profile.get("reason"),
                        "sample_s": round(sample_s, 3),
                        "peak_vram_mb": gpu_vram_mb(),
                    }
                )
                last_path = Path(str(shot.get("last_frame_path") or ""))
                if not last_path.is_file():
                    last_path = _ensure_last_frame(root, shot, dest)
                if last_path is not None:
                    _link_next_chain(shots, index, last_path)
                    _db_execute(
                        conn,
                        "UPDATE segments SET last_frame_path = ? WHERE id = ?",
                        (str(last_path), sid),
                    )
                attempt = 0
                verdict = "pass"
                last_path: Path | None = None
                while True:
                    verdict, _, last_path = _h3_normalize_and_qc(
                        root,
                        conn,
                        shot,
                        dest,
                        prev,
                        version=version,
                        rel=rel,
                        story_id=story_id,
                        progress=progress,
                    )
                    if verdict != "retry" or attempt >= H3_MAX_RETRIES or router is None:
                        break
                    attempt += 1
                    c_verdict, c_details = continuity_qc(prev, shot, used_mode)
                    issues = list((c_details.get("issues") or [])) or ["visual_retry"]
                    try:
                        from anime_factory.video_visual_qc import map_reasons_to_strategy

                        report = dest.parent / "visual_qc.json"
                        if report.is_file():
                            payload = json.loads(report.read_text(encoding="utf-8"))
                            reasons = list(payload.get("reasons") or [])
                            if reasons:
                                issues = list(dict.fromkeys([*issues, *reasons]))
                            strategy_hint = payload.get("retry_strategy") or map_reasons_to_strategy(reasons)
                            if strategy_hint and strategy_hint not in issues:
                                issues.append(strategy_hint)
                    except Exception:  # noqa: BLE001
                        pass
                    repair = plan_repair(shot, issues, attempt)
                    open_repair_task(conn, sid, issues, attempt, shot.get("h3_prompt"), repair)
                    shot = dict(
                        shot,
                        seed=repair["seed"],
                        h3_prompt=repair["h3_prompt"],
                        h3_mode=repair.get("h3_mode") or shot.get("h3_mode"),
                    )
                    shots[index] = shot
                    version, rel = next_generation_path(root, sid)
                    dest = root / rel
                    used_mode = select_mode(shot)
                    session_tracker.note_mode(used_mode)
                    print(
                        {
                            "h3_repair": sid,
                            "attempt": attempt,
                            "strategy": repair["strategy"],
                            "backend": "h3",
                        },
                        flush=True,
                    )
                    _submit_h3_gpu(router, shot, root, dest, progress)
                    last_path = _ensure_last_frame(root, shot, dest)
                    if last_path is not None:
                        _link_next_chain(shots, index, last_path)
                    repaired.append(sid)
                if verdict == "pass":
                    cpu.submit(
                        _h3_upload_files,
                        story_id,
                        root,
                        sid,
                        dest,
                        rel,
                        last_path,
                        progress,
                    )
                else:
                    failed.append({"id": sid, "verdict": verdict})
                if next_shot is not None and int(next_shot.get("chain_index") or 0) > 0:
                    prep_pool.schedule(dict(next_shot))
            except (BudgetExceeded, ProgressStalled, StartupTimeout):
                try:
                    flush_gpu_artifacts(story_id, root, conn, progress=progress)
                except Exception:
                    pass
                raise
            except Exception as exc:  # noqa: BLE001 — do not silent-truncate the episode
                if str(exc).startswith("h3_fail_closed:"):
                    raise
                classified = recycle_failure_class(exc)
                print(
                    {"h3_fail": sid, "error": str(exc)[:800], "failure_class": classified},
                    flush=True,
                )
                if classified:
                    raise
                failed.append({"id": sid, "error": str(exc)[:800]})
        posted_uploads = cpu.drain()
        for item in posted_uploads:
            if isinstance(item, dict) and item.get("sid"):
                done.append(_h3_commit_passing(conn, story_id, root, item, progress=progress))
            elif item:
                done.append(str(item))
    finally:
        try:
            cpu.close()
        except Exception:
            pass
        prep_pool.close()
    expected = model_load_count_for_modes(modes_for_shots(shots))
    actual = session_tracker.model_load_count
    if expected and actual > expected:
        raise RuntimeError(
            f"{FAIL_CLOSED}: expected model_load_count<={expected}, got {actual} "
            f"({session_tracker.loaded_unets})"
        )
    instrument = instrument_h3_session(shots, warmed=actual > 0)
    instrument["model_load_count"] = actual
    instrument["shot_metrics"] = shot_metrics
    print(json.dumps({"h3_session": instrument}, ensure_ascii=False), flush=True)
    return {
        "shots_total": len(shots),
        "shots_done": len(done) + len(skipped),
        "generated": done,
        "skipped_existing": skipped,
        "repaired": repaired,
        "failed": failed,
        "resolution": resolution,
        "model_load_count": actual,
        "h3_session": instrument,
        "mode_groups": mode_groups,
        "shot_metrics": shot_metrics,
        "gpu_done": not failed and (len(done) + len(skipped)) > 0,
    }


def run_anim(
    story_id: str,
    root: Path,
    conn,
    router: ComfyRouter | None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    shots = _board_shots(root)
    try:
        backend = lock_video_backend(root=root)
    except VideoBackendLockError as exc:
        raise RuntimeError(str(exc)) from exc
    if backend == "longlive":
        return _run_anim_longlive(story_id, root, conn, progress)
    return _run_anim_h3(story_id, root, conn, router, progress=progress)


def _run_checked(
    command: list[str],
    progress: ProgressCallback | None = None,
    stage: str = "compose",
) -> None:
    proc = subprocess.Popen(command)
    try:
        while True:
            try:
                code = proc.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                if progress:
                    progress(stage)
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def _ensure_compose_subtitles(
    plan: ComposePlan,
    shots: list[dict[str, Any]],
    langs: tuple[str, ...],
) -> None:
    """Backfill current-main ComposePlan paths without translating another language."""
    planned = {Path(path).name: Path(path) for path in plan.srt_outputs}
    for lang in langs:
        name = f"{EP}.{lang}.srt"
        path = planned.get(name)
        if path is None or path.is_file():
            continue
        cues: list[tuple[float, float, str]] = []
        cursor = 0.0
        for shot in shots:
            duration = float(shot.get("duration") or 8.0)
            line = shot.get("line") or shot.get("dialogue") or shot.get("text")
            text = (
                str(line.get(lang) or "").strip()
                if isinstance(line, dict)
                else str(line or "").strip()
            )
            if text:
                cues.append((cursor, cursor + duration, text))
            cursor += duration
        if not cues:
            raise RuntimeError(f"subtitle dialogue missing for language: {lang}")
        write_srt(path, cues)


def compose_gate(anim: dict[str, Any] | None) -> tuple[int, bool]:
    """QC/compose must not go green on 0 mp4s. remaining>0 keeps the lease busy."""
    payload = anim if isinstance(anim, dict) else {}
    total = int(payload.get("shots_total") or 0)
    done = int(payload.get("shots_done") or 0)
    failed = bool(payload.get("failed"))
    remaining = max(0, total - done)
    allow = remaining == 0 and done > 0 and not failed
    if not allow and remaining == 0:
        remaining = max(1, total)
    return remaining, allow


def run_compose(
    story_id: str,
    root: Path,
    conn,
    langs: tuple[str, ...] = ("zh", "en", "ja"),
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    pairs = collect_shot_paths(conn, EP)
    local: list[Path] = []
    local_by_id: dict[str, Path] = {}
    for sid, p in pairs:
        rel = p.split(f"stories/{story_id}/", 1)[-1] if "stories/" in p else p
        candidate = root / rel
        if not candidate.is_file():
            continue
        local.append(candidate)
        if sid:
            local_by_id[str(sid)] = candidate
    if not local:
        raise MissingShotError("compose blocked; 0 shot mp4s on disk")
    work = root / "episodes" / EP
    work.mkdir(parents=True, exist_ok=True)
    shots = _board_shots(root)
    # Pair board rows to picture by segment id. zip(shots, local) shifted every
    # clip after a gap against its dialogue when COMPOSE_ALLOW_MISSING_SHOTS
    # silently dropped a shot.
    if shots and local_by_id:
        shots = [s for s in shots if str(s.get("id") or "") in local_by_id]
        if not shots:
            raise MissingShotError(
                f"compose blocked; none of the {len(local)} shot mp4s match a board segment id"
            )
        local = [local_by_id[str(s.get("id"))] for s in shots]
    clip_durations: dict[str, float] = {}
    audio_metas: dict[str, dict] = {}
    probed_durations: dict[str, float] = {}
    for shot, video in zip(shots, local):
        sid = str(shot.get("id") or "")
        probed = _probe_video(video)
        if sid:
            clip_durations[sid] = float(probed.get("duration") or shot.get("duration") or 8.0)
            probed_durations[sid] = clip_durations[sid]
            shot["duration"] = clip_durations[sid]
        audio_metas[str(video)] = _probe_audio(video)
    assert_no_h3_dialogue(local, audio_metas)
    clip_durations = fit_clip_durations_to_speech(work, shots, langs, clip_durations)
    padded: list[Path] = []
    pad_dir = work / "compose_pad"
    for shot, video in zip(shots, local):
        sid = str(shot.get("id") or "")
        target = float(clip_durations.get(sid) or shot.get("duration") or 8.0)
        probed_s = float(probed_durations.get(sid) or target)
        shot["duration"] = target
        if sid:
            clip_durations[sid] = target
        pad_s = target - probed_s
        if sid and pad_s >= 0.05:
            pad_dir.mkdir(parents=True, exist_ok=True)
            dest = pad_dir / f"{sid}.mp4"
            if progress:
                progress(f"compose_pad:{sid}")
            _run_checked(extend_clip_cmd(video, dest, pad_s), progress=progress, stage="compose")
            padded.append(dest)
        else:
            padded.append(video)
    if shots:
        local = padded
        overlap = chain_overlap_frames(shots)
        trim_dir = work / "compose_trim"
        trimmed: list[Path] = []
        for shot, video, drop in zip(shots, local, overlap):
            sid = str(shot.get("id") or "")
            if drop <= 0:
                trimmed.append(video)
                continue
            trim_dir.mkdir(parents=True, exist_ok=True)
            dest = trim_dir / f"{sid or video.stem}.mp4"
            if progress:
                progress(f"compose_trim:{sid}")
            _run_checked(drop_first_frames_cmd(video, dest, drop), progress=progress, stage="compose")
            trimmed.append(dest)
            # Dropping the duplicated join frame shortens the picture. Decrement
            # the timeline as well or every later line drifts 41.7ms per chain
            # join — visibly out of sync by the end of a 600s episode.
            dropped_s = drop / float(VIDEO_FPS)
            shot["duration"] = max(0.0, float(shot.get("duration") or 8.0) - dropped_s)
            if sid:
                clip_durations[sid] = shot["duration"]
        local = trimmed
    # Backend-aware MOSS handoff: LongLive clears ownership + vacates leftovers;
    # H3 stops Comfy without setting longlive_owns_gpu so Comfy can restore.
    try:
        video_backend = select_video_backend(root=root)
    except Exception:  # noqa: BLE001 — compose can still proceed with H3 default
        video_backend = "h3"
    moss_handoff_done = {"done": False}

    def _before_moss_sfx() -> None:
        release_gpu_for_moss_sfx(backend=video_backend)
        moss_handoff_done["done"] = True

    sfx_prepared = prepare_episode_sfx(
        shots,
        root,
        EP,
        story_id=story_id,
        conn=conn,
        clip_durations=clip_durations or None,
        before_moss=_before_moss_sfx,
    )
    if moss_handoff_done["done"]:
        restore_gpu_after_moss_sfx(backend=video_backend)
    mixed, timeline = compose_episode_audio(
        work,
        EP,
        shots,
        langs=langs,
        require_speech=True,
        clip_durations=clip_durations or None,
        video_paths=local,
        audio_metas=audio_metas,
        sfx_clips=sfx_prepared["sfx_clips"],
        ambience_clips=sfx_prepared["ambience_clips"],
        sfx_cues=sfx_prepared["sfx_cues"],
    )
    sfx_results = sfx_prepared.get("results") or {}
    resolved_sources = [
        {
            "cue_key": key,
            "source": (result.provenance.source if result.provenance else None),
            "cached": bool(result.provenance.cached) if result.provenance else False,
        }
        for key, result in sfx_results.items()
        if result.status == "resolved"
    ]
    sfx_evidence = sfx_prepared.get("evidence") or {}
    sfx_credits = sfx_prepared.get("credits") or sfx_evidence.get("credits") or []
    timeline_path = work / "audio" / "timeline.json"
    evidence_path = work / "audio" / "sfx_evidence.json"
    credits_path = work / "audio" / "sfx_credits.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_payload = {
        **sfx_evidence,
        "credits": sfx_credits,
        "resolved_cues": resolved_sources,
        "missing": sfx_prepared.get("missing") or sfx_evidence.get("missing") or [],
    }
    evidence_path.write_text(json.dumps(evidence_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    credits_path.write_text(json.dumps(sfx_credits, ensure_ascii=False, indent=2), encoding="utf-8")
    # Attach credits onto the timeline manifest when present so R2 evidence is one place.
    if timeline_path.is_file():
        try:
            timeline_payload = json.loads(timeline_path.read_text(encoding="utf-8"))
            if isinstance(timeline_payload, dict):
                timeline_payload["sfx_credits"] = sfx_credits
                timeline_payload["sfx_evidence"] = {
                    "coverage": evidence_payload.get("coverage"),
                    "cue_total": evidence_payload.get("cue_total"),
                    "cue_resolved": evidence_payload.get("cue_resolved"),
                    "cue_missing": evidence_payload.get("cue_missing"),
                    "missing": evidence_payload.get("missing"),
                    "sources": evidence_payload.get("sources"),
                }
                timeline_path.write_text(
                    json.dumps(timeline_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        except (OSError, ValueError):
            pass
    concat = work / "concat.txt"
    concat.write_text("".join(f"file '{p}'\n" for p in local), encoding="utf-8")
    plan: ComposePlan = build_compose_plan(work, EP, local, langs=langs, require_audio=True)
    _ensure_compose_subtitles(plan, shots, langs)
    (work / "final").mkdir(parents=True, exist_ok=True)
    _run_checked(plan.encode, progress=progress)
    for mux in plan.muxes:
        _run_checked(mux, progress=progress)
    uploads = []
    final_keys = []
    subtitle_keys = []
    evidence_keys = []
    failed_uploads = []
    for lang in langs:
        for suffix, content_type in (
            ("mp4", "video/mp4"),
            ("srt", "application/x-subrip"),
        ):
            artifact = work / "final" / f"{EP}.{lang}.{suffix}"
            key = join_story(
                story_id,
                f"episodes/{EP}/final/{EP}.{lang}.{suffix}",
            )
            if not artifact.is_file():
                failed_uploads.append(f"missing:{key}")
                continue
            uploaded = put_file(key, artifact, content_type)
            uploads.append(uploaded)
            if isinstance(uploaded, dict) and uploaded.get("ok"):
                if suffix == "mp4":
                    final_keys.append(key)
                else:
                    subtitle_keys.append(key)
                if progress:
                    progress(f"r2_uploaded:{key}")
            else:
                failed_uploads.append(key)
    for local_name, rel_suffix, content_type in (
        ("timeline.json", f"episodes/{EP}/audio/timeline.json", "application/json"),
        ("sfx_evidence.json", f"episodes/{EP}/audio/sfx_evidence.json", "application/json"),
        ("sfx_credits.json", f"episodes/{EP}/audio/sfx_credits.json", "application/json"),
    ):
        artifact = work / "audio" / local_name
        if not artifact.is_file():
            continue
        key = join_story(story_id, rel_suffix)
        uploaded = put_file(key, artifact, content_type)
        uploads.append(uploaded)
        if isinstance(uploaded, dict) and uploaded.get("ok"):
            evidence_keys.append(key)
            if progress:
                progress(f"r2_uploaded:{key}")
        else:
            failed_uploads.append(key)
    if failed_uploads:
        raise RuntimeError(f"compose R2 upload failed: {failed_uploads}")
    return {
        "uploaded": uploads,
        "final_keys": final_keys,
        "subtitle_keys": subtitle_keys,
        "evidence_keys": evidence_keys,
        "n_shots": len(local),
        "mixed_wavs": mixed,
        "sfx": {
            "timeline_path": timeline_path.relative_to(work).as_posix(),
            "timeline_exists": timeline_path.is_file(),
            "sfx_event_count": len(timeline.sfx),
            "sfx_clip_count": len(sfx_prepared.get("sfx_clips") or []),
            "ambience_clip_count": len(sfx_prepared.get("ambience_clips") or []),
            "resolved_cues": resolved_sources,
            "credits": sfx_credits,
            "coverage": evidence_payload.get("coverage"),
            "missing": evidence_payload.get("missing") or [],
            "evidence_path": evidence_path.relative_to(work).as_posix(),
        },
    }


def render_episode_shorts(
    story_id: str,
    root: Path,
    langs: tuple[str, ...],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Render A-line shorts when available; an unmerged library must not block the episode."""
    try:
        from anime_factory.shorts import ShortPlan, plan_shorts, render_short
    except ImportError as exc:
        return {
            "available": False,
            "keys": [],
            "uploads": [],
            "errors": [f"shorts_library_unavailable:{exc}"],
        }
    _ = ShortPlan
    board_path = root / "episodes" / EP / "board.json"
    if not board_path.is_file():
        return {
            "available": True,
            "keys": [],
            "uploads": [],
            "errors": [f"board_missing:{board_path}"],
        }
    board = json.loads(board_path.read_text(encoding="utf-8"))
    ep_root = root / "episodes" / EP
    plans = list(plan_shorts(board, count=3, min_s=20.0, max_s=45.0))
    keys: list[str] = []
    uploads: list[dict[str, Any]] = []
    errors: list[str] = []
    for lang in langs:
        for plan in plans:
            if progress:
                progress(f"shorts:{lang}")
            try:
                rendered = Path(render_short(plan, ep_root, lang, youtube_url=None))
                path = rendered if rendered.is_absolute() else ep_root / rendered
                if not path.is_file():
                    raise FileNotFoundError(path)
                rel = path.relative_to(ep_root).as_posix()
                key = join_story(story_id, f"episodes/{EP}/{rel}")
                uploaded = put_file(key, path, "video/mp4")
                uploads.append(uploaded)
                if isinstance(uploaded, dict) and uploaded.get("ok"):
                    keys.append(key)
                    if progress:
                        progress(f"r2_uploaded:{key}")
                else:
                    errors.append(f"upload_failed:{key}")
                thumb = path.with_suffix(".jpg")
                if thumb.is_file():
                    thumb_rel = thumb.relative_to(ep_root).as_posix()
                    thumb_key = join_story(story_id, f"episodes/{EP}/{thumb_rel}")
                    thumb_upload = put_file(thumb_key, thumb, "image/jpeg")
                    uploads.append(thumb_upload)
                    if isinstance(thumb_upload, dict) and thumb_upload.get("ok"):
                        if progress:
                            progress(f"r2_uploaded:{thumb_key}")
                    else:
                        errors.append(f"upload_failed:{thumb_key}")
            except (BudgetExceeded, ProgressStalled, StartupTimeout):
                raise
            except Exception as exc:  # noqa: BLE001 — shorts are additive, preserve the final
                errors.append(f"{lang}:{type(exc).__name__}:{exc}")
    return {
        "available": True,
        "keys": keys,
        "uploads": uploads,
        "errors": errors,
    }


def ensure_episode_rows(story_id: str, root: Path, conn) -> None:
    """Insert a newly batched episode without resetting completed resume rows."""
    now = utcnow()
    conn.execute(
        """
        INSERT OR IGNORE INTO story (id, title, world_mode, langs, created_at, updated_at)
        VALUES (?, ?, 'fiction', '["zh","en","ja"]', ?, ?)
        """,
        (story_id, story_id, now, now),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO episodes (episode_code, title, status, created_at, updated_at)
        VALUES (?, ?, 'running', ?, ?)
        """,
        (EP, EP, now, now),
    )
    scene_id = f"{EP}-sc01"
    conn.execute(
        "INSERT OR IGNORE INTO scenes (id, episode_code, seq) VALUES (?, ?, 1)",
        (scene_id, EP),
    )
    existing = conn.execute(
        "SELECT COUNT(*) AS n FROM segments WHERE episode_code = ?",
        (EP,),
    ).fetchone()
    if int(existing["n"] if existing else 0) > 0:
        conn.commit()
        return
    shots = []
    for shot in _board_shots(root):
        row = dict(shot)
        row.setdefault("scene_id", scene_id)
        row.setdefault("duration", 8.0)
        row.setdefault("h3_mode", "fl2va_first")
        shots.append(row)
    if shots:
        persist_board(conn, EP, shots, video_backend=select_video_backend(root=root))
    conn.commit()


def ensure_story_db(story_id: str, root: Path) -> Any:
    """Rebuild story.sqlite from board.json when the hosted path did not upload it."""
    db_path = root / "story.sqlite"
    conn = open_db(db_path)
    migrate(conn)
    ensure_episode_rows(story_id, root, conn)
    return conn


def _shot_has_spoken_line(shot: dict) -> bool:
    line = shot.get("line") if isinstance(shot, dict) else None
    if not isinstance(line, dict):
        return False
    return any(str(v or "").strip() for v in line.values())


def _shot_has_video_prompt(shot: dict) -> bool:
    if not isinstance(shot, dict):
        return False
    return any(
        str(shot.get(key) or "").strip()
        for key in ("h3_mode", "h3_prompt", "prompt", "first_frame_prompt")
    )


def _tts_manifest_complete(audio: Path) -> bool:
    manifest = audio / "tts_manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — unreadable manifest is not complete
            data = {}
        if isinstance(data, dict) and data.get("deferred") is True:
            return False
        return True
    return any(audio.rglob("*.wav")) or any(audio.rglob("*.mp3"))


def pre_gpu_artifacts_ready(root: Path, episode_code: str) -> bool:
    """Skip hosted bible when the board is already GPU-runnable.

    Spoken boards still need real TTS (not a deferred placeholder). Silent H3
    canary boards ship shots with prompts and n_spoken=0 — those must not call
    the director LLM on an isolated box.
    """
    ep = Path(root) / "episodes" / episode_code
    board = ep / "board.json"
    if not board.is_file():
        return False
    try:
        data = json.loads(board.read_text(encoding="utf-8"))
        shots = list(data.get("shots") or [])
        if not shots:
            return False
        spoken = [shot for shot in shots if _shot_has_spoken_line(shot)]
        if spoken:
            return _tts_manifest_complete(ep / "audio")
        return all(_shot_has_video_prompt(shot) for shot in shots)
    except Exception:  # noqa: BLE001 — missing/invalid board is not ready
        return False


def run_pre_gpu_if_needed(
    story_id: str,
    root: Path,
    episode_code: str,
    langs: tuple[str, ...] | list[str] | None = None,
    control: ControlPlane | None = None,
    title: str | None = None,
    logline: str | None = None,
    *,
    skip_pull: bool = False,
) -> dict[str, Any]:
    """Hosted bible→board (CosyVoice TTS) on this box when the laptop never ran produce.

    design/keyframe stills wait for Flux weights + Comfy, not for H3. Callers run
    generate_missing_stills after this, then join_h3_weights before anim.
    """
    if not skip_pull:
        pull_story(story_id, root, episode_code=episode_code)
    if str(os.environ.get("AF_SKIP_PRE_GPU") or "").strip().lower() in {"1", "true", "yes"}:
        return {"skipped": True, "reason": "af_skip_pre_gpu"}
    if pre_gpu_artifacts_ready(root, episode_code):
        return {"skipped": True, "reason": "pre_gpu_ready"}
    return produce_episode(
        story_id,
        root,
        control=control,
        live=True,
        upload=True,
        lease_gpu=False,
        episode_code=episode_code,
        title=title,
        logline=logline,
        langs=langs,
        until_stage="board",
    )


def _existing_compose_from_disk(
    story_id: str,
    root: Path,
    episode_code: str,
    langs: tuple[str, ...],
) -> dict[str, Any]:
    ep = _normalize_episode_code(episode_code)
    final_keys: list[str] = []
    subtitle_keys: list[str] = []
    final = Path(root) / "episodes" / ep / "final"
    for lang in langs:
        mp4 = final / f"{ep}.{lang}.mp4"
        if mp4.is_file() and mp4.stat().st_size > 0:
            final_keys.append(join_story(story_id, f"episodes/{ep}/final/{ep}.{lang}.mp4"))
        srt = final / f"{ep}.{lang}.srt"
        if srt.is_file() and srt.stat().st_size > 0:
            subtitle_keys.append(join_story(story_id, f"episodes/{ep}/final/{ep}.{lang}.srt"))
    return {
        "final_keys": final_keys,
        "subtitle_keys": subtitle_keys,
        "skipped_existing": True,
    }


def _idle_episode_result(
    story_id: str,
    episode_code: str,
    langs: tuple[str, ...],
    *,
    reason: str,
    pulled: int = 0,
    compose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "story_id": story_id,
        "episode_code": episode_code,
        "langs": list(langs),
        "pulled": pulled,
        "pre_gpu": {"skipped": True, "reason": reason},
        "stills": {"skipped": True, "reason": reason},
        "anim": {
            "shots_total": 0,
            "shots_done": 0,
            "generated": [],
            "skipped_existing": [],
            "failed": [],
            "skipped_cycle": True,
        },
        "compose": compose
        or {"skipped_existing": True, "reason": reason, "final_keys": []},
        "remaining": 0,
        "truncated": False,
        "skipped_cycle": True,
        "skip_reason": reason,
    }


def run_gpu_episode(
    story_id: str,
    root: Path | None = None,
    control: ControlPlane | None = None,
    episode_code: str | None = None,
    langs: tuple[str, ...] | list[str] | None = None,
    progress: ProgressCallback | None = None,
    finish_story: bool = True,
    skip_shorts: bool = False,
    story_status: str | None = None,
    until_stage: str | None = None,
    episode_status: str | None = None,
) -> dict[str, Any]:
    global EP
    ep = _normalize_episode_code(episode_code)
    EP = ep
    episode_langs = tuple(
        str(lang).strip().lower()
        for lang in (langs or ("zh", "en", "ja"))
        if str(lang).strip()
    ) or ("zh", "en", "ja")
    root = Path(root or os.environ.get("AF_STORY_ROOT") or f"/work/{story_id}")
    hint = {
        "story_id": story_id,
        "episode_code": EP,
        "status": story_status,
        "story_status": story_status,
        "episode_status": episode_status,
        "until_stage": until_stage,
    }
    idle = gpu_cycle_idle_reason(hint, root=root, episode_code=EP, langs=episode_langs)
    if idle:
        compose = (
            _existing_compose_from_disk(story_id, root, EP, episode_langs)
            if episode_finals_ready(root, EP, episode_langs)
            else None
        )
        return _idle_episode_result(
            story_id,
            EP,
            episode_langs,
            reason=idle,
            compose=compose,
        )
    if progress:
        progress("pull")
    pulled = pull_story(story_id, root, episode_code=EP)
    if episode_finals_ready(root, EP, episode_langs):
        return _idle_episode_result(
            story_id,
            EP,
            episode_langs,
            reason="remaining=0",
            pulled=len(pulled),
            compose=_existing_compose_from_disk(story_id, root, EP, episode_langs),
        )
    pre_gpu = run_pre_gpu_if_needed(
        story_id,
        root,
        EP,
        langs=episode_langs,
        control=control,
        skip_pull=True,
    )
    if pre_gpu.get("blocked"):
        return {
            "story_id": story_id,
            "episode_code": EP,
            "langs": list(episode_langs),
            "pulled": len(pulled),
            "pre_gpu": pre_gpu,
            "stills": None,
            "anim": {"failed": [{"error": "pre_gpu_blocked"}], "shots_total": 0, "shots_done": 0},
            "compose": None,
            "remaining": 1,
            "truncated": False,
        }
    db_path = root / "story.sqlite"
    if db_path.is_file():
        conn = open_db(db_path)
        migrate(conn)
        ensure_episode_rows(story_id, root, conn)
    else:
        conn = ensure_story_db(story_id, root)
    router = None
    try:
        router = ComfyRouter()
    except Exception:  # noqa: BLE001
        router = None
    if progress:
        progress("design")
    if control:
        control.job(story_id, "design", "running", episode_code=EP)
    try:
        stills = generate_missing_stills(
            story_id,
            root,
            conn,
            router,
            progress=progress,
        )
    except Exception as exc:  # noqa: BLE001 — do not stamp keyframe succeeded with 0 files
        err = str(exc)[:500]
        design_fail = (
            "produced no character" in err
            or "no character rows" in err
            or "empty library" in err
        )
        if control:
            if design_fail:
                control.job(story_id, "design", "blocked", error=err, episode_code=EP)
            else:
                control.job(story_id, "design", "succeeded", episode_code=EP)
                control.job(story_id, "keyframe", "blocked", error=err, episode_code=EP)
        try:
            _checkpoint_story(conn, story_id, root, progress=progress, upload=True)
        except Exception:
            pass
        return {
            "story_id": story_id,
            "episode_code": EP,
            "langs": list(episode_langs),
            "pulled": len(pulled),
            "pre_gpu": pre_gpu,
            "stills": {"ok": False, "error": str(exc)[:500]},
            "anim": {"failed": [{"error": f"keyframe:{exc}"}], "shots_total": 0, "shots_done": 0},
            "compose": None,
            "remaining": 1,
            "truncated": False,
        }
    if control:
        control.job(story_id, "design", "succeeded", episode_code=EP)
        control.job(story_id, "keyframe", "succeeded", episode_code=EP)
        control.job(story_id, "anim", "running", episode_code=EP)
    try:
        backend = lock_video_backend(root=root)
    except VideoBackendLockError as exc:
        raise RuntimeError(str(exc)) from exc
    if backend == "longlive":
        unload_still_models(router)
        stop_comfy_for_longlive()
        if progress:
            progress("weights:longlive")
        try:
            ensure_longlive(progress=progress)
        except Exception as exc:  # noqa: BLE001 — do not retry pip -e / SIGKILL
            raise RuntimeError(f"{FAIL_CLOSED}:{exc}") from exc
    else:
        if progress:
            progress("weights:h3_join")
        join_h3_weights()
        ensure_h3_dits_for_shots(_board_shots(root))
        unload_still_models(router)
    anim = run_anim(story_id, root, conn, router, progress=progress)
    if backend != "longlive" and anim.get("gpu_done"):
        try:
            flush_gpu_artifacts(story_id, root, conn, progress=progress)
        except Exception:
            pass
    remaining, allow_compose = compose_gate(anim)
    if control:
        if allow_compose:
            control.job(story_id, "anim", "succeeded", episode_code=EP)
            control.job(story_id, "qc", "succeeded", episode_code=EP)
        else:
            err = (
                str(anim["failed"][:3])
                if anim.get("failed")
                else f"partial {anim.get('shots_done') or 0}/{anim.get('shots_total') or 0} remaining={remaining}"
            )
            control.job(story_id, "anim", "running", error=err, episode_code=EP)
            control.job(story_id, "qc", "blocked", error=err, episode_code=EP)
    compose = None
    if allow_compose:
        if progress:
            progress("compose")
        if control:
            control.job(story_id, "compose", "running", episode_code=EP)
        try:
            compose = run_compose(
                story_id,
                root,
                conn,
                langs=episode_langs,
                progress=progress,
            )
            if skip_shorts:
                compose["shorts"] = {"keys": [], "errors": [], "skipped": "short"}
            else:
                shorts = render_episode_shorts(
                    story_id,
                    root,
                    episode_langs,
                    progress=progress,
                )
                compose["shorts"] = shorts
            if control:
                control.job(story_id, "compose", "succeeded", episode_code=EP)
            if control and finish_story:
                try:
                    control.action(story_id, "finish", episode_code=EP)
                except Exception as exc:  # noqa: BLE001
                    compose = dict(compose or {}, finish_error=str(exc))
        except Exception as exc:  # noqa: BLE001 — keep the lease; do not re-render picture
            err = f"{type(exc).__name__}:{exc}"[:500]
            if control:
                control.job(story_id, "compose", "failed", error=err, episode_code=EP)
            remaining = max(int(remaining or 0), 1)
            compose = {"ok": False, "error": err, "final_keys": []}
    _checkpoint_story(
        conn,
        story_id,
        root,
        progress=progress,
        upload=True,
    )
    return {
        "story_id": story_id,
        "episode_code": EP,
        "langs": list(episode_langs),
        "pulled": len(pulled),
        "pre_gpu": pre_gpu,
        "stills": stills,
        "anim": anim,
        "compose": compose,
        "remaining": remaining,
        "truncated": False,
        "gpu_done": bool((anim or {}).get("gpu_done")),
    }


def _episode_artifacts(result: dict[str, Any] | None) -> dict[str, Any]:
    result = result or {}
    compose = result.get("compose") if isinstance(result.get("compose"), dict) else {}
    shorts = compose.get("shorts") if isinstance(compose.get("shorts"), dict) else {}
    final_keys = compose.get("final_keys") if isinstance(compose.get("final_keys"), list) else []
    subtitle_keys = (
        compose.get("subtitle_keys")
        if isinstance(compose.get("subtitle_keys"), list)
        else []
    )
    anim = result.get("anim") if isinstance(result.get("anim"), dict) else {}
    artifacts: dict[str, Any] = {
        "final": final_keys[0] if final_keys else None,
        "subtitles": subtitle_keys,
        "shorts": list(shorts.get("keys") or []),
    }
    if isinstance(anim.get("resolution"), dict):
        artifacts["h3_resolution"] = anim["resolution"]
    if shorts.get("errors"):
        artifacts["shorts_errors"] = list(shorts["errors"])
    return artifacts


def _episode_never_destroy_error(result: dict[str, Any] | None) -> str | None:
    anim = result.get("anim") if isinstance(result, dict) else None
    failures = anim.get("failed") if isinstance(anim, dict) else None
    text = json.dumps(failures or [], ensure_ascii=False, default=str).lower()
    classified = never_destroy_error(text)
    if classified:
        return classified
    if any(
        marker in text
        for marker in (
            "no_comfy_router",
            "connection refused",
            "failed to establish a new connection",
        )
    ):
        return "install_failure"
    return None


def _report_with_retry(
    report: ReportCallback,
    payload: dict[str, Any],
    runtime: LeaseRuntime,
    attempts: int = 3,
) -> dict[str, Any]:
    last: dict[str, Any] = {"ok": False, "error": "report_not_attempted"}
    for attempt in range(max(1, attempts)):
        try:
            last = report(payload)
        except Exception as exc:  # noqa: BLE001 — retry before allowing lease teardown
            last = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}
        if isinstance(last, dict) and last.get("ok", True) is not False:
            return last
        if attempt + 1 < attempts:
            runtime.assert_within_budget()
            time.sleep(max(0.0, float(os.environ.get("GPU_REPORT_RETRY_SECONDS") or 2)))
    return last


def run_gpu_batch(
    batch: dict[str, Any],
    runtime: LeaseRuntime,
    *,
    control: ControlPlane | None = None,
    claim: ClaimCallback | None = None,
    report: ReportCallback | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Run every episode in a claimed batch, preserving per-shot R2 checkpoints."""
    story_id = str(batch.get("story_id") or "").strip()
    batch_id = str(batch.get("batch_id") or "").strip()
    episodes = [ep for ep in (batch.get("episodes") or []) if isinstance(ep, dict)]
    story_root = Path(
        root
        or os.environ.get("AF_STORY_ROOT")
        or (f"/work/{story_id}" if story_id else "/work")
    )
    results: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    report_errors: list[str] = []
    stop_limit: str | None = None
    stop_checkpoint: dict[str, Any] | None = None
    destroy_error: str | None = None
    skip_cycle = False
    longlive_closed = False
    if not story_id or not batch_id:
        raise ValueError("batch_id and story_id are required")
    runtime.set_active_work(story_id, story_root)

    def _skip_reason(episode: dict[str, Any]) -> str | None:
        code = str(episode.get("episode_code") or DEFAULT_EPISODE)
        if runtime.already_completed(story_id, code):
            return "remaining=0"
        langs = episode.get("langs") if isinstance(episode.get("langs"), list) else None
        return gpu_cycle_idle_reason(
            {
                **episode,
                "story_id": story_id,
                "status": episode.get("story_status") or batch.get("status") or episode.get("status"),
                "story_status": episode.get("story_status") or batch.get("status"),
                "episode_status": episode.get("episode_status") or episode.get("status"),
                "until_stage": episode.get("until_stage"),
            },
            root=story_root,
            episode_code=code,
            langs=langs,
        )

    actionable = [ep for ep in episodes if not _skip_reason(ep)]
    runtime.adopt_batch(batch, busy=bool(actionable))
    if not actionable:
        skip_cycle = True
        for episode in episodes:
            code = str(episode.get("episode_code") or DEFAULT_EPISODE)
            reason = _skip_reason(episode) or "remaining=0"
            runtime.remember_completed(story_id, code)
            results.append(
                {
                    "episode_code": code,
                    "status": "done",
                    "error": None,
                    "result": _idle_episode_result(
                        story_id,
                        _normalize_episode_code(code),
                        tuple(str(lang).strip().lower() for lang in (episode.get("langs") or ("zh", "en", "ja")) if str(lang).strip())
                        or ("zh", "en", "ja"),
                        reason=reason,
                    ),
                }
            )
        runtime.mark_idle()
        return {
            "batch_id": batch_id,
            "story_id": story_id,
            "episodes_total": len(episodes),
            "episodes_attempted": 0,
            "episodes_done": len(results),
            "results": results,
            "reports": reports,
            "report_errors": report_errors,
            "limit": None,
            "stop_checkpoint": None,
            "destroy_reason": "success",
            "destroy_error": None,
            "skip_cycle": True,
        }

    for episode in episodes:
        episode_code = str(episode.get("episode_code") or DEFAULT_EPISODE)
        skip_reason = _skip_reason(episode)
        if skip_reason:
            skip_cycle = True
            runtime.remember_completed(story_id, episode_code)
            results.append(
                {
                    "episode_code": episode_code,
                    "status": "done",
                    "error": None,
                    "result": _idle_episode_result(
                        story_id,
                        _normalize_episode_code(episode_code),
                        tuple(
                            str(lang).strip().lower()
                            for lang in (episode.get("langs") or ("zh", "en", "ja"))
                            if str(lang).strip()
                        )
                        or ("zh", "en", "ja"),
                        reason=skip_reason,
                    ),
                }
            )
            continue
        requested_stage = str(episode.get("stage") or "anim")
        current_stage = (
            requested_stage
            if requested_stage in {"design", "keyframe", "anim", "qc", "compose"}
            else "anim"
        )
        episode_result: dict[str, Any] | None = None
        status = "failed"
        error: str | None = None
        stop = False

        def progress(stage: str) -> None:
            nonlocal current_stage
            reported_stage = stage.split(":", 1)[0]
            if reported_stage in {"design", "keyframe", "anim", "qc", "compose"}:
                current_stage = reported_stage
            runtime.observe_progress(stage)

        try:
            runtime.assert_within_budget()
            runtime.mark_busy()
            if claim:
                claimed = claim(story_id, episode_code)
                if isinstance(claimed, dict) and claimed.get("ok", True) is False:
                    raise RuntimeError(f"claim failed: {claimed.get('error') or claimed}")
            langs = episode.get("langs") if isinstance(episode.get("langs"), list) else None
            skip_shorts = is_short_kind(episode.get("kind"))
            episode_result = run_gpu_episode(
                story_id,
                root=story_root,
                control=control,
                episode_code=episode_code,
                langs=langs,
                progress=progress,
                finish_story=False,
                skip_shorts=skip_shorts,
                story_status=str(
                    episode.get("story_status") or batch.get("status") or episode.get("status") or ""
                )
                or None,
                until_stage=str(episode.get("until_stage") or "") or None,
                episode_status=str(episode.get("episode_status") or "") or None,
            )
            if int(episode_result.get("remaining") or 0) == 0:
                status = "done"
                current_stage = "compose"
                runtime.remember_completed(story_id, episode_code)
            else:
                remaining = int(episode_result.get("remaining") or 0)
                status = "running"
                error = f"anim incomplete; remaining={remaining}"
                current_stage = "anim"
                destroy_error = _episode_never_destroy_error(episode_result) or destroy_error
                runtime.last_error = destroy_error
                runtime.mark_busy()
                stop = bool(destroy_error)
                failed = []
                if isinstance(episode_result.get("anim"), dict):
                    failed = list(episode_result["anim"].get("failed") or [])
                if any(is_longlive_fatal_error(item) for item in failed) or is_longlive_fatal_error(error):
                    longlive_closed = True
                    stop = True
                recycle_hit = recycle_failure_class(error)
                if recycle_hit is None:
                    for item in failed:
                        recycle_hit = recycle_failure_class(item)
                        if recycle_hit:
                            break
                if recycle_hit:
                    error = f"{recycle_hit}:{error}"
                    stop_limit = recycle_hit
                    stop_checkpoint = best_effort_upload_checkpoint(story_id, story_root)
                    stop = True
        except BudgetExceeded as exc:
            error = str(exc)
            stop_limit = exc.limit
            stop = True
        except ProgressStalled as exc:
            error = str(exc)
            stop_limit = exc.limit
            stop_checkpoint = best_effort_upload_checkpoint(story_id, story_root)
            stop = True
        except StartupTimeout as exc:
            error = str(exc)
            stop_limit = exc.limit
            stop_checkpoint = best_effort_upload_checkpoint(story_id, story_root)
            stop = True
        except Exception as exc:  # noqa: BLE001 — report one episode failure and continue safely
            error = f"{type(exc).__name__}:{exc}"
            classified = recycle_failure_class(exc) or recycle_failure_class(error)
            if classified:
                error = f"{classified}:{error}"
                stop_limit = classified
                stop_checkpoint = best_effort_upload_checkpoint(story_id, story_root)
                stop = True
            elif is_longlive_fatal_error(exc) or is_longlive_fatal_error(error):
                if FAIL_CLOSED not in error:
                    error = f"{FAIL_CLOSED}:{error}"
                longlive_closed = True
                stop = True
            else:
                destroy_error = never_destroy_error(exc) or destroy_error
                runtime.last_error = destroy_error
                stop = bool(destroy_error)

        payload = {
            "vast_instance_id": runtime.instance_id,
            "batch_id": batch_id,
            "episode_code": episode_code,
            "stage": current_stage,
            "status": status,
            "artifacts": _episode_artifacts(episode_result),
            "error": error,
        }
        report_response: dict[str, Any] | None = None
        if report:
            try:
                report_response = _report_with_retry(report, payload, runtime)
            except (BudgetExceeded, ProgressStalled, StartupTimeout) as exc:
                report_response = {"ok": False, "error": str(exc)}
                stop_limit = exc.limit
                if isinstance(exc, (ProgressStalled, StartupTimeout)):
                    stop_checkpoint = best_effort_upload_checkpoint(
                        story_id,
                        story_root,
                    )
                stop = True
            if report_response.get("ok", True) is False:
                report_errors.append(
                    f"{episode_code}:{report_response.get('error') or 'report failed'}"
                )
        reports.append({"payload": payload, "response": report_response})
        results.append(
            {
                "episode_code": episode_code,
                "status": status,
                "error": error,
                "result": episode_result,
            }
        )
        if stop:
            break

    runtime.last_error = destroy_error
    remaining_any = any(
        int((item.get("result") or {}).get("remaining") or 0) > 0
        if isinstance(item.get("result"), dict)
        else item["status"] != "done"
        for item in results
    )
    all_done = len(results) == len(episodes) and all(item["status"] == "done" for item in results)
    budget_stop = stop_limit in {
        "max_minutes",
        "max_usd",
        "global_max_usd",
        "progress_stalled",
        "startup_timeout",
    }
    if remaining_any or not all_done:
        runtime.mark_busy()
    else:
        runtime.mark_idle()
    recycle_stop = stop_limit in RECYCLE_FAILURE_CLASSES
    if report_errors:
        destroy_reason = None
    elif longlive_closed or recycle_stop:
        # Stop GPU 0% pip-retry / SIGKILL loops. Scheduler cooldown must not re-lease.
        destroy_reason = "explicit_abort"
        destroy_error = None
    elif budget_stop:
        destroy_reason = "explicit_abort"
    elif remaining_any or not all_done:
        # Remaining H3 work keeps the box. One Comfy 400 must not deallocate Vast.
        destroy_reason = None
    else:
        destroy_reason = "success"
    return {
        "batch_id": batch_id,
        "story_id": story_id,
        "episodes_total": len(episodes),
        "episodes_attempted": len(results),
        "episodes_done": sum(item["status"] == "done" for item in results),
        "results": results,
        "reports": reports,
        "report_errors": report_errors,
        "limit": stop_limit,
        "stop_checkpoint": stop_checkpoint,
        "destroy_reason": destroy_reason,
        "destroy_error": destroy_error,
        "skip_cycle": skip_cycle,
    }

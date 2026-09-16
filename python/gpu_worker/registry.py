"""In-memory GPU worker registry + idle scale-to-zero (dry-run)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from gpu_worker.images import default_register_capabilities
from gpu_worker.vast_client import VastClient

STATUSES = ("booting", "ready", "busy", "idle", "draining", "dead")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class GpuWorker:
    vast_instance_id: str
    gpu_type: str = "RTX_5090"
    status: str = "booting"
    capabilities: dict = field(default_factory=lambda: default_register_capabilities())
    assigned_story_id: str | None = None
    assigned_episode: str | None = None
    assigned_scenes: list[str] = field(default_factory=list)
    last_heartbeat: datetime | None = None
    lease_hourly_usd: float = 0.0
    started_at: datetime | None = None


class GpuRegistry:
    def __init__(self, vast: VastClient | None = None, heartbeat_timeout_s: int = 90, idle_ttl_s: int = 900):
        self.workers: dict[str, GpuWorker] = {}
        self.vast = vast or VastClient(dry_run=True)
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.idle_ttl_s = idle_ttl_s
        self.task_queue: list[dict] = []

    def register(self, instance_id: str, gpu_type: str = "RTX_5090", capabilities: dict | None = None) -> GpuWorker:
        w = GpuWorker(
            vast_instance_id=instance_id,
            gpu_type=gpu_type,
            status="ready",
            capabilities=capabilities or default_register_capabilities(),
            last_heartbeat=_now(),
            started_at=_now(),
        )
        self.workers[instance_id] = w
        return w

    def heartbeat(self, instance_id: str, status: str | None = None) -> GpuWorker:
        w = self.workers[instance_id]
        w.last_heartbeat = _now()
        if status:
            w.status = status
        return w

    def enqueue(self, task: dict) -> None:
        self.task_queue.append(task)

    def ready_count(self) -> int:
        return sum(1 for w in self.workers.values() if w.status in {"ready", "busy", "idle"})

    def scale_for_tasks(self) -> int:
        """No tasks → gpu_workers count 0 (scale to zero)."""
        if not self.task_queue:
            registered = set(self.workers)
            for wid in list(self.workers):
                self.vast.destroy(wid, registered)
                del self.workers[wid]
            return 0
        return len(self.workers)

    def reap_timeouts(self) -> list[str]:
        dead = []
        now = _now()
        for wid, w in list(self.workers.items()):
            if w.last_heartbeat and now - w.last_heartbeat > timedelta(seconds=self.heartbeat_timeout_s):
                w.status = "dead"
                dead.append(wid)
        return dead

    def idle_scale_to_zero(self) -> int:
        if self.task_queue:
            return len(self.workers)
        now = _now()
        for wid, w in list(self.workers.items()):
            last = w.last_heartbeat or w.started_at or now
            if w.status in {"idle", "ready"} and now - last >= timedelta(seconds=self.idle_ttl_s):
                self.vast.destroy(wid, set(self.workers))
                del self.workers[wid]
        return len(self.workers)

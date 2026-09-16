"""Scene-modulo shard. Per-backend concurrency = 1. Heartbeat timeout requeues that worker's scenes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkerSlot:
    worker_id: str
    scenes: list[str] = field(default_factory=list)
    inflight: int = 0
    alive: bool = True


def comfy_max_concurrent(n_backends: int) -> int:
    """COMFYUI_MAX_CONCURRENT equals the number of real backends."""
    return max(n_backends, 0)


PER_BACKEND_CONCURRENCY = 1


def assign_scenes(scene_ids: list[str], n_cards: int) -> dict[int, list[str]]:
    n_cards = max(int(n_cards), 1)
    buckets: dict[int, list[str]] = {i: [] for i in range(n_cards)}
    for i, sid in enumerate(scene_ids):
        buckets[i % n_cards].append(sid)
    return buckets


def assign_chains(chain_ids: list[str], n_cards: int) -> dict[int, list[str]]:
    """A chain never splits across cards; the unit of sharding is scene/chain."""
    return assign_scenes(chain_ids, n_cards)


def assign_to_workers(scene_ids: list[str], worker_ids: list[str]) -> dict[str, list[str]]:
    buckets = assign_scenes(scene_ids, len(worker_ids) or 1)
    out: dict[str, list[str]] = {}
    if not worker_ids:
        return {"": scene_ids}
    for i, wid in enumerate(worker_ids):
        out[wid] = buckets.get(i, [])
    return out


def requeue_on_timeout(
    assignments: dict[str, list[str]],
    dead_worker_id: str,
    live_worker_ids: list[str],
) -> dict[str, list[str]]:
    """Requeue ONLY the dead worker's scenes. Other workers keep theirs."""
    orphan = list(assignments.get(dead_worker_id) or [])
    next_map = {wid: list(scenes) for wid, scenes in assignments.items() if wid != dead_worker_id}
    if not live_worker_ids:
        next_map["unassigned"] = orphan
        return next_map
    for i, sid in enumerate(orphan):
        next_map.setdefault(live_worker_ids[i % len(live_worker_ids)], []).append(sid)
    return next_map


def can_submit(slot: WorkerSlot) -> bool:
    return slot.alive and slot.inflight < PER_BACKEND_CONCURRENCY

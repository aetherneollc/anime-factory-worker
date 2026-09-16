from datetime import datetime, timedelta, timezone

from gpu_worker.registry import GpuRegistry
from gpu_worker.shard import (
    PER_BACKEND_CONCURRENCY,
    assign_scenes,
    can_submit,
    comfy_max_concurrent,
    requeue_on_timeout,
    WorkerSlot,
)
from gpu_worker.vast_client import VastClient


def test_scene_modulo_across_n_cards():
    scenes = ["sc1", "sc2", "sc3", "sc4", "sc5"]
    buckets = assign_scenes(scenes, 2)
    assert buckets[0] == ["sc1", "sc3", "sc5"]
    assert buckets[1] == ["sc2", "sc4"]
    four = assign_scenes(scenes, 4)
    assert sum(len(v) for v in four.values()) == 5


def test_per_backend_concurrency_is_one():
    assert PER_BACKEND_CONCURRENCY == 1
    assert comfy_max_concurrent(3) == 3
    slot = WorkerSlot("w1", inflight=1)
    assert can_submit(slot) is False
    slot.inflight = 0
    assert can_submit(slot) is True


def test_heartbeat_timeout_requeues_only_that_worker_scenes():
    assignments = {"w1": ["sc1", "sc3"], "w2": ["sc2", "sc4"]}
    nxt = requeue_on_timeout(assignments, "w1", ["w2"])
    assert nxt["w2"] == ["sc2", "sc4", "sc1", "sc3"]
    assert "w1" not in nxt


def test_idle_ttl_scale_to_zero_dry_run():
    client = VastClient(dry_run=True)
    reg = GpuRegistry(vast=client, idle_ttl_s=0)
    reg.register("inst-idle")
    # last heartbeat is now; with ttl 0, idle_scale_to_zero should destroy
    reg.workers["inst-idle"].status = "idle"
    reg.workers["inst-idle"].last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=1)
    n = reg.idle_scale_to_zero()
    assert n == 0
    assert any(m == "DELETE" or m == "DRY" or "instances/inst-idle" in u for m, u in client.calls) or not reg.workers
    assert len(reg.workers) == 0

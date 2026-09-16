"""GPU scheduler — offer pick must honor GPU_MODEL_POLICY (no SKU fallback)."""

from gpu_worker.registry import GpuRegistry
from gpu_worker.scheduler import ensure_capacity
from gpu_worker.vast_client import VastClient

_PRE_LEASE = {s: "passed" for s in ("bible", "canon", "script", "tts", "timing", "board")}


def _offer(**partial):
    return {
        "id": "offer",
        "gpu_name": "RTX 5090",
        "gpu_ram": 32768,
        "duration": 4,
        "reliability": 0.995,
        "inet_down": 400,
        "inet_up": 200,
        "disk_space": 250,
        "geolocation": "US",
        "dph_total": 0.639,
        **partial,
    }


def test_ensure_capacity_no_offer_when_policy_blocks_only_inventory(monkeypatch):
    monkeypatch.setenv("GPU_MODEL_POLICY", "5090")
    puts = []

    def opener(req):
        if req.get_method() == "PUT" and "/asks/" in req.full_url:
            puts.append(req.full_url)
        return {"offers": [_offer(id="4090", gpu_name="RTX 4090", gpu_ram=49152, dph_total=0.2)]}

    client = VastClient("fake", opener=opener, dry_run=False)
    reg = GpuRegistry(vast=client)
    out = ensure_capacity(reg, client, need=1, stage_flags=_PRE_LEASE)
    assert out["action"] == "no_offer"
    assert out["reason"] == "scored_out"
    assert out["put_asks"] is False
    assert puts == []


def test_ensure_capacity_no_offer_when_5090_over_dph_cap(monkeypatch):
    monkeypatch.setenv("GPU_MODEL_POLICY", "5090")
    puts = []

    def opener(req):
        if req.get_method() == "PUT" and "/asks/" in req.full_url:
            puts.append(req.full_url)
        return {"offers": [_offer(id="expensive", dph_total=4.0, geolocation="Norway")]}

    client = VastClient("fake", opener=opener, dry_run=False)
    reg = GpuRegistry(vast=client)
    out = ensure_capacity(reg, client, need=1, stage_flags=_PRE_LEASE)
    assert out["action"] == "no_offer"
    assert out["reason"] == "scored_out"
    assert puts == []

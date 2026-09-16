"""Search → lease only when dry-run is off. Production default VAST_DRY_RUN=1.

Live leases go through gpu_worker.boot (pre-lease gates, one instance,
no SKU flip). Dry-run still short-circuits before PUT /asks/.
Refuse lease unless the Hub image is comfy+h3+kolors.
"""

from __future__ import annotations

from gpu_worker.boot import LeaseSession, assert_lease_capabilities, assert_pre_lease, request_lease
from gpu_worker.images import AGENT_IMAGE, image_capabilities
from gpu_worker.offers import parse_gpu_model_policy, pick_one_offer
from gpu_worker.registry import GpuRegistry
from gpu_worker.vast_client import VastClient


def ensure_capacity(
    registry: GpuRegistry,
    client: VastClient,
    need: int = 1,
    stage_flags: dict[str, str] | None = None,
    image: str | None = None,
    session: LeaseSession | None = None,
) -> dict:
    ready = [w for w in registry.workers.values() if w.status in {"ready", "idle"}]
    if ready:
        return {"action": "reuse", "ids": [w.vast_instance_id for w in ready]}
    offers = client.search_offers()
    if client.dry_run:
        return {"action": "dry_run_skip_lease", "offers": offers, "put_asks": False}
    assert_pre_lease(stage_flags)
    sess = session or LeaseSession(image=image or f"{AGENT_IMAGE}:main")
    if not sess.capabilities:
        sess.capabilities = image_capabilities(sess.image)
    assert_lease_capabilities(sess.capabilities)
    offer_id = "0"
    offer_list = []
    if isinstance(offers, dict):
        offer_list = list(offers.get("offers") or [])
    chosen = pick_one_offer(offer_list, gpu_model_policy=parse_gpu_model_policy())
    if not chosen:
        return {
            "action": "no_offer",
            "offers": offers,
            "put_asks": False,
            "reason": "scored_out",
        }
    offer_id = str(chosen.get("id") or "0")
    leased = request_lease(client, offer_id, sess, stage_flags)
    leased["capabilities"] = sess.capabilities
    return leased

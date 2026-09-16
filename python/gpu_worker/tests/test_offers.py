"""Vast offer scoring — mirrors cf-control gpu_scheduler."""

from gpu_worker.offers import (
    OFFER_SCORE_VERSION,
    UPLOAD_GB_PER_EP,
    WEIGHTS_GB,
    offer_matches_gpu_policy,
    pick_one_offer,
    score_offer,
)


def _eligible(**partial):
    return {
        "gpu_name": "RTX 5090",
        "gpu_ram": 32768,
        "duration": 4,
        "reliability": 0.995,
        "inet_down": 400,
        "inet_up": 200,
        "disk_space": 250,
        "geolocation": "US",
        **partial,
    }


def test_rejects_dph_above_default_cap():
    norway = _eligible(id="norway", dph_total=4.0, geolocation="Norway")
    cheap = _eligible(id="ok", dph_total=0.639, geolocation="US")
    assert score_offer(norway)["reject_reason"] == "max_dph"
    assert score_offer(cheap).get("reject_reason") is None
    assert pick_one_offer([norway, cheap])["id"] == "ok"
    assert pick_one_offer([norway]) is None


def test_max_dph_override():
    mid = _eligible(id="mid", dph_total=1.5)
    assert pick_one_offer([mid], max_usd=50.0, max_dph=1.0) is None
    assert pick_one_offer([mid], max_usd=50.0, max_dph=2.0)["id"] == "mid"


def test_default_5090_policy_rejects_4090():
    rtx4090 = _eligible(id="4090", gpu_name="RTX 4090", gpu_ram=49152, dph_total=0.3)
    rtx5090 = _eligible(id="5090", dph_total=0.45)
    assert offer_matches_gpu_policy(rtx4090, "5090") is False
    assert pick_one_offer([rtx4090, rtx5090], gpu_model_policy="5090")["id"] == "5090"


def test_rejects_24gb_4090():
    small = _eligible(id="small", gpu_name="RTX 4090", gpu_ram=24576, dph_total=0.2)
    assert offer_matches_gpu_policy(small, "5090") is False
    assert pick_one_offer([small]) is None


def test_5090_policy_no_fallback_when_only_4090():
    rtx4090 = _eligible(id="4090", gpu_name="RTX 4090", gpu_ram=49152, dph_total=0.1)
    assert pick_one_offer([rtx4090], gpu_model_policy="5090") is None


def test_5090_policy_no_fallback_when_only_over_dph_cap():
    pricey = _eligible(id="pricey", dph_total=4.0, geolocation="Norway")
    assert pick_one_offer([pricey], gpu_model_policy="5090") is None


def test_california_no_trap_total_matches_lease_only():
    california = _eligible(id="ca", dph_total=0.393, inet_down=330, geolocation="California, US")
    vietnam = _eligible(id="vn", dph_total=0.476, inet_down=7239, geolocation="Vietnam")
    ca = score_offer(california, 3)
    vn = score_offer(vietnam, 3)
    assert ca.get("reject_reason") is None
    assert vn.get("reject_reason") is None
    assert ca["egress_usd"] == 0
    assert vn["egress_usd"] == 0
    assert ca["egress_trap"] is False
    assert ca["expected_total_usd"] == ca["lease_h"] * ca["dph_total"]
    assert vn["expected_total_usd"] == vn["lease_h"] * vn["dph_total"]
    assert ca["expected_total_usd"] < vn["expected_total_usd"]
    assert pick_one_offer([vietnam, california], episode_count=3)["id"] == "ca"
    assert ca["version"] == "expected_total_v3"
    assert ca["version"] == OFFER_SCORE_VERSION


def test_high_bw_cost_rejected_or_ranked_after_ca():
    california = _eligible(id="ca", dph_total=0.393, inet_down=330, geolocation="California, US")
    expensive = _eligible(id="trap", dph_total=0.15, inet_down=8000, bw_cost=0.5, geolocation="Somewhere")
    moderate = _eligible(id="pricey-bw", dph_total=0.15, inet_down=8000, bw_cost=0.2, geolocation="Somewhere")
    trap = score_offer(expensive, 3)
    assert trap["egress_trap"] is True
    assert trap["usd_per_gb"] == 0.5
    assert trap["transfer_gb"] == WEIGHTS_GB + UPLOAD_GB_PER_EP * 3
    assert trap["egress_usd"] == 81 * 0.5
    assert trap["reject_reason"] == "egress"
    assert pick_one_offer([expensive, california], episode_count=3)["id"] == "ca"

    mid = score_offer(moderate, 3)
    ca = score_offer(california, 3)
    assert mid.get("reject_reason") is None
    assert mid["egress_trap"] is True
    assert mid["expected_total_usd"] > ca["expected_total_usd"]
    assert pick_one_offer([moderate, california], episode_count=3)["id"] == "ca"


def test_trap_heuristic_imputes_when_cheap_and_fast():
    bait = _eligible(id="bait", dph_total=0.12, inet_down=5000, geolocation="Unknown")
    scored = score_offer(bait, 3)
    assert scored.get("reject_reason") is None
    assert scored["egress_trap"] is True
    assert scored["usd_per_gb"] == 0.08
    assert scored["egress_usd"] == (WEIGHTS_GB + UPLOAD_GB_PER_EP * 3) * 0.08
    assert scored["expected_total_usd"] == scored["lease_h"] * scored["dph_total"] + scored["egress_usd"]

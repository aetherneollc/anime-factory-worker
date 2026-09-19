"""Vast offer search and cost-first scoring (mirrors cf-control gpu_scheduler)."""

from __future__ import annotations

import os
from functools import cmp_to_key
from typing import Any

# H3 native path needs 32GB (gpu_worker.h3.H3_NATIVE_VRAM_MB). Vast /bundles/ wants dict filters.
MIN_GPU_RAM_MB = 32000
MIN_CPU_RAM_MB = 65536
MIN_CUDA_VERSION = 13.0
MIN_DURATION_DAYS = 3
DEFAULT_LEASE_EPISODE_COUNT = 3
DEFAULT_MAX_LEASE_USD = 40.0
MAX_LEASE_USD_CEILING = 100.0
DEFAULT_MAX_DPH_USD = 1.0
COMPUTE_HOURS_PER_EPISODE = 9.65
BOOT_HOURS = 1.0
WEIGHTS_GB = 63
UPLOAD_GB_PER_EP = 6
EGRESS_TRAP_USD_PER_GB = 0.02
EGRESS_TRAP_CHEAP_DPH = 0.28
EGRESS_TRAP_INET_DOWN_MBPS = 2000
EGRESS_TRAP_IMPUTED_USD_PER_GB = 0.08
INET_DOWN_FACTOR = 0.7
EFFECTIVE_DOWN_CAP_MBPS = 1000
DURATION_BUFFER_HOURS = 12
COST_TIE_RATIO = 0.05
SKU_MULTIPLIER_5090 = 1.0
SKU_MULTIPLIER_4090_48 = 1.45
OFFER_SCORE_VERSION = "expected_total_v3"

VERIFIED_GPU_SKUS = ("4090", "5090")
REJECT_GPU_SKUS = ("v100", "cmp", "p40", "p100", "m40", "titan v", "mi25", "mi50")

DEFAULT_GPU_MODEL_POLICY = "5090"
GPU_PROFILE_DEFS: tuple[tuple[str, str, str, bool], ...] = (
    ("5090", "h3-comfy-cu130-sm120", "RTX 5090", True),
    ("4090_48", "h3-comfy-cu128-sm89", "RTX 4090 48GB", False),
)

DEFAULT_SEARCH_FILTERS: dict[str, Any] = {
    "verified": {"eq": True},
    "rentable": {"eq": True},
    "rented": {"eq": False},
    "num_gpus": {"eq": 1},
    "gpu_ram": {"gte": MIN_GPU_RAM_MB},
    "cpu_ram": {"gte": MIN_CPU_RAM_MB},
    "cuda_max_good": {"gte": MIN_CUDA_VERSION},
    "duration": {"gte": MIN_DURATION_DAYS},
    "reliability": {"gte": 0.95},
    "disk_space": {"gte": 200},
    "inet_down": {"gte": 250},
    "inet_up": {"gte": 100},
    "geolocation": {"notin": ["CN"]},
    "type": "ondemand",
    "limit": 100,
}


def search_payload(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(DEFAULT_SEARCH_FILTERS)
    if extra:
        payload.update(extra)
    return payload


def _gpu_frac(offer: dict) -> float:
    return _offer_num(offer, "gpu_frac", "gpu_fraction")


def _gpu_name(offer: dict) -> str:
    return str(offer.get("gpu_name") or offer.get("gpu_name_long") or offer.get("gpu") or "")


def _cpu_ram_mb(offer: dict) -> int:
    raw = offer.get("cpu_ram") or offer.get("cpu_ram_mb") or 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _cuda_version(offer: dict) -> float:
    raw = offer.get("cuda_max_good") or offer.get("cuda_vers") or offer.get("cuda_version") or 0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _gpu_ram_mb(offer: dict) -> int:
    raw = offer.get("gpu_ram") or offer.get("gpu_ram_mb") or 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _offer_num(offer: dict, *keys: str) -> float:
    for key in keys:
        raw = offer.get(key)
        if raw is None:
            continue
        try:
            n = float(raw)
        except (TypeError, ValueError):
            continue
        if n == n:
            return n
    return float("nan")


def _dph(offer: dict) -> float:
    raw = offer.get("dph_total") or offer.get("dph")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def _usd_per_gb_field(offer: dict) -> float:
    return _offer_num(offer, "inet_down_cost", "bw_cost", "bandwidth_cost")


def resolve_egress_rate(offer: dict, dph: float, inet_down: float) -> tuple[float, bool]:
    explicit = _usd_per_gb_field(offer)
    if explicit == explicit and explicit >= 0:
        return explicit, explicit >= EGRESS_TRAP_USD_PER_GB
    if dph == dph and dph < EGRESS_TRAP_CHEAP_DPH and inet_down > EGRESS_TRAP_INET_DOWN_MBPS:
        return EGRESS_TRAP_IMPUTED_USD_PER_GB, True
    return 0.0, False


def _duration_days(offer: dict) -> float:
    raw = offer.get("duration") or offer.get("duration_days") or 0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _inet_down(offer: dict) -> float:
    raw = offer.get("inet_down") or offer.get("inet_down_mbps") or 0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _reliability(offer: dict) -> float:
    raw = offer.get("reliability") or offer.get("machine_reliability") or 0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _geolocation(offer: dict) -> str:
    return str(offer.get("geolocation") or offer.get("geoloc") or offer.get("country") or offer.get("country_code") or "").strip()


def _is_cn_geolocation(offer: dict) -> bool:
    raw = " ".join(
        str(offer.get(k) or "").strip().lower()
        for k in ("geolocation", "geoloc", "country", "country_code")
        if offer.get(k)
    )
    if not raw:
        return False
    if any(tok in raw for tok in ("taiwan", "hong kong", "macau", "macao")):
        return False
    return "cn" in raw.split() or "china" in raw or "中国" in raw or "prc" in raw.split()


def parse_gpu_model_policy(value: str | None = None) -> str:
    raw = str(value or os.environ.get("GPU_MODEL_POLICY", DEFAULT_GPU_MODEL_POLICY)).strip().lower()
    if raw in ("5090", "rtx5090", "rtx 5090", ""):
        return "5090"
    if raw in ("4090_48", "4090-48", "409048", "rtx 4090 48gb"):
        return "4090_48"
    if raw in ("auto_compatible", "auto"):
        return "auto_compatible"
    return DEFAULT_GPU_MODEL_POLICY


def _validated_profiles(policy: str) -> list[tuple[str, str, str, bool]]:
    if policy == "auto_compatible":
        return [p for p in GPU_PROFILE_DEFS if p[3]]
    for prof in GPU_PROFILE_DEFS:
        if prof[0] == policy:
            return [prof] if prof[3] else []
    return []


def offer_matches_gpu_policy(offer: dict, policy: str | None = None) -> bool:
    pol = parse_gpu_model_policy(policy)
    name = _gpu_name(offer).lower()
    ram = _gpu_ram_mb(offer)
    if ram > 0 and ram < MIN_GPU_RAM_MB:
        return False
    if "4090" in name and ram > 0 and ram < 40000 and "48" not in name:
        return False
    profiles = _validated_profiles(pol)
    if not profiles:
        return False
    if pol == "5090":
        return "5090" in name
    if pol == "4090_48":
        return "4090" in name and (ram >= 48000 or "48" in name)
    for prof in profiles:
        sku = prof[0]
        if sku == "5090" and "5090" in name:
            return True
        if sku == "4090_48" and "4090" in name and (ram >= 48000 or "48" in name):
            return True
    return False


def gpu_allowlist() -> tuple[str, ...]:
    raw = os.environ.get("VAST_GPU_ALLOWLIST", "").strip()
    if not raw:
        return VERIFIED_GPU_SKUS
    parts = tuple(p.strip().lower() for p in raw.split(",") if p.strip())
    return parts or VERIFIED_GPU_SKUS


def gpu_sku_verdict(gpu_name: str, allowlist: tuple[str, ...] | None = None) -> str:
    name = str(gpu_name or "").lower()
    if not name:
        return "confirm"
    if any(s in name for s in REJECT_GPU_SKUS):
        return "reject"
    allow = allowlist or gpu_allowlist()
    if any(s in name for s in allow):
        return "verified"
    return "confirm"


def sku_multiplier(gpu_name: str, gpu_ram_mb: int = 0) -> float:
    name = str(gpu_name or "").lower()
    if "4090" in name and (gpu_ram_mb >= 40000 or "48" in name):
        return SKU_MULTIPLIER_4090_48
    if "4090" in name:
        return SKU_MULTIPLIER_4090_48
    return SKU_MULTIPLIER_5090


def max_lease_usd() -> float:
    raw = os.environ.get("VAST_MAX_LEASE_USD", "")
    try:
        n = float(raw) if raw not in (None, "") else DEFAULT_MAX_LEASE_USD
    except (TypeError, ValueError):
        n = DEFAULT_MAX_LEASE_USD
    if n <= 0:
        return DEFAULT_MAX_LEASE_USD
    return min(MAX_LEASE_USD_CEILING, n)


def max_dph_usd() -> float:
    raw = os.environ.get("VAST_MAX_DPH_USD", "")
    try:
        n = float(raw) if raw not in (None, "") else DEFAULT_MAX_DPH_USD
    except (TypeError, ValueError):
        n = DEFAULT_MAX_DPH_USD
    if n <= 0:
        return DEFAULT_MAX_DPH_USD
    return n


def lease_episode_count(raw: int | None = None) -> int:
    if raw is None:
        env_raw = os.environ.get("GPU_BATCH_MIN_EPISODES") or os.environ.get("LEASE_EPISODE_COUNT")
        try:
            n = int(env_raw) if env_raw not in (None, "") else DEFAULT_LEASE_EPISODE_COUNT
        except (TypeError, ValueError):
            n = DEFAULT_LEASE_EPISODE_COUNT
    else:
        n = raw
    return max(1, int(n))


def score_offer(
    offer: dict,
    episode_count: int | None = None,
    max_usd: float | None = None,
    max_dph: float | None = None,
) -> dict[str, Any]:
    eps = lease_episode_count(episode_count)
    budget = max_usd if max_usd is not None else max_lease_usd()
    dph_cap = max_dph if max_dph is not None else max_dph_usd()
    duration_days = _duration_days(offer)
    inet_down = _inet_down(offer)
    reliability = _reliability(offer)
    dph = _dph(offer)
    ram = _gpu_ram_mb(offer)
    mult = sku_multiplier(_gpu_name(offer), ram)
    transfer_gb = WEIGHTS_GB + UPLOAD_GB_PER_EP * eps
    base: dict[str, Any] = {
        "version": OFFER_SCORE_VERSION,
        "episode_count": eps,
        "sku_multiplier": mult,
        "dph_total": dph if dph == dph else 0.0,
        "inet_down": inet_down,
        "effective_down": 0.0,
        "download_h": 0.0,
        "compute_h": 0.0,
        "lease_h": 0.0,
        "transfer_gb": transfer_gb,
        "usd_per_gb": 0.0,
        "egress_usd": 0.0,
        "egress_trap": False,
        "expected_total_usd": 0.0,
        "duration_days": duration_days,
        "reliability": reliability,
        "geolocation": _geolocation(offer),
    }
    if _is_cn_geolocation(offer):
        return {**base, "reject_reason": "cn"}
    cpu_ram = _cpu_ram_mb(offer)
    if cpu_ram > 0 and cpu_ram < MIN_CPU_RAM_MB:
        return {**base, "reject_reason": "cpu_ram", "cpu_ram": cpu_ram}
    cuda_vers = _cuda_version(offer)
    if cuda_vers > 0 and cuda_vers < MIN_CUDA_VERSION:
        return {**base, "reject_reason": "cuda_version", "cuda_max_good": cuda_vers}
    gpu_frac = _gpu_frac(offer)
    if gpu_frac == gpu_frac and 0 < gpu_frac < 1:
        return {**base, "reject_reason": "gpu_frac", "gpu_frac": gpu_frac}
    if duration_days < MIN_DURATION_DAYS:
        return {**base, "reject_reason": "duration"}
    if inet_down <= 0:
        return {**base, "reject_reason": "inet_down"}
    if dph != dph or dph < 0:
        return {**base, "reject_reason": "max_usd"}
    if dph > dph_cap:
        return {**base, "reject_reason": "max_dph"}

    effective_down = min(inet_down * INET_DOWN_FACTOR, EFFECTIVE_DOWN_CAP_MBPS)
    download_h = (WEIGHTS_GB * 8000) / effective_down / 3600
    compute_h = COMPUTE_HOURS_PER_EPISODE * eps * mult
    lease_h = BOOT_HOURS + download_h + compute_h
    usd_per_gb, egress_trap = resolve_egress_rate(offer, dph, inet_down)
    egress_usd = transfer_gb * usd_per_gb
    expected = lease_h * dph + egress_usd
    scored = {
        **base,
        "effective_down": effective_down,
        "download_h": download_h,
        "compute_h": compute_h,
        "lease_h": lease_h,
        "usd_per_gb": usd_per_gb,
        "egress_usd": egress_usd,
        "egress_trap": egress_trap,
        "expected_total_usd": expected,
    }
    if duration_days * 24 < lease_h + DURATION_BUFFER_HOURS:
        return {**scored, "reject_reason": "lease_window"}
    if egress_usd > DEFAULT_MAX_LEASE_USD:
        return {**scored, "reject_reason": "egress"}
    if expected > budget:
        return {**scored, "reject_reason": "egress" if egress_usd > 0 else "max_usd"}
    return scored


def _cost_within_tie(a: float, b: float) -> bool:
    lo = min(a, b)
    if lo <= 0:
        return True
    return abs(a - b) / lo <= COST_TIE_RATIO


def _duration_tie_key(score: dict[str, Any]) -> float:
    need_days = max(MIN_DURATION_DAYS, (score["lease_h"] + DURATION_BUFFER_HOURS) / 24)
    return min(score["duration_days"], need_days)


def compare_scored_offers(a: dict[str, Any], b: dict[str, Any]) -> int:
    if not _cost_within_tie(a["expected_total_usd"], b["expected_total_usd"]):
        return -1 if a["expected_total_usd"] < b["expected_total_usd"] else 1
    dph = a["dph_total"] - b["dph_total"]
    if dph:
        return -1 if dph < 0 else 1
    rel = b["reliability"] - a["reliability"]
    if rel:
        return -1 if rel < 0 else 1
    down = b["inet_down"] - a["inet_down"]
    if down:
        return -1 if down < 0 else 1
    dt = _duration_tie_key(b) - _duration_tie_key(a)
    return -1 if dt < 0 else (1 if dt > 0 else 0)


def classify_offers(offers: list[dict]) -> dict[str, list[dict]]:
    allow = gpu_allowlist()
    verified: list[dict] = []
    confirm: list[dict] = []
    rejected: list[dict] = []
    for offer in offers:
        if not offer:
            continue
        ram = _gpu_ram_mb(offer)
        if ram < MIN_GPU_RAM_MB:
            continue
        verdict = gpu_sku_verdict(_gpu_name(offer), allow)
        if verdict == "verified":
            verified.append(offer)
        elif verdict == "confirm":
            confirm.append(offer)
        else:
            rejected.append(offer)
    return {"verified": verified, "confirm": confirm, "rejected": rejected}


def pick_one_offer(
    offers: list[dict],
    episode_count: int | None = None,
    max_usd: float | None = None,
    max_dph: float | None = None,
    gpu_model_policy: str | None = None,
) -> dict | None:
    """Pick the cheapest expected-total offer that passes hard constraints."""
    policy = parse_gpu_model_policy(gpu_model_policy)
    pool = [o for o in classify_offers(offers)["verified"] if offer_matches_gpu_policy(o, policy)]
    dph_cap = max_dph if max_dph is not None else max_dph_usd()
    scored: list[tuple[dict, dict[str, Any]]] = []
    for offer in pool:
        score = score_offer(offer, episode_count, max_usd, dph_cap)
        if score.get("reject_reason"):
            continue
        scored.append((offer, score))
    if not scored:
        return None
    scored.sort(key=cmp_to_key(lambda row_a, row_b: compare_scored_offers(row_a[1], row_b[1])))
    return scored[0][0]


def offer_sample(offers: list[dict], n: int = 5) -> list[dict]:
    out = []
    for o in offers[:n]:
        out.append(
            {
                "id": o.get("id"),
                "gpu": _gpu_name(o),
                "dph": o.get("dph_total") or o.get("dph"),
                "gpu_ram": _gpu_ram_mb(o),
            }
        )
    return out

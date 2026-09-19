#!/usr/bin/env python3
"""GPU canary watchdog — 1-episode RTX 5090 smoke on digest-pinned Hub image.

Default is dry-run: search/score only, never PUT /asks/. Live mode requires
``--live``, an explicit confirmation phrase, and ``VAST_API_KEY``.

Always destroys the instance it created (SIGINT, success, failure, timeout).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import sys
import time
from urllib.error import HTTPError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TextIO

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from anime_factory.config import load_dotenv
from anime_factory.r2_paths import join_story
from gpu_worker.boot import evaluate_readiness
from gpu_worker.images import AGENT_IMAGE, parse_content_digest
from gpu_worker.offers import pick_one_offer, score_offer
from gpu_worker.vast_client import VastClient, VastSafetyError

# Re-export canary cap locally so tests patch one module.
CANARY_MAX_DPH_USD = 1.0
CANARY_EPISODE_COUNT = 1
CANARY_GPU_MODEL_POLICY = "5090"
CANARY_BACKENDS = ("h3", "longlive")
CANARY_H3_PROFILE = "h3-comfy-cu130-sm120"
CANARY_LONGLIVE_PROFILE = "longlive-nvfp4-sm120"
CANARY_GPU_PROFILE = CANARY_H3_PROFILE
CANARY_PROFILE_BY_BACKEND = {
    "h3": CANARY_H3_PROFILE,
    "longlive": CANARY_LONGLIVE_PROFILE,
}
CANARY_PUBLIC_H3_IMAGE = "ghcr.io/aetherneollc/anime-factory-worker-h3"
CANARY_PUBLIC_LONGLIVE_IMAGE = "ghcr.io/aetherneollc/anime-factory-worker-longlive"
DEFAULT_STORY_ID = "story-canary-7516b66"
DEFAULT_CANARY_SHOT_ID = "s001"
DEFAULT_EPISODE = "EP001"
LIVE_CONFIRM_PHRASE = "I-UNDERSTAND-GPU-CANARY-SPEND"

CANARY_MAX_WALL_S = 90 * 60
CANARY_MAX_SPEND_USD = 1.0
CANARY_OFFER_SCORE_BUDGET_USD = 15.0
DEFAULT_MAX_WALL_S = CANARY_MAX_WALL_S
DEFAULT_MAX_SPEND_USD = CANARY_MAX_SPEND_USD
DEFAULT_POLL_INTERVAL_S = 10.0
SPEND_CLEANUP_MARGIN_S = 60.0
CANARY_LEASE_LABEL_PREFIX = "anime-factory-canary-"
INSTANCE_RESOLVE_MAX_ATTEMPTS = 5
INSTANCE_RESOLVE_RETRY_S = 2.0
DELETE_CONFIRM_MAX_ATTEMPTS = 5
V1_GONE_STREAK = 3
# After instance_gone / OCI create failure, try a different machine. Cap is
# still $1; spent USD accumulates across retries.
CANARY_GONE_RETRIES = 2
# Match packages/cf-control seasons.ts PULL_STUCK_MS — Vast `loading` is docker pull.
CANARY_PULL_STATES = frozenset({"loading", "creating", "created", "pending"})
CANARY_PULL_STALL_S = 12 * 60
# 7.2GB Hub pull at ~40Mbps is ~25 min, plus Vast ssh-wrap apt; do not kill a host still progressing.
CANARY_PULL_MAX_S = 45 * 60
CANARY_PULL_ERROR_RE = re.compile(
    r"tls handshake|net/http:\s*tls|\beof\b|i/o timeout|context deadline|"
    r"error pulling|failed to pull|image pull|connection reset|"
    r"connection refused|no such host|unauthorized|authentication required|"
    r"pull(?:ing)? denied|access denied|manifest unknown|repository not found|"
    r"service unavailable|registry.*unavailable|"
    r"oci runtime|runtime create failed|failed to create shim|"
    r"failed to create task|nvidia-container",
    re.I,
)
# Vast ssh/jupyter wrap prints apt postinst noise ("policy-rc.d denied") while still booting.
CANARY_WRAP_NOISE_RE = re.compile(
    r"policy-rc\.d denied|#\d+\s+\d|Creating SSH|Setting up polkitd|invoke-rc\.d",
    re.I,
)
CANARY_OCI_ERROR_RE = re.compile(
    r"oci runtime|runtime create failed|failed to create shim|"
    r"failed to create task|nvidia-container",
    re.I,
)
CANARY_RETRY_POLL_REASONS = frozenset({"instance_gone"})
CANARY_EXCLUDE_TTL_S = 24 * 3600

ISOLATED_ENV_FORBIDDEN_KEYS = frozenset(
    {
        "CONTROL_PLANE_URL",
        "STUDIO_USER",
        "STUDIO_PASSWORD",
        "STUDIO_AGENT_KEY",
        "CLOUDFLARE_TUNNEL_TOKEN",
        "TORCH_INDEX_URL",
    }
)
DELETE_MAX_RETRIES = 5
DELETE_RETRY_BACKOFF_S = 2.0

_DIGEST_SUFFIX_RE = r"@sha256:([0-9a-f]{64})$"
CANARY_IMAGE_REPOS: dict[str, tuple[str, ...]] = {
    "h3": (CANARY_PUBLIC_H3_IMAGE, AGENT_IMAGE),
    "longlive": (CANARY_PUBLIC_LONGLIVE_IMAGE,),
}
CANARY_IMAGE_REF_RES: dict[str, tuple[re.Pattern[str], ...]] = {
    backend: tuple(
        re.compile(rf"^{re.escape(repo)}{_DIGEST_SUFFIX_RE}$", re.IGNORECASE)
        for repo in repos
    )
    for backend, repos in CANARY_IMAGE_REPOS.items()
}

SECRET_ENV_KEYS = frozenset(
    {
        "VAST_API_KEY",
        "R2_SECRET_ACCESS_KEY",
        "R2_ACCESS_KEY_ID",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "STUDIO_AGENT_KEY",
        "STUDIO_PASSWORD",
        "CLOUDFLARE_TUNNEL_TOKEN",
    }
)

SENSITIVE_SUBSTRINGS = (
    "secret",
    "password",
    "token",
    "api_key",
    "access_key",
)


class CanaryConfigError(ValueError):
    pass


class CanaryImageError(CanaryConfigError):
    pass


def normalize_canary_backend(backend: str | None) -> str:
    raw = str(backend or "h3").strip().lower()
    if raw not in CANARY_BACKENDS:
        raise CanaryConfigError(f"backend must be one of {CANARY_BACKENDS}; got {backend!r}")
    return raw


def _image_repo_for_ref(image_ref: str) -> str | None:
    raw = (image_ref or "").strip()
    at = raw.rfind("@sha256:")
    if at < 0:
        return None
    return raw[:at]


def validate_lease_image_ref(image_ref: str, backend: str = "h3") -> tuple[str, str]:
    """Require digest-pinned image for the selected backend. Reject tags and mismatches."""
    wanted = normalize_canary_backend(backend)
    raw = (image_ref or "").strip()
    if not raw:
        raise CanaryImageError("image ref required (digest pin, no tag)")
    if ":" in raw.rsplit("/", 1)[-1] and "@sha256:" not in raw.lower():
        raise CanaryImageError(f"refuses floating tag image: {raw!r}")
    for other_backend, patterns in CANARY_IMAGE_REF_RES.items():
        if other_backend == wanted:
            continue
        for pattern in patterns:
            if pattern.match(raw):
                raise CanaryImageError(
                    f"image/backend mismatch: {raw!r} is for {other_backend!r}, not {wanted!r}"
                )
    matched_repo: str | None = None
    hexpart: str | None = None
    for pattern in CANARY_IMAGE_REF_RES[wanted]:
        match = pattern.match(raw)
        if match:
            matched_repo = _image_repo_for_ref(raw)
            hexpart = match.group(1).lower()
            break
    if not hexpart or not matched_repo:
        allowed = ", ".join(CANARY_IMAGE_REPOS[wanted])
        raise CanaryImageError(
            f"image must be one of {allowed}@sha256:<64 hex> for backend {wanted!r}; got {raw!r}"
        )
    digest = f"sha256:{hexpart}"
    return f"{matched_repo}@sha256:{hexpart}", digest


def clamp_canary_max_dph(requested: float | None = None) -> float:
    """Hard cap $1/h — env/CLI cannot raise the canary ceiling."""
    if requested is None:
        env_raw = os.environ.get("VAST_MAX_DPH_USD", "")
        try:
            requested = float(env_raw) if env_raw not in (None, "") else CANARY_MAX_DPH_USD
        except (TypeError, ValueError):
            requested = CANARY_MAX_DPH_USD
    if requested <= 0:
        return CANARY_MAX_DPH_USD
    return min(CANARY_MAX_DPH_USD, float(requested))


def clamp_canary_max_spend(requested: float | None = None) -> float:
    """Hard cap $1 — env/CLI cannot raise the canary spend ceiling."""
    if requested is None:
        env_raw = os.environ.get("GPU_CANARY_MAX_SPEND_USD") or os.environ.get("VAST_MAX_LEASE_USD", "")
        try:
            requested = float(env_raw) if env_raw not in (None, "") else CANARY_MAX_SPEND_USD
        except (TypeError, ValueError):
            requested = CANARY_MAX_SPEND_USD
    if requested <= 0:
        return CANARY_MAX_SPEND_USD
    return min(CANARY_MAX_SPEND_USD, float(requested))


def clamp_canary_max_wall_s(requested: float | None = None) -> float:
    """Hard cap 90 minutes — env/CLI cannot raise the canary wall-clock ceiling."""
    if requested is None:
        env_raw = os.environ.get("GPU_CANARY_MAX_WALL_MINUTES", "")
        try:
            requested = float(env_raw) * 60.0 if env_raw not in (None, "") else CANARY_MAX_WALL_S
        except (TypeError, ValueError):
            requested = CANARY_MAX_WALL_S
    if requested <= 0:
        return CANARY_MAX_WALL_S
    return min(CANARY_MAX_WALL_S, float(requested))


def redact_value(key: str, value: Any) -> Any:
    key_l = key.lower()
    if key in SECRET_ENV_KEYS or any(s in key_l for s in SENSITIVE_SUBSTRINGS):
        if value in (None, ""):
            return value
        text = str(value)
        if len(text) <= 4:
            return "***"
        return text[:2] + "***" + text[-2:]
    return value


def redact_mapping(data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {}
    return {k: redact_value(k, v) for k, v in data.items()}


def redact_for_log(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: redact_for_log(redact_value(k, v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_for_log(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(redact_for_log(v) for v in obj)
    return obj


def build_canary_env(*, image_digest: str, story_id: str, backend: str = "h3") -> dict[str, str]:
    """Isolated canary env: R2 + AF_ONCE only — never production control-plane keys."""
    video_backend = normalize_canary_backend(backend)
    profile = CANARY_PROFILE_BY_BACKEND[video_backend]
    env: dict[str, str] = {
        "AF_STORY_ID": story_id,
        "AF_EPISODE": DEFAULT_EPISODE,
        "AF_ONCE": "1",
        "AF_VIDEO_BACKEND": video_backend,
        "AF_IMAGE_CAPABILITY": video_backend,
        "VAST_ALLOW_REPLACE": "0",
        "VAST_DRY_RUN": "0",
        "ANIME_FACTORY_LIVE_VAST": "1",
        "AF_EXPECTED_IMAGE_DIGEST": image_digest,
        "AF_REPORTED_IMAGE_DIGEST": image_digest,
        "AF_GPU_PROFILE": profile,
        "AF_START_COMFY": "1",
        "ANIME_FACTORY_GPU_STILLS": "1",
        "AF_SKIP_PRE_GPU": "1",
        "COMFYUI_BASE_URL": "http://127.0.0.1:8199",
        "R2_ENDPOINT": os.environ.get("R2_ENDPOINT") or "",
        "R2_BUCKET": os.environ.get("R2_BUCKET") or "",
        "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID") or "",
        "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY") or "",
    }
    if video_backend == "longlive":
        env["AF_VIDEO_BACKEND_LOCKED"] = "longlive"
    hf = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if hf:
        env["HF_TOKEN"] = hf
    for forbidden in ISOLATED_ENV_FORBIDDEN_KEYS:
        env.pop(forbidden, None)
    return env


def inject_lease_watchdog_env(
    lease_env: dict[str, str],
    *,
    offer: dict,
    api_key: str,
    max_spend_usd: float,
    max_wall_s: float,
) -> dict[str, str]:
    """Add Vast self-destruct keys after offer selection (never in build_canary_env)."""
    merged = dict(lease_env)
    if api_key:
        merged["VAST_API_KEY"] = api_key
    dph = float(offer.get("dph_total") or offer.get("dph") or 0.0)
    merged["VAST_LEASE_HOURLY_USD"] = str(dph)
    merged["VAST_MAX_LEASE_MINUTES"] = str(int(max(1, max_wall_s / 60.0)))
    merged["VAST_MAX_LEASE_USD"] = str(max_spend_usd)
    merged["VAST_WATCH_USD"] = str(max_spend_usd)
    return merged


def r2_canary_input_keys(
    story_id: str,
    episode: str = DEFAULT_EPISODE,
    backend: str = "h3",
) -> tuple[str, ...]:
    """R2 text/plan inputs required before a live canary lease."""
    keys = [
        join_story(story_id, "bible/period.md"),
        join_story(story_id, "bible/world.md"),
        join_story(story_id, "canon/wiki/characters/hero.md"),
        join_story(story_id, "canon/wiki/locations/loc_dorm.md"),
        join_story(story_id, f"episodes/{episode}/audio/tts_manifest.json"),
        join_story(story_id, f"episodes/{episode}/board.json"),
        join_story(story_id, f"episodes/{episode}/script.json"),
    ]
    if normalize_canary_backend(backend) == "longlive":
        keys.append(join_story(story_id, "factory.json"))
    return tuple(keys)


def r2_canary_output_keys(story_id: str, episode: str = DEFAULT_EPISODE) -> tuple[str, ...]:
    """Fixed generated artifacts that must exist for canary completion."""
    from anime_factory.r2_paths import character_asset_rel, scene_asset_rel

    return (
        join_story(story_id, "assets/index.json"),
        join_story(story_id, character_asset_rel("hero", "sheet_front.png")),
        join_story(story_id, character_asset_rel("hero", "sheet_side.png")),
        join_story(story_id, character_asset_rel("hero", "sheet_back.png")),
        join_story(story_id, character_asset_rel("hero", "sheet_turnaround.png")),
        join_story(story_id, scene_asset_rel("loc_dorm", "plate_base.png")),
        join_story(story_id, f"episodes/{episode}/keyframes/s001/last.png"),
        join_story(story_id, "story.sqlite"),
    )


GENERATION_MP4_RE = re.compile(r"generation-\d+\.mp4$")
LEGACY_SHOT_MP4_RE = re.compile(r"v\d+\.mp4$")


def _r2_key_nonempty(key: str) -> bool:
    try:
        from anime_factory.r2_client import list_prefix
    except ImportError:
        return False
    for item in list_prefix(key, max_keys=5):
        if str(item.get("key") or "") == key and int(item.get("size") or 0) > 0:
            return True
    return False


def r2_generation_shot_prefix(story_id: str, shot_id: str = "s001") -> str:
    """H3 writer stores clips at stories/<id>/shots/<shot>/generation-*.mp4."""
    return join_story(story_id, f"shots/{shot_id}/")


def _r2_shot_video_matches(key: str, backend: str) -> bool:
    if normalize_canary_backend(backend) == "longlive":
        return bool(GENERATION_MP4_RE.search(key) or LEGACY_SHOT_MP4_RE.search(key))
    return bool(GENERATION_MP4_RE.search(key))


def _r2_has_shot_video_mp4(
    story_id: str,
    episode: str = DEFAULT_EPISODE,
    shot_id: str = DEFAULT_CANARY_SHOT_ID,
    backend: str = "h3",
) -> bool:
    del episode  # generation keys are story-scoped, not under episodes/
    try:
        from anime_factory.r2_client import list_prefix
    except ImportError:
        return False
    prefix = r2_generation_shot_prefix(story_id, shot_id)
    for item in list_prefix(prefix, max_keys=50):
        key = str(item.get("key") or "")
        if _r2_shot_video_matches(key, backend) and int(item.get("size") or 0) > 0:
            return True
    return False


def r2_canary_complete(
    story_id: str,
    episode: str = DEFAULT_EPISODE,
    backend: str = "h3",
) -> bool:
    if not all(_r2_key_nonempty(key) for key in r2_canary_output_keys(story_id, episode)):
        return False
    if not _r2_has_shot_video_mp4(story_id, episode, backend=backend):
        return False
    return all(_r2_key_nonempty(key) for key in r2_canary_input_keys(story_id, episode, backend))


def _r2_list_keys(prefix: str, max_keys: int = 200) -> list[str]:
    try:
        from anime_factory.r2_client import list_prefix
    except ImportError:
        return []
    return [
        str(item.get("key") or "")
        for item in list_prefix(prefix, max_keys=max_keys)
        if str(item.get("key") or "") and int(item.get("size") or 0) > 0
    ]


def _r2_probe_boto3() -> str | None:
    from anime_factory.r2_client import _client

    client, cfg = _client()
    if client is None:
        return "r2_client_unavailable"
    try:
        client.list_objects_v2(Bucket=cfg["bucket"], MaxKeys=1)
    except Exception as exc:  # noqa: BLE001
        return f"r2_client_probe_failed:{exc.__class__.__name__}"
    return None


def r2_output_artifacts_present(
    story_id: str,
    episode: str = DEFAULT_EPISODE,
    backend: str = "h3",
) -> list[str]:
    """Return generation artifacts that forbid a fresh canary lease.

    Pre-seeded still assets (hero sheets, loc plates, index.json, story.sqlite)
    are allowed so a retry can skip the Flux still pipeline and go straight to H3.
    """
    found: list[str] = []
    if _r2_has_shot_video_mp4(story_id, episode, backend=backend):
        if normalize_canary_backend(backend) == "longlive":
            found.append(r2_generation_shot_prefix(story_id, DEFAULT_CANARY_SHOT_ID) + "{generation-*,v*.mp4}")
        else:
            found.append(r2_generation_shot_prefix(story_id, DEFAULT_CANARY_SHOT_ID) + "generation-*.mp4")
    keyframes_prefix = join_story(story_id, f"episodes/{episode}/keyframes/s001/")
    found.extend(_r2_list_keys(keyframes_prefix, max_keys=50))
    final_prefix = join_story(story_id, f"episodes/{episode}/final/")
    found.extend(_r2_list_keys(final_prefix, max_keys=50))
    return found


def r2_live_preflight(
    story_id: str,
    episode: str = DEFAULT_EPISODE,
    backend: str = "h3",
) -> dict[str, Any]:
    endpoint = (os.environ.get("R2_ENDPOINT") or "").strip()
    access_key = (os.environ.get("R2_ACCESS_KEY_ID") or "").strip()
    secret_key = (os.environ.get("R2_SECRET_ACCESS_KEY") or "").strip()
    if not endpoint or not access_key or not secret_key:
        return {
            "ok": False,
            "reason": "r2_creds_missing",
            "missing_env": [
                name
                for name, val in (
                    ("R2_ENDPOINT", endpoint),
                    ("R2_ACCESS_KEY_ID", access_key),
                    ("R2_SECRET_ACCESS_KEY", secret_key),
                )
                if not val
            ],
        }
    probe_err = _r2_probe_boto3()
    if probe_err:
        return {"ok": False, "reason": probe_err}
    missing_inputs = [
        key for key in r2_canary_input_keys(story_id, episode, backend) if not _r2_key_nonempty(key)
    ]
    if missing_inputs:
        return {"ok": False, "reason": "r2_inputs_missing", "missing": missing_inputs}
    outputs = r2_output_artifacts_present(story_id, episode, backend)
    if outputs:
        return {"ok": False, "reason": "r2_outputs_present", "artifacts": outputs}
    return {
        "ok": True,
        "reason": "ready",
        "input_keys": list(r2_canary_input_keys(story_id, episode, backend)),
    }


def make_canary_lease_label() -> str:
    return f"{CANARY_LEASE_LABEL_PREFIX}{secrets.token_hex(4)}"


def _extract_scalar_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("id", "instance_id", "contract_id", "new_contract"):
            inner = value.get(key)
            parsed = _extract_scalar_id(inner)
            if parsed:
                return parsed
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def parse_lease_instance_id(result: dict | None) -> str | None:
    if not isinstance(result, dict):
        return None
    for key in ("new_contract", "id", "instance_id", "contract_id"):
        parsed = _extract_scalar_id(result.get(key))
        if parsed:
            return parsed
    return None


def v1_instance_row(payload: dict | None, instance_id: str) -> dict | None:
    if not isinstance(payload, dict):
        return None
    rows = payload.get("instances")
    if rows is None:
        rows = payload.get("data")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        parsed = _extract_scalar_id(row.get("id")) or _extract_scalar_id(row.get("instance_id"))
        if parsed == instance_id:
            return row
    return None


def v1_instance_state(row: dict | None) -> str:
    if not isinstance(row, dict):
        return "unknown"
    for key in ("actual_status", "cur_state", "status"):
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return "unknown"


def v1_status_msg(row: dict | None, payload: dict | None = None) -> str:
    for src in (row, payload):
        if not isinstance(src, dict):
            continue
        for key in ("status_msg", "status_message", "error"):
            val = src.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()
    return ""


def v1_machine_id(row: dict | None) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("machine_id") or row.get("machineId") or "").strip()


def offer_machine_id(offer: dict | None) -> str:
    if not isinstance(offer, dict):
        return ""
    return str(offer.get("machine_id") or offer.get("machineId") or "").strip()


def canary_exclude_path() -> Path:
    raw = (os.environ.get("GPU_CANARY_EXCLUDE_MACHINES") or "").strip()
    if raw:
        return Path(raw)
    return Path.home() / ".cache" / "anime-factory" / "canary-exclude-machines.json"


def load_excluded_machines(*, now: float | None = None) -> set[str]:
    stamp = time.time() if now is None else now
    path = canary_exclude_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return set()
    if not isinstance(data, dict):
        return set()
    kept: set[str] = set()
    for mid, ts in data.items():
        try:
            stamped = float(ts)
        except (TypeError, ValueError):
            continue
        key = str(mid).strip()
        if key and stamp - stamped < CANARY_EXCLUDE_TTL_S:
            kept.add(key)
    return kept


def persist_excluded_machines(ids: set[str], *, now: float | None = None) -> None:
    stamp = time.time() if now is None else now
    path = canary_exclude_path()
    existing: dict[str, float] = {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for mid, ts in raw.items():
                try:
                    existing[str(mid).strip()] = float(ts)
                except (TypeError, ValueError):
                    continue
    except (OSError, json.JSONDecodeError, TypeError):
        existing = {}
    out: dict[str, float] = {}
    for mid, ts in existing.items():
        if mid in ids and stamp - ts < CANARY_EXCLUDE_TTL_S:
            out[mid] = ts
    for mid in ids:
        key = str(mid).strip()
        if key and key not in out:
            out[key] = stamp
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return


def is_pull_state(state: str | None) -> bool:
    return str(state or "").strip().lower() in CANARY_PULL_STATES


def instance_ids_from_v1(payload: dict | None) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    rows = payload.get("instances")
    if rows is None:
        rows = payload.get("data")
    if not isinstance(rows, list):
        return set()
    out: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        parsed = _extract_scalar_id(row.get("id")) or _extract_scalar_id(row.get("instance_id"))
        if parsed:
            out.add(parsed)
    return out


def find_instance_id_by_label(payload: dict | None, label: str, baseline: set[str]) -> str | None:
    if not isinstance(payload, dict):
        return None
    rows = payload.get("instances") or payload.get("data") or []
    matches: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("label") or "") != label:
            continue
        parsed = _extract_scalar_id(row.get("id")) or _extract_scalar_id(row.get("instance_id"))
        if parsed and parsed not in baseline:
            matches.append(parsed)
    if len(matches) == 1:
        return matches[0]
    return None


def vast_delete_accepted(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("dry_run"):
        return True
    if result.get("success") is False or result.get("ok") is False:
        return False
    if result.get("error"):
        return False
    msg = str(result.get("msg") or result.get("message") or "").strip().lower()
    if msg and any(token in msg for token in ("error", "fail", "denied", "forbidden")):
        return False
    return True


@dataclass
class CanaryConfig:
    live: bool = False
    confirm: str = ""
    api_key: str = ""
    image_ref: str = ""
    backend: str = "h3"
    story_id: str = DEFAULT_STORY_ID
    max_wall_s: float = DEFAULT_MAX_WALL_S
    max_spend_usd: float = DEFAULT_MAX_SPEND_USD
    max_dph: float = CANARY_MAX_DPH_USD
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    r2_completion: bool = True
    ssh_completion: bool = False


@dataclass
class CanaryWatchdog:
    config: CanaryConfig
    client: VastClient | None = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    r2_checker: Callable[[str, str], bool] | None = None
    progress_stream: TextIO | None = None
    _allowlist: set[str] = field(default_factory=set, init=False)
    _instance_id: str | None = field(default=None, init=False)
    _lease_label: str | None = field(default=None, init=False)
    _started_at: float | None = field(default=None, init=False)
    _abort: bool = field(default=False, init=False)
    _delete_errors: list[str] = field(default_factory=list, init=False)
    _v1_missing: int = field(default=0, init=False)
    _exclude_offer_ids: set[str] = field(default_factory=set, init=False)
    _exclude_machine_ids: set[str] = field(default_factory=set, init=False)
    _spent_usd: float = field(default=0.0, init=False)
    _lease_started_at: float | None = field(default=None, init=False)
    _pull_since: float | None = field(default=None, init=False)
    _pull_msg: str = field(default="", init=False)
    _pull_msg_changed_at: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.r2_checker is None:
            backend = normalize_canary_backend(self.config.backend)
            self.r2_checker = lambda sid, ep, bk=backend: r2_canary_complete(sid, ep, bk)
        if self.client is None:
            key = self.config.api_key or os.environ.get("VAST_API_KEY") or ""
            # Real POST /bundles/ search even in CLI dry-run; run() never PUTs unless --live.
            self.client = VastClient(api_key=key, dry_run=False)
        self._exclude_machine_ids.update(load_excluded_machines())

    def validate_config(self) -> tuple[str, str]:
        if self.config.live:
            if not (self.config.api_key or os.environ.get("VAST_API_KEY")):
                raise CanaryConfigError("live mode requires VAST_API_KEY")
            if self.config.confirm.strip() != LIVE_CONFIRM_PHRASE:
                raise CanaryConfigError(
                    f"live mode requires --confirm {LIVE_CONFIRM_PHRASE!r}"
                )
            if self.config.ssh_completion:
                raise CanaryConfigError("live mode forbids --ssh-completion")
            if not self.config.r2_completion:
                raise CanaryConfigError("live mode forbids --no-r2-completion")
            if os.environ.get("VAST_ALLOW_REPLACE", "0") not in ("0", "false", "False"):
                raise CanaryConfigError("VAST_ALLOW_REPLACE must be 0 for canary")
        normalize_canary_backend(self.config.backend)
        image_ref, digest = validate_lease_image_ref(self.config.image_ref, self.config.backend)
        self.config.max_dph = clamp_canary_max_dph(self.config.max_dph)
        self.config.max_spend_usd = clamp_canary_max_spend(self.config.max_spend_usd)
        self.config.max_wall_s = clamp_canary_max_wall_s(self.config.max_wall_s)
        parsed = parse_content_digest(digest)
        if not parsed:
            raise CanaryImageError(f"invalid digest after validation: {digest!r}")
        return image_ref, parsed

    def _offer_list(self) -> list[dict]:
        body = self.client.search_offers()
        if isinstance(body, dict):
            return list(body.get("offers") or [])
        return []

    def select_offer(self, offers: list[dict]) -> dict | None:
        pool = []
        for offer in offers:
            oid = str(offer.get("id") or "")
            if oid and oid in self._exclude_offer_ids:
                continue
            machine = offer_machine_id(offer)
            if machine and machine in self._exclude_machine_ids:
                continue
            pool.append(offer)
        return pick_one_offer(
            pool,
            episode_count=CANARY_EPISODE_COUNT,
            max_usd=CANARY_OFFER_SCORE_BUDGET_USD,
            max_dph=self.config.max_dph,
            gpu_model_policy=CANARY_GPU_MODEL_POLICY,
        )

    def _estimate_running_cost(self, offer: dict, elapsed_h: float) -> float:
        dph = float(offer.get("dph_total") or offer.get("dph") or 0.0)
        return max(0.0, elapsed_h * dph)

    def _effective_spend_cap(self, offer: dict) -> float:
        dph = float(offer.get("dph_total") or offer.get("dph") or 0.0)
        margin_usd = (dph / 3600.0) * SPEND_CLEANUP_MARGIN_S
        return max(0.0, self.config.max_spend_usd - margin_usd)

    def _register_instance(self, instance_id: str) -> None:
        self._instance_id = str(instance_id)
        self._allowlist.add(self._instance_id)
        self._lease_started_at = self.clock()
        self._v1_missing = 0
        self._pull_since = None
        self._pull_msg = ""
        self._pull_msg_changed_at = None

    def _running_cost(self, offer: dict) -> float:
        started = self._lease_started_at if self._lease_started_at is not None else self._started_at
        if started is None:
            started = self.clock()
        elapsed_h = max(0.0, (self.clock() - started) / 3600.0)
        return self._spent_usd + self._estimate_running_cost(offer, elapsed_h)

    def _note_excluded_machine(self, offer: dict, v1_row: dict | None = None, extra: dict | None = None) -> None:
        before = set(self._exclude_machine_ids)
        for src in (v1_row, extra, offer):
            if not isinstance(src, dict):
                continue
            mid = v1_machine_id(src) or offer_machine_id(src)
            if mid:
                self._exclude_machine_ids.add(mid)
        if self._exclude_machine_ids != before:
            persist_excluded_machines(self._exclude_machine_ids)

    def _pull_stuck_outcome(
        self,
        *,
        now: float,
        offer: dict,
        last_payload: dict | None,
        v1_row: dict | None,
        state: str,
        msg: str,
    ) -> dict | None:
        if not is_pull_state(state):
            self._pull_since = None
            self._pull_msg = ""
            self._pull_msg_changed_at = None
            return None
        if self._pull_since is None:
            self._pull_since = now
        if self._pull_msg_changed_at is None:
            self._pull_msg_changed_at = now
        if msg != self._pull_msg:
            self._pull_msg = msg
            self._pull_msg_changed_at = now
        if msg and CANARY_PULL_ERROR_RE.search(msg) and not CANARY_WRAP_NOISE_RE.search(msg):
            self._note_excluded_machine(offer, v1_row, last_payload)
            kind = "oci" if CANARY_OCI_ERROR_RE.search(msg) else "tls"
            return {
                "status": "failed",
                "reason": "instance_gone",
                "pull_stuck": kind,
                "status_msg": msg,
                "machine_id": v1_machine_id(v1_row) or offer_machine_id(offer),
                "last": last_payload,
            }
        stall_from = self._pull_msg_changed_at
        if stall_from is None:
            stall_from = self._pull_since
        if stall_from is None:
            stall_from = now
        stalled_s = now - stall_from
        loading_from = self._pull_since if self._pull_since is not None else now
        loading_s = now - loading_from
        if stalled_s >= CANARY_PULL_STALL_S:
            self._note_excluded_machine(offer, v1_row, last_payload)
            return {
                "status": "failed",
                "reason": "instance_gone",
                "pull_stuck": "status_msg_stalled",
                "status_msg": msg,
                "stalled_s": round(stalled_s, 1),
                "machine_id": v1_machine_id(v1_row) or offer_machine_id(offer),
                "last": last_payload,
            }
        if loading_s >= CANARY_PULL_MAX_S:
            self._note_excluded_machine(offer, v1_row, last_payload)
            return {
                "status": "failed",
                "reason": "instance_gone",
                "pull_stuck": "loading_exceeded",
                "status_msg": msg,
                "stalled_s": round(loading_s, 1),
                "machine_id": v1_machine_id(v1_row) or offer_machine_id(offer),
                "last": last_payload,
            }
        return None

    def _v1_baseline_ids(self) -> set[str]:
        try:
            payload = self.client.list_instances_v1()
        except Exception:  # noqa: BLE001
            return set()
        return instance_ids_from_v1(payload)

    def _resolve_instance_id(self, lease_result: dict, baseline: set[str], label: str) -> str | None:
        parsed = parse_lease_instance_id(lease_result)
        if parsed:
            return parsed
        for attempt in range(1, INSTANCE_RESOLVE_MAX_ATTEMPTS + 1):
            try:
                payload = self.client.list_instances_v1()
            except Exception:  # noqa: BLE001
                payload = None
            found = find_instance_id_by_label(payload, label, baseline)
            if found:
                return found
            if attempt < INSTANCE_RESOLVE_MAX_ATTEMPTS:
                self.sleep(INSTANCE_RESOLVE_RETRY_S)
        return None

    def _instance_visible_v1(self, instance_id: str) -> bool:
        try:
            payload = self.client.list_instances_v1()
        except Exception:  # noqa: BLE001
            return True
        return instance_id in instance_ids_from_v1(payload)

    def _emit_poll_progress(
        self,
        offer: dict,
        last_payload: dict | None,
        *,
        r2_done: bool,
        v1_row: dict | None = None,
    ) -> None:
        stream = self.progress_stream if self.progress_stream is not None else sys.stderr
        elapsed_from = self._lease_started_at if self._lease_started_at is not None else self._started_at
        if elapsed_from is None:
            elapsed_from = self.clock()
        elapsed_s = max(0.0, self.clock() - elapsed_from)
        cost = self._running_cost(offer)
        if v1_row:
            state = v1_instance_state(v1_row)
        else:
            state = (
                (last_payload or {}).get("cur_state")
                or (last_payload or {}).get("status")
                or "unknown"
            )
        line = {
            "canary_progress": True,
            "instance_id": self._instance_id,
            "state": state,
            "elapsed_s": round(elapsed_s, 1),
            "estimated_usd": round(cost, 4),
            "r2_complete": r2_done,
            "status_msg": v1_status_msg(v1_row, last_payload)[:180],
            "machine_id": v1_machine_id(v1_row) or offer_machine_id(offer),
            "gpu_util": (v1_row or {}).get("gpu_util") if v1_row else None,
            "cpu_util": (v1_row or {}).get("cpu_util") if v1_row else None,
            "disk_usage": (v1_row or {}).get("disk_usage") if v1_row else None,
        }
        print(json.dumps(redact_for_log(line), ensure_ascii=False), file=stream, flush=True)

    def lease_selected(self, offer: dict, *, image_ref: str, lease_env: dict[str, str]) -> dict:
        offer_id = str(offer.get("id") or "")
        label = make_canary_lease_label()
        self._lease_label = label
        baseline = self._v1_baseline_ids()
        body = self.client.lease_body(image=image_ref, env=lease_env, extra={"label": label})
        try:
            result = self.client.lease(offer_id, body)
        except HTTPError as exc:
            if int(getattr(exc, "code", 0) or 0) in {400, 404, 409, 410}:
                self._exclude_offer_ids.add(offer_id)
                return {
                    "action": "stale_offer",
                    "offer_id": offer_id,
                    "put_asks": True,
                    "error": str(exc),
                }
            raise
        if result.get("dry_run") or result.get("skipped"):
            return {
                "action": "dry_run_skip_lease",
                "offer_id": offer_id,
                "put_asks": False,
                "lease": result,
            }
        instance_id = self._resolve_instance_id(result, baseline, label)
        if not instance_id:
            try:
                payload = self.client.list_instances_v1()
            except Exception:  # noqa: BLE001
                payload = None
            labeled = find_instance_id_by_label(payload, label, set())
            if labeled:
                self._register_instance(labeled)
                destroyed = self.destroy_registered()
                if labeled in self._allowlist:
                    self._allowlist.discard(labeled)
                self._instance_id = None
                return {
                    "action": "cleanup_unknown",
                    "reason": "instance_id_unresolved",
                    "offer_id": offer_id,
                    "lease_label": label,
                    "put_asks": True,
                    "lease": result,
                    "destroy": destroyed,
                    "recovered_id": labeled,
                }
            return {
                "action": "cleanup_unknown",
                "reason": "instance_id_unresolved",
                "offer_id": offer_id,
                "lease_label": label,
                "put_asks": True,
                "lease": result,
                "destroy": {"skipped": True, "reason": "no_registered_instance"},
            }
        self._register_instance(instance_id)
        return {
            "action": "leased",
            "offer_id": offer_id,
            "instance_id": instance_id,
            "lease_label": label,
            "put_asks": True,
            "lease": result,
        }

    def _wall_deadline(self) -> float:
        """Per-lease wall. Spent USD still accumulates across gone-retries."""
        started = self._lease_started_at if self._lease_started_at is not None else self._started_at
        if started is None:
            started = self.clock()
        return started + self.config.max_wall_s

    def poll_until_done(self, offer: dict) -> dict:
        assert self._instance_id
        deadline = self._wall_deadline()
        last_payload: dict | None = None
        self._pull_since = None
        self._pull_msg = ""
        self._pull_msg_changed_at = None
        while not self._abort:
            now = self.clock()
            if now >= deadline:
                return {"status": "timeout", "reason": "wall_clock", "last": last_payload}
            spend_cap = self._effective_spend_cap(offer)
            cost = self._running_cost(offer)
            try:
                last_payload = self.client.get_instance(self._instance_id)
            except Exception as exc:  # noqa: BLE001
                last_payload = {"error": str(exc)}
            if self._abort:
                break
            v1_row: dict | None = None
            try:
                v1_payload = self.client.list_instances_v1()
                if self._instance_id not in instance_ids_from_v1(v1_payload):
                    self._v1_missing += 1
                else:
                    self._v1_missing = 0
                    v1_row = v1_instance_row(v1_payload, self._instance_id)
            except Exception:  # noqa: BLE001
                pass
            if self._abort:
                break
            r2_done = bool(
                self.config.r2_completion and self.r2_checker(self.config.story_id, DEFAULT_EPISODE)
            )
            self._emit_poll_progress(offer, last_payload, r2_done=r2_done, v1_row=v1_row)
            if cost >= spend_cap:
                return {
                    "status": "timeout",
                    "reason": "max_spend",
                    "estimated_usd": cost,
                    "spend_cap_usd": spend_cap,
                    "last": last_payload,
                }
            if r2_done:
                return {"status": "success", "reason": "r2_canary", "last": last_payload}
            if self._v1_missing >= V1_GONE_STREAK:
                self._note_excluded_machine(offer, v1_row, last_payload)
                return {
                    "status": "failed",
                    "reason": "instance_gone",
                    "last": last_payload,
                    "v1_missing": self._v1_missing,
                }
            state = v1_instance_state(v1_row) if v1_row else str(
                (last_payload or {}).get("actual_status")
                or (last_payload or {}).get("cur_state")
                or (last_payload or {}).get("status")
                or ""
            )
            msg = v1_status_msg(v1_row, last_payload)
            stuck = self._pull_stuck_outcome(
                now=now,
                offer=offer,
                last_payload=last_payload,
                v1_row=v1_row,
                state=state,
                msg=msg,
            )
            if stuck:
                return stuck
            if self.config.ssh_completion:
                ready, why = evaluate_readiness(last_payload, probes=_probes_from_payload(last_payload))
                if ready:
                    return {"status": "success", "reason": "ssh_ready", "last": last_payload}
                if why and why.startswith("running_but_no_ssh"):
                    pass
            else:
                stop_states = {
                    (last_payload or {}).get("cur_state"),
                    (last_payload or {}).get("actual_status"),
                    (last_payload or {}).get("status"),
                }
                if v1_row:
                    stop_states.update(
                        {
                            v1_row.get("actual_status"),
                            v1_row.get("cur_state"),
                            v1_row.get("status"),
                        }
                    )
                if any(s in {"exited", "stopped", "offline"} for s in stop_states if s):
                    return {"status": "failed", "reason": "instance_stopped", "last": last_payload}
            self.sleep(min(self.config.poll_interval_s, max(0.0, deadline - now)))
        return {"status": "aborted", "reason": "sigint", "last": last_payload}

    def destroy_registered(self) -> dict:
        if not self._instance_id:
            return {"skipped": True, "reason": "no_instance"}
        errors: list[str] = []
        last: dict | None = None
        for attempt in range(1, DELETE_MAX_RETRIES + 1):
            try:
                last = self.client.destroy(self._instance_id, self._allowlist)
                if not vast_delete_accepted(last):
                    errors.append(f"delete_rejected:{last}")
                elif not self._instance_visible_v1(self._instance_id):
                    return {
                        "ok": True,
                        "attempts": attempt,
                        "confirmed_v1": True,
                        "result": last,
                    }
                else:
                    errors.append("delete_unconfirmed_still_listed")
            except VastSafetyError as exc:
                errors.append(str(exc))
                break
            except HTTPError as exc:
                errors.append(str(exc))
                if not self._instance_visible_v1(self._instance_id):
                    return {
                        "ok": True,
                        "attempts": attempt,
                        "confirmed_v1": True,
                        "already_gone": True,
                        "result": last,
                    }
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
            if attempt < DELETE_MAX_RETRIES:
                self.sleep(DELETE_RETRY_BACKOFF_S * attempt)
        self._delete_errors.extend(errors)
        return {"ok": False, "attempts": DELETE_MAX_RETRIES, "errors": errors, "result": last}

    def run(self) -> dict:
        self._started_at = self.clock()
        outcome: dict[str, Any] = {"live": self.config.live, "dry_run": not self.config.live}
        try:
            image_ref, digest = self.validate_config()
            lease_env = build_canary_env(
                image_digest=digest,
                story_id=self.config.story_id,
                backend=self.config.backend,
            )
            offers = self._offer_list()
            chosen = self.select_offer(offers)
            if not chosen:
                outcome.update(
                    {
                        "action": "wait_no_offer",
                        "put_asks": False,
                        "offer_count": len(offers),
                        "max_dph": self.config.max_dph,
                        "gpu_policy": CANARY_GPU_MODEL_POLICY,
                    }
                )
                return outcome
            score = score_offer(
                chosen,
                episode_count=CANARY_EPISODE_COUNT,
                max_usd=CANARY_OFFER_SCORE_BUDGET_USD,
                max_dph=self.config.max_dph,
            )
            outcome["selected"] = {
                "id": chosen.get("id"),
                "gpu": chosen.get("gpu_name") or chosen.get("gpu"),
                "dph": chosen.get("dph_total") or chosen.get("dph"),
                "expected_total_usd": score.get("expected_total_usd"),
                "offer_score_budget_usd": CANARY_OFFER_SCORE_BUDGET_USD,
                "watchdog_max_spend_usd": self.config.max_spend_usd,
                "watchdog_max_wall_s": self.config.max_wall_s,
            }
            lease_env = inject_lease_watchdog_env(
                lease_env,
                offer=chosen,
                api_key=self.config.api_key or os.environ.get("VAST_API_KEY") or "",
                max_spend_usd=self.config.max_spend_usd,
                max_wall_s=self.config.max_wall_s,
            )
            outcome["lease_env"] = redact_mapping(lease_env)
            preflight = r2_live_preflight(self.config.story_id, DEFAULT_EPISODE, self.config.backend)
            outcome["preflight"] = {
                "ok": preflight.get("ok"),
                "reason": preflight.get("reason"),
                **{
                    k: preflight[k]
                    for k in ("missing_env", "missing", "artifacts", "input_keys")
                    if k in preflight
                },
            }
            if not self.config.live:
                outcome.update(
                    {
                        "action": "dry_run_would_lease",
                        "put_asks": False,
                        "image_ref": image_ref,
                    }
                )
                return outcome
            if not preflight.get("ok"):
                outcome.update(
                    {
                        "action": "preflight_failed",
                        "reason": preflight.get("reason"),
                        "put_asks": False,
                    }
                )
                return outcome
            gone_tries = 0
            while True:
                if self._abort:
                    outcome["final_status"] = "aborted"
                    outcome["reason"] = "sigint"
                    return outcome
                leased = self.lease_selected(chosen, image_ref=image_ref, lease_env=lease_env)
                outcome.update(leased)
                retry_host = False
                if leased.get("action") == "stale_offer":
                    self._note_excluded_machine(chosen)
                    retry_host = gone_tries < CANARY_GONE_RETRIES and not self._abort
                    if not retry_host:
                        if gone_tries:
                            outcome["gone_retries"] = gone_tries
                        return outcome
                elif leased.get("action") != "leased":
                    return outcome
                else:
                    poll = self.poll_until_done(chosen)
                    outcome.setdefault("polls", []).append(poll)
                    outcome["poll"] = poll
                    outcome["final_status"] = poll.get("status")
                    retry_host = (
                        poll.get("reason") in CANARY_RETRY_POLL_REASONS
                        and gone_tries < CANARY_GONE_RETRIES
                        and not self._abort
                    )
                    if not retry_host:
                        if self._abort:
                            outcome["final_status"] = "aborted"
                            outcome["reason"] = "sigint"
                        if gone_tries:
                            outcome["gone_retries"] = gone_tries
                        return outcome
                    self._spent_usd = self._running_cost(chosen)
                    old_id = self._instance_id
                    destroyed = self.destroy_registered()
                    if not destroyed.get("ok") and not destroyed.get("skipped"):
                        outcome["destroy"] = destroyed
                        outcome["final_status"] = "failed"
                        outcome["reason"] = "destroy_failed_before_retry"
                        return outcome
                    if old_id:
                        self._allowlist.discard(old_id)
                    self._instance_id = None
                    self._v1_missing = 0
                    self._pull_since = None
                    self._pull_msg = ""
                    self._pull_msg_changed_at = None
                    if self._abort:
                        outcome["gone_retries"] = gone_tries + 1
                        outcome["final_status"] = "aborted"
                        outcome["reason"] = "sigint"
                        return outcome
                    extra = poll if isinstance(poll, dict) else None
                    self._note_excluded_machine(chosen, extra=extra)
                offer_id = str(leased.get("offer_id") or chosen.get("id") or "")
                if offer_id:
                    self._exclude_offer_ids.add(offer_id)
                gone_tries += 1
                offers = self._offer_list()
                chosen = self.select_offer(offers)
                if not chosen:
                    outcome["gone_retries"] = gone_tries
                    outcome["action"] = "wait_no_offer"
                    return outcome
                score = score_offer(
                    chosen,
                    episode_count=CANARY_EPISODE_COUNT,
                    max_usd=CANARY_OFFER_SCORE_BUDGET_USD,
                    max_dph=self.config.max_dph,
                )
                outcome["selected"] = {
                    "id": chosen.get("id"),
                    "gpu": chosen.get("gpu_name") or chosen.get("gpu"),
                    "dph": chosen.get("dph_total") or chosen.get("dph"),
                    "expected_total_usd": score.get("expected_total_usd"),
                    "offer_score_budget_usd": CANARY_OFFER_SCORE_BUDGET_USD,
                    "watchdog_max_spend_usd": self.config.max_spend_usd,
                    "watchdog_max_wall_s": self.config.max_wall_s,
                }
                lease_env = inject_lease_watchdog_env(
                    lease_env,
                    offer=chosen,
                    api_key=self.config.api_key or os.environ.get("VAST_API_KEY") or "",
                    max_spend_usd=self.config.max_spend_usd,
                    max_wall_s=self.config.max_wall_s,
                )
                outcome["lease_env"] = redact_mapping(lease_env)
        finally:
            destroyed = self.destroy_registered()
            outcome["destroy"] = destroyed
            if self._delete_errors:
                outcome["delete_errors"] = list(self._delete_errors)

    def request_abort(self) -> None:
        self._abort = True


def _probes_from_payload(payload: dict | None):
    from gpu_worker.boot import ReadyProbe

    probe = ReadyProbe()
    if not payload:
        return probe
    probe.ssh_echo_ok = bool(payload.get("ssh_host") or payload.get("ssh_port"))
    probe.nvidia_smi_ok = bool(payload.get("gpu_name") or payload.get("num_gpus"))
    probe.comfy_system_stats_ok = False
    probe.comfy_required = False
    return probe


def _install_abort_signals(watchdog: CanaryWatchdog) -> None:
    def _handler(signum, frame):  # noqa: ARG001
        watchdog.request_abort()

    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)
    for sig in signals:
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPU canary watchdog (default dry-run)")
    p.add_argument("--live", action="store_true", help="Actually PUT /asks/ (requires confirm + API key)")
    p.add_argument("--confirm", default="", help=f"Required with --live: {LIVE_CONFIRM_PHRASE!r}")
    p.add_argument(
        "--image",
        required=True,
        help=(
            f"Digest pin: {CANARY_PUBLIC_H3_IMAGE}@sha256:<64hex>, "
            f"{CANARY_PUBLIC_LONGLIVE_IMAGE}@sha256:<64hex>, "
            f"or legacy {AGENT_IMAGE}@sha256:<64hex> (h3 only)"
        ),
    )
    p.add_argument(
        "--backend",
        choices=list(CANARY_BACKENDS),
        default="h3",
        help="Video line: h3 (MiniMax) or longlive (NVFP4)",
    )
    p.add_argument("--story-id", default=DEFAULT_STORY_ID)
    p.add_argument(
        "--max-wall-minutes",
        type=float,
        default=DEFAULT_MAX_WALL_S / 60.0,
        help="Clamped to 90 minutes for canary",
    )
    p.add_argument(
        "--max-spend-usd",
        type=float,
        default=DEFAULT_MAX_SPEND_USD,
        help="Clamped to $1 for canary watchdog (offer scoring uses separate budget)",
    )
    p.add_argument("--max-dph", type=float, default=CANARY_MAX_DPH_USD, help="Clamped to $1 for canary")
    p.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)
    p.add_argument("--no-r2-completion", action="store_true")
    p.add_argument("--ssh-completion", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_arg_parser().parse_args(argv)
    cfg = CanaryConfig(
        live=bool(args.live),
        confirm=args.confirm or "",
        api_key=os.environ.get("VAST_API_KEY") or "",
        image_ref=args.image,
        backend=args.backend,
        story_id=args.story_id,
        max_wall_s=clamp_canary_max_wall_s(max(60.0, float(args.max_wall_minutes) * 60.0)),
        max_spend_usd=clamp_canary_max_spend(float(args.max_spend_usd)),
        max_dph=float(args.max_dph),
        poll_interval_s=float(args.poll_interval_s),
        r2_completion=not args.no_r2_completion,
        ssh_completion=bool(args.ssh_completion),
    )
    watchdog = CanaryWatchdog(config=cfg)
    _install_abort_signals(watchdog)
    try:
        result = watchdog.run()
    except (CanaryConfigError, CanaryImageError, VastSafetyError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(redact_for_log(result), ensure_ascii=False, indent=2))
    if result.get("action") == "wait_no_offer":
        return 0
    if result.get("action") in {"preflight_failed", "cleanup_unknown"}:
        return 2
    if result.get("delete_errors"):
        return 3
    if result.get("destroy", {}).get("ok") is False:
        return 3
    if result.get("final_status") in {"failed", "timeout", "aborted"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

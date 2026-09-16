"""Shared cue/result types, license policy, deterministic QC and cache for SFX.

Freesound (anime_factory.sfx_freesound) and MOSS-SoundEffect v2
(anime_factory.sfx_moss) both produce a PCM WAV and both must pass the same
deterministic QC (duration, silence/RMS, clipping, malformed audio) before an
orchestrator (anime_factory.sfx_orchestrate) will accept the result. Cache and
provenance live here so both sources record the same audit shape.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from anime_factory.r2_paths import safe_cue_slug
from anime_factory.tts import decode_pcm16_mono, encode_pcm16_mono, resample_pcm16_mono

log = logging.getLogger("anime_factory.sfx")

# Pinned MOSS-SoundEffect v2 identity. Lives here (not in sfx_moss) because it
# is part of the cue *cache contract*: a model/source policy change must
# invalidate cached renders, and sfx_common owns cache identity.
MOSS_MODEL_REPO = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
MOSS_SOURCE_COMMIT = "934d6826b084c46a0d033402174d5f8ac4ed2519"

# Source/model policy string baked into every cue's cache identity. Bumping
# the Freesound asset tier or the MOSS pin re-resolves every cue instead of
# silently reusing renders produced under the old policy.
SFX_SOURCE_POLICY = f"freesound-preview-hq+{MOSS_MODEL_REPO}@{MOSS_SOURCE_COMMIT}"

# Deterministic QC defaults. Freesound previews and MOSS renders are judged
# the same way — a source must not get a QC pass it would not earn elsewhere.
SFX_SAMPLE_RATE = 44100
MIN_SFX_SECONDS = 0.15
MAX_SFX_SECONDS = 30.0
MIN_SFX_RMS = 40.0
MAX_CLIP_RATIO = 0.02
CLIP_THRESHOLD = 32000


class SfxError(RuntimeError):
    """Base for all SFX pipeline failures. Never carries a secret in its message."""


class MalformedAudioError(SfxError):
    """Downloaded/generated blob does not decode to usable PCM."""


class SfxQCError(SfxError):
    """Duration, silence, or clipping QC rejected a candidate render."""

    def __init__(self, message: str, issues: list[str] | None = None):
        super().__init__(message)
        self.issues = list(issues or [])


class LicenseRejectedError(SfxError):
    """CC-BY-NC (or unrecognized) license. CC0 preferred; CC-BY optional; NC always rejected."""


@dataclass(frozen=True)
class SfxCue:
    """One SFX/ambience need: a board cue, or a shared reusable ambience bed."""

    cue_key: str
    query: str
    tags: tuple[str, ...] = ()
    duration_target: float = 2.0
    duration_tolerance: float = 0.6
    bus: str = "sfx"  # "sfx" | "ambience"
    allow_cc_by: bool = True
    episode_code: str = ""
    segment_id: str | None = None
    shared: bool = False
    seed: int | None = None


@dataclass(frozen=True)
class SfxCandidate:
    """A scored Freesound search hit, before download."""

    freesound_id: int
    name: str
    tags: tuple[str, ...]
    duration: float
    license: str
    username: str
    preview_url: str
    samplerate: int | None = None
    avg_rating: float | None = None
    num_ratings: int | None = None


@dataclass(frozen=True)
class SfxProvenance:
    source: str  # "freesound" | "moss"
    content_hash: str
    sample_rate: int
    duration: float
    license: str | None = None
    creator: str | None = None
    source_url: str | None = None
    freesound_id: int | None = None
    query: str | None = None
    tags: tuple[str, ...] = ()
    score: float | None = None
    seed: int | None = None
    model: str | None = None
    model_commit: str | None = None
    prompt: str | None = None
    cached: bool = False


@dataclass(frozen=True)
class SfxResult:
    cue_key: str
    status: str  # "resolved" | "rejected" | "failed"
    local_path: str | None = None
    provenance: SfxProvenance | None = None
    reason: str | None = None


@dataclass(frozen=True)
class SfxQCReport:
    passed: bool
    issues: tuple[str, ...]
    duration: float
    rms: float
    clip_ratio: float
    sample_rate: int


# ---------------------------------------------------------------------------
# License policy. CC0 always wins; CC-BY is opt-in per cue; BY-NC/unknown always reject.
# ---------------------------------------------------------------------------


def classify_license(license_url: str | None) -> str:
    low = str(license_url or "").strip().lower()
    if not low:
        return "unknown"
    if "publicdomain/zero" in low or low in {"cc0", "http://creativecommons.org/publicdomain/zero/1.0/"}:
        return "cc0"
    if "-nc" in low or "noncommercial" in low or "non-commercial" in low:
        return "by-nc"
    if "/by/" in low or low.rstrip("/").endswith("/by") or low == "by":
        return "by"
    return "unknown"


def license_allowed(license_url: str | None, allow_cc_by: bool = True) -> bool:
    """CC0 always allowed. CC-BY allowed only when the cue opts in. BY-NC/unknown never."""
    kind = classify_license(license_url)
    if kind == "cc0":
        return True
    if kind == "by":
        return bool(allow_cc_by)
    return False


# ---------------------------------------------------------------------------
# Deterministic QC — duration, silence/RMS, clipping, malformed audio.
# Shared by the Freesound and MOSS paths so neither gets an easier bar.
# ---------------------------------------------------------------------------


def _wav_rms(samples: list[int]) -> float:
    if len(samples) < 2:
        return 0.0
    total = sum(s * s for s in samples)
    return (float(total) / len(samples)) ** 0.5


def _clip_ratio(samples: list[int]) -> float:
    if not samples:
        return 0.0
    clipped = sum(1 for s in samples if abs(s) >= CLIP_THRESHOLD)
    return clipped / len(samples)


def run_audio_qc(
    wav_bytes: bytes,
    *,
    target_duration: float | None = None,
    duration_tolerance: float = 0.6,
    min_seconds: float = MIN_SFX_SECONDS,
    max_seconds: float = MAX_SFX_SECONDS,
    min_rms: float = MIN_SFX_RMS,
    max_clip_ratio: float = MAX_CLIP_RATIO,
) -> SfxQCReport:
    rate, samples = decode_pcm16_mono(wav_bytes)
    issues: list[str] = []
    if not samples:
        issues.append("malformed")
        return SfxQCReport(passed=False, issues=tuple(issues), duration=0.0, rms=0.0, clip_ratio=0.0, sample_rate=rate)
    duration = len(samples) / float(rate or 1)
    rms = _wav_rms(samples)
    clip_ratio = _clip_ratio(samples)
    if duration < min_seconds:
        issues.append("too_short")
    if duration > max_seconds:
        issues.append("too_long")
    if target_duration and target_duration > 0:
        if abs(duration - target_duration) > max(duration_tolerance, target_duration * 0.5):
            issues.append("duration_mismatch")
    if rms < min_rms:
        issues.append("silence")
    if clip_ratio > max_clip_ratio:
        issues.append("clipping")
    return SfxQCReport(
        passed=not issues,
        issues=tuple(issues),
        duration=duration,
        rms=rms,
        clip_ratio=clip_ratio,
        sample_rate=rate,
    )


def assert_audio_qc(
    wav_bytes: bytes,
    *,
    label: str,
    target_duration: float | None = None,
    duration_tolerance: float = 0.6,
) -> SfxQCReport:
    report = run_audio_qc(wav_bytes, target_duration=target_duration, duration_tolerance=duration_tolerance)
    if not report.passed:
        if "malformed" in report.issues:
            raise MalformedAudioError(f"{label}: audio did not decode to usable PCM")
        raise SfxQCError(f"{label}: QC failed {list(report.issues)} (duration={report.duration:.2f}s, rms={report.rms:.1f})", list(report.issues))
    return report


def resample_wav_bytes(wav_bytes: bytes, dst_rate: int = SFX_SAMPLE_RATE) -> bytes:
    """Force any decodable mono PCM16 WAV onto one fixed sample rate."""
    rate, samples = decode_pcm16_mono(wav_bytes)
    if not samples:
        raise MalformedAudioError("cannot resample: audio did not decode to usable PCM")
    if rate == dst_rate:
        return encode_pcm16_mono(samples, rate)
    resampled = resample_pcm16_mono(samples, rate, dst_rate)
    return encode_pcm16_mono(resampled, dst_rate)


def content_hash(wav_bytes: bytes) -> str:
    return hashlib.sha256(wav_bytes).hexdigest()


CUE_CONTRACT_VERSION = "cue:v2"


def cue_contract_hash(cue: SfxCue, source_policy: str = SFX_SOURCE_POLICY) -> str:
    """Cache identity over the full cue contract, not `cue_key` alone.

    A changed query/tags/duration/tolerance/bus/license-policy/shared flag or a
    model/source policy bump must miss the cache and re-resolve — reusing an
    old render for an edited prompt is a silent wrong-sound bug.
    """
    payload = "|".join(
        (
            CUE_CONTRACT_VERSION,
            str(cue.cue_key or ""),
            str(cue.query or ""),
            ",".join(cue.tags),
            f"{float(cue.duration_target or 0.0):.3f}",
            f"{float(cue.duration_tolerance or 0.0):.3f}",
            str(cue.bus or "sfx"),
            "cc0+ccby" if cue.allow_cc_by else "cc0-only",
            "shared" if cue.shared else "episode",
            str(source_policy or ""),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_seed(cue_key: str, salt: str = "") -> int:
    """Stable, reproducible seed derived from a cue key. No RNG, no clock."""
    digest = hashlib.sha256(f"{cue_key}:{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# ---------------------------------------------------------------------------
# Episode-local / shared cache. Content-hash addressed so identical renders
# (or identical Freesound picks) across cues dedupe on disk automatically.
# ---------------------------------------------------------------------------


def cache_dir_for(root: Path, cue: SfxCue) -> Path:
    if cue.shared:
        return Path(root) / "shared" / "sfx" / "cache"
    ep = cue.episode_code or "EP001"
    return Path(root) / "episodes" / ep / "audio" / "sfx" / "cache"


def cache_manifest_path(cache_root: Path, cue: SfxCue) -> Path:
    """Manifest name = safe slug + 16-hex contract hash.

    The slug keeps the file greppable; the contract hash is the actual cache
    key, so a raw `cue_key` (which may contain `/`, `..`, unicode…) is never a
    path component and an edited cue contract never matches an old manifest.
    """
    slug = safe_cue_slug(cue.cue_key)
    return Path(cache_root) / f"{slug}.{cue_contract_hash(cue)[:16]}.json"


def cache_lookup(cache_root: Path, cue: SfxCue) -> SfxResult | None:
    """A cache hit must skip both the Freesound search and any MOSS generation."""
    manifest = cache_manifest_path(cache_root, cue)
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("contract") != cue_contract_hash(cue):
        return None
    local_path = payload.get("local_path")
    if not local_path or not Path(local_path).is_file():
        return None
    prov = dict(payload.get("provenance") or {})
    prov["tags"] = tuple(prov.get("tags") or ())
    prov["cached"] = True
    return SfxResult(
        cue_key=cue.cue_key,
        status="resolved",
        local_path=local_path,
        provenance=SfxProvenance(**prov),
    )


def cache_write(cache_root: Path, digest: str, wav_bytes: bytes) -> Path:
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{digest}.wav"
    if not path.is_file():
        path.write_bytes(wav_bytes)
    return path


def cache_record(cache_root: Path, cue: SfxCue, local_path: Path, provenance: SfxProvenance) -> None:
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = cache_manifest_path(root, cue)
    payload = {
        "cue_key": cue.cue_key,
        "contract": cue_contract_hash(cue),
        "query": cue.query,
        "tags": list(cue.tags),
        "duration_target": cue.duration_target,
        "duration_tolerance": cue.duration_tolerance,
        "bus": cue.bus,
        "allow_cc_by": cue.allow_cc_by,
        "shared": cue.shared,
        "local_path": str(local_path),
        "provenance": asdict(provenance),
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def redact(value: str | None) -> str:
    """For log lines that must name a secret's presence without its value."""
    return "<redacted>" if value else "<empty>"


# Env var names whose values are treated as secrets for redaction. Length >= 6
# avoids nuking every "1"/"true" flag value out of error text.
_SECRET_ENV_NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.I)
_MIN_SECRET_LEN = 6


def _secret_env_values() -> list[str]:
    values = []
    for name, value in os.environ.items():
        if value and len(value) >= _MIN_SECRET_LEN and _SECRET_ENV_NAME_RE.search(name):
            values.append(value)
    return values


def redact_secrets(text: str | None, extra: Sequence[str] = ()) -> str:
    """Strip every secret-looking env value (and `extra` values) out of `text`.

    Applied to network exception text, subprocess stderr tails, and DB failure
    reasons so an API token or secret env value can never reach a log line, a
    cache manifest, an `sfx_cues.provenance_json` row, or raised error text.
    """
    out = str(text or "")
    candidates = {str(v) for v in extra if v and len(str(v)) >= _MIN_SECRET_LEN}
    candidates.update(_secret_env_values())
    for value in sorted(candidates, key=len, reverse=True):
        out = out.replace(value, "<redacted>")
    return out

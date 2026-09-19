"""Freesound-first SFX search/select/download/normalize/QC pipeline.

Freesound API v2 (https://freesound.org/docs/api/) is queried with
`FREESOUND_API_KEY` from the environment; the token is attached as an
`Authorization: Token …` request header (the documented API-key scheme) —
never as a URL query parameter, where it would land in proxy/server logs and
exception reprs. Network exception text is redacted before it can reach a log
line, a DB failure reason, a cache manifest, or raised error text. Only the
HQ preview is fetched — this worker has no Freesound OAuth token, so the
authenticated original-file download is not available; the HQ preview is the
highest-quality asset reachable with an API key alone.

Host policy is fixed: API search goes to https://freesound.org only, and
preview downloads are restricted to HTTPS URLs on Freesound-controlled hosts
(freesound.org / *.freesound.org). Response and download sizes are bounded.

CC0 is preferred. CC-BY is optional per cue (recording creator, source URL,
and license so attribution can be assembled downstream). CC-BY-NC (and any
license we cannot positively classify) is always rejected — see
`sfx_common.license_allowed`.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from anime_factory.sfx_common import (
    MalformedAudioError,
    SFX_SAMPLE_RATE,
    SfxCandidate,
    SfxCue,
    SfxProvenance,
    SfxQCError,
    SfxResult,
    assert_audio_qc,
    cache_dir_for,
    cache_lookup,
    cache_record,
    cache_write,
    classify_license,
    content_hash,
    license_allowed,
    redact_secrets,
)
from anime_factory.tts import decode_pcm16_mono, encode_pcm16_mono

log = logging.getLogger("anime_factory.sfx.freesound")

FREESOUND_BASE_URL = "https://freesound.org/apiv2"
FREESOUND_API_HOST = "freesound.org"
FREESOUND_SEARCH_PATH = "/search/text/"
FREESOUND_SEARCH_FIELDS = (
    "id,name,tags,duration,license,username,previews,avg_rating,num_ratings,samplerate,channels,type"
)
FREESOUND_PAGE_SIZE = 15
# Bounded reads: a search response is small JSON; a preview is an mp3/ogg of
# at most ~30s of audio. Anything past these caps is a broken/hostile payload.
MAX_SEARCH_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_PREVIEW_BYTES = 48 * 1024 * 1024
REQUEST_TIMEOUT_S = 30

HttpOpener = Callable[[Request], "bytes | dict"]


class FreesoundConfigError(RuntimeError):
    """FREESOUND_API_KEY is not set. Never includes the key (there is none)."""


class FreesoundNoCandidateError(RuntimeError):
    """No search hit cleared the license/QC bar for this cue."""


class FreesoundNetworkError(RuntimeError):
    """A search/preview request failed. Message is always secret-redacted."""


class FreesoundHostPolicyError(RuntimeError):
    """URL is not HTTPS on a Freesound-controlled host."""


def freesound_api_key() -> str:
    """Read the API key from the environment only. Callers must never log it."""
    return (os.environ.get("FREESOUND_API_KEY") or "").strip()


def assert_freesound_url(url: str, *, allow_subdomains: bool) -> None:
    """HTTPS + freesound.org (API) or *.freesound.org (preview CDN) only."""
    parts = urlsplit(str(url or ""))
    host = (parts.hostname or "").lower()
    ok_host = host == FREESOUND_API_HOST or (
        allow_subdomains and host.endswith("." + FREESOUND_API_HOST)
    )
    if parts.scheme != "https" or not ok_host:
        raise FreesoundHostPolicyError(
            f"refusing non-Freesound/non-HTTPS URL host={host!r} scheme={parts.scheme!r}"
        )


def _candidate_from_json(row: dict) -> SfxCandidate | None:
    previews = row.get("previews") or {}
    preview_url = previews.get("preview-hq-mp3") or previews.get("preview-hq-ogg") or previews.get("preview-lq-mp3")
    if not preview_url:
        return None
    return SfxCandidate(
        freesound_id=int(row.get("id") or 0),
        name=str(row.get("name") or ""),
        tags=tuple(str(t) for t in (row.get("tags") or [])),
        duration=float(row.get("duration") or 0.0),
        license=str(row.get("license") or ""),
        username=str(row.get("username") or ""),
        preview_url=str(preview_url),
        samplerate=int(row["samplerate"]) if row.get("samplerate") else None,
        avg_rating=float(row["avg_rating"]) if row.get("avg_rating") is not None else None,
        num_ratings=int(row["num_ratings"]) if row.get("num_ratings") is not None else None,
    )


def score_candidate(candidate: SfxCandidate, cue: SfxCue) -> float:
    """Higher is better. CC0 outranks an equally-good CC-BY hit; closer duration
    and better metadata (tags, rating, sample rate) push a candidate up."""
    score = 0.0
    query_tokens = {t.lower() for t in cue.query.split() if t}
    name_tokens = {t.lower() for t in candidate.name.replace("-", " ").replace("_", " ").split() if t}
    score += 2.0 * len(query_tokens & name_tokens)
    cue_tags = {t.lower() for t in cue.tags}
    cand_tags = {t.lower() for t in candidate.tags}
    score += 1.5 * len(cue_tags & cand_tags)
    if cue.duration_target and candidate.duration:
        score += max(0.0, 3.0 - abs(candidate.duration - cue.duration_target))
    if candidate.avg_rating:
        score += min(candidate.avg_rating, 5.0) * 0.4
    if candidate.samplerate and candidate.samplerate >= 44100:
        score += 0.5
    if classify_license(candidate.license) == "cc0":
        score += 5.0
    return score


class FreesoundClient:
    """Search + preview download. `opener` makes every network call mockable.

    Live by default in production: a client built without an `opener` performs
    real HTTPS requests, so a missing mock cannot silently turn every cue into
    a MOSS fallback. Tests pass `opener=` and never touch the network;
    `live=False` remains available as an explicit offline kill-switch.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        opener: HttpOpener | None = None,
        live: bool | None = None,
        base_url: str = FREESOUND_BASE_URL,
    ):
        self.api_key = api_key if api_key is not None else freesound_api_key()
        self.opener = opener
        self.live = (opener is None) if live is None else bool(live)
        self.base_url = base_url.rstrip("/")

    def _get(self, req: Request, *, limit: int) -> bytes:
        if self.opener is not None:
            data = self.opener(req)
            raw = data if isinstance(data, (bytes, bytearray)) else json.dumps(data).encode("utf-8")
            if len(raw) > limit:
                raise FreesoundNetworkError(f"response exceeds {limit} byte cap")
            return bytes(raw)
        if not self.live:
            return b"{}"
        try:
            with urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:  # noqa: S310 — host asserted by caller
                raw = resp.read(limit + 1)
        except Exception as exc:  # noqa: BLE001 — sanitized: no token/env secret may escape
            # `from None`: the original exception (whose repr/args can embed
            # header dicts or URLs) must not ride along into tracebacks/logs.
            raise FreesoundNetworkError(
                f"{type(exc).__name__}: {redact_secrets(str(exc), extra=(self.api_key,))}"
            ) from None
        if len(raw) > limit:
            raise FreesoundNetworkError(f"response exceeds {limit} byte cap")
        return raw

    def search(self, query: str, tags: Sequence[str] = (), page_size: int = FREESOUND_PAGE_SIZE) -> list[SfxCandidate]:
        if not self.api_key and not self.opener:
            raise FreesoundConfigError("FREESOUND_API_KEY is not set")
        params = {
            "query": query,
            "fields": FREESOUND_SEARCH_FIELDS,
            "page_size": page_size,
        }
        if tags:
            params["filter"] = " ".join(f"tag:{t}" for t in tags)
        url = f"{self.base_url}{FREESOUND_SEARCH_PATH}?{urlencode(params)}"
        assert_freesound_url(url, allow_subdomains=False)
        req = Request(url, method="GET")
        if self.api_key:
            # Documented API-key scheme. Never `?token=` — query strings end up
            # in access logs, proxies, and URLError text.
            req.add_header("Authorization", f"Token {self.api_key}")
        raw = self._get(req, limit=MAX_SEARCH_RESPONSE_BYTES)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            payload = {}
        candidates = []
        for row in payload.get("results") or []:
            candidate = _candidate_from_json(row)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def fetch_preview(self, preview_url: str) -> bytes:
        # Preview URLs are pre-signed by Freesound and need no Authorization
        # header — and must not get one: only freesound.org itself may ever
        # see the token, and previews live on Freesound-controlled CDN hosts.
        assert_freesound_url(preview_url, allow_subdomains=True)
        req = Request(preview_url, method="GET")
        return self._get(req, limit=MAX_PREVIEW_BYTES)


def _is_pcm_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def normalize_to_pcm_wav(
    raw: bytes,
    *,
    sample_rate: int = SFX_SAMPLE_RATE,
    ffmpeg_runner: Callable[[list[str]], "subprocess.CompletedProcess"] | None = None,
) -> bytes:
    """ffmpeg-normalize a Freesound preview (mp3/ogg) into mono PCM16 WAV.

    A preview that already decodes as PCM16 WAV at the target rate is passed
    through without a redundant re-encode; anything else — real Freesound
    mp3/ogg previews — goes through ffmpeg exactly once.
    """
    if _is_pcm_wav(raw):
        rate, samples = decode_pcm16_mono(raw)
        if samples and rate == sample_rate:
            return encode_pcm16_mono(samples, rate)
    runner = ffmpeg_runner or _run_ffmpeg
    with tempfile.TemporaryDirectory(prefix="af-sfx-") as tmp:
        src = Path(tmp) / "in.bin"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(raw)
        args = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            "s16",
            "-f",
            "wav",
            str(dst),
        ]
        try:
            result = runner(args)
        except (OSError, subprocess.SubprocessError) as exc:
            raise MalformedAudioError(
                f"ffmpeg normalize failed to start: {redact_secrets(str(exc))}"
            ) from None
        if getattr(result, "returncode", 1) != 0 or not dst.is_file():
            stderr = getattr(result, "stderr", b"") or b""
            tail = stderr[-300:].decode("utf-8", "replace") if isinstance(stderr, (bytes, bytearray)) else str(stderr)[-300:]
            raise MalformedAudioError(
                f"ffmpeg normalize did not produce a WAV: {redact_secrets(tail)}"
            )
        return dst.read_bytes()


def _run_ffmpeg(args: list[str]) -> "subprocess.CompletedProcess":
    return subprocess.run(args, capture_output=True, timeout=60)


def rank_candidates(candidates: list[SfxCandidate], cue: SfxCue) -> list[SfxCandidate]:
    allowed = [c for c in candidates if license_allowed(c.license, cue.allow_cc_by)]
    return sorted(allowed, key=lambda c: score_candidate(c, cue), reverse=True)


def resolve_via_freesound(
    client: FreesoundClient,
    cue: SfxCue,
    root: Path,
    *,
    normalizer: Callable[[bytes], bytes] | None = None,
) -> SfxResult:
    """Freesound search -> license filter -> score -> download -> normalize -> QC.

    A cache hit for the cue's full contract (see `sfx_common.cue_contract_hash`)
    returns immediately without calling `client.search` or
    `client.fetch_preview` at all.
    """
    cache_root = cache_dir_for(root, cue)
    cached = cache_lookup(cache_root, cue)
    if cached is not None:
        return cached

    candidates = client.search(cue.query, tags=cue.tags)
    ranked = rank_candidates(candidates, cue)
    if not ranked:
        raise FreesoundNoCandidateError(
            f"{cue.cue_key}: no CC0/CC-BY candidate for query {cue.query!r} "
            f"(allow_cc_by={cue.allow_cc_by}, {len(candidates)} raw hits)"
        )

    normalize = normalizer or normalize_to_pcm_wav
    last_error: Exception | None = None
    for candidate in ranked:
        try:
            # A fetch error/timeout, an off-policy preview host, an oversized
            # download, a malformed blob, or a QC failure on one candidate all
            # mean the same thing: move on to the next ranked candidate.
            raw = client.fetch_preview(candidate.preview_url)
            wav = normalize(raw)
            assert_audio_qc(
                wav,
                label=f"{cue.cue_key}:fs{candidate.freesound_id}",
                target_duration=cue.duration_target,
                duration_tolerance=cue.duration_tolerance,
                bus=cue.bus,
            )
        except (SfxQCError, MalformedAudioError, FreesoundNetworkError, FreesoundHostPolicyError) as exc:
            log.warning("freesound candidate rejected cue=%s fs_id=%s reason=%s", cue.cue_key, candidate.freesound_id, exc)
            last_error = exc
            continue
        digest = content_hash(wav)
        local_path = cache_write(cache_root, digest, wav)
        rate, samples = decode_pcm16_mono(wav)
        provenance = SfxProvenance(
            source="freesound",
            content_hash=digest,
            sample_rate=rate,
            duration=len(samples) / float(rate or 1),
            license=candidate.license,
            creator=candidate.username,
            source_url=f"https://freesound.org/s/{candidate.freesound_id}/",
            freesound_id=candidate.freesound_id,
            query=cue.query,
            tags=candidate.tags,
            score=score_candidate(candidate, cue),
            cached=False,
        )
        cache_record(cache_root, cue, local_path, provenance)
        return SfxResult(cue_key=cue.cue_key, status="resolved", local_path=str(local_path), provenance=provenance)
    raise FreesoundNoCandidateError(
        f"{cue.cue_key}: all {len(ranked)} candidate(s) failed QC/decode; last error: {last_error}"
    )

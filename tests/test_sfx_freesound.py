"""Freesound pipeline: mocked network, license policy, QC fallback, cache, secrets."""

from __future__ import annotations

import json
import logging

import pytest

from anime_factory.sfx_common import (
    MalformedAudioError,
    SfxCue,
    cache_dir_for,
)
from anime_factory.sfx_freesound import (
    FreesoundClient,
    FreesoundNoCandidateError,
    resolve_via_freesound,
    score_candidate,
)
from anime_factory.tts import voiced_dummy_wav


def _search_payload(results: list[dict]) -> bytes:
    return json.dumps({"results": results}).encode("utf-8")


def _candidate_row(
    sound_id: int,
    *,
    name: str = "footstep on gravel",
    tags: tuple[str, ...] = ("footstep", "gravel"),
    duration: float = 1.2,
    license_url: str = "http://creativecommons.org/publicdomain/zero/1.0/",
    username: str = "someuser",
    preview: str = "https://cdn.freesound.org/preview.mp3",
    samplerate: int = 44100,
    avg_rating: float = 4.2,
) -> dict:
    return {
        "id": sound_id,
        "name": name,
        "tags": list(tags),
        "duration": duration,
        "license": license_url,
        "username": username,
        "previews": {"preview-hq-mp3": preview},
        "samplerate": samplerate,
        "avg_rating": avg_rating,
        "num_ratings": 10,
    }


def _good_wav() -> bytes:
    return voiced_dummy_wav(1.2, freq=220.0)


def _opener_factory(search_results: list[dict], preview_bytes: dict[str, bytes], calls: list[str]):
    def opener(req):
        url = req.full_url
        calls.append(url)
        if "/search/" in url:
            return _search_payload(search_results)
        for key, blob in preview_bytes.items():
            if key in url:
                return blob
        return b""

    return opener


def test_cc0_preferred_over_cc_by(tmp_path):
    rows = [
        _candidate_row(1, license_url="http://creativecommons.org/licenses/by/4.0/", name="footstep gravel"),
        _candidate_row(2, license_url="http://creativecommons.org/publicdomain/zero/1.0/", name="footstep gravel"),
    ]
    calls: list[str] = []
    previews = {"preview.mp3": _good_wav()}
    opener = _opener_factory(rows, previews, calls)
    client = FreesoundClient("test-key", opener=opener)
    cue = SfxCue(cue_key="footstep_1", query="footstep gravel", tags=("footstep",), duration_target=1.2)
    result = resolve_via_freesound(client, cue, tmp_path)
    assert result.status == "resolved"
    assert result.provenance.freesound_id == 2
    assert result.provenance.license.startswith("http://creativecommons.org/publicdomain/zero")


def test_noncommercial_license_always_rejected(tmp_path):
    rows = [
        _candidate_row(3, license_url="http://creativecommons.org/licenses/by-nc/4.0/", name="footstep gravel"),
    ]
    calls: list[str] = []
    opener = _opener_factory(rows, {"preview.mp3": _good_wav()}, calls)
    client = FreesoundClient("test-key", opener=opener)
    cue = SfxCue(cue_key="footstep_nc", query="footstep gravel", duration_target=1.2)
    with pytest.raises(FreesoundNoCandidateError):
        resolve_via_freesound(client, cue, tmp_path)


def test_cc_by_rejected_when_cue_disallows_it(tmp_path):
    rows = [_candidate_row(4, license_url="http://creativecommons.org/licenses/by/4.0/")]
    opener = _opener_factory(rows, {"preview.mp3": _good_wav()}, [])
    client = FreesoundClient("test-key", opener=opener)
    cue = SfxCue(cue_key="footstep_no_by", query="footstep gravel", duration_target=1.2, allow_cc_by=False)
    with pytest.raises(FreesoundNoCandidateError):
        resolve_via_freesound(client, cue, tmp_path)


def test_bad_candidate_falls_back_to_next(tmp_path):
    """Top-scored candidate downloads malformed audio; the ranked runner-up must be tried."""
    rows = [
        _candidate_row(5, license_url="http://creativecommons.org/publicdomain/zero/1.0/", preview="https://cdn.freesound.org/bad.mp3"),
        _candidate_row(6, license_url="http://creativecommons.org/publicdomain/zero/1.0/", preview="https://cdn.freesound.org/good.mp3", avg_rating=1.0),
    ]
    previews = {"bad.mp3": b"not-audio-garbage", "good.mp3": _good_wav()}
    opener = _opener_factory(rows, previews, [])
    client = FreesoundClient("test-key", opener=opener)
    cue = SfxCue(cue_key="footstep_fallback", query="footstep gravel", duration_target=1.2)

    def normalizer(raw: bytes) -> bytes:
        if raw == b"not-audio-garbage":
            raise MalformedAudioError("garbage")
        return raw

    result = resolve_via_freesound(client, cue, tmp_path, normalizer=normalizer)
    assert result.status == "resolved"
    assert result.provenance.freesound_id == 6


def test_all_candidates_failing_qc_raises(tmp_path):
    rows = [_candidate_row(7, license_url="http://creativecommons.org/publicdomain/zero/1.0/")]
    opener = _opener_factory(rows, {"preview.mp3": b"garbage"}, [])
    client = FreesoundClient("test-key", opener=opener)
    cue = SfxCue(cue_key="footstep_allbad", query="footstep gravel", duration_target=1.2)

    def normalizer(raw: bytes) -> bytes:
        raise MalformedAudioError("bad")

    with pytest.raises(FreesoundNoCandidateError):
        resolve_via_freesound(client, cue, tmp_path, normalizer=normalizer)


def test_cache_hit_skips_search_and_download(tmp_path):
    rows = [_candidate_row(8, license_url="http://creativecommons.org/publicdomain/zero/1.0/")]
    calls: list[str] = []
    opener = _opener_factory(rows, {"preview.mp3": _good_wav()}, calls)
    client = FreesoundClient("test-key", opener=opener)
    cue = SfxCue(cue_key="door_creak", query="door creak", duration_target=1.2)

    first = resolve_via_freesound(client, cue, tmp_path)
    assert first.status == "resolved"
    n_calls_after_first = len(calls)
    assert n_calls_after_first >= 1

    def exploding_opener(req):
        raise AssertionError(f"network must not be called on a cache hit: {req.full_url}")

    client2 = FreesoundClient("test-key", opener=exploding_opener)
    second = resolve_via_freesound(client2, cue, tmp_path)
    assert second.status == "resolved"
    assert second.provenance.cached is True
    assert second.local_path == first.local_path


def test_score_candidate_rewards_duration_closeness_and_tags():
    from anime_factory.sfx_common import SfxCandidate

    cue = SfxCue(cue_key="k", query="door creak", tags=("door", "creak"), duration_target=2.0)
    close = SfxCandidate(
        freesound_id=1,
        name="door creak",
        tags=("door", "creak"),
        duration=2.0,
        license="http://creativecommons.org/publicdomain/zero/1.0/",
        username="u",
        preview_url="p",
        samplerate=44100,
        avg_rating=4.0,
    )
    far = SfxCandidate(
        freesound_id=2,
        name="unrelated",
        tags=(),
        duration=20.0,
        license="http://creativecommons.org/publicdomain/zero/1.0/",
        username="u",
        preview_url="p",
        samplerate=8000,
        avg_rating=None,
    )
    assert score_candidate(close, cue) > score_candidate(far, cue)


def test_live_by_default_without_opener_and_mocked_with_opener():
    assert FreesoundClient("k").live is True  # production default: real requests
    assert FreesoundClient("k", opener=lambda req: b"{}").live is False
    assert FreesoundClient("k", live=False).live is False
    assert FreesoundClient("k", opener=lambda req: b"{}", live=True).live is True


def test_search_uses_authorization_header_never_url_token(tmp_path):
    secret = "sf-super-secret-key-987654"
    captured = []

    def opener(req):
        captured.append(req)
        if "/search/" in req.full_url:
            return _search_payload([_candidate_row(11)])
        return _good_wav()

    client = FreesoundClient(secret, opener=opener)
    cue = SfxCue(cue_key="header_check", query="footstep gravel", duration_target=1.2)
    result = resolve_via_freesound(client, cue, tmp_path)
    assert result.status == "resolved"
    search_reqs = [r for r in captured if "/search/" in r.full_url]
    assert search_reqs
    for req in search_reqs:
        assert secret not in req.full_url  # never a ?token= query param
        assert req.get_header("Authorization") == f"Token {secret}"
    # Preview requests must NOT carry the token off the API host.
    preview_reqs = [r for r in captured if "/search/" not in r.full_url]
    for req in preview_reqs:
        assert req.get_header("Authorization") is None


def test_search_host_policy_rejects_non_freesound_base():
    from anime_factory.sfx_freesound import FreesoundHostPolicyError

    client = FreesoundClient("k", opener=lambda req: b"{}", base_url="https://evil.example/apiv2")
    with pytest.raises(FreesoundHostPolicyError):
        client.search("door creak")
    http_client = FreesoundClient("k", opener=lambda req: b"{}", base_url="http://freesound.org/apiv2")
    with pytest.raises(FreesoundHostPolicyError):
        http_client.search("door creak")


def test_preview_host_policy_only_freesound_https(tmp_path):
    """A candidate whose preview lives off Freesound-controlled HTTPS is skipped;
    a Freesound-CDN candidate still resolves."""
    rows = [
        _candidate_row(21, preview="https://evil.example/steal.mp3", avg_rating=5.0),
        _candidate_row(22, preview="https://cdn.freesound.org/ok.mp3", avg_rating=1.0),
    ]
    fetched: list[str] = []

    def opener(req):
        url = req.full_url
        fetched.append(url)
        if "/search/" in url:
            return _search_payload(rows)
        return _good_wav()

    client = FreesoundClient("k", opener=opener)
    cue = SfxCue(cue_key="host_policy", query="footstep gravel", duration_target=1.2)
    result = resolve_via_freesound(client, cue, tmp_path)
    assert result.provenance.freesound_id == 22
    assert not any("evil.example" in u for u in fetched)


def test_network_error_on_one_candidate_falls_to_next(tmp_path):
    from anime_factory.sfx_freesound import FreesoundNetworkError

    rows = [
        _candidate_row(31, preview="https://cdn.freesound.org/flaky.mp3", avg_rating=5.0),
        _candidate_row(32, preview="https://cdn.freesound.org/solid.mp3", avg_rating=1.0),
    ]

    def opener(req):
        url = req.full_url
        if "/search/" in url:
            return _search_payload(rows)
        if "flaky.mp3" in url:
            raise FreesoundNetworkError("TimeoutError: timed out")  # what a live timeout wraps into
        return _good_wav()

    client = FreesoundClient("k", opener=opener)
    cue = SfxCue(cue_key="net_fallback", query="footstep gravel", duration_target=1.2)
    result = resolve_via_freesound(client, cue, tmp_path)
    assert result.status == "resolved"
    assert result.provenance.freesound_id == 32


def test_oversized_preview_rejected_and_next_candidate_used(tmp_path, monkeypatch):
    import anime_factory.sfx_freesound as fs_mod

    monkeypatch.setattr(fs_mod, "MAX_PREVIEW_BYTES", 1024)
    rows = [
        _candidate_row(41, preview="https://cdn.freesound.org/huge.mp3", avg_rating=5.0),
        _candidate_row(42, preview="https://cdn.freesound.org/small.mp3", avg_rating=1.0),
    ]
    small = _good_wav()
    assert len(small) > 1024  # a real wav exceeds the shrunken cap...

    def opener(req):
        url = req.full_url
        if "/search/" in url:
            return _search_payload(rows)
        if "huge.mp3" in url:
            return b"x" * 4096
        return small

    client = FreesoundClient("k", opener=opener)
    cue = SfxCue(cue_key="size_cap", query="footstep gravel", duration_target=1.2)
    # Both previews exceed the 1 KiB cap here, so resolution must fail loudly —
    # never accept a blob past the bound.
    with pytest.raises(FreesoundNoCandidateError):
        resolve_via_freesound(client, cue, tmp_path)


def test_live_network_exception_is_redacted_and_unchained(tmp_path, monkeypatch):
    import urllib.error

    from anime_factory.sfx_freesound import FreesoundNetworkError
    import anime_factory.sfx_freesound as fs_mod

    secret = "sf-token-abcdef123456"

    def exploding_urlopen(req, timeout=0):
        raise urllib.error.URLError(f"boom while sending Token {secret}")

    monkeypatch.setattr(fs_mod, "urlopen", exploding_urlopen)
    client = FreesoundClient(secret, live=True)
    with pytest.raises(FreesoundNetworkError) as excinfo:
        client.search("door creak")
    message = str(excinfo.value)
    assert secret not in message
    assert "<redacted>" in message
    # `from None`: the raw urllib exception (with the token) must not ride
    # along as __cause__ into tracebacks and logs.
    assert excinfo.value.__cause__ is None


def test_env_secret_values_redacted(monkeypatch):
    from anime_factory.sfx_common import redact_secrets

    monkeypatch.setenv("SOME_SERVICE_TOKEN", "env-secret-value-42")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "another-secret-666")
    text = "failed: env-secret-value-42 and another-secret-666 leaked"
    cleaned = redact_secrets(text)
    assert "env-secret-value-42" not in cleaned
    assert "another-secret-666" not in cleaned
    assert cleaned.count("<redacted>") == 2


def test_normalizer_stderr_is_redacted(monkeypatch):
    import subprocess

    from anime_factory.sfx_freesound import normalize_to_pcm_wav

    monkeypatch.setenv("FREESOUND_API_KEY", "fs-secret-xyz-778899")

    def failing_runner(args):
        return subprocess.CompletedProcess(
            args, returncode=1, stdout=b"", stderr=b"error: token fs-secret-xyz-778899 rejected"
        )

    with pytest.raises(MalformedAudioError) as excinfo:
        normalize_to_pcm_wav(b"not-a-wav", ffmpeg_runner=failing_runner)
    assert "fs-secret-xyz-778899" not in str(excinfo.value)


def test_cache_identity_includes_cue_contract_not_just_key(tmp_path):
    """Same cue_key but a changed query must MISS the cache and re-search —
    a stale render for an edited prompt is a silent wrong-sound bug."""
    rows_a = [_candidate_row(51, name="footstep gravel")]
    calls_a: list[str] = []
    client_a = FreesoundClient("k", opener=_opener_factory(rows_a, {"preview.mp3": _good_wav()}, calls_a))
    cue_a = SfxCue(cue_key="step", query="footstep gravel", duration_target=1.2)
    first = resolve_via_freesound(client_a, cue_a, tmp_path)
    assert first.status == "resolved"

    rows_b = [_candidate_row(52, name="footstep snow")]
    calls_b: list[str] = []
    client_b = FreesoundClient("k", opener=_opener_factory(rows_b, {"preview.mp3": _good_wav()}, calls_b))
    cue_b = SfxCue(cue_key="step", query="footstep snow", duration_target=1.2)  # same key, new prompt
    second = resolve_via_freesound(client_b, cue_b, tmp_path)
    assert any("/search/" in u for u in calls_b)  # cache did not falsely hit
    assert second.provenance.freesound_id == 52

    # And the original contract still hits its own cache.
    def exploding(req):
        raise AssertionError("unchanged contract must be a cache hit")

    third = resolve_via_freesound(FreesoundClient("k", opener=exploding), cue_a, tmp_path)
    assert third.provenance.cached is True


def test_traversal_cue_key_cannot_escape_cache_dir(tmp_path):
    rows = [_candidate_row(61)]
    client = FreesoundClient("k", opener=_opener_factory(rows, {"preview.mp3": _good_wav()}, []))
    evil = SfxCue(cue_key="../../escape/../../tmp/evil", query="footstep gravel", duration_target=1.2, episode_code="EP001")
    result = resolve_via_freesound(client, evil, tmp_path)
    assert result.status == "resolved"
    cache_root = cache_dir_for(tmp_path, evil).resolve()
    for created in tmp_path.rglob("*"):
        assert created.resolve().is_relative_to(tmp_path.resolve())
    manifests = list(cache_root.glob("*.json"))
    assert manifests, "manifest must live inside the cue cache dir"
    for manifest in manifests:
        assert ".." not in manifest.name


def test_freesound_api_key_never_logged(tmp_path, caplog):
    secret = "sf-super-secret-token-xyz"
    rows = [_candidate_row(9, license_url="http://creativecommons.org/publicdomain/zero/1.0/", preview="https://cdn.freesound.org/bad2.mp3")]
    opener = _opener_factory(rows, {"bad2.mp3": b"garbage"}, [])
    client = FreesoundClient(secret, opener=opener)
    cue = SfxCue(cue_key="secret_check", query="footstep gravel", duration_target=1.2)

    def normalizer(raw: bytes) -> bytes:
        raise MalformedAudioError("bad")

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(FreesoundNoCandidateError):
            resolve_via_freesound(client, cue, tmp_path, normalizer=normalizer)
    for record in caplog.records:
        assert secret not in record.getMessage()
    assert secret not in json.dumps(str(cache_dir_for(tmp_path, cue)))

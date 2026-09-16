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

"""Bus-aware SFX QC: action stays strict; quiet ambience beds must not false-fail."""

from __future__ import annotations

import math

import pytest

from anime_factory.sfx_common import (
    MIN_AMBIENCE_RMS,
    MIN_SFX_RMS,
    SFX_SOURCE_POLICY,
    SfxCue,
    SfxQCError,
    assert_audio_qc,
    cue_contract_hash,
    loop_crossfade_wav,
    run_audio_qc,
)
from anime_factory.tts import decode_pcm16_mono, encode_pcm16_mono, voiced_dummy_wav


def _sine_wav(seconds: float, *, amp: int, freq: float = 200.0, rate: int = 44100) -> bytes:
    n = max(int(seconds * rate), 1)
    samples = [int(amp * math.sin(2 * math.pi * freq * (i / rate))) for i in range(n)]
    return encode_pcm16_mono(samples, rate)


def test_quiet_classroom_ambience_passes_bus_aware_qc():
    # RMS ≈ amp/√2. amp=25 → ~17.7, matching the false-fail classroom bed.
    quiet = _sine_wav(2.0, amp=25, freq=120.0)
    rate, samples = decode_pcm16_mono(quiet)
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
    assert 15.0 <= rms <= 25.0
    assert rms < MIN_SFX_RMS  # would fail the old action gate
    assert rms >= MIN_AMBIENCE_RMS

    report = run_audio_qc(quiet, target_duration=18.0, bus="ambience")
    assert report.passed, report.issues
    assert_audio_qc(quiet, label="classroom", target_duration=18.0, bus="ambience")


def test_action_sfx_still_rejects_quiet_rms():
    quiet = _sine_wav(1.0, amp=25, freq=400.0)
    report = run_audio_qc(quiet, target_duration=1.0, bus="sfx")
    assert not report.passed
    assert "silence" in report.issues
    with pytest.raises(SfxQCError):
        assert_audio_qc(quiet, label="keys", target_duration=1.0, bus="sfx")


def test_digital_silence_fails_ambience_too():
    silent = encode_pcm16_mono([0] * 44100, 44100)
    report = run_audio_qc(silent, target_duration=1.0, bus="ambience")
    assert not report.passed
    assert "silence" in report.issues


def test_loop_crossfade_extends_short_bed():
    src = voiced_dummy_wav(1.0, sample_rate=44100, freq=180.0)
    out = loop_crossfade_wav(src, 3.5, crossfade_s=0.2)
    rate, samples = decode_pcm16_mono(out)
    assert rate == 44100
    assert abs(len(samples) / rate - 3.5) < 0.02


def test_source_policy_bump_invalidates_pre_busqc_cache_identity():
    cue = SfxCue(cue_key="classroom", query="quiet classroom", bus="ambience", duration_target=4.0)
    assert "busqc-v2" in SFX_SOURCE_POLICY
    digest = cue_contract_hash(cue)
    old = cue_contract_hash(cue, source_policy="freesound-preview-hq+OpenMOSS-Team/MOSS-SoundEffect-v2.0@934d6826b084c46a0d033402174d5f8ac4ed2519")
    assert digest != old

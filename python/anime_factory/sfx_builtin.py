"""Small original interface sounds for a bounded, CPU-only final fallback.

Only recognized phone UI events are supported. Physical sounds and ambience
must still come from the requested audio sources; they never become beeps.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from anime_factory.sfx_common import (
    SfxCue, SfxProvenance, SfxResult, assert_audio_qc, cache_dir_for,
    cache_record, cache_write, content_hash,
)
from anime_factory.tts import encode_pcm16_mono


def resolve_builtin_phone_cue(cue: SfxCue, root: Path) -> SfxResult | None:
    text = cue.query.lower()
    if cue.bus != "sfx" or not re.search(r"\b(phone|smartphone|mobile)\b", text):
        return None
    if re.search(r"\b(transfer|sent)\b", text):
        kind = "sent"
    elif re.search(r"\b(notification|notify|chime)\b", text):
        kind = "notification"
    elif re.search(r"\b(hang up|hangup)\b", text):
        kind = "hangup"
    else:
        return None
    duration = float(cue.duration_target)
    if not 0.15 <= duration <= 2.0:
        return None
    rate = 44100
    samples = []
    for i in range(round(duration * rate)):
        t = i / rate
        u = t / duration
        envelope = math.sin(math.pi * u) ** 2
        if kind == "sent":
            # Upward sweep with a soft attack and tail, not a branded UI sound.
            phase = 2 * math.pi * (480 * t + 420 * t * t / duration)
        elif kind == "notification":
            freq = 660 if u < 0.5 else 880
            phase = 2 * math.pi * freq * t
            envelope *= math.sin(math.pi * ((u * 2) % 1)) ** 2
        else:
            phase = 2 * math.pi * (700 * t - 200 * t * t / duration)
        samples.append(round(3600 * envelope * math.sin(phase)))
    wav = encode_pcm16_mono(samples, rate)
    assert_audio_qc(wav, label=f"{cue.cue_key}:builtin", target_duration=duration)
    digest = content_hash(wav)
    cache_root = cache_dir_for(root, cue)
    path = cache_write(cache_root, digest, wav)
    provenance = SfxProvenance(
        source="builtin", content_hash=digest, sample_rate=rate,
        duration=len(samples) / rate, creator="anime-factory-worker",
        query=cue.query, prompt=cue.query, model=f"phone-ui-{kind}-v1",
    )
    cache_record(cache_root, cue, path, provenance)
    return SfxResult(cue_key=cue.cue_key, status="resolved", local_path=str(path), provenance=provenance)

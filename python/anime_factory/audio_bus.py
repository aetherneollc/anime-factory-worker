"""48 kHz dialogue/SFX/ambience bus mixing.

Dialogue is per-language (the locked CosyVoice2 masters `compose.py` already
builds). SFX and ambience are language-independent: `build_shared_bed` is
called once per episode and the resulting stem is reused for every
language's final master, so switching dub tracks never shifts the effects
bed. Buses are combined with safe headroom, short fades, and an
envelope-follower duck (SFX/ambience attenuate under active dialogue) rather
than a flat additive overlay, which is what silently clipped or buried
dialogue under a loud effects bed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from anime_factory.tts import decode_pcm16_mono, encode_pcm16_mono, resample_pcm16_mono

BUS_SAMPLE_RATE = 48000
DIALOGUE_GAIN_DB = 0.0
SFX_GAIN_DB = -6.0
AMBIENCE_GAIN_DB = -12.0
DUCK_ATTENUATION_DB = -10.0
DUCK_THRESHOLD = 400.0
DUCK_ATTACK_S = 0.05
DUCK_RELEASE_S = 0.35
DEFAULT_FADE_S = 0.02
# Safe int16 ceiling with headroom below hard clip (32767) for the loudnorm
# pass compose.build_compose_plan applies downstream.
PEAK_LIMIT = 32000


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def to_bus_rate(samples: list[int], src_rate: int, dst_rate: int = BUS_SAMPLE_RATE) -> list[int]:
    if not samples or src_rate == dst_rate:
        return list(samples)
    return resample_pcm16_mono(samples, src_rate, dst_rate)


def apply_gain(samples: list[int], db: float) -> list[int]:
    if not samples or db == 0.0:
        return list(samples)
    factor = db_to_linear(db)
    return [int(max(-32768, min(32767, round(s * factor)))) for s in samples]


def apply_fade(samples: list[int], rate: int, fade_in_s: float = DEFAULT_FADE_S, fade_out_s: float = DEFAULT_FADE_S) -> list[int]:
    n = len(samples)
    if n == 0:
        return list(samples)
    out = list(samples)
    fade_in_n = max(0, min(n, int(fade_in_s * rate)))
    fade_out_n = max(0, min(n, int(fade_out_s * rate)))
    for i in range(fade_in_n):
        out[i] = int(out[i] * (i / fade_in_n))
    for i in range(fade_out_n):
        idx = n - 1 - i
        out[idx] = int(out[idx] * (i / fade_out_n))
    return out


def _envelope(samples: Sequence[int], rate: int, window_s: float) -> list[float]:
    """Rectified moving-average envelope at output-sample resolution."""
    n = len(samples)
    if n == 0:
        return []
    window = max(1, int(window_s * rate))
    env = [0.0] * n
    acc = 0
    for i in range(n):
        acc += abs(samples[i])
        if i >= window:
            acc -= abs(samples[i - window])
        env[i] = acc / min(i + 1, window)
    return env


def duck_under_dialogue(
    bed: list[int],
    dialogue: Sequence[int],
    rate: int,
    *,
    threshold: float = DUCK_THRESHOLD,
    attenuation_db: float = DUCK_ATTENUATION_DB,
    attack_s: float = DUCK_ATTACK_S,
    release_s: float = DUCK_RELEASE_S,
) -> list[int]:
    """Attenuate `bed` (SFX/ambience) wherever `dialogue` energy is active.

    Envelope-follower sidechain: bed gain glides toward `attenuation_db`
    while the dialogue envelope is above `threshold`, and glides back to 0 dB
    in silence, smoothed by attack/release so ducking never clicks.
    """
    n = len(bed)
    if n == 0:
        return list(bed)
    env = _envelope(dialogue, rate, 0.02)
    dlen = len(env)
    target_gain = db_to_linear(attenuation_db)
    attack_step = 1.0 / max(1, int(attack_s * rate))
    release_step = 1.0 / max(1, int(release_s * rate))
    gain = 1.0
    out = [0] * n
    for i in range(n):
        active = i < dlen and env[i] > threshold
        goal = target_gain if active else 1.0
        if gain > goal:
            gain = max(goal, gain - attack_step)
        elif gain < goal:
            gain = min(goal, gain + release_step)
        out[i] = int(max(-32768, min(32767, round(bed[i] * gain))))
    return out


def mix_add(*tracks: Sequence[int]) -> list[int]:
    """Wide (unclamped) sample-wise sum. Python ints never overflow, so the
    true mixed waveform survives intact; the ONE place amplitude is reduced is
    the final `safe_gain_stage` before encode. Clamping each intermediate sum
    used to bake hard-clip distortion into the mix before the "safe" gain ran.
    """
    n = max((len(t) for t in tracks), default=0)
    out = [0] * n
    for t in tracks:
        for i, v in enumerate(t):
            out[i] += v
    return out


def safe_gain_stage(samples: list[int], limit: int = PEAK_LIMIT) -> list[int]:
    """The single final gain/limit: scale down (never up) so the mixed peak
    stays at or under `limit`. Deterministic — same input, same output."""
    if not samples:
        return list(samples)
    peak = max((abs(s) for s in samples), default=0)
    if peak <= limit:
        return list(samples)
    factor = limit / float(peak)
    return [int(round(s * factor)) for s in samples]


@dataclass(frozen=True)
class BusMixResult:
    pcm: bytes
    sample_rate: int
    duration_s: float
    peak: int


def build_shared_bed(
    duration_s: float,
    sfx_clips: Sequence[tuple[float, bytes]] = (),
    ambience_clips: Sequence[tuple[float, bytes]] = (),
    *,
    sample_rate: int = BUS_SAMPLE_RATE,
) -> bytes:
    """One SFX+ambience stem, independent of dialogue language.

    Every language's final master reuses this exact stem via
    `mix_language_master` — that is what keeps the effects bed byte-identical
    across zh/en/ja masters.
    """
    n = max(int(duration_s * sample_rate), 1)
    # Accumulate with unbounded Python ints — overlapping clips must sum
    # exactly, not clip sample-by-sample. One safe gain stage at the end is
    # the only level reduction, so no distortion is baked into the stem.
    sfx_bed = [0] * n
    amb_bed = [0] * n
    for start_s, blob in sfx_clips:
        rate, samples = decode_pcm16_mono(blob)
        if not samples:
            continue
        samples = apply_fade(apply_gain(to_bus_rate(samples, rate, sample_rate), SFX_GAIN_DB), sample_rate)
        offset = int(start_s * sample_rate)
        for i, v in enumerate(samples):
            j = offset + i
            if 0 <= j < n:
                sfx_bed[j] += v
    for start_s, blob in ambience_clips:
        rate, samples = decode_pcm16_mono(blob)
        if not samples:
            continue
        samples = apply_gain(to_bus_rate(samples, rate, sample_rate), AMBIENCE_GAIN_DB)
        offset = int(start_s * sample_rate)
        for i, v in enumerate(samples):
            j = offset + i
            if 0 <= j < n:
                amb_bed[j] += v
    bed = safe_gain_stage(mix_add(sfx_bed, amb_bed))
    return encode_pcm16_mono(bed, sample_rate)


def mix_language_master(
    dialogue_wav: bytes,
    shared_bed_wav: bytes | None,
    *,
    sample_rate: int = BUS_SAMPLE_RATE,
    duck: bool = True,
) -> BusMixResult:
    """Combine one language's dialogue bus with the shared SFX+ambience bed.

    Dialogue stays at 0 dB; the bed ducks under active dialogue so effects
    never bury a line; the sum is peak-limited to a safe int16 ceiling —
    never a raw additive overlay that can hard-clip.
    """
    d_rate, d_samples_raw = decode_pcm16_mono(dialogue_wav)
    d_samples = apply_gain(to_bus_rate(d_samples_raw, d_rate, sample_rate), DIALOGUE_GAIN_DB) if d_samples_raw else []
    if shared_bed_wav:
        b_rate, b_samples_raw = decode_pcm16_mono(shared_bed_wav)
        b_samples = to_bus_rate(b_samples_raw, b_rate, sample_rate) if b_samples_raw else []
    else:
        b_samples = []
    n = max(len(d_samples), len(b_samples))
    d_padded = d_samples + [0] * (n - len(d_samples))
    b_padded = b_samples + [0] * (n - len(b_samples))
    bed = duck_under_dialogue(b_padded, d_padded, sample_rate) if duck and b_padded else b_padded
    mixed = safe_gain_stage(mix_add(d_padded, bed))
    pcm = encode_pcm16_mono(mixed, sample_rate)
    peak = max((abs(v) for v in mixed), default=0)
    return BusMixResult(pcm=pcm, sample_rate=sample_rate, duration_s=n / float(sample_rate), peak=peak)

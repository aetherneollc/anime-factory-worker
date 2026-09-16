"""24 fps frame timeline, ±1 frame alignment, and 48 kHz dialogue/SFX/ambience buses."""

from __future__ import annotations

from pathlib import Path

import pytest

from anime_factory.audio_bus import (
    BUS_SAMPLE_RATE,
    build_shared_bed,
    duck_under_dialogue,
    mix_language_master,
    safe_gain_stage,
)
from anime_factory.compose import build_episode_frame_timeline, compose_episode_audio, measured_line_wav_durations
from anime_factory.timeline import (
    FrameAlignmentError,
    assert_frame_aligned,
    build_episode_timeline,
    frames_to_seconds,
    seconds_to_frames,
    write_episode_timeline,
)
from anime_factory.tts import decode_pcm16_mono, voiced_dummy_wav


def test_seconds_frames_round_trip_24fps():
    assert seconds_to_frames(1.0) == 24
    assert seconds_to_frames(8.0) == 192
    assert abs(frames_to_seconds(192) - 8.0) < 1e-9


def test_assert_frame_aligned_tolerance():
    assert_frame_aligned(100, 101)  # exactly ±1, must pass
    assert_frame_aligned(100, 100)
    with pytest.raises(FrameAlignmentError):
        assert_frame_aligned(100, 102)


def test_shot_frame_offsets_are_cumulative_and_frame_aligned():
    shots = [
        {"id": "s001", "duration": 8.0},
        {"id": "s002", "duration": 5.5},
    ]
    timeline = build_episode_timeline("EP001", shots, langs=("zh",))
    assert timeline.shots[0].start_frame == 0
    assert timeline.shots[0].end_frame == 192  # 8.0s * 24fps
    assert timeline.shots[1].start_frame == 192
    # 5.5 * 24 = 132 frames exactly
    assert timeline.shots[1].end_frame == 192 + 132
    assert timeline.total_frames == timeline.shots[-1].end_frame


def test_dialogue_events_derive_from_measured_wav_not_estimate():
    shots = [{"id": "s001", "duration": 8.0, "character_id": "ke"}]
    # Measured wav is 6.0s, not the 8.0s shot estimate.
    wav_durations = {("s001", "zh"): 6.0}
    timeline = build_episode_timeline("EP001", shots, wav_durations=wav_durations, langs=("zh",))
    assert len(timeline.dialogue) == 1
    ev = timeline.dialogue[0]
    lead_in_frames = seconds_to_frames(0.35)
    expected_start = 0 + lead_in_frames
    expected_end = expected_start + seconds_to_frames(6.0)
    assert_frame_aligned(ev.start_frame, expected_start)
    assert_frame_aligned(ev.end_frame, expected_end)
    assert ev.mouth_motion is True  # on-camera, spoken -> silent mouth-motion hint
    assert ev.character_id == "ke"


def test_dialogue_off_camera_has_no_mouth_motion_hint():
    shots = [{"id": "s001", "duration": 8.0, "character_id": "ke", "on_camera": False}]
    timeline = build_episode_timeline("EP001", shots, wav_durations={("s001", "zh"): 4.0}, langs=("zh",))
    assert timeline.dialogue[0].mouth_motion is False


def test_no_lipsync_only_boolean_hint_exposed():
    """The manifest never carries phoneme/viseme timing — only the mouth_motion bool."""
    shots = [{"id": "s001", "duration": 8.0, "character_id": "ke"}]
    timeline = build_episode_timeline("EP001", shots, wav_durations={("s001", "zh"): 4.0}, langs=("zh",))
    from anime_factory.timeline import timeline_to_manifest

    manifest = timeline_to_manifest(timeline)
    dialogue_keys = set(manifest["dialogue"][0].keys())
    assert dialogue_keys == {"shot_id", "character_id", "lang", "start_frame", "end_frame", "on_camera", "mouth_motion"}
    assert "phoneme" not in dialogue_keys
    assert "viseme" not in dialogue_keys


def test_sfx_events_from_cues_convert_seconds_to_frames():
    cues = [{"cue_key": "door_creak", "bus": "sfx", "onset_s": 2.0, "duration_s": 1.5, "shared": False}]
    shots = [{"id": "s001", "duration": 8.0}]
    timeline = build_episode_timeline("EP001", shots, sfx_cues=cues)
    assert len(timeline.sfx) == 1
    ev = timeline.sfx[0]
    assert ev.start_frame == seconds_to_frames(2.0)
    assert ev.end_frame == seconds_to_frames(3.5)


def test_write_episode_timeline_roundtrip(tmp_path):
    shots = [{"id": "s001", "duration": 8.0, "character_id": "ke"}]
    timeline = build_episode_timeline("EP001", shots, wav_durations={("s001", "zh"): 4.0}, langs=("zh",))
    path = write_episode_timeline(tmp_path, timeline)
    assert path.is_file()
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["episode_code"] == "EP001"
    assert payload["fps"] == 24


def test_measured_line_wav_durations_reads_disk_not_estimate(tmp_path):
    audio_dir = tmp_path / "audio" / "lines"
    audio_dir.mkdir(parents=True)
    wav = voiced_dummy_wav(3.3, freq=200.0)
    (audio_dir / "s001.zh.wav").write_bytes(wav)
    shots = [{"id": "s001", "duration": 8.0, "character_id": "ke"}]
    durations = measured_line_wav_durations(tmp_path, shots, ("zh",))
    assert abs(durations[("s001", "zh")] - 3.3) < 0.05


def test_dialogue_frame_span_aligns_with_shot_boundary_when_clip_fits_speech():
    """With the compose-fitted picture duration (wav + lead-in/tail headroom,
    what `fit_clip_durations_to_speech` produces), the dialogue event must end
    at or before the fitted shot boundary — the timeline builder now enforces
    this fail-closed instead of merely documenting it."""
    shots = [{"id": "s001", "duration": 4.0, "character_id": "ke"}]
    wav_durations = {("s001", "zh"): 4.0}
    # fit_clip_durations_to_speech would stretch a 4.0s clip to 4.5s for a
    # 4.0s wav (SPEECH_PAD_TAIL); the 0.35s lead-in then still fits inside.
    timeline = build_episode_timeline(
        "EP001", shots, wav_durations=wav_durations, langs=("zh",), clip_durations={"s001": 4.5}
    )
    shot = timeline.shots[0]
    dialogue = timeline.dialogue[0]
    assert dialogue.end_frame <= shot.end_frame + 1
    lead_in_frames = seconds_to_frames(0.35)
    assert_frame_aligned(dialogue.end_frame - lead_in_frames - seconds_to_frames(4.0), 0)


def test_dialogue_past_fitted_shot_boundary_fails_closed():
    """A wav that genuinely overruns its fitted picture must raise, not ship a
    manifest whose 'single source of truth' contradicts the picture."""
    shots = [{"id": "s001", "duration": 4.0, "character_id": "ke"}]
    with pytest.raises(FrameAlignmentError):
        build_episode_timeline(
            "EP001",
            shots,
            wav_durations={("s001", "zh"): 4.0},  # 0.35 lead-in + 4.0s > 4.0s shot
            langs=("zh",),
            clip_durations={"s001": 4.0},
        )


def test_sfx_event_past_episode_end_fails_closed():
    shots = [{"id": "s001", "duration": 3.0}]
    cues = [{"cue_key": "boom", "bus": "sfx", "onset_s": 2.5, "duration_s": 2.0, "shared": False}]
    with pytest.raises(FrameAlignmentError):
        build_episode_timeline("EP001", shots, sfx_cues=cues)


def test_sfx_event_negative_onset_fails_closed():
    shots = [{"id": "s001", "duration": 3.0}]
    cues = [{"cue_key": "boom", "bus": "sfx", "onset_s": -1.0, "duration_s": 0.5, "shared": False}]
    with pytest.raises(FrameAlignmentError):
        build_episode_timeline("EP001", shots, sfx_cues=cues)


def test_mouth_motion_is_metadata_only_no_lipsync_fields():
    """`mouth_motion` stays a boolean hint; validation adds no lip-sync data."""
    shots = [{"id": "s001", "duration": 8.0, "character_id": "ke"}]
    timeline = build_episode_timeline("EP001", shots, wav_durations={("s001", "zh"): 4.0}, langs=("zh",))
    assert timeline.dialogue[0].mouth_motion is True
    from anime_factory.timeline import timeline_to_manifest

    manifest = timeline_to_manifest(timeline)
    assert set(manifest["dialogue"][0]) == {
        "shot_id", "character_id", "lang", "start_frame", "end_frame", "on_camera", "mouth_motion"
    }


def test_build_episode_frame_timeline_uses_fitted_clip_durations(tmp_path):
    audio_dir = tmp_path / "audio" / "lines"
    audio_dir.mkdir(parents=True)
    wav = voiced_dummy_wav(4.0, freq=200.0)
    (audio_dir / "s001.zh.wav").write_bytes(wav)
    shots = [{"id": "s001", "duration": 8.0, "character_id": "ke", "line": {"zh": "你好"}}]
    timeline = build_episode_frame_timeline(tmp_path, "EP001", shots, ("zh",))
    # The 4.0s wav comfortably fits the 8.0s shot (no freeze-pad needed), so
    # the fitted picture duration stays the original 8.0s shot duration.
    shot = timeline.shots[0]
    assert_frame_aligned(shot.end_frame, seconds_to_frames(8.0))
    dialogue = timeline.dialogue[0]
    assert dialogue.end_frame <= shot.end_frame


# ---------------------------------------------------------------------------
# Audio buses
# ---------------------------------------------------------------------------


def test_shared_bed_is_identical_regardless_of_language():
    sfx = [(1.0, voiced_dummy_wav(0.5, freq=800.0))]
    bed_a = build_shared_bed(3.0, sfx_clips=sfx)
    bed_b = build_shared_bed(3.0, sfx_clips=sfx)
    assert bed_a == bed_b  # deterministic, byte-identical stem


def test_mix_language_master_never_hard_clips():
    dialogue = voiced_dummy_wav(2.0, freq=300.0)
    bed = build_shared_bed(2.0, sfx_clips=[(0.0, voiced_dummy_wav(2.0, freq=900.0))])
    result = mix_language_master(dialogue, bed)
    assert result.peak <= 32000
    _, samples = decode_pcm16_mono(result.pcm)
    assert samples
    assert max(abs(s) for s in samples) <= 32000


def test_duck_under_dialogue_reduces_bed_energy_when_dialogue_active():
    rate = BUS_SAMPLE_RATE
    n = rate * 2
    loud_bed = [10000] * n
    dialogue = [15000] * n  # active throughout
    ducked = duck_under_dialogue(loud_bed, dialogue, rate)
    # After the attack ramps in, the ducked bed must be well below the source.
    tail = ducked[rate:]  # second half, safely past attack ramp
    assert sum(abs(v) for v in tail) < sum(abs(v) for v in loud_bed[rate:]) * 0.6


def test_duck_under_dialogue_leaves_bed_alone_in_silence():
    rate = BUS_SAMPLE_RATE
    n = rate
    bed = [5000] * n
    silence = [0] * n
    ducked = duck_under_dialogue(bed, silence, rate)
    assert ducked[-1] == bed[-1]


def test_safe_gain_stage_only_scales_down():
    samples = [40000, -40000, 100]
    scaled = safe_gain_stage(samples, limit=32000)
    assert max(abs(v) for v in scaled) <= 32000
    quiet = [10, -10, 5]
    assert safe_gain_stage(quiet, limit=32000) == quiet


def test_mix_add_accumulates_wide_without_intermediate_clamp():
    """Intermediate sums must survive intact; only the single final
    safe_gain_stage may reduce level. Clamping inside mix_add baked hard-clip
    distortion into the mix before the 'safe' gain ever ran."""
    from anime_factory.audio_bus import mix_add

    wide = mix_add([30000, -30000], [30000, -30000])
    assert wide == [60000, -60000]  # true sum, not [32767, -32768]
    limited = safe_gain_stage(wide, limit=32000)
    assert max(abs(v) for v in limited) <= 32000
    # Waveform shape preserved: symmetric input stays symmetric after gain.
    assert limited[0] == -limited[1]


def test_shared_bed_hot_overlaps_scale_instead_of_flat_topping():
    """Three overlapping identical hot clips (sum >> int16 ceiling): the bed
    must be one uniformly scaled copy of the sum (single final gain), not a
    flat-topped clip-by-clip accumulation."""
    import math

    from anime_factory.tts import encode_pcm16_mono

    rate = BUS_SAMPLE_RATE
    n = rate // 2
    # 30000-amp sine; at the -6 dB SFX bus gain each clip lands ~15k, so three
    # overlapped copies sum to ~45k — far past int16 without wide accumulation.
    loud = encode_pcm16_mono(
        [int(30000 * math.sin(2 * math.pi * 200.0 * i / rate)) for i in range(n)], rate
    )
    single = build_shared_bed(0.6, sfx_clips=[(0.0, loud)])
    triple = build_shared_bed(0.6, sfx_clips=[(0.0, loud), (0.0, loud), (0.0, loud)])
    _, s1 = decode_pcm16_mono(single)
    _, s3 = decode_pcm16_mono(triple)
    assert max(abs(v) for v in s3) <= 32000
    ratios = [s3[i] / s1[i] for i in range(len(s1)) if abs(s1[i]) > 4000]
    assert ratios
    # One uniform gain factor across loud samples — no per-sample flattening.
    assert max(ratios) - min(ratios) < 0.05


def test_shared_sfx_stem_reused_across_language_masters(tmp_path):
    shots = [
        {
            "id": "s001",
            "duration": 3.0,
            "character_id": "ke",
            "line": {"zh": "你好世界一二三", "en": "hello there world", "ja": "こんにちは世界"},
        }
    ]
    for lang, text, freq in (("zh", "你好世界一二三", 220.0), ("en", "hello there world", 260.0), ("ja", "こんにちは世界", 300.0)):
        p = tmp_path / "audio" / "lines" / f"s001.{lang}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(voiced_dummy_wav(2.0, freq=freq))
    sfx_clips = [(0.5, voiced_dummy_wav(1.0, freq=700.0))]
    mixed, timeline = compose_episode_audio(
        tmp_path,
        "EP001",
        shots,
        langs=("zh", "en", "ja"),
        sfx_clips=sfx_clips,
    )
    assert set(mixed.keys()) == {"zh", "en", "ja"}
    for path_str in mixed.values():
        assert Path(path_str).is_file()
    assert (tmp_path / "audio" / "timeline.json").is_file()
    assert timeline.episode_code == "EP001"

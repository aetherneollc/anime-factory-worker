"""ffmpeg mux plan: video concat + one mapped TTS mux and one srt per project language.

Dialogue must land on the timeline. Silent episode beds are not a substitute
for CosyVoice line wavs. Compose fails closed if speech is missing or near-silent.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from anime_factory.config import load_settings
from anime_factory.langs import normalize_langs
from anime_factory.models import TARGET_EPISODE_SECONDS, VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH
from anime_factory.tts import (
    COSYVOICE_SAMPLE_RATE,
    VOICE_CLONE_TRANSCRIPT,
    assert_line_duration_plausible,
    decode_pcm16_mono,
    encode_pcm16_mono,
    is_clone_transcript,
    line_expected_seconds,
    pcm_duration_seconds,
    resample_pcm16_mono,
    wav_sha256,
)

log = logging.getLogger("anime_factory.compose")
_SUMPROD = getattr(math, "sumprod", None)  # 3.12+

LOUDNESS_I = -16
# Mix at the CosyVoice2 native rate; ffmpeg owns the delivery resample.
MIX_SAMPLE_RATE = COSYVOICE_SAMPLE_RATE
# Compose preflight (tts box includes timing internally). The boot state
# machine additionally requires an explicit `timing=passed` flag — see
# gpu_worker.boot.PRE_LEASE_STAGES — before any Vast lease.
NON_GPU_STAGES = ("bible", "canon", "script", "tts", "board", "design", "keyframe")

# 10-minute / 75-shot baseline. Short packs scale down; do not apply 40/70 to a 120s cut.
MIN_DIALOGUE_LINES = 40
MIN_SPEECH_SECONDS = 70.0
# int16 peak-normalized RMS of mixed episode speech (silence is ~0)
MIN_SPEECH_RMS = 80.0
SHOT_SLOT_SECONDS = 8.0
# H3 clips are often shorter than JA/EN CosyVoice. Freeze-extend picture only
# for a short tail. A pad past MAX_FREEZE_PAD_SECONDS is a director duration
# bug (retime TTS or split the beat) — never hold a mismatched store shot for 7s.
SPEECH_PAD_TAIL = 0.5
MAX_FREEZE_PAD_SECONDS = 1.5
MAX_SPEECH_OVERFLOW_RATIO = 3.0
MAX_SPEECH_OVERFLOW_SECONDS = 10.0

# 864/480 = 1.8, not 16:9. A bare scale=864:480 squashed low-VRAM 512x288 and
# LongLive 1280x704 takes; letterbox off-ratio sources instead of distorting faces.
SCALE_PAD_FILTER = (
    f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,"
    f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
    f"setsar=1,fps={VIDEO_FPS}"
)


class MissingShotError(RuntimeError):
    pass


class DialogueMissingError(RuntimeError):
    pass


class SilentAudioError(RuntimeError):
    pass


class MixDecodeError(RuntimeError):
    """mix_clips could not decode a line wav. Silently skipping it mixes a silent bed."""


class IdenticalMasterError(RuntimeError):
    """zh / ja / en masters came out byte-identical, so at most one of them is real."""


class ClonedDialogueError(RuntimeError):
    """Line wav is the CosyVoice clone transcript or a duplicate clip.

    Implausible lengths raise tts.LineDurationError instead.
    """


class AvDesyncError(RuntimeError):
    """Speech duration or identity does not match the rendered clip."""


class FreezePadError(AvDesyncError):
    """Picture freeze-extend exceeded the director duration cap."""


class H3SpeechLeakError(RuntimeError):
    """H3/Hailuo generated speech leaked into the picture mix. CosyVoice2 only."""


@dataclass
class ComposePlan:
    encode: list[str]
    muxes: list[list[str]]
    srt_outputs: list[str]
    filter_graph: str
    loudness_i: int = LOUDNESS_I


def shot_line_text(shot: dict, lang: str = "zh") -> str:
    line = shot.get("line")
    if isinstance(line, dict):
        return str(line.get(lang) or line.get("zh") or line.get("text") or "").strip()
    if isinstance(line, str):
        return line.strip()
    text = shot.get("text") or shot.get("dialogue")
    return str(text or "").strip()


def spoken_shots(shots: list[dict]) -> list[dict]:
    return [s for s in shots if shot_line_text(s)]


def estimate_speech_seconds(shots: list[dict]) -> float:
    total = 0.0
    for shot in spoken_shots(shots):
        zh = shot_line_text(shot, "zh")
        # CosyVoice ~4-5 CJK chars/s; floor 1.2s per spoken shot.
        total += max(len(zh) * 0.22, 1.2)
    return total


def episode_runtime(shots: Sequence[dict]) -> tuple[int, float]:
    n_shots = len(shots)
    duration_s = sum(float(s.get("duration") or SHOT_SLOT_SECONDS) for s in shots)
    if duration_s <= 0:
        duration_s = float(TARGET_EPISODE_SECONDS if n_shots <= 0 else n_shots * SHOT_SLOT_SECONDS)
    return n_shots, duration_s


def dialogue_gate_minima(
    shots: Sequence[dict] | None = None,
    *,
    n_shots: int | None = None,
    duration_s: float | None = None,
) -> tuple[int, float]:
    """Scale the 10-min / 75-shot / 40-line / 70s-speech compose gate.

    A 120s / 15-shot oneshot needs 15 audible lines, not 40. A 600s episode
    still needs 40 lines and ~70s of speech. Never require more lines than shots.

    min_lines = min(N, max(round(40 * duration/600), min(max(N, duration/8), 40)))
    """
    if shots is not None:
        n_shots, duration_s = episode_runtime(shots)
    else:
        n_shots = int(n_shots or 0)
        duration_s = float(duration_s or 0.0)
        if duration_s <= 0:
            duration_s = float(n_shots * SHOT_SLOT_SECONDS if n_shots else TARGET_EPISODE_SECONDS)
        if n_shots <= 0:
            n_shots = max(1, int(round(duration_s / SHOT_SLOT_SECONDS)))
    scaled_lines = max(1, int(round(MIN_DIALOGUE_LINES * duration_s / TARGET_EPISODE_SECONDS)))
    by_duration = max(1, int(round(duration_s / SHOT_SLOT_SECONDS)))
    per_shot = max(n_shots, by_duration)
    min_lines = min(n_shots, max(scaled_lines, min(per_shot, MIN_DIALOGUE_LINES)))
    min_seconds = MIN_SPEECH_SECONDS * duration_s / TARGET_EPISODE_SECONDS
    min_seconds = min(duration_s * 0.8, max(min_seconds, 1.0))
    return min_lines, min_seconds


def assert_shots_have_dialogue(
    shots: list[dict],
    min_lines: int | None = None,
    min_seconds: float | None = None,
) -> None:
    need_lines, need_seconds = dialogue_gate_minima(shots)
    if min_lines is None:
        min_lines = need_lines
    if min_seconds is None:
        min_seconds = need_seconds
    spoken = spoken_shots(shots)
    if len(spoken) < min_lines:
        raise DialogueMissingError(
            f"script gate: {len(spoken)} spoken shots, need >= {min_lines} audible lines before TTS"
        )
    seconds = estimate_speech_seconds(shots)
    if seconds < min_seconds:
        raise DialogueMissingError(
            f"script gate: ~{seconds:.0f}s estimated speech, need >= {min_seconds:.0f}s"
        )


def shot_start_times(shots: list[dict]) -> dict[str, float]:
    t = 0.0
    out: dict[str, float] = {}
    for shot in shots:
        sid = str(shot.get("id") or "")
        if sid:
            out[sid] = t
        t += float(shot.get("duration") or 8.0)
    return out


def wav_rms(path: Path) -> float:
    _rate, samples = decode_pcm16_mono(path.read_bytes())
    if len(samples) < 2:
        return 0.0
    # math.sumprod keeps a 600s 24 kHz master out of a per-sample Python loop.
    total = _SUMPROD(samples, samples) if _SUMPROD else sum(s * s for s in samples)
    return (float(total) / len(samples)) ** 0.5


def assert_wav_not_silent(path: Path, min_rms: float = MIN_SPEECH_RMS) -> None:
    if not path.is_file():
        raise SilentAudioError(f"missing dialogue wav: {path}")
    rms = wav_rms(path)
    if rms < min_rms:
        raise SilentAudioError(f"near-silence dialogue track {path.name} rms={rms:.1f} < {min_rms}")


def _decode_pcm16_mono(data: bytes) -> tuple[int, list[int]]:
    return decode_pcm16_mono(data)


def _encode_pcm16_mono(samples: list[int], rate: int) -> bytes:
    return encode_pcm16_mono(samples, rate)


def _line_expected_seconds(text: str, lang: str) -> float:
    return line_expected_seconds(text, lang)


def assert_line_wavs_are_spoken_lines(
    workdir: Path,
    shots: list[dict],
    langs: Sequence[str] | None = None,
) -> None:
    """Fail closed if CosyVoice cloned the ref transcript, or reused one wav for every shot."""
    for lang in normalize_langs(langs):
        hashes: list[str] = []
        for shot in shots:
            sid = str(shot.get("id") or "")
            text = shot_line_text(shot, lang)
            if not sid or not text:
                continue
            if is_clone_transcript(text):
                raise ClonedDialogueError(
                    f"board line {sid}.{lang} is clone transcript {VOICE_CLONE_TRANSCRIPT!r}"
                )
            path = next((p for p in _line_wav_candidates(workdir, sid, lang) if p.is_file()), None)
            if path is None:
                continue
            data = path.read_bytes()
            digest = wav_sha256(data)
            hashes.append(digest)
            # Truncated-at-cap, empty and runaway takes all fail here rather than
            # reaching the mixer and blowing up as a FreezePadError loop.
            assert_line_duration_plausible(
                pcm_duration_seconds(data), text, lang, sid=path.name
            )
        if len(hashes) >= 2 and len(set(hashes)) == 1:
            raise ClonedDialogueError(f"all {lang} line wavs are identical; clone/ref leaked")
        dup = sorted({h for h in hashes if hashes.count(h) > 1})
        if dup:
            raise ClonedDialogueError(f"duplicate {lang} line wav hashes: {len(dup)}")


def shot_parent_id(shot: dict) -> str:
    return str(shot.get("shot_id") or shot.get("id") or "").strip()


def spoken_line_units(shots: Sequence[dict], lang: str = "zh") -> int:
    """Distinct CosyVoice lines to mux. H3 chain clones of the same text count once;
    split-dialogue segments (s001 vs s002 under EP001-01) count separately."""
    seen: set[tuple[str, str]] = set()
    n = 0
    for shot in shots:
        text = shot_line_text(shot, lang)
        if not text:
            continue
        sid = str(shot.get("id") or "").strip()
        parent = shot_parent_id(shot)
        seen.add((parent or sid, text))
        n = len(seen)
    return n


def _line_wav_for_shot(workdir: Path, shot: dict, lang: str) -> Path | None:
    """Prefer the H3 segment id (s002.zh.wav) over the director parent (EP001-01.zh.wav)."""
    ids: list[str] = []
    for key in ("id", "shot_id"):
        sid = str(shot.get(key) or "").strip()
        if sid and sid not in ids:
            ids.append(sid)
    for sid in list(ids):
        if "-s" in sid:
            parent = sid.rsplit("-s", 1)[0]
            if parent and parent not in ids:
                ids.append(parent)
    for sid in ids:
        path = next((p for p in _line_wav_candidates(workdir, sid, lang) if p.is_file()), None)
        if path is not None:
            log.debug("line wav hit %s.%s -> %s", ids[0], lang, path)
            return path
    log.debug("line wav miss for %s.%s (tried ids %s)", ids[0] if ids else "?", lang, ids)
    return None


def _has_character_picture(shot: dict) -> bool:
    if str(shot.get("character_id") or "").strip():
        return True
    refs = list(shot.get("refs") or [])
    return any("char" in str(r) or "sheet" in str(r) or "costume" in str(r) for r in refs)


def assert_speech_matches_picture(shot: dict, lang: str = "zh") -> None:
    """Refuse CosyVoice on an empty plate / establishing clip (音画不相干)."""
    if not shot_line_text(shot, lang):
        return
    purpose = str(shot.get("purpose") or "")
    refs = list(shot.get("refs") or [])
    plates_only = bool(refs) and all("plate" in str(r) or "scene" in str(r) for r in refs)
    if purpose in {"establish", "hook"} and not _has_character_picture(shot):
        raise AvDesyncError(
            f"{shot.get('id')}: speech about a character while the clip is an empty plate"
        )
    if plates_only and not _has_character_picture(shot):
        raise AvDesyncError(
            f"{shot.get('id')}: CosyVoice line on a plate-only clip; picture has no character"
        )


def assert_no_h3_dialogue(
    video_paths: Sequence[Path] | None = None,
    audio_metas: dict[str, dict] | None = None,
) -> None:
    """Fail closed if Hailuo/H3 speech would enter the master. CosyVoice2 is the mix."""
    metas = audio_metas or {}
    for raw in video_paths or []:
        path = Path(raw)
        name = path.name.lower()
        if path.suffix.lower() in {".wav", ".mp3", ".flac"} and "h3" in name:
            raise H3SpeechLeakError(f"H3/Hailuo wav {path.name} must not be muxed; CosyVoice2 only")
        meta = metas.get(str(path)) or metas.get(path.name) or {}
        if meta.get("has_audio_stream") and float(meta.get("audio_rms") or 0) >= MIN_SPEECH_RMS:
            raise H3SpeechLeakError(
                f"H3/Hailuo speech leaked in {path.name}; strip picture audio and mux locked CosyVoice2"
            )


def mix_clips(
    duration_s: float,
    clips: list[tuple[float, bytes]],
    sample_rate: int = MIX_SAMPLE_RATE,
    *,
    strict: bool = True,
) -> bytes:
    """Overlay line wavs onto a bed at the CosyVoice2 native rate.

    An undecodable blob is a hard error: skipping it is how EP001 shipped three
    byte-identical 600s masters of pure digital silence while 34 good 24 kHz
    line wavs sat on disk.
    """
    n = max(int(duration_s * sample_rate), 1)
    bed = [0] * n
    skipped: list[int] = []
    for index, (start_s, blob) in enumerate(clips):
        rate, samples = _decode_pcm16_mono(blob)
        if not samples:
            skipped.append(index)
            continue
        if rate != sample_rate and rate > 0:
            samples = resample_pcm16_mono(samples, rate, sample_rate)
        offset = int(start_s * sample_rate)
        for i, val in enumerate(samples):
            j = offset + i
            if 0 <= j < n:
                mixed = bed[j] + val
                bed[j] = max(-32768, min(32767, mixed))
    if skipped:
        message = (
            f"mix_clips could not decode {len(skipped)}/{len(clips)} line wavs "
            f"(clip indices {skipped[:8]}); a silent bed is not an acceptable substitute"
        )
        if strict:
            raise MixDecodeError(message)
        log.warning(message)
    return _encode_pcm16_mono(bed, sample_rate)


def _line_wav_candidates(workdir: Path, shot_id: str, lang: str) -> list[Path]:
    return [
        workdir / "audio" / "lines" / f"{shot_id}.{lang}.wav",
        workdir / "audio" / f"{shot_id}.{lang}.wav",
        workdir / f"{shot_id}.{lang}.wav",
    ]


def clip_duration_for_speech(clip_s: float, wav_s: float, *, sid: str = "") -> float:
    """Return the picture length needed so locked CosyVoice fits this clip.

    A short tail freeze-extends the last frame. A pad longer than
    MAX_FREEZE_PAD_SECONDS is a director duration bug: retime TTS or split
    the beat. Never fill the gap with Hailuo/H3 speech.
    """
    clip_s = float(clip_s or 0.0)
    wav_s = float(wav_s or 0.0)
    if wav_s <= 0:
        return clip_s
    pad = max(0.0, wav_s - clip_s)
    if pad > MAX_FREEZE_PAD_SECONDS + 1e-6:
        raise FreezePadError(
            f"{sid or 'clip'} freeze pad {pad:.1f}s > {MAX_FREEZE_PAD_SECONDS}s "
            f"(wav {wav_s:.1f}s, clip {clip_s:.1f}s); retime TTS or split the beat, "
            "never fill with H3/Hailuo speech"
        )
    cap = max(clip_s * MAX_SPEECH_OVERFLOW_RATIO, clip_s + MAX_SPEECH_OVERFLOW_SECONDS)
    if wav_s > cap:
        raise AvDesyncError(
            f"{sid or 'clip'} wav is {wav_s:.1f}s but clip is {clip_s:.1f}s; "
            "retime/pad/split the same CosyVoice2 voice, never fill with H3 speech"
        )
    if wav_s + SPEECH_PAD_TAIL > clip_s:
        return wav_s + SPEECH_PAD_TAIL
    return clip_s


def max_line_wav_seconds(workdir: Path, shot: dict, langs: Sequence[str] | None = None) -> float:
    best = 0.0
    for lang in normalize_langs(langs):
        path = _line_wav_for_shot(workdir, shot, lang)
        if path is None:
            continue
        best = max(best, pcm_duration_seconds(path.read_bytes()))
    return best


def fit_clip_durations_to_speech(
    workdir: Path,
    shots: Sequence[dict],
    langs: Sequence[str] | None = None,
    clip_durations: dict[str, float] | None = None,
) -> dict[str, float]:
    """Stretch per-segment picture duration to the longest locked line wav."""
    fitted: dict[str, float] = dict(clip_durations or {})
    for shot in shots:
        sid = str(shot.get("id") or "").strip()
        if not sid:
            continue
        clip_s = float(fitted.get(sid) or shot.get("duration") or 8.0)
        wav_s = max_line_wav_seconds(workdir, shot, langs)
        fitted[sid] = clip_duration_for_speech(clip_s, wav_s, sid=sid)
    return fitted


def extend_clip_cmd(src: Path, dest: Path, pad_s: float) -> list[str]:
    """Freeze the last frame so the silent H3 take covers the CosyVoice line."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"tpad=stop_mode=clone:stop_duration={pad_s:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]


def chain_overlap_frames(shots: Sequence[dict]) -> list[int]:
    """Drop frame 0 of later clips that share chain_id. Cross-scene / new chain is a hard cut."""
    seen: set[str] = set()
    out: list[int] = []
    for shot in shots:
        cid = str(shot.get("chain_id") or "").strip()
        drop = 1 if cid and cid in seen else 0
        if cid:
            seen.add(cid)
        out.append(drop)
    return out


def drop_first_frames_cmd(src: Path, dest: Path, frames: int = 1) -> list[str]:
    """Trim duplicated join frames. Hard cut only — never xfade / fade."""
    n = max(1, int(frames))
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"select=gte(n\\,{n}),setpts=PTS-STARTPTS",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]


def mix_dialogue_timeline(
    workdir: Path,
    episode_code: str,
    shots: list[dict],
    langs: Sequence[str] | None = None,
    require_speech: bool = True,
    clip_durations: dict[str, float] | None = None,
    video_paths: Sequence[Path] | None = None,
    audio_metas: dict[str, dict] | None = None,
) -> dict[str, str]:
    """Overlay locked CosyVoice2 wavs onto actual clip lengths. Never mux H3/Hailuo speech."""
    aligned = []
    for shot in shots:
        row = dict(shot)
        sid = str(row.get("id") or "")
        if clip_durations and sid in clip_durations:
            row["duration"] = float(clip_durations[sid])
        aligned.append(row)
    if require_speech:
        assert_shots_have_dialogue(aligned)
        assert_line_wavs_are_spoken_lines(workdir, aligned, langs)
    assert_no_h3_dialogue(video_paths, audio_metas)
    starts = shot_start_times(aligned)
    duration = sum(float(s.get("duration") or 8.0) for s in aligned) or TARGET_EPISODE_SECONDS
    mixed: dict[str, str] = {}
    workdir.mkdir(parents=True, exist_ok=True)
    spans: dict[str, float] = {}
    for shot in aligned:
        parent = shot_parent_id(shot)
        if not parent:
            continue
        spans[parent] = spans.get(parent, 0.0) + float(shot.get("duration") or 8.0)
    for lang in normalize_langs(langs):
        clips: list[tuple[float, bytes]] = []
        placed: set[str] = set()
        missing: list[str] = []
        need_lines = max(spoken_line_units(aligned, lang), 1)
        for shot in aligned:
            if not shot_line_text(shot, lang):
                continue
            assert_speech_matches_picture(shot, lang)
            path = _line_wav_for_shot(workdir, shot, lang)
            sid = str(shot.get("id") or "")
            parent = shot_parent_id(shot)
            if path is None:
                log.warning(
                    "no %s line wav for %s; tried %s",
                    lang,
                    sid or parent,
                    [str(p) for p in _line_wav_candidates(workdir, sid or parent, lang)],
                )
                if sid:
                    missing.append(sid)
                continue
            wav_key = str(path.resolve())
            if wav_key in placed:
                continue
            data = path.read_bytes()
            wav_s = pcm_duration_seconds(data)
            clip_s = float(shot.get("duration") or 8.0)
            wav_id = path.name.split(".")[0]
            span = spans.get(parent, clip_s) if parent and wav_id == parent else clip_s
            pad = max(0.0, wav_s - span)
            if pad > MAX_FREEZE_PAD_SECONDS + 1e-6:
                raise FreezePadError(
                    f"{path.name} freeze pad {pad:.1f}s > {MAX_FREEZE_PAD_SECONDS}s "
                    f"(wav {wav_s:.1f}s, clip {sid or parent} {span:.1f}s); "
                    "retime TTS or split the beat, never fill with H3/Hailuo speech"
                )
            placed.add(wav_key)
            clips.append((starts.get(sid, 0.0) + 0.35, data))
        if missing and require_speech:
            raise DialogueMissingError(
                f"compose: {len(missing)} {lang} lines have dialogue but no line wav on disk: "
                f"{missing[:12]}; re-run TTS instead of mixing a silent bed"
            )
        if require_speech and len(clips) < need_lines:
            extra = f"; missing {missing}" if missing else ""
            raise DialogueMissingError(
                f"compose: {len(clips)} {lang} line wavs on disk, need >= {need_lines}{extra}"
            )
        if not clips:
            raise DialogueMissingError(f"compose: no {lang} line wavs to mux")
        payload = mix_clips(duration, clips, MIX_SAMPLE_RATE)
        out = workdir / f"{episode_code}.{lang}.wav"
        out.write_bytes(payload)
        # Unconditional: require_speech=False used to license a silent master.
        assert_wav_not_silent(out)
        mixed[lang] = str(out)
    assert_masters_differ(mixed)
    return mixed


def assert_masters_differ(mixed: dict[str, str]) -> None:
    """zh / ja / en masters must not be byte-identical.

    EP001's three 600s masters shared one md5 (ac54498f...) because each was the
    same all-zero bed. Identical masters mean at most one language is real.
    """
    digests: dict[str, list[str]] = {}
    for lang, path in (mixed or {}).items():
        blob = Path(path)
        if not blob.is_file():
            raise SilentAudioError(f"missing {lang} master: {path}")
        digests.setdefault(wav_sha256(blob.read_bytes()), []).append(lang)
    for langs in digests.values():
        if len(langs) > 1:
            raise IdenticalMasterError(
                f"masters for {sorted(langs)} are byte-identical; "
                "one language's line wavs were mixed for all of them"
            )


def build_compose_plan(
    workdir: Path,
    episode_code: str,
    shot_paths: list[Path],
    langs: Sequence[str] | None = None,
    require_audio: bool = False,
) -> ComposePlan:
    _ = shot_paths
    concat = workdir / "concat.txt"
    encoded = workdir / f"{episode_code}.video.mp4"
    filter_graph = f"loudnorm=I={LOUDNESS_I}:TP=-1.5:LRA=11"
    encode = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-vf",
        SCALE_PAD_FILTER,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(encoded),
    ]
    blob = " ".join(str(x) for x in encode).lower()
    if "xfade" in blob:
        raise RuntimeError("compose encode must hard-cut; xfade is forbidden")
    muxes = []
    srts = []
    for lang in normalize_langs(langs):
        audio = workdir / f"{episode_code}.{lang}.wav"
        if require_audio:
            assert_wav_not_silent(audio)
        out = workdir / "final" / f"{episode_code}.{lang}.mp4"
        srt = workdir / "final" / f"{episode_code}.{lang}.srt"
        muxes.append(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(encoded),
                "-i",
                str(audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-af",
                filter_graph,
                "-c:a",
                "aac",
                str(out),
            ]
        )
        srts.append(str(srt))
    return ComposePlan(encode=encode, muxes=muxes, srt_outputs=srts, filter_graph=filter_graph)


def collect_shot_paths(conn: sqlite3.Connection, episode_code: str) -> list[tuple[str, str]]:
    """(segment_id, video_path) in seq order.

    Compose must pair picture with board rows by id: with
    COMPOSE_ALLOW_MISSING_SHOTS=true a dropped shot made zip(shots, paths)
    shift every later clip against its dialogue.
    """
    rows = conn.execute(
        "SELECT id, video_path, status FROM segments WHERE episode_code = ? ORDER BY seq",
        (episode_code,),
    ).fetchall()
    missing = [r["id"] for r in rows if r["status"] != "completed" or not r["video_path"]]
    if missing and not load_settings().compose_allow_missing_shots:
        raise MissingShotError(f"compose blocked; missing shots: {missing}")
    return [(str(r["id"]), r["video_path"]) for r in rows if r["video_path"]]


def collect_shots(conn: sqlite3.Connection, episode_code: str) -> list[str]:
    return [path for _sid, path in collect_shot_paths(conn, episode_code)]


def non_gpu_stages_passed(flags: dict[str, str]) -> bool:
    return all(flags.get(stage) == "passed" for stage in NON_GPU_STAGES)


def subtitle_cues(shots: list[dict], lang: str, lead_in: float = 0.35) -> list[tuple[float, float, str]]:
    starts = shot_start_times(shots)
    cues: list[tuple[float, float, str]] = []
    seen: set[tuple[str, str]] = set()
    for shot in shots:
        text = shot_line_text(shot, lang)
        sid = str(shot.get("id") or "")
        parent = shot_parent_id(shot)
        if not text or not sid:
            continue
        key = (parent or sid, text)
        if key in seen:
            continue
        seen.add(key)
        start = starts.get(sid, 0.0) + lead_in
        same = [
            float(s.get("duration") or 8.0)
            for s in shots
            if shot_parent_id(s) == parent and shot_line_text(s, lang) == text
        ]
        span = sum(same) if same else float(shot.get("duration") or 8.0)
        cues.append((start, start + max(span - lead_in, 1.0), text))
    return cues


def write_episode_srts(
    workdir: Path,
    episode_code: str,
    shots: list[dict],
    langs: Sequence[str] | None = None,
) -> dict[str, str]:
    """One srt per project language, named to match build_compose_plan().srt_outputs."""
    out: dict[str, str] = {}
    for lang in normalize_langs(langs):
        path = workdir / "final" / f"{episode_code}.{lang}.srt"
        write_srt(path, subtitle_cues(shots, lang))
        out[lang] = str(path)
    return out


def write_srt(path: Path, cues: list[tuple[float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (start, end, text) in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{_ts(start)} --> {_ts(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"

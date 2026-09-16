"""24 fps integer-frame episode timeline: one manifest for video duration,
subtitle timing, and dialogue/SFX audio placement.

Frame counts are derived from *measured* WAV durations — the actual
synthesized dialogue and resolved SFX audio already on disk — not from the
director's target `duration` estimate. A frame-accurate timeline cannot be
built from a plan the real audio may have overrun or underrun; compose
already fits picture duration to the locked speech (`fit_clip_durations_to_speech`)
so by the time this manifest is built, "planned" and "measured" should
already agree to within the ±1 frame tolerance enforced by
`assert_frame_aligned`.

No lip-sync is implemented or implied here. `DialogueEvent.mouth_motion` is
only a boolean hint — whether an on-camera line should get natural idle/talk
mouth-motion metadata downstream — never phoneme/viseme timing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from anime_factory.models import VIDEO_FPS

FPS = VIDEO_FPS
FRAME_ALIGNMENT_TOLERANCE = 1  # ±1 frame, per spec


class FrameAlignmentError(RuntimeError):
    """Two frame counts that should represent the same instant disagree by >1 frame."""


def seconds_to_frames(seconds: float, fps: int = FPS) -> int:
    return int(round(float(seconds or 0.0) * fps))


def frames_to_seconds(frames: int, fps: int = FPS) -> float:
    return float(frames) / float(fps or FPS)


def assert_frame_aligned(
    a_frames: int, b_frames: int, *, label: str = "frames", tolerance: int = FRAME_ALIGNMENT_TOLERANCE
) -> None:
    if abs(int(a_frames) - int(b_frames)) > tolerance:
        raise FrameAlignmentError(
            f"{label}: {a_frames} vs {b_frames} frames differ by more than ±{tolerance} frame(s)"
        )


@dataclass(frozen=True)
class ShotFrame:
    shot_id: str
    start_frame: int
    end_frame: int
    duration_s: float


@dataclass(frozen=True)
class DialogueEvent:
    shot_id: str
    character_id: str
    lang: str
    start_frame: int
    end_frame: int
    on_camera: bool = True

    @property
    def mouth_motion(self) -> bool:
        """Natural silent mouth-motion hint for on-camera dialogue. Not lip-sync:
        no phoneme/viseme timing is produced or consumed, only this boolean."""
        return self.on_camera


@dataclass(frozen=True)
class SfxEvent:
    cue_key: str
    bus: str
    start_frame: int
    end_frame: int
    shared: bool = False


@dataclass(frozen=True)
class EpisodeTimeline:
    episode_code: str
    fps: int
    total_frames: int
    shots: tuple[ShotFrame, ...]
    dialogue: tuple[DialogueEvent, ...]
    sfx: tuple[SfxEvent, ...]

    @property
    def total_seconds(self) -> float:
        return frames_to_seconds(self.total_frames, self.fps)


def shot_frame_offsets(
    shots: Sequence[dict], durations: dict[str, float] | None = None, fps: int = FPS
) -> list[ShotFrame]:
    """Cumulative frame-grid offsets. `durations[shot_id]` overrides `shot['duration']`
    — pass the compose-fitted picture durations, which already match the locked speech."""
    out: list[ShotFrame] = []
    cursor = 0
    for shot in shots:
        sid = str(shot.get("id") or "")
        duration = float((durations or {}).get(sid, shot.get("duration") or 8.0))
        n_frames = max(1, seconds_to_frames(duration, fps))
        out.append(ShotFrame(shot_id=sid, start_frame=cursor, end_frame=cursor + n_frames, duration_s=duration))
        cursor += n_frames
    return out


def dialogue_events_from_wavs(
    shots: Sequence[dict],
    shot_frames: Sequence[ShotFrame],
    wav_durations: dict[tuple[str, str], float],
    langs: Sequence[str],
    lead_in_s: float = 0.35,
    fps: int = FPS,
) -> list[DialogueEvent]:
    """One event per (shot, lang) that has a measured wav duration > 0."""
    frame_by_shot = {f.shot_id: f for f in shot_frames}
    events: list[DialogueEvent] = []
    for shot in shots:
        sid = str(shot.get("id") or "")
        frame = frame_by_shot.get(sid)
        if frame is None:
            continue
        cid = str(shot.get("character_id") or "")
        on_camera = bool(cid) and bool(shot.get("on_camera", True))
        for lang in langs:
            wav_s = wav_durations.get((sid, lang))
            if not wav_s:
                continue
            start_frame = frame.start_frame + seconds_to_frames(lead_in_s, fps)
            end_frame = start_frame + seconds_to_frames(wav_s, fps)
            events.append(
                DialogueEvent(
                    shot_id=sid,
                    character_id=cid,
                    lang=lang,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    on_camera=on_camera,
                )
            )
    return events


def sfx_events_from_cues(cues: Sequence[dict], fps: int = FPS) -> list[SfxEvent]:
    """`cues` rows: {"cue_key", "bus", "onset_s", "duration_s", "shared"}."""
    out: list[SfxEvent] = []
    for cue in cues:
        onset = float(cue.get("onset_s") or 0.0)
        duration = float(cue.get("duration_s") or 0.0)
        out.append(
            SfxEvent(
                cue_key=str(cue.get("cue_key") or ""),
                bus=str(cue.get("bus") or "sfx"),
                start_frame=seconds_to_frames(onset, fps),
                end_frame=seconds_to_frames(onset + duration, fps),
                shared=bool(cue.get("shared")),
            )
        )
    return out


def build_episode_timeline(
    episode_code: str,
    shots: Sequence[dict],
    *,
    wav_durations: dict[tuple[str, str], float] | None = None,
    sfx_cues: Sequence[dict] | None = None,
    langs: Sequence[str] = ("zh",),
    clip_durations: dict[str, float] | None = None,
    fps: int = FPS,
) -> EpisodeTimeline:
    shot_frames = shot_frame_offsets(shots, clip_durations, fps)
    total_frames = shot_frames[-1].end_frame if shot_frames else 0
    dialogue = dialogue_events_from_wavs(shots, shot_frames, wav_durations or {}, langs, fps=fps)
    sfx = sfx_events_from_cues(sfx_cues or [], fps=fps)
    return EpisodeTimeline(
        episode_code=episode_code,
        fps=fps,
        total_frames=total_frames,
        shots=tuple(shot_frames),
        dialogue=tuple(dialogue),
        sfx=tuple(sfx),
    )


def timeline_to_manifest(timeline: EpisodeTimeline) -> dict:
    return {
        "episode_code": timeline.episode_code,
        "fps": timeline.fps,
        "total_frames": timeline.total_frames,
        "total_seconds": timeline.total_seconds,
        "shots": [
            {
                "shot_id": s.shot_id,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "duration_s": s.duration_s,
            }
            for s in timeline.shots
        ],
        "dialogue": [
            {
                "shot_id": d.shot_id,
                "character_id": d.character_id,
                "lang": d.lang,
                "start_frame": d.start_frame,
                "end_frame": d.end_frame,
                "on_camera": d.on_camera,
                "mouth_motion": d.mouth_motion,
            }
            for d in timeline.dialogue
        ],
        "sfx": [
            {
                "cue_key": e.cue_key,
                "bus": e.bus,
                "start_frame": e.start_frame,
                "end_frame": e.end_frame,
                "shared": e.shared,
            }
            for e in timeline.sfx
        ],
    }


def write_episode_timeline(episode_dir: Path, timeline: EpisodeTimeline) -> Path:
    """`episode_dir` is the per-episode workdir (e.g. `<root>/episodes/<EP>`),
    the same directory `compose.mix_dialogue_timeline` reads/writes under."""
    path = Path(episode_dir) / "audio" / "timeline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline_to_manifest(timeline), ensure_ascii=False, indent=2), encoding="utf-8")
    return path

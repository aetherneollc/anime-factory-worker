"""Film director: acts/coverage (master + shot-reverse + OTS), eyeline axis, match_cut.

Does not pad to 600s and does not cut one spoken line into an isolated I2V shot.
Long coverage shots split into chained segments at persist time.
"""

from __future__ import annotations

from typing import Any, Sequence

from anime_factory.directors.common import (
    ESTABLISHING_SECONDS,
    emit_shot,
    group_consecutive_speaker_lines,
    line_on_camera,
    merge_line_langs,
    opposite_eyeline,
    spoken_items,
    world_maps,
)
from anime_factory.models import H3_MAX_SECONDS, scrub_copycat


def _scene_listener(scene: dict, speaker: str) -> str:
    for cid in scene.get("characters") or []:
        if str(cid) and str(cid) != speaker:
            return str(cid)
    return ""


def board_film(
    script: dict,
    *,
    episode_code: str,
    langs: Sequence[str],
    target_seconds: float | None = None,
    shot_seconds: float = H3_MAX_SECONDS,
) -> list[dict[str, Any]]:
    del shot_seconds  # H3 max is a segment constraint, not a film shot length
    scenes, cast, locations, interiors = world_maps(script)
    if not scenes:
        raise ValueError(f"{episode_code}: script has no scenes to board")
    spoken = spoken_items(scenes, langs, episode_code)
    if not spoken:
        raise ValueError(f"{episode_code}: script has no spoken lines to board")

    shots: list[dict[str, Any]] = []
    total_hint = max(8, len(spoken) * 2 + len(scenes))

    def emit(**kwargs: Any) -> dict[str, Any]:
        return emit_shot(
            shots,
            episode_code=episode_code,
            locations=locations,
            interiors=interiors,
            cast=cast,
            total_hint=total_hint,
            **kwargs,
        )

    for si, scene in enumerate(scenes):
        scene_lines = [item for item in spoken if item[0] == si]
        emit(
            scene=scene,
            size="wide",
            camera="Static Shot" if si % 2 == 0 else "Slow Pan",
            character_id="",
            staging=str(scene.get("synopsis") or "establishing only, no people entering frame"),
            duration=ESTABLISHING_SECONDS,
            line=None,
            purpose="establish",
        )
        eyeline = "right"
        for gi, group in enumerate(group_consecutive_speaker_lines(scene_lines)):
            speaker = str(group[0][2].get("character_id") or "")
            duration = sum(item[3] for item in group)
            staging = scrub_copycat(
                str(group[0][2].get("staging") or "speaking, locked eyeline, coverage")
            )
            emit(
                scene=scene,
                size="MCU" if gi % 2 == 0 else "medium",
                camera="Static Shot",
                character_id=speaker,
                staging=staging,
                duration=duration,
                line=merge_line_langs([item[2] for item in group], langs),
                purpose="coverage",
                eyeline=eyeline,
                on_camera=line_on_camera(group[0][2]),
                visual_en=str(group[0][2].get("visual_en") or ""),
                motion_zh=str(group[0][2].get("motion_zh") or ""),
            )
            listener = _scene_listener(scene, speaker) or str(
                (group[0][1].get("characters") or [""])[-1] or ""
            )
            if listener and listener != speaker:
                emit(
                    scene=scene,
                    size="OTS" if gi % 2 == 0 else "CU",
                    camera="Static Shot",
                    character_id=listener,
                    staging="silent reverse / over-shoulder, hold eyeline, no new subjects",
                    duration=min(3.0, max(2.0, duration * 0.35)),
                    line=None,
                    purpose="reverse",
                    eyeline=opposite_eyeline(eyeline),
                )
            eyeline = opposite_eyeline(eyeline)
        if shots and si + 1 < len(scenes):
            shots[-1]["match_cut"] = True

    if target_seconds and shots:
        from anime_factory.directors.common import scale_shot_durations

        natural = sum(float(s["duration"]) for s in shots)
        # Scale only when the film's declared runtime is close to coverage length.
        if natural > 0 and 0.7 <= (float(target_seconds) / natural) <= 1.3:
            scale_shot_durations(shots, float(target_seconds))
    return shots

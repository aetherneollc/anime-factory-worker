"""Series director: season/episode grammar, 600s target, same-scene identity lock.

Filler holds are allowed so an episode can hit its runtime. Each spoken line is a
shot that may split into chained segments; character identity is ref2va + sheet.
"""

from __future__ import annotations

from typing import Any, Sequence

from anime_factory.directors.common import (
    ESTABLISHING_SECONDS,
    FILLER_SIZES,
    MIN_SHOT_SECONDS,
    SPOKEN_SIZES,
    _filler_plan,
    emit_shot,
    line_langs,
    line_on_camera,
    spoken_items,
    world_maps,
)
from anime_factory.models import H3_MAX_SECONDS, TARGET_EPISODE_SECONDS, scrub_copycat


def board_series(
    script: dict,
    *,
    episode_code: str,
    langs: Sequence[str],
    target_seconds: float = TARGET_EPISODE_SECONDS,
    shot_seconds: float = H3_MAX_SECONDS,
) -> list[dict[str, Any]]:
    scenes, cast, locations, interiors = world_maps(script)
    if not scenes:
        raise ValueError(f"{episode_code}: script has no scenes to board")
    spoken = spoken_items(scenes, langs, episode_code)
    if not spoken:
        raise ValueError(f"{episode_code}: script has no spoken lines to board")

    spoken_seconds = sum(item[3] for item in spoken)
    establishing_seconds = len(scenes) * ESTABLISHING_SECONDS
    budget = float(target_seconds) - spoken_seconds - establishing_seconds
    filler_total, filler_seconds = _filler_plan(budget, shot_seconds)

    per_scene_lines: dict[int, int] = {}
    for si, _scene, _line, _dur in spoken:
        per_scene_lines[si] = per_scene_lines.get(si, 0) + 1
    quota: dict[int, int] = {}
    left = filler_total
    for si in sorted(per_scene_lines):
        share = int(filler_total * per_scene_lines[si] / len(spoken))
        quota[si] = min(share, left)
        left -= quota[si]
    for si in sorted(per_scene_lines):
        if left <= 0:
            break
        quota[si] += 1
        left -= 1

    shots: list[dict[str, Any]] = []
    total_shots = len(spoken) + len(scenes) + filler_total

    def emit(
        scene: dict,
        size: str,
        camera: str,
        character_id: str,
        staging: str,
        duration: float,
        line: dict | None,
        purpose: str,
        on_camera: bool | None = None,
        visual_en: str = "",
        motion_zh: str = "",
    ) -> None:
        emit_shot(
            shots,
            episode_code=episode_code,
            scene=scene,
            locations=locations,
            interiors=interiors,
            cast=cast,
            size=size,
            camera=camera,
            character_id=character_id,
            staging=staging,
            duration=duration,
            line=line,
            total_hint=total_shots,
            purpose=purpose,
            on_camera=on_camera,
            visual_en=visual_en,
            motion_zh=motion_zh,
        )

    for si, scene in enumerate(scenes):
        scene_lines = [item for item in spoken if item[0] == si]
        emit(
            scene,
            "wide",
            "Static Shot" if si % 2 == 0 else "Slow Pan",
            "",
            str(scene.get("synopsis") or "establishing only, no people entering frame"),
            ESTABLISHING_SECONDS,
            None,
            "establish",
        )
        fillers_left = quota.get(si, 0)
        stride = max(1, len(scene_lines) // (fillers_left + 1)) if fillers_left else 0
        for li, (_si, _scene, line, duration) in enumerate(scene_lines):
            cid = str(line.get("character_id") or "")
            size = str(line.get("size") or SPOKEN_SIZES[li % len(SPOKEN_SIZES)])
            camera = str(line.get("camera") or "Static Shot")
            staging = scrub_copycat(str(line.get("staging") or "speaking, desk-level eyeline"))
            # line_langs drops everything but the dialogue text, so the director's
            # own picture description has to be passed alongside it.
            emit(
                scene,
                size,
                camera,
                cid,
                staging,
                duration,
                line_langs(line, langs),
                "dialogue",
                line_on_camera(line),
                visual_en=str(line.get("visual_en") or ""),
                motion_zh=str(line.get("motion_zh") or ""),
            )
            if fillers_left and stride and (li + 1) % stride == 0:
                emit(
                    scene,
                    FILLER_SIZES[fillers_left % len(FILLER_SIZES)],
                    "Static Shot",
                    cid,
                    "silent reaction beat, no new dialogue, no new subjects",
                    filler_seconds,
                    None,
                    "hold",
                )
                fillers_left -= 1
        while fillers_left:
            emit(
                scene,
                FILLER_SIZES[fillers_left % len(FILLER_SIZES)],
                "Static Shot",
                str((scene.get("characters") or [""])[0] or ""),
                "silent hold on the set, no new subjects",
                filler_seconds,
                None,
                "hold",
            )
            fillers_left -= 1

    drift = float(target_seconds) - sum(float(s["duration"]) for s in shots)
    if shots and abs(drift) > 0.01:
        tail = shots[-1]
        adjusted = float(tail["duration"]) + drift
        if adjusted >= MIN_SHOT_SECONDS:
            tail["duration"] = round(adjusted, 2)
            tail["cuts"][0]["seconds"] = tail["duration"]
    return shots

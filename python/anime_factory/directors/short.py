"""Short director: CineFlow long takes, not one-line-one-H3.

Pack a scene's consecutive dialogue into few ≤8s shots with First/Then
intra-shot cuts. Same scene_id shares chain_id; a new scene_id is a hard cut.
"""

from __future__ import annotations

from typing import Any, Sequence

from anime_factory.directors.common import (
    CINEFLOW_BRIDGE,
    ESTABLISHING_SECONDS,
    LEAD_IN_SECONDS,
    MIN_SHOT_SECONDS,
    STAGING_FOR_BEAT,
    TAIL_SECONDS,
    cineflow_cut_script,
    emit_shot,
    line_beat_id,
    line_on_camera,
    line_speech_seconds,
    merge_line_langs,
    pack_consecutive_dialogue,
    scale_shot_durations,
    scene_dialogue_chain_id,
    spoken_items,
    world_maps,
)
from anime_factory.models import H3_MAX_SECONDS, TARGET_EPISODE_SECONDS, scrub_copycat

_CONTINUE_CAMERAS = ("Static Shot", "Slow Pan", "Push-in")


def board_short(
    script: dict,
    *,
    episode_code: str,
    langs: Sequence[str],
    target_seconds: float | None = None,
    shot_seconds: float = H3_MAX_SECONDS,
) -> list[dict[str, Any]]:
    del shot_seconds
    scenes, cast, locations, interiors = world_maps(script)
    if not scenes:
        raise ValueError(f"{episode_code}: script has no scenes to board")
    spoken = spoken_items(scenes, langs, episode_code)
    if not spoken:
        raise ValueError(f"{episode_code}: script has no spoken lines to board")

    # A 600s default is the series episode length. Shorts must not inherit it as padding.
    creative = None if target_seconds is None else float(target_seconds)
    if creative is not None and creative >= TARGET_EPISODE_SECONDS * 0.9:
        creative = None

    shots: list[dict[str, Any]] = []
    total_hint = max(4, len(scenes) + max(1, (len(spoken) + 2) // 3))

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
        scene_id = str(scene.get("id") or f"{episode_code}-sc{si + 1:02d}")
        scene_lines = [item for item in spoken if item[0] == si]
        emit(
            scene=scene,
            size="wide",
            camera="Static Shot",
            character_id="",
            staging=str(scene.get("synopsis") or "hook / establish the one location, no extra people"),
            duration=min(ESTABLISHING_SECONDS, 4.0) if si == 0 else 3.0,
            line=None,
            purpose="hook" if si == 0 else "establish",
            chain_id=f"{scene_id}-plate",
        )
        dialogue_chain = scene_dialogue_chain_id(scene, episode_code)
        for gi, group in enumerate(pack_consecutive_dialogue(scene_lines)):
            on_screen: list[str] = []
            cuts: list[dict[str, Any]] = []
            speaker = ""
            for item in group:
                line = item[2]
                cid = str(line.get("character_id") or "")
                on_cam = line_on_camera(line)
                if not speaker and cid:
                    speaker = cid
                if on_cam and cid and cid not in on_screen:
                    on_screen.append(cid)
                beat = line_beat_id(line)
                staging = scrub_copycat(
                    str(line.get("staging") or STAGING_FOR_BEAT.get(beat) or "speaking, locked eyeline")
                )
                size = str(line.get("size") or ("MCU" if len(cuts) % 2 == 0 else "CU"))
                camera = str(line.get("camera") or _CONTINUE_CAMERAS[gi % len(_CONTINUE_CAMERAS)])
                cuts.append(
                    {
                        "seq": len(cuts) + 1,
                        "beats": [1, 1],
                        "seconds": round(float(item[3]), 2),
                        "size": size,
                        "camera": camera,
                        "characters": [cid] if on_cam and cid else [],
                        "staging": staging,
                    }
                )
            # A CineFlow take covers several lines; the first described one leads the still.
            visual_en = next((str(item[2].get("visual_en") or "") for item in group if item[2].get("visual_en")), "")
            motion_zh = next((str(item[2].get("motion_zh") or "") for item in group if item[2].get("motion_zh")), "")
            speech = sum(line_speech_seconds(item[2]) for item in group)
            raw = max(MIN_SHOT_SECONDS, speech + LEAD_IN_SECONDS + TAIL_SECONDS)
            duration = raw if len(group) == 1 else min(H3_MAX_SECONDS, raw)
            size = str(cuts[0]["size"]) if cuts else "MCU"
            camera = str(cuts[0]["camera"]) if cuts else "Static Shot"
            staging = cineflow_cut_script(cuts) or (str(cuts[0]["staging"]) if cuts else "speaking")
            if gi > 0:
                staging = f"延续上一镜头, continue same scene long take. {staging}"
            elif CINEFLOW_BRIDGE not in staging and len(cuts) == 1:
                staging = f"First: {staging}"
            emit(
                scene=scene,
                size=size,
                camera=camera,
                character_id=speaker,
                staging=staging,
                duration=duration,
                line=merge_line_langs([item[2] for item in group], langs),
                purpose="dialogue",
                on_camera=any(line_on_camera(item[2]) for item in group) if group else True,
                on_screen_ids=on_screen,
                chain_id=dialogue_chain,
                cut_list=cuts,
                visual_en=visual_en,
                motion_zh=motion_zh,
            )

    if creative:
        scale_shot_durations(shots, creative)
    return shots

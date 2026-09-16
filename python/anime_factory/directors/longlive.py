"""LongLive 2.0 director: few ultra-long AR takes, not MiniMax H3 8s slices.

One take ≈ one scene / continuous location. Duration follows speech, capped at
NVlabs default 384 latent frames (~64s @ 24fps on a 5090 with local_attn_size=32).
Prompt is CineFlow First/Then inside one generation. Same-scene split only when
the take would exceed the cap (last frame → next first frame). Cross-scene is a
hard cut with a new first frame from the character turnaround / scene plate.
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
from anime_factory.models import (
    H3_MAX_SECONDS,
    LONGLIVE_MAX_SECONDS,
    LONGLIVE_SHORT_MAX_SECONDS,
    LONGLIVE_SHORT_TARGET_SECONDS,
    TARGET_EPISODE_SECONDS,
    scrub_copycat,
)

_CONTINUE_CAMERAS = ("Static Shot", "Slow Pan", "Push-in")


def pack_longlive_takes(
    scene_lines: Sequence[tuple[int, dict, dict, float]],
    max_s: float | None = None,
) -> list[list[tuple[int, dict, dict, float]]]:
    """Pack a scene into few AR takes. Split only for the 64s cap or remote vs on-camera."""
    limit = float(max_s) if max_s and max_s > 0 else LONGLIVE_MAX_SECONDS
    return pack_consecutive_dialogue(scene_lines, max_s=limit)


def board_longlive(
    script: dict,
    *,
    episode_code: str,
    langs: Sequence[str],
    target_seconds: float | None = None,
    shot_seconds: float = LONGLIVE_MAX_SECONDS,
) -> list[dict[str, Any]]:
    take_max = max(H3_MAX_SECONDS, float(shot_seconds or LONGLIVE_MAX_SECONDS))
    short_mode = False
    if target_seconds is not None:
        creative_hint = float(target_seconds)
        if 0 < creative_hint <= LONGLIVE_SHORT_TARGET_SECONDS * 1.25:
            short_mode = True
            take_max = min(take_max, LONGLIVE_SHORT_MAX_SECONDS)
    scenes, cast, locations, interiors = world_maps(script)
    if not scenes:
        raise ValueError(f"{episode_code}: script has no scenes to board")
    spoken = spoken_items(scenes, langs, episode_code)
    if not spoken:
        raise ValueError(f"{episode_code}: script has no spoken lines to board")

    creative = None if target_seconds is None else float(target_seconds)
    if creative is not None and creative >= TARGET_EPISODE_SECONDS * 0.9:
        creative = None

    shots: list[dict[str, Any]] = []
    total_hint = max(2, len(scenes) + 1)

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
        establish = scrub_copycat(
            str(scene.get("synopsis") or "hook / establish the one location, no extra people")
        )
        dialogue_chain = scene_dialogue_chain_id(scene, episode_code)
        groups = pack_longlive_takes(scene_lines, max_s=take_max)
        if not groups:
            emit(
                scene=scene,
                size="wide",
                camera="Static Shot",
                character_id="",
                staging=f"First: {establish}",
                duration=min(ESTABLISHING_SECONDS, take_max),
                line=None,
                purpose="establish",
                chain_id=dialogue_chain,
            )
            continue
        for gi, group in enumerate(groups):
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
            visual_en = next((str(item[2].get("visual_en") or "") for item in group if item[2].get("visual_en")), "")
            motion_zh = next((str(item[2].get("motion_zh") or "") for item in group if item[2].get("motion_zh")), "")
            speech = sum(line_speech_seconds(item[2]) for item in group)
            raw = max(MIN_SHOT_SECONDS, speech + LEAD_IN_SECONDS + TAIL_SECONDS)
            duration = round(min(take_max, raw), 2)
            size = str(cuts[0]["size"]) if cuts else "MCU"
            camera = str(cuts[0]["camera"]) if cuts else "Static Shot"
            staging = cineflow_cut_script(cuts) or (str(cuts[0]["staging"]) if cuts else "speaking")
            if gi == 0:
                head = f"First: {establish}"
                body = staging if staging.lower().startswith(("first:", "then:", "finally:")) else f"Then: {staging}"
                staging = f"{head} {CINEFLOW_BRIDGE} {body}"
            else:
                staging = f"延续上一镜头, continue same scene long take. {staging}"
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
    if short_mode:
        from anime_factory.longlive_workflow import LongLiveWorkflowError, pack_short_takes

        try:
            packed = pack_short_takes(shots, max_take_s=take_max)
        except LongLiveWorkflowError:
            return shots
        by_id = {str(s.get("id") or ""): s for s in shots}
        merged: list[dict[str, Any]] = []
        for take in packed:
            head_id = str((take.get("source_shot_ids") or [take.get("id")])[0] or "")
            head = dict(by_id.get(head_id) or shots[0])
            head.update(
                {
                    "id": take["take_id"],
                    "take_id": take["take_id"],
                    "duration": take["duration"],
                    "start_frame": take["start_frame"],
                    "end_frame": take["end_frame"],
                    "cuts": take.get("cuts") or head.get("cuts"),
                    "cast": take.get("cast"),
                    "continue_from": take.get("continue_from"),
                    "keyframe_source": take.get("keyframe_source"),
                    "scene_id": take.get("scene_id") or head.get("scene_id"),
                    "h3_prompt": take.get("prompt") or head.get("h3_prompt"),
                }
            )
            merged.append(head)
        return merged
    return shots

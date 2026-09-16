"""Machine-readable LongLive 2.0 short workflow: 2–3 silent takes ≤60s.

A ~120s short packs by scene / dramatic beat. Native 1280×704 is only
scale/padded at compose. First take is QC keyframe I2V; same-scene
continuation may use the prior last frame; scene changes are hard cuts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from anime_factory.models import (
    LONGLIVE_SHORT_MAX_SECONDS,
    LONGLIVE_SHORT_MAX_TAKES,
    LONGLIVE_SHORT_MIN_TAKES,
    LONGLIVE_SHORT_TARGET_SECONDS,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / "workflows"
WORKFLOW_NAME = "longlive2_short"
SCHEMA_NAME = "longlive2_short"
LONGLIVE_NATIVE_WIDTH = 1280
LONGLIVE_NATIVE_HEIGHT = 704
# A same-scene take continues by conditioning on the previous take's last frame
# (upstream i2v: one clean first latent). Each take is an independent rollout —
# CausalDiffusionInferencePipeline.inference resets its KV cache on every call, so
# there is no latent or attention carry-over between takes to claim.
CONTINUATION_MECHANISM = "i2v_first_frame"


class LongLiveWorkflowError(ValueError):
    """Workflow JSON failed validation."""


def load_longlive_short_workflow(path: Path | None = None) -> dict[str, Any]:
    dest = path or (WORKFLOWS_DIR / f"{WORKFLOW_NAME}.json")
    data = json.loads(Path(dest).read_text(encoding="utf-8"))
    validate_longlive_short_workflow(data)
    return data


def validate_longlive_short_workflow(data: dict[str, Any] | Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise LongLiveWorkflowError("longlive2_short workflow must be an object")
    if data.get("schema") != SCHEMA_NAME:
        raise LongLiveWorkflowError(f"expected schema {SCHEMA_NAME!r}")
    if int(data.get("native_width") or 0) != LONGLIVE_NATIVE_WIDTH:
        raise LongLiveWorkflowError(f"native_width must be {LONGLIVE_NATIVE_WIDTH}")
    if int(data.get("native_height") or 0) != LONGLIVE_NATIVE_HEIGHT:
        raise LongLiveWorkflowError(f"native_height must be {LONGLIVE_NATIVE_HEIGHT}")
    if data.get("audio") or data.get("silent") is False:
        raise LongLiveWorkflowError("longlive2_short output must be silent")
    if data.get("continuation_mechanism") != CONTINUATION_MECHANISM:
        raise LongLiveWorkflowError(
            f"continuation_mechanism must be {CONTINUATION_MECHANISM!r}: takes are "
            "independent rollouts conditioned on the prior last frame, not a shared "
            "latent or KV context"
        )
    if float(data.get("max_take_seconds") or 0) > LONGLIVE_SHORT_MAX_SECONDS + 1e-9:
        raise LongLiveWorkflowError("max_take_seconds must be <= 60")
    compose = data.get("compose") or {}
    if not compose.get("scale_pad_only"):
        raise LongLiveWorkflowError("native 1280x704 may only scale/pad at compose")
    if int(compose.get("delivery_width") or 0) != VIDEO_WIDTH:
        raise LongLiveWorkflowError(f"compose delivery_width must be {VIDEO_WIDTH}")
    if int(compose.get("delivery_height") or 0) != VIDEO_HEIGHT:
        raise LongLiveWorkflowError(f"compose delivery_height must be {VIDEO_HEIGHT}")
    return data


def _frame_at(seconds: float) -> int:
    return max(0, int(round(float(seconds) * VIDEO_FPS)))


def annotate_take_frames(takes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cursor = 0.0
    out: list[dict[str, Any]] = []
    for i, take in enumerate(takes):
        row = dict(take)
        duration = float(row.get("duration") or 0)
        start = _frame_at(cursor)
        end = _frame_at(cursor + duration)
        row["take_id"] = str(row.get("take_id") or row.get("id") or f"take-{i + 1:02d}")
        row["start_frame"] = int(row.get("start_frame") or start)
        row["end_frame"] = int(row.get("end_frame") or end)
        row["beats"] = list(row.get("beats") or row.get("cuts") or [])
        row["cuts"] = list(row.get("cuts") or row.get("beats") or [])
        row["cast"] = list(row.get("cast") or row.get("on_screen_ids") or [])
        if row.get("character_id") and row["character_id"] not in row["cast"]:
            row["cast"] = [row["character_id"], *row["cast"]]
        row["first_frame"] = row.get("first_frame") or row.get("first_frame_path") or ""
        row["prompt"] = row.get("prompt") or row.get("h3_prompt") or ""
        scene_id = str(row.get("scene_id") or "")
        prev_scene = str(out[-1].get("scene_id") or "") if out else ""
        if i == 0:
            expected_source = "qc_keyframe"
            row["continue_from"] = None
        elif scene_id and scene_id == prev_scene:
            expected_source = "prior_last_frame"
            row["continue_from"] = out[-1]["take_id"]
        else:
            expected_source = "new_keyframe_hard_cut"
            row["continue_from"] = None
        provided = str(row.get("keyframe_source") or "")
        if provided and provided != expected_source:
            raise LongLiveWorkflowError(
                f"take {row['take_id']} keyframe_source {provided!r} must be {expected_source}"
            )
        row["keyframe_source"] = expected_source
        out.append(row)
        cursor += duration
    return out


def validate_packed_takes(
    takes: Sequence[dict[str, Any]],
    *,
    max_take_s: float = LONGLIVE_SHORT_MAX_SECONDS,
    min_takes: int = LONGLIVE_SHORT_MIN_TAKES,
    max_takes: int = LONGLIVE_SHORT_MAX_TAKES,
) -> list[dict[str, Any]]:
    rows = annotate_take_frames(list(takes))
    if not rows:
        raise LongLiveWorkflowError("longlive short has no takes")
    if not (min_takes <= len(rows) <= max_takes):
        raise LongLiveWorkflowError(
            f"longlive short must pack into {min_takes}-{max_takes} takes, got {len(rows)}"
        )
    for row in rows:
        if float(row.get("duration") or 0) > max_take_s + 1e-9:
            raise LongLiveWorkflowError(
                f"take {row.get('take_id')} exceeds {max_take_s}s"
            )
        if not row.get("take_id"):
            raise LongLiveWorkflowError("take missing take_id")
        if int(row.get("end_frame") or 0) <= int(row.get("start_frame") or 0):
            raise LongLiveWorkflowError(f"take {row['take_id']} has empty frame range")
    prev_scene = ""
    for i, row in enumerate(rows):
        scene_id = str(row.get("scene_id") or "")
        if i == 0:
            if row.get("keyframe_source") != "qc_keyframe":
                raise LongLiveWorkflowError("first take must use QC keyframe I2V")
        elif scene_id and scene_id == prev_scene:
            if row.get("keyframe_source") != "prior_last_frame":
                raise LongLiveWorkflowError(
                    f"same-scene take {row['take_id']} must continue from prior last frame"
                )
        else:
            if row.get("keyframe_source") != "new_keyframe_hard_cut":
                raise LongLiveWorkflowError(
                    f"scene-change take {row['take_id']} must use a new keyframe hard cut"
                )
        prev_scene = scene_id
    return rows


def pack_short_takes(
    shots: Sequence[dict[str, Any]],
    *,
    max_take_s: float = LONGLIVE_SHORT_MAX_SECONDS,
    min_takes: int = LONGLIVE_SHORT_MIN_TAKES,
    max_takes: int = LONGLIVE_SHORT_MAX_TAKES,
    target_seconds: float = LONGLIVE_SHORT_TARGET_SECONDS,
) -> list[dict[str, Any]]:
    """Pack director shots into 2–3 ≤60s takes, preferring scene/beat boundaries."""
    del target_seconds  # duration comes from shots; 120s is the contract, not padding
    items = [dict(s) for s in shots if float(s.get("duration") or 0) > 0]
    if not items:
        raise LongLiveWorkflowError("no shots to pack")

    def scene_key(shot: dict[str, Any]) -> str:
        return str(shot.get("scene_id") or shot.get("chain_id") or "")

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_dur = 0.0
    current_scene: str | None = None
    for shot in items:
        dur = float(shot.get("duration") or 0)
        scene = scene_key(shot)
        scene_break = current and current_scene is not None and scene != current_scene
        overflow = current and current_dur + dur > max_take_s + 1e-9
        if scene_break or overflow:
            groups.append(current)
            current = []
            current_dur = 0.0
            current_scene = None
        if not current:
            current_scene = scene
        current.append(shot)
        current_dur += dur
    if current:
        groups.append(current)

    def merge_group(parts: list[dict[str, Any]], index: int) -> dict[str, Any]:
        duration = round(sum(float(s.get("duration") or 0) for s in parts), 2)
        cuts: list[dict[str, Any]] = []
        cast: list[str] = []
        for shot in parts:
            for cid in [shot.get("character_id"), *(shot.get("on_screen_ids") or [])]:
                text = str(cid or "").strip()
                if text and text not in cast:
                    cast.append(text)
            for cut in shot.get("cuts") or []:
                if isinstance(cut, dict):
                    cuts.append(dict(cut))
        head = parts[0]
        return {
            "id": str(head.get("id") or f"take-{index:02d}"),
            "take_id": f"take-{index:02d}",
            "duration": duration,
            "scene_id": scene_key(head),
            "chain_id": head.get("chain_id") or scene_key(head),
            "character_id": head.get("character_id"),
            "first_frame_path": head.get("first_frame_path"),
            "h3_prompt": head.get("h3_prompt") or head.get("prompt") or "",
            "prompt": head.get("h3_prompt") or head.get("prompt") or "",
            "cuts": cuts,
            "beats": cuts,
            "cast": cast,
            "source_shot_ids": [str(s.get("id") or "") for s in parts],
        }

    while len(groups) > max_takes:
        best_i = 0
        best_dur = float("inf")
        for i in range(len(groups) - 1):
            left = sum(float(s.get("duration") or 0) for s in groups[i])
            right = sum(float(s.get("duration") or 0) for s in groups[i + 1])
            same = scene_key(groups[i][0]) == scene_key(groups[i + 1][0])
            combined = left + right
            if combined <= max_take_s + 1e-9 and (same or combined < best_dur):
                best_dur = combined if same else combined + 1000
                best_i = i
        if best_dur == float("inf"):
            break
        groups[best_i] = groups[best_i] + groups[best_i + 1]
        del groups[best_i + 1]

    while len(groups) < min_takes:
        longest_i = max(range(len(groups)), key=lambda i: sum(float(s.get("duration") or 0) for s in groups[i]))
        parts = groups[longest_i]
        if len(parts) < 2:
            break
        mid = max(1, len(parts) // 2)
        groups[longest_i:longest_i + 1] = [parts[:mid], parts[mid:]]

    takes = [merge_group(parts, i + 1) for i, parts in enumerate(groups)]
    return validate_packed_takes(takes, max_take_s=max_take_s, min_takes=min_takes, max_takes=max_takes)

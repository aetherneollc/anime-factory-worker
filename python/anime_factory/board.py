"""Board: Shot (director) → Segment (one H3 call ≤8s) → cuts.

`board_from_script` dispatches film / series / short grammars, and shorts on
LongLive use the ultra-long AR take director. persist_board splits duration
over the line max (8s H3 / 64s LongLive) into chained segments instead of throwing.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Sequence

from anime_factory.db import utcnow
from anime_factory.directors import board_from_script as director_board
from anime_factory.directors.common import (
    CAMERA_TERM_RE,
    CJK_RE,
    ESTABLISHING_SECONDS,
    FILLER_SIZES,
    H3_TAIL,
    LEAD_IN_SECONDS,
    MIN_SHOT_SECONDS,
    REMOTE_BEATS,
    RESOLUTION_RE,
    SHOT_INDEX_RE,
    SIZE_FRAMING,
    SPOKEN_SIZES,
    STORE_COPY_MARKERS,
    choose_h3_mode,
    expand_shots_to_segments,
    hydrate_shot_identity,
    line_beat_id,
    line_blob,
    line_langs,
    line_shot_seconds,
    line_text,
    split_duration,
)
from anime_factory.models import H3_MAX_REFS, H3_MAX_SECONDS, TARGET_EPISODE_SECONDS, normalize_story_kind
from anime_factory.video_backend import max_seconds_for_backend

# Re-export helpers used by tests and produce.
__all__ = [
    "ESTABLISHING_SECONDS",
    "FILLER_SIZES",
    "H3_TAIL",
    "LEAD_IN_SECONDS",
    "MIN_SHOT_SECONDS",
    "PromptCollapseError",
    "SIZE_FRAMING",
    "SPOKEN_SIZES",
    "SpeakerFrameError",
    "StillPromptError",
    "assert_shot_prompt_diversity",
    "assert_speaker_matches_frame",
    "assert_still_prompt_clean",
    "board_from_script",
    "board_totals",
    "choose_h3_mode",
    "expand_shots_to_segments",
    "hydrate_shot_identity",
    "line_langs",
    "line_shot_seconds",
    "line_text",
    "persist_board",
    "shot_motion_prompt",
    "shot_visual_prompt",
    "split_duration",
]


def board_from_script(
    script: dict,
    *,
    episode_code: str,
    langs: Sequence[str],
    target_seconds: float = TARGET_EPISODE_SECONDS,
    shot_seconds: float = H3_MAX_SECONDS,
    kind: str | None = "series",
    video_backend: str | None = None,
) -> list[dict[str, Any]]:
    """Director shot list. Segment split happens in persist_board / expand_shots_to_segments."""
    shots = director_board(
        script,
        episode_code=episode_code,
        langs=langs,
        target_seconds=target_seconds,
        shot_seconds=shot_seconds,
        kind=kind,
        video_backend=video_backend,
    )
    assert_shot_prompt_diversity(shots)
    for shot in shots:
        assert_speaker_matches_frame(shot)
        assert_still_prompt_clean(
            shot_visual_prompt(shot), label=str(shot.get("id") or shot.get("shot_id") or "shot")
        )
    return shots


class PromptCollapseError(RuntimeError):
    pass


class SpeakerFrameError(RuntimeError):
    pass


_SHOT_NUM_RE = re.compile(r",?\s*shot\s+\d+\s+of\s+\d+", re.I)
_KEEP_FACE_RE = re.compile(r",?\s*keep face/costume[^,]*(?:,|$)", re.I)
_CAMERA_MOTION_RE = re.compile(r",?\s*camera motion must change[^,]*(?:,|$)", re.I)
_RES_RE = re.compile(r",?\s*\d{3,4}x\d{3,4}\b", re.I)
_CAMERA_CLAUSE_RE = re.compile(r",?\s*camera\s+[^,]+", re.I)
_IMAGE_TAG_RE = re.compile(r"<<<image_\d+>>>", re.I)
_LOCK_REF_RE = re.compile(
    r",?\s*(?:locked character-sheet and scene-plate refs|no face drift|no costume swap|禁止五官漂移/?换装)[^,]*(?:,|$)",
    re.I,
)


class StillPromptError(RuntimeError):
    """A still prompt still carries video-side syntax the image model cannot read."""


def shot_visual_prompt(shot: dict) -> str:
    """The still prompt for this shot. Never h3_prompt — that is the video script.

    Taking h3_prompt here is how the still model was handed `camera dolly in,
    ..., do not freeze a still, 864x480` plus a `<<<image_1>>>` header.
    """
    return str(shot.get("first_frame_prompt") or shot.get("prompt") or "").strip()


def shot_motion_prompt(shot: dict) -> str:
    """The H3 video prompt: Chinese, @tags, camera direction, H3_TAIL."""
    return str(shot.get("h3_prompt") or "").strip()


def assert_still_prompt_clean(prompt: str, *, label: str = "still") -> str:
    """Refuse a still prompt that contains video-side syntax.

    Tripwire for regressions in the split: CJK, `<<<image_N>>>`, `@tag`,
    `shot N of M`, a resolution string, or camera/motion direction.
    """
    text = str(prompt or "").strip()
    if not text:
        raise StillPromptError(f"{label}: still prompt is empty")
    found: list[str] = []
    if CJK_RE.search(text):
        found.append("cjk")
    if "<<<" in text:
        found.append("image_ref_tag")
    if "@" in text:
        found.append("at_tag")
    if SHOT_INDEX_RE.search(text):
        found.append("shot_index")
    if RESOLUTION_RE.search(text):
        found.append("resolution")
    camera = CAMERA_TERM_RE.search(text)
    if camera:
        found.append(f"camera_term:{camera.group(0).lower()}")
    if found:
        raise StillPromptError(f"{label}: still prompt carries video-side syntax {found}: {text[:160]}")
    return text


def _diversity_text(shot: dict) -> str:
    """Whatever describes this shot, for collapse/speaker QC only (not for drawing)."""
    return str(
        shot.get("h3_prompt") or shot.get("first_frame_prompt") or shot.get("prompt") or ""
    ).strip()


def visual_fingerprint(shot: dict) -> str:
    """Location + action. Strip shot-index suffixes that used to fake diversity."""
    raw = _diversity_text(shot)
    raw = _IMAGE_TAG_RE.sub(" ", raw)
    raw = _LOCK_REF_RE.sub(",", raw)
    raw = _SHOT_NUM_RE.sub("", raw)
    raw = _KEEP_FACE_RE.sub(",", raw)
    raw = _CAMERA_MOTION_RE.sub(",", raw)
    raw = _RES_RE.sub(" ", raw)
    raw = _CAMERA_CLAUSE_RE.sub(" ", raw)
    if H3_TAIL:
        raw = raw.replace(H3_TAIL, " ")
    return re.sub(r"[, ]+", " ", raw).strip().lower()


def _is_chain_continue(prev: dict, cur: dict) -> bool:
    if int(cur.get("chain_index") or 0) <= 0:
        return False
    left = str(prev.get("chain_id") or "")
    right = str(cur.get("chain_id") or "")
    return bool(left) and left == right


def _spoken_line_count(shots: list[dict]) -> int:
    seen: set[tuple[str, str]] = set()
    for shot in shots:
        line = shot.get("line")
        if isinstance(line, dict):
            text = str(line.get("zh") or line.get("text") or "").strip()
        else:
            text = str(line or "").strip()
        if not text:
            continue
        parent = str(shot.get("shot_id") or shot.get("id") or "")
        seen.add((parent, text))
    return len(seen)


def assert_dramatic_beat_visuals(shots: list[dict]) -> None:
    """Arrive / phone / wechat / cat cannot reuse the same store-speaking prompt."""
    beat_fps: dict[str, set[str]] = {}
    for shot in shots:
        if int(shot.get("chain_index") or 0) > 0:
            continue
        line = shot.get("line")
        if not line:
            continue
        beat = line_beat_id(line if isinstance(line, dict) else {"zh": str(line)})
        if beat == "dialogue":
            continue
        fp = visual_fingerprint(shot)
        if fp:
            beat_fps.setdefault(beat, set()).add(fp)
    if len(beat_fps) < 2:
        return
    union: set[str] = set()
    for fps in beat_fps.values():
        union |= fps
    if len(union) < len(beat_fps):
        raise PromptCollapseError(
            f"dramatic beats {sorted(beat_fps)} share {len(union)} visual prompts; "
            "arrive/phone/wechat/cat must change location or action in h3_prompt"
        )


def assert_shot_prompt_diversity(shots: list[dict], min_unique_ratio: float = 0.75) -> None:
    """Fail 15 segs / 2 unique visuals. Same-speaker chain tails may keep the last frame."""
    fingerprints = [visual_fingerprint(s) for s in shots]
    nonempty = [fp for fp in fingerprints if fp]
    for prev, cur, left_fp, right_fp in zip(shots, shots[1:], fingerprints, fingerprints[1:]):
        if left_fp and left_fp == right_fp and not _is_chain_continue(prev, cur):
            raise PromptCollapseError("consecutive shots share the same visual prompt")
    unique_all = len(set(nonempty))
    heads = [s for s in shots if int(s.get("chain_index") or 0) == 0]
    head_fps = [fp for fp in (visual_fingerprint(s) for s in heads) if fp]
    n_spoken = _spoken_line_count(shots)
    if (len(heads) >= 8 or n_spoken >= 8) and unique_all < 3 and nonempty:
        raise PromptCollapseError(
            f"board visual collapse: {unique_all} unique visual prompts for "
            f"{n_spoken} spoken lines / {len(shots)} segs"
        )
    if len(head_fps) >= 5 and len(set(head_fps)) < 3:
        raise PromptCollapseError(
            f"board visual collapse: {len(set(head_fps))} unique prompts of {len(head_fps)}"
        )
    _ = min_unique_ratio  # kept for call-site compatibility; collapse is unique<3
    assert_dramatic_beat_visuals(shots)


def assert_speaker_matches_frame(shot: dict) -> None:
    """小美 on the phone must not be a store CU unless on_camera is explicit True."""
    line = shot.get("line")
    if not line:
        return
    blob = line_blob(line if isinstance(line, dict) else {"zh": str(line)})
    beat = line_beat_id(line if isinstance(line, dict) else {"zh": str(line)})
    remote = beat in REMOTE_BEATS or any(
        marker.lower() in blob.lower()
        for marker in ("phone", "wechat", "微信", "电话", "睡着", "asleep", "outside", "外面", "朋友圈")
    )
    if not remote:
        return
    if shot.get("on_camera") is True:
        return
    prompt = f"{visual_fingerprint(shot)} {_diversity_text(shot)}".lower()
    if "off-camera" in prompt or "do not show" in prompt:
        return
    store_speaking = any(tok in prompt for tok in STORE_COPY_MARKERS) or (
        "speaking" in prompt and ("store" in prompt or "便利店" in prompt)
    )
    if store_speaking:
        sid = shot.get("id") or shot.get("shot_id") or "?"
        raise SpeakerFrameError(
            f"{sid}: remote/asleep/outside line is boarded as on-camera store speaking; "
            "set on_camera true only if they are actually in frame"
        )


def board_totals(shots: list[dict]) -> dict[str, Any]:
    spoken = [s for s in shots if s.get("line")]
    return {
        "n_shots": len(shots),
        "n_spoken": len(spoken),
        "total_s": round(sum(float(s.get("duration") or 0) for s in shots), 2),
    }


def _shots_from_items(items: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for item in items:
        shot_id = str(item.get("shot_id") or item.get("id") or "")
        if not shot_id:
            continue
        if shot_id not in by_id:
            row = dict(item)
            row["id"] = shot_id
            row["duration"] = float(item.get("duration") or 0)
            by_id[shot_id] = row
            order.append(shot_id)
        else:
            by_id[shot_id]["duration"] = round(
                float(by_id[shot_id]["duration"]) + float(item.get("duration") or 0), 2
            )
    return [by_id[sid] for sid in order]


def persist_board(
    conn: sqlite3.Connection,
    episode_code: str,
    items: list[dict],
    *,
    kind: str | None = None,
    video_backend: str | None = None,
) -> list[dict]:
    """Write Shot rows, then video segments. duration over the line max splits; does not throw."""
    resolved_kind = normalize_story_kind(kind) if kind else None
    limit = max_seconds_for_backend(video_backend) if video_backend is not None else max_seconds_for_backend()
    looks_split = bool(items) and all("chain_index" in row for row in items)
    over_max = any(float(row.get("duration") or 0) > limit for row in items)
    if looks_split and not over_max:
        shots = _shots_from_items(items)
        segments = items
    else:
        shots = items
        segments = expand_shots_to_segments(items, max_s=limit)

    cast_map: dict[str, dict] = {}
    for row in list(shots) + list(segments):
        cid = str(row.get("character_id") or "").strip()
        if cid:
            cast_map[cid] = {}
        for cut in row.get("cuts") or []:
            for other in cut.get("characters") or []:
                if str(other).strip():
                    cast_map[str(other).strip()] = {}
    shots = [hydrate_shot_identity(row, cast=cast_map) for row in shots]
    segments = [hydrate_shot_identity(row, cast=cast_map) for row in segments]

    assert_shot_prompt_diversity(shots)
    assert_shot_prompt_diversity(segments)
    for row in shots:
        assert_speaker_matches_frame(row)
    for row in segments:
        assert_speaker_matches_frame(row)

    now = utcnow()
    if resolved_kind:
        conn.execute(
            """
            INSERT INTO productions (id, story_id, kind, status, created_at, updated_at)
            VALUES (?, ?, ?, 'boarding', ?, ?)
            ON CONFLICT(id) DO UPDATE SET kind = excluded.kind, updated_at = excluded.updated_at
            """,
            (episode_code, episode_code, resolved_kind, now, now),
        )
        try:
            conn.execute(
                "UPDATE episodes SET kind = ? WHERE episode_code = ?",
                (resolved_kind, episode_code),
            )
        except sqlite3.OperationalError:
            pass

    conn.execute("DELETE FROM cuts WHERE segment_id IN (SELECT id FROM segments WHERE episode_code = ?)", (episode_code,))
    conn.execute("DELETE FROM segments WHERE episode_code = ?", (episode_code,))
    conn.execute("DELETE FROM shots WHERE episode_code = ?", (episode_code,))

    for i, shot in enumerate(shots, start=1):
        sid = shot.get("shot_id") or shot.get("id") or f"E{episode_code[-2:]}-{i:02d}"
        conn.execute(
            """
            INSERT INTO shots (id, episode_code, scene_id, seq, duration, purpose, size, camera,
                h3_mode, character_id, match_cut, chain_id, first_frame_prompt, h3_prompt,
                refs_json, plate_id, costume_ids_json, line_json, eyeline, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared')
            ON CONFLICT(id) DO UPDATE SET duration = excluded.duration, h3_mode = excluded.h3_mode
            """,
            (
                sid,
                episode_code,
                shot.get("scene_id") or f"{episode_code}-sc01",
                shot.get("seq", i),
                float(shot.get("duration") or 0),
                shot.get("purpose"),
                ((shot.get("cuts") or [{}])[0] or {}).get("size"),
                ((shot.get("cuts") or [{}])[0] or {}).get("camera"),
                choose_h3_mode({**shot, "chain_index": 0}),
                shot.get("character_id"),
                1 if shot.get("match_cut") else 0,
                shot.get("chain_id") or f"{sid}-chain",
                shot.get("first_frame_prompt"),
                shot.get("h3_prompt"),
                json.dumps(list(shot.get("refs") or [])[:H3_MAX_REFS], ensure_ascii=False),
                shot.get("plate_id"),
                json.dumps(shot.get("costume_ids") or {}, ensure_ascii=False),
                json.dumps(shot.get("line") or {}, ensure_ascii=False),
                shot.get("eyeline"),
            ),
        )

    for i, seg in enumerate(segments, start=1):
        sid = seg.get("id") or f"E{episode_code[-2:]}-{i:02d}"
        duration = float(seg.get("duration") or 0)
        refs = list(seg.get("refs") or [])[:H3_MAX_REFS]
        chain_index = int(seg.get("chain_index") or 0)
        status = "pending_chain" if chain_index > 0 else "prepared"
        conn.execute(
            """
            INSERT INTO segments (id, episode_code, scene_id, seq, duration, h3_mode,
                first_frame_prompt, h3_prompt, refs_json, plate_id, costume_ids_json,
                video_path, keyframe_path, last_frame_path, seed, retry_count, status,
                shot_id, chain_id, chain_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET duration = excluded.duration, status = excluded.status,
                shot_id = excluded.shot_id, chain_id = excluded.chain_id,
                chain_index = excluded.chain_index, h3_mode = excluded.h3_mode
            """,
            (
                sid,
                episode_code,
                seg.get("scene_id") or f"{episode_code}-sc01",
                seg.get("seq", i),
                duration,
                choose_h3_mode(seg),
                seg.get("first_frame_prompt"),
                seg.get("h3_prompt"),
                json.dumps(refs, ensure_ascii=False),
                seg.get("plate_id"),
                json.dumps(seg.get("costume_ids") or {}, ensure_ascii=False),
                seg.get("video_path"),
                seg.get("keyframe_path"),
                seg.get("last_frame_path") or None,
                seg.get("seed"),
                status,
                seg.get("shot_id") or sid,
                seg.get("chain_id") or f"{seg.get('shot_id') or sid}-chain",
                chain_index,
            ),
        )
        conn.execute("DELETE FROM cuts WHERE segment_id = ?", (sid,))
        for j, cut in enumerate(seg.get("cuts") or [], start=1):
            beats = cut.get("beats") or [cut.get("beats_start"), cut.get("beats_end")]
            conn.execute(
                """
                INSERT INTO cuts (segment_id, seq, beats_start, beats_end, seconds, size, camera,
                                  characters_json, props_json, frame_prompt, keyframe_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    cut.get("seq", j),
                    beats[0] if beats else None,
                    beats[-1] if beats else None,
                    cut.get("seconds"),
                    cut.get("size"),
                    cut.get("camera"),
                    json.dumps(cut.get("characters") or [], ensure_ascii=False),
                    json.dumps(cut.get("props") or [], ensure_ascii=False),
                    cut.get("frame_prompt"),
                    cut.get("keyframe_path"),
                ),
            )
    conn.commit()
    return segments

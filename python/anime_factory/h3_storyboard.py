"""H3 storyboard v2: deterministic cut/prompt/manifest compilation and pre-GPU gates.

Picture-only. Dialogue stays CosyVoice2; never feed spoken lines into H3 prompts.
Apache-2.0 reimplementation of structured-storyboard methodology (not a Node skill dump).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "h3_storyboard_v2"
FPS = 24
CUT_MIN_S = 2.0
CUT_MAX_S = 5.0
SEGMENT_MAX_S = 8.0
MAX_ON_SCREEN = 3
MAX_REFS = 9
DEFAULT_TEXT_POLICY = "no_generated_text"
GATES_REL = "validation.gates.jsonl"

H3_CAMERA_ENUM = frozenset(
    {
        "Static Shot",
        "Pan Left",
        "Pan Right",
        "Tilt Up",
        "Tilt Down",
        "Zoom In",
        "Zoom Out",
        "Push In",
        "Pull Out",
        "Tracking Shot",
        "Orbit Left",
        "Orbit Right",
        "Handheld",
    }
)

SIZE_ENUM = frozenset({"ECU", "CU", "MCU", "MS", "MLS", "LS", "ELS", "WS"})

_CJK_RE = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]+"
)
_VISIBLE_TEXT_RE = re.compile(
    r"\b(whiteboard|blackboard|chalkboard|signage|subtitle|caption|logo text|"
    r"chinese characters on|write on board|手写|白板|黑板|字幕)\b",
    re.I,
)

CONTINUE_CLAUSE = (
    "延续上一镜头, continue from the previous last frame, "
    "same identity, same costume, same set"
)
MOTION_KEEP_ALIVE = (
    "camera and subject motion are the changing terms this shot; "
    "do not freeze a still photograph; keep face and costume of the named character refs; "
    "no face drift, no costume swap; do not generate spoken dialogue; no text, no watermark"
)
DEFAULT_FRAME_PROMPT = (
    "original cinematic animation for this story, named character and scene refs already bound"
)


def _round2(value: float) -> float:
    return round(float(value), 2)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("en", "zh", "ja", "text"):
            raw = value.get(key)
            if raw and str(raw).strip():
                return str(raw).strip()
        return ""
    return str(value or "").strip()


def normalize_camera(camera: Any) -> str:
    text = str(camera or "").strip() or "Static Shot"
    if text in H3_CAMERA_ENUM:
        return text
    lowered = text.lower().replace("-", " ").replace("_", " ")
    aliases = {
        "static": "Static Shot",
        "locked off": "Static Shot",
        "lockedoff": "Static Shot",
        "pan left": "Pan Left",
        "pan right": "Pan Right",
        "tilt up": "Tilt Up",
        "tilt down": "Tilt Down",
        "zoom in": "Zoom In",
        "zoom out": "Zoom Out",
        "push in": "Push In",
        "pull out": "Pull Out",
        "pull back": "Pull Out",
        "tracking": "Tracking Shot",
        "tracking shot": "Tracking Shot",
        "orbit left": "Orbit Left",
        "orbit right": "Orbit Right",
        "handheld": "Handheld",
    }
    return aliases.get(lowered, "Static Shot")


def normalize_size(size: Any) -> str:
    text = str(size or "").strip().upper()
    if text in SIZE_ENUM:
        return text
    aliases = {
        "CLOSEUP": "CU",
        "CLOSE-UP": "CU",
        "MEDIUM": "MS",
        "WIDE": "WS",
        "LONG": "LS",
        "FULL": "LS",
    }
    return aliases.get(text, "MS")


def seconds_to_frames(seconds: float, *, fps: int = FPS) -> int:
    return max(1, int(round(float(seconds) * fps)))


def frames_to_seconds(frames: int, *, fps: int = FPS) -> float:
    return _round2(float(frames) / float(fps))


def h3_length_frames(seconds: float, *, fps: int = FPS) -> int:
    """Snap to MiniMax H3's 17k+5 frame grid."""
    frames = max(5, int(round(float(seconds) * fps)))
    while frames % 17 != 5:
        frames += 1
    return frames


def speech_seconds_for_line(line: Any, *, max_seconds: float = SEGMENT_MAX_S) -> float:
    zh = ""
    if isinstance(line, dict):
        zh = str(line.get("zh") or line.get("text") or "").strip()
    else:
        zh = str(line or "").strip()
    if zh:
        return max(zh.__len__() * 0.22, 0.4)
    return min(max_seconds, 2.0)


def cut_keyframe_name(seq: int) -> str:
    """Per-cut composition anchor filename: f01.png, f02.png, ..."""
    return f"f{int(seq):02d}.png"


def segment_keyframe_name() -> str:
    return "f1.png"


def normalize_cut(cut: dict[str, Any], *, seq: int, default_seconds: float) -> dict[str, Any]:
    row = dict(cut)
    row["seq"] = int(row.get("seq") or seq)
    seconds = float(row.get("seconds") or default_seconds or CUT_MIN_S)
    row["seconds"] = _round2(max(0.1, seconds))
    row["size"] = normalize_size(row.get("size"))
    row["camera"] = normalize_camera(row.get("camera"))
    chars = [str(c).strip() for c in (row.get("characters") or []) if str(c).strip()]
    row["characters"] = chars
    props = [str(p).strip() for p in (row.get("props") or []) if str(p).strip()]
    row["props"] = props
    frame = _text(row.get("frame_prompt") or row.get("staging") or row.get("visual_en"))
    row["frame_prompt"] = frame
    if "beats" not in row or row.get("beats") is None:
        row["beats"] = [row["seq"], row["seq"]]
    if "sfx" not in row:
        row["sfx"] = row.get("sfx_events") or []
    beat_ids = row.get("beat_ids") or row.get("claimed_beats")
    if beat_ids is None and row.get("line") is not None:
        beat_ids = [f"line-{row['seq']}"]
    row["beat_ids"] = [str(b).strip() for b in (beat_ids or []) if str(b).strip()]
    row["keyframe"] = str(row.get("keyframe") or cut_keyframe_name(row["seq"]))
    row["text_policy"] = str(row.get("text_policy") or DEFAULT_TEXT_POLICY)
    return row


def ensure_cuts_for_segment(segment: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize cuts; synthesize a single cut when legacy boards omit them."""
    duration = float(segment.get("duration") or SEGMENT_MAX_S)
    raw = [c for c in (segment.get("cuts") or []) if isinstance(c, dict)]
    if not raw:
        frame = _text(
            segment.get("first_frame_prompt")
            or segment.get("h3_prompt")
            or segment.get("prompt")
            or segment.get("action")
        )
        # Never fall back to dialogue `line` for the frame prompt.
        chars: list[str] = []
        cid = str(segment.get("character_id") or "").strip()
        if cid and segment.get("on_camera") is not False:
            chars.append(cid)
        for other in segment.get("characters") or []:
            text = str(other or "").strip()
            if text and text not in chars:
                chars.append(text)
        raw = [
            {
                "seq": 1,
                "seconds": duration,
                "size": segment.get("size") or "MS",
                "camera": segment.get("camera") or "Static Shot",
                "characters": chars,
                "props": list(segment.get("props") or []),
                "frame_prompt": frame,
                "beat_ids": [str(segment.get("id") or "beat-1")],
                "line": segment.get("line") if isinstance(segment.get("line"), dict) else None,
                "sfx": segment.get("sfx"),
            }
        ]
    total = sum(float(c.get("seconds") or 0) for c in raw) or duration
    cuts: list[dict[str, Any]] = []
    for i, cut in enumerate(raw, start=1):
        share = float(cut.get("seconds") or 0)
        if share <= 0 and total > 0:
            share = duration * (1.0 / len(raw))
        elif abs(total - duration) > 0.05 and total > 0:
            share = duration * (share / total)
        cuts.append(normalize_cut(cut, seq=i, default_seconds=share))
    # Re-normalize residual onto the last cut so sums match exactly.
    if cuts:
        head = sum(c["seconds"] for c in cuts[:-1])
        cuts[-1]["seconds"] = _round2(max(0.1, duration - head))
    return cuts


def cut_frame_ranges(
    cuts: Sequence[dict[str, Any]],
    *,
    fps: int = FPS,
    total_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministic Picture alignment rows keyed by cut sequence."""
    rows: list[dict[str, Any]] = []
    cursor = 0
    for cut in cuts:
        frames = seconds_to_frames(float(cut.get("seconds") or CUT_MIN_S), fps=fps)
        start = cursor
        end = cursor + frames
        rows.append(
            {
                "seq": int(cut.get("seq") or len(rows) + 1),
                "start_frame": start,
                "end_frame": end,
                "start_s": frames_to_seconds(start, fps=fps),
                "end_s": frames_to_seconds(end, fps=fps),
                "seconds": _round2(float(cut.get("seconds") or 0)),
                "camera": normalize_camera(cut.get("camera")),
                "size": normalize_size(cut.get("size")),
                "keyframe": str(cut.get("keyframe") or cut_keyframe_name(int(cut.get("seq") or 1))),
            }
        )
        cursor = end
    if total_frames is not None and rows and cursor != total_frames:
        # Stretch the last cut to the H3 length grid without inventing new timestamps.
        delta = int(total_frames) - cursor
        rows[-1]["end_frame"] = int(rows[-1]["end_frame"]) + delta
        rows[-1]["end_s"] = frames_to_seconds(int(rows[-1]["end_frame"]), fps=fps)
    return rows


def _camera_motion_clause(cam: str) -> str:
    text = normalize_camera(cam)
    if text == "Static Shot":
        return "subtle body and face motion, speaking gesture, do not freeze as a still"
    return f"camera move: {text}"


def compile_picture_prompt(segment: dict[str, Any]) -> str:
    """Deterministic H3 picture prompt from cuts. Never reads dialogue `line`."""
    cuts = ensure_cuts_for_segment(segment)
    parts: list[str] = []
    for i, cut in enumerate(cuts):
        frame = _text(cut.get("frame_prompt"))
        if not frame:
            continue
        if _CJK_RE.search(frame):
            # Drop CJK so CLIP/H3 do not paint glyphs; keep English remainder.
            frame = _CJK_RE.sub(" ", frame)
            frame = re.sub(r"\s{2,}", " ", frame).strip(" ,")
        size = normalize_size(cut.get("size"))
        cam = normalize_camera(cut.get("camera"))
        clause = f"{frame}, framing {size}, camera: {cam}, {_camera_motion_clause(cam)}"
        if len(cuts) > 1:
            label = "First" if i == 0 else ("Finally" if i == len(cuts) - 1 else "Then")
            clause = f"{label}: {clause}"
        parts.append(clause)
    prompt = " [cut] ".join(parts) if parts else DEFAULT_FRAME_PROMPT
    # Prefer an explicit compiled visual field when present (and not dialogue).
    for key in ("h3_prompt", "first_frame_prompt", "action", "prompt"):
        # Skip fields that are clearly the packed dialogue-polluted cineflow body
        # only when cuts already provided frame prompts.
        if parts:
            break
        val = segment.get(key)
        text = _text(val)
        if text:
            prompt = text
            break
    if int(segment.get("chain_index") or 0) > 0 or segment.get("chain_source_last_frame"):
        if CONTINUE_CLAUSE not in prompt:
            prompt = f"{CONTINUE_CLAUSE}. {prompt}"
    prompt = f"{prompt}, {MOTION_KEEP_ALIVE}"
    if DEFAULT_TEXT_POLICY == "no_generated_text":
        prompt = f"{prompt}, no readable text, no whiteboard writing, no logos"
    return re.sub(r"\s{2,}", " ", prompt).strip(" ,")


def build_reference_manifest(segment: dict[str, Any]) -> dict[str, Any]:
    """Fixed H3 ref order: temporal keyframe → character locks → scene → props.

    Temporal entries are recorded for the package; Comfy ref LoadImage slots should
    use ``ref_bind_names`` (characters/scenes/props, plus real temporal files only).
    """
    sid = str(segment.get("id") or "")
    cuts = ensure_cuts_for_segment(segment)
    temporal: list[str] = []
    first = str(segment.get("first_frame_path") or segment.get("keyframe_path") or "").strip()
    if first and not first.endswith("/") and Path(first).name:
        temporal.append(Path(first).name)
    for cut in cuts:
        name = str(cut.get("keyframe") or cut_keyframe_name(int(cut.get("seq") or 1)))
        if name not in temporal:
            temporal.append(name)
    if not temporal:
        temporal.append(segment_keyframe_name())

    characters: list[str] = []
    scenes: list[str] = []
    props: list[str] = []
    # Prefer explicit refs order from the board; do not invent duplicate tokens.
    for rid in list(segment.get("refs") or []):
        text = str(rid or "").strip()
        if not text:
            continue
        if text.startswith("plate_"):
            if text not in scenes:
                scenes.append(text)
            continue
        if text.startswith("char_") or "sheet" in text or "costume" in text:
            name = Path(text).name if "/" in text or "\\" in text else text
            if name not in characters:
                characters.append(name)
            continue
        # Unknown ref token — keep in props pack so binders can resolve it.
        if text not in props:
            props.append(text)
    cid = str(segment.get("character_id") or "").strip()
    if cid and segment.get("on_camera") is not False:
        token = f"char_{cid}_sheet"
        # Skip when a sheet file or char_{cid}_* token is already in the pack.
        if not any(
            token in c
            or c.startswith(f"char_{cid}_")
            or Path(c).name.startswith(f"char_{cid}_")
            or Path(c).name.startswith(f"{cid}_")
            or Path(c).stem == token
            for c in characters
        ):
            characters.append(token)

    plate = str(segment.get("plate_id") or "").strip()
    if plate:
        plate_token = plate if plate.startswith("plate_") else f"plate_{plate}"
        lid = plate_token[len("plate_") :] if plate_token.startswith("plate_") else plate
        if not any(
            plate_token == s
            or plate_token in s
            or Path(s).stem == plate_token
            or Path(s).name.startswith(f"{lid}_")
            or Path(s).stem.startswith(f"{lid}_")
            for s in scenes
        ):
            scenes.append(plate_token)

    for cut in cuts:
        for prop in cut.get("props") or []:
            name = str(prop or "").strip()
            if name and name not in props:
                props.append(name)
    for prop in segment.get("props") or []:
        name = str(prop or "").strip()
        if name and name not in props:
            props.append(name)

    # Bind order for Comfy ref pack: real temporal files first, then locks.
    # Synthetic f01/f1 names that are only package labels do not consume ref slots.
    temporal_bind: list[str] = []
    for name in temporal:
        low = name.lower()
        if low in {"f1.png"} or (low.startswith("f") and low[1:3].isdigit() and low.endswith(".png")):
            # Include only when the segment already points at a concrete first_frame file.
            if first and Path(first).name == name:
                temporal_bind.append(name)
            continue
        temporal_bind.append(name)

    ordered = [*temporal_bind, *characters, *scenes, *props]
    seen: set[str] = set()
    unique: list[str] = []
    for item in ordered:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return {
        "order": ["temporal", "character", "scene", "prop"],
        "temporal": temporal,
        "characters": characters,
        "scenes": scenes,
        "props": props,
        "refs": unique[:MAX_REFS],
        "segment_id": sid,
    }


def ref_bind_name(raw: str) -> str:
    """Comfy LoadImage filename for a ref token or path. Always a real file name."""
    text = str(raw or "").strip()
    if not text:
        return ""
    name = Path(text).name.strip() or text
    if "." not in name:
        return f"{name}.png"
    return name


def _is_synthetic_keyframe_name(name: str) -> bool:
    low = str(name or "").lower()
    if low in {"f1.png", "f1"}:
        return True
    return bool(low.startswith("f") and len(low) >= 5 and low[1:3].isdigit() and low.endswith(".png"))


def ref_bind_names(segment: dict[str, Any]) -> list[str]:
    """Comfy ref LoadImage slots: character/scene/prop files, not dummy f1/f01 labels.

    Temporal keyframes bind on the first-frame guide node when a real file exists.
    """
    manifest = segment.get("reference_manifest")
    if not isinstance(manifest, dict):
        manifest = build_reference_manifest(segment)
    names: list[str] = []
    seen: set[str] = set()

    def add(raw: Any) -> None:
        name = ref_bind_name(str(raw or ""))
        if not name:
            return
        key = name.lower()
        if key in seen or _is_synthetic_keyframe_name(name):
            return
        seen.add(key)
        names.append(name)

    groups = [
        *(manifest.get("characters") or []),
        *(manifest.get("scenes") or []),
        *(manifest.get("props") or []),
    ]
    if groups:
        for raw in groups:
            add(raw)
    else:
        for raw in list(manifest.get("refs") or segment.get("refs") or []):
            add(raw)
    return names[:MAX_REFS]


def compile_segment_v2(segment: dict[str, Any]) -> dict[str, Any]:
    """Return a segment enriched with v2 cuts, prompt, alignment, and manifest."""
    out = dict(segment)
    duration = float(out.get("duration") or SEGMENT_MAX_S)
    out["duration"] = _round2(min(SEGMENT_MAX_S, max(0.1, duration)))
    cuts = ensure_cuts_for_segment(out)
    out["cuts"] = cuts
    length = h3_length_frames(out["duration"])
    out["picture_alignment"] = cut_frame_ranges(cuts, total_frames=length)
    out["reference_manifest"] = build_reference_manifest(out)
    out["h3_prompt"] = compile_picture_prompt(out)
    # Keep still prompt English-only from the first cut framing.
    head = cuts[0] if cuts else {}
    still = _text(head.get("frame_prompt") or out.get("first_frame_prompt"))
    if still:
        out["first_frame_prompt"] = still
    out["text_policy"] = str(out.get("text_policy") or DEFAULT_TEXT_POLICY)
    out["schema_version"] = SCHEMA_VERSION
    return out


def gate_result(
    code: str,
    *,
    ok: bool,
    message: str,
    shot_id: str = "",
    cut_seq: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "code": code,
        "ok": bool(ok),
        "message": message,
        "shot_id": shot_id,
    }
    if cut_seq is not None:
        row["cut_seq"] = int(cut_seq)
    if details:
        row["details"] = details
    return row


def validate_segment_gates(segment: dict[str, Any], *, cast_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    sid = str(segment.get("id") or "")
    gates: list[dict[str, Any]] = []
    duration = float(segment.get("duration") or 0)
    cuts = ensure_cuts_for_segment(segment)
    cast_set = {str(c).strip() for c in (cast_ids or []) if str(c).strip()}

    gates.append(
        gate_result(
            "segment_duration_le_8s",
            ok=duration <= SEGMENT_MAX_S + 1e-6,
            message=f"segment duration {duration}s exceeds {SEGMENT_MAX_S}s",
            shot_id=sid,
            details={"duration": duration},
        )
    )

    cut_sum = _round2(sum(float(c.get("seconds") or 0) for c in cuts))
    gates.append(
        gate_result(
            "cuts_sum_equals_segment",
            ok=abs(cut_sum - _round2(duration)) <= 0.05,
            message=f"cut seconds sum {cut_sum} != segment {duration}",
            shot_id=sid,
            details={"cut_sum": cut_sum, "duration": duration},
        )
    )

    for cut in cuts:
        seq = int(cut.get("seq") or 0)
        sec = float(cut.get("seconds") or 0)
        ok_range = CUT_MIN_S - 1e-6 <= sec <= CUT_MAX_S + 1e-6 or len(cuts) == 1
        gates.append(
            gate_result(
                "cut_duration_2_to_5s",
                ok=ok_range,
                message=f"cut {seq} duration {sec}s outside {CUT_MIN_S}-{CUT_MAX_S}s",
                shot_id=sid,
                cut_seq=seq,
                details={"seconds": sec},
            )
        )
        chars = [str(c).strip() for c in (cut.get("characters") or []) if str(c).strip()]
        gates.append(
            gate_result(
                "on_screen_character_limit",
                ok=len(chars) <= MAX_ON_SCREEN,
                message=f"cut {seq} has {len(chars)} on-screen characters (max {MAX_ON_SCREEN})",
                shot_id=sid,
                cut_seq=seq,
                details={"characters": chars},
            )
        )
        frame = _text(cut.get("frame_prompt"))
        text_policy = str(cut.get("text_policy") or segment.get("text_policy") or DEFAULT_TEXT_POLICY)
        wants_text = bool(_VISIBLE_TEXT_RE.search(frame) or _CJK_RE.search(frame))
        if text_policy == "no_generated_text":
            gates.append(
                gate_result(
                    "no_generated_text",
                    ok=not wants_text,
                    message=f"cut {seq} asks the model to paint readable text",
                    shot_id=sid,
                    cut_seq=seq,
                )
            )
        cam = normalize_camera(cut.get("camera"))
        gates.append(
            gate_result(
                "h3_camera_enum",
                ok=cam in H3_CAMERA_ENUM,
                message=f"cut {seq} camera {cut.get('camera')!r} not in H3 enum",
                shot_id=sid,
                cut_seq=seq,
            )
        )
        line = cut.get("line")
        if isinstance(line, dict) and any(str(v).strip() for v in line.values()):
            need = speech_seconds_for_line(line, max_seconds=sec)
            gates.append(
                gate_result(
                    "tts_fits_cut",
                    ok=need <= sec + 0.15,
                    message=f"cut {seq} TTS needs ~{need:.2f}s but cut is {sec}s",
                    shot_id=sid,
                    cut_seq=seq,
                    details={"speech_s": need, "cut_s": sec},
                )
            )
        if segment.get("on_camera") is False:
            speaker = str(segment.get("character_id") or "").strip()
            if speaker and speaker in chars:
                gates.append(
                    gate_result(
                        "remote_speaker_not_on_camera",
                        ok=False,
                        message=f"remote speaker {speaker} listed on-camera in cut {seq}",
                        shot_id=sid,
                        cut_seq=seq,
                    )
                )

    manifest = build_reference_manifest({**segment, "cuts": cuts})
    gates.append(
        gate_result(
            "reference_count_le_9",
            ok=len(manifest.get("refs") or []) <= MAX_REFS,
            message=f"reference_manifest has {len(manifest.get('refs') or [])} refs (max {MAX_REFS})",
            shot_id=sid,
            details={"refs": manifest.get("refs")},
        )
    )

    # Adjacent composition collapse: identical frame prompts + size + camera.
    for i in range(1, len(cuts)):
        a, b = cuts[i - 1], cuts[i]
        same = (
            _text(a.get("frame_prompt")).lower() == _text(b.get("frame_prompt")).lower()
            and normalize_size(a.get("size")) == normalize_size(b.get("size"))
            and normalize_camera(a.get("camera")) == normalize_camera(b.get("camera"))
        )
        gates.append(
            gate_result(
                "adjacent_composition_collapse",
                ok=not same,
                message=f"cuts {a.get('seq')} and {b.get('seq')} share identical composition",
                shot_id=sid,
                cut_seq=int(b.get("seq") or i + 1),
            )
        )

    if cast_set:
        unknown = []
        for cut in cuts:
            for cid in cut.get("characters") or []:
                if str(cid).strip() and str(cid).strip() not in cast_set:
                    unknown.append(str(cid).strip())
        gates.append(
            gate_result(
                "characters_in_cast",
                ok=not unknown,
                message=f"unknown character ids: {unknown}",
                shot_id=sid,
                details={"unknown": unknown},
            )
        )

    return gates


def validate_board_gates(
    segments: Sequence[dict[str, Any]],
    *,
    cast: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cast_ids = [str(c.get("id") or "").strip() for c in (cast or []) if isinstance(c, dict)]
    gates: list[dict[str, Any]] = []
    claimed: dict[str, str] = {}
    scenes: list[str] = []

    for seg in segments:
        compiled = compile_segment_v2(dict(seg))
        gates.extend(validate_segment_gates(compiled, cast_ids=cast_ids))
        scene = str(compiled.get("scene_id") or compiled.get("location_id") or "").strip()
        scenes.append(scene)
        for cut in compiled.get("cuts") or []:
            for beat in cut.get("beat_ids") or []:
                key = str(beat).strip()
                if not key:
                    continue
                if key in claimed:
                    gates.append(
                        gate_result(
                            "duplicate_beat_claim",
                            ok=False,
                            message=f"beat {key} claimed by {claimed[key]} and {compiled.get('id')}",
                            shot_id=str(compiled.get("id") or ""),
                            cut_seq=int(cut.get("seq") or 0),
                        )
                    )
                else:
                    claimed[key] = str(compiled.get("id") or "")
            # Cross-scene cut inside one segment is forbidden.
            cut_scene = str(cut.get("scene_id") or scene).strip()
            if scene and cut_scene and cut_scene != scene:
                gates.append(
                    gate_result(
                        "cross_scene_cut",
                        ok=False,
                        message=f"cut scene {cut_scene} != segment scene {scene}",
                        shot_id=str(compiled.get("id") or ""),
                        cut_seq=int(cut.get("seq") or 0),
                    )
                )

    # Hard scene changes between consecutive segments are OK; flag only if a single
    # segment somehow listed multiple scenes via cuts (handled above).

    failed = [g for g in gates if not g.get("ok")]
    return {
        "version": SCHEMA_VERSION,
        "ok": not failed,
        "gates": gates,
        "failed": failed,
        "claimed_beats": sorted(claimed.keys()),
    }


def shots_to_v2_segments(shots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lift flat/packed shots into v2 segments with normalized cuts."""
    segments: list[dict[str, Any]] = []
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        seg = compile_segment_v2(dict(shot))
        # Preserve packed multi-line provenance when merge left a cineflow body
        # but no cuts — already handled by ensure_cuts_for_segment.
        segments.append(seg)
    return segments


def build_board_v2_payload(
    *,
    episode_code: str,
    shots: Sequence[dict[str, Any]],
    cast: Sequence[dict[str, Any]] | None = None,
    title: str = "",
    logline: str = "",
    synopsis: str = "",
    langs: Sequence[str] | None = None,
    target_s: float | None = None,
    drafted_by: str = "hosted_preprod",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    segments = shots_to_v2_segments(shots)
    validation = validate_board_gates(segments, cast=cast)
    payload: dict[str, Any] = {
        "episode_code": episode_code,
        "title": title,
        "logline": logline,
        "synopsis": synopsis,
        "langs": list(langs or []),
        "drafted_by": drafted_by,
        "target_s": target_s,
        "schema_version": SCHEMA_VERSION,
        "shots": list(shots),
        "segments": segments,
        "n_shots": len(shots),
        "n_segments": len(segments),
        "total_s": _round2(sum(float(s.get("duration") or 0) for s in segments)),
        "validation": {
            "version": validation["version"],
            "ok": validation["ok"],
            "gates": validation["gates"],
        },
    }
    if extra:
        for key, value in extra.items():
            if key in {"shots", "segments", "validation", "schema_version"}:
                continue
            payload[key] = value
    return payload


def append_gates_jsonl(story_root: Path | str, gates: Sequence[dict[str, Any]], *, episode_code: str = "EP001") -> Path:
    root = Path(story_root)
    dest = root / "episodes" / episode_code / GATES_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as fh:
        for gate in gates:
            fh.write(json.dumps(gate, ensure_ascii=False) + "\n")
    return dest


def source_hash_payload(parts: dict[str, Any]) -> str:
    """Stable hash for identity/scene/prop contracts used by asset_lock invalidation."""
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def character_source_hash(
    *,
    identity_prompt: str,
    age: int | None = None,
    name: str = "",
    exclusions: Sequence[str] | None = None,
) -> str:
    return source_hash_payload(
        {
            "identity_prompt": str(identity_prompt or "").strip(),
            "age": int(age) if age is not None else None,
            "name": str(name or "").strip(),
            "exclusions": [str(x).strip() for x in (exclusions or []) if str(x).strip()],
        }
    )


def scene_source_hash(
    *,
    plate_prompt: str,
    lighting: str = "",
    anchors: Sequence[str] | None = None,
    props: Sequence[str] | None = None,
) -> str:
    return source_hash_payload(
        {
            "plate_prompt": str(plate_prompt or "").strip(),
            "lighting": str(lighting or "").strip(),
            "anchors": [str(a).strip() for a in (anchors or []) if str(a).strip()],
            "props": [str(p).strip() for p in (props or []) if str(p).strip()],
        }
    )


def overlay_metadata_for_text(cut: dict[str, Any]) -> dict[str, Any] | None:
    """When text is required, emit compose overlay metadata instead of asking H3 to paint glyphs."""
    frame = _text(cut.get("frame_prompt"))
    if not (_VISIBLE_TEXT_RE.search(frame) or _CJK_RE.search(frame)):
        return None
    return {
        "type": "compose_text_overlay",
        "cut_seq": int(cut.get("seq") or 0),
        "text": frame,
        "policy": "compose_overlay_not_h3",
    }

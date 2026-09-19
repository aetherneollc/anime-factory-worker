"""Shared director helpers: prompts, line timing, shot emit, H3 duration split."""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from anime_factory.models import (
    H3_GEN_HEIGHT,
    H3_GEN_WIDTH,
    H3_MAX_REFS,
    H3_MAX_SECONDS,
    scrub_copycat,
)
from anime_factory.translate import estimate_speech_seconds

MIN_SHOT_SECONDS = 3.0
LEAD_IN_SECONDS = 0.35
TAIL_SECONDS = 0.45
ESTABLISHING_SECONDS = 5.0
MIN_SEGMENT_SECONDS = 2.0

SIZE_FRAMING = {
    "CU": "close-up",
    "MCU": "medium close-up",
    "medium": "medium shot",
    "wide": "wide establishing shot",
    "OTS": "over-the-shoulder",
}
SPOKEN_SIZES = ("MCU", "CU", "medium")
FILLER_SIZES = ("medium", "CU")
H3_TAIL = (
    f"keep face/costume of named character refs, no face drift, no costume swap, "
    f"camera motion must change, {H3_GEN_WIDTH}x{H3_GEN_HEIGHT}"
)
CINEFLOW_BRIDGE = "[动势匹配转场]"
CINEFLOW_STEPS = ("First", "Then", "Finally")
CUT_SIZE_ZH = {
    "CU": "特写",
    "MCU": "中近景",
    "medium": "中景",
    "wide": "全景",
    "OTS": "过肩",
}

# Dramatic beats the 舔狗 board collapsed into one store CU. Order is specific-first.
BEAT_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wechat", ("微信", "wechat", "朋友圈", "moments")),
    ("phone", ("电话", "phone", "没接", "打给", "on the phone", "发消息", "短信")),
    ("asleep", ("睡着", "asleep", "睡了", "fell asleep")),
    ("cat", ("猫", "cat", "kitten")),
    ("arrive", ("到了", "arrive", "到门口", "我到")),
    ("reject", ("回去吧", "不饿", "别这样", "reject", "以后别")),
    ("rain", ("雨", "rain", "伞", "umbrella")),
    ("door", ("门", "door", "门铃", "intercom")),
    ("store", ("便利店", "convenience store", "关东煮")),
)
REMOTE_BEATS = frozenset({"phone", "wechat", "asleep", "outside"})
STORE_COPY_MARKERS = (
    "convenience store",
    "same store",
    "便利店",
    "speaking, same store",
    "locked eyeline",
)
STAGING_FOR_BEAT = {
    "arrive": "arrive at the rain-soaked doorway, hold umbrella, hand on the door, not a store counter CU",
    "phone": "phone UI in another room, speaker off-camera at the store, do not put them in the convenience-store CU",
    "wechat": "WeChat / Moments overlay, phone in hand, not a convenience-store speaking CU",
    "reject": "apartment door cracked open from inside, reject the delivery, not the store counter",
    "asleep": "dark bedroom, figure asleep under a blanket, not speaking at the store",
    "cat": "stray cat at the wet curb by the doorway, low angle, store interior is not the subject",
    "rain": "heavy rain on the street, raincoat, puddles, not the store interior",
    "door": "exterior door and intercom, rain on the awning, not a store speaking CU",
    "store": "night convenience store counter, clerk under fluorescent light, in-person coverage",
    "dialogue": "coverage in the established location, distinct blocking, eyeline level",
}


def line_blob(line: dict | str | None) -> str:
    if isinstance(line, str):
        return line
    if not isinstance(line, dict):
        return ""
    parts = [
        line.get("zh"),
        line.get("en"),
        line.get("ja"),
        line.get("text"),
        line.get("staging"),
    ]
    return " ".join(str(p or "") for p in parts)


def line_beat_id(line: dict | str | None) -> str:
    blob = line_blob(line).lower()
    if not blob.strip():
        return "dialogue"
    for beat, markers in BEAT_MARKERS:
        for marker in markers:
            if marker.lower() in blob:
                return beat
    return "dialogue"


def line_on_camera(line: dict | None) -> bool:
    if not isinstance(line, dict):
        return True
    if line.get("on_camera") is not None:
        return bool(line.get("on_camera"))
    return line_beat_id(line) not in REMOTE_BEATS


def sheet_ref(character_id: str) -> str:
    cid = str(character_id or "").strip()
    return f"char_{cid}_sheet" if cid else ""


def _sheet_refs(segment: dict) -> list[str]:
    out: list[str] = []
    for raw in segment.get("refs") or []:
        text = str(raw or "").strip()
        if not text:
            continue
        if "char_" in text or "sheet" in text or "costume" in text:
            out.append(text)
    return out


def _on_screen_standin(scene: dict | None, speaker_id: str, cast: dict | None) -> str:
    """Off-camera phone/wechat drops the speaker, not the person still in frame."""
    speaker = str(speaker_id or "").strip()
    others: list[str] = []
    if isinstance(scene, dict):
        others.extend(str(c).strip() for c in (scene.get("characters") or []) if str(c).strip())
    if isinstance(cast, dict):
        others.extend(str(cid).strip() for cid in cast if str(cid).strip())
    for cid in others:
        if cid and cid != speaker:
            return cid
    return ""


def pictured_character_id(shot: dict, scene: dict | None = None, cast: dict | None = None) -> str:
    """Who Hailuo should lock. Hosted boards omit on_camera; trust character_id unless False."""
    speaker = str(shot.get("character_id") or "").strip()
    purpose = str(shot.get("purpose") or "").strip().lower()
    if purpose in {"establish", "establishing"}:
        return ""
    if shot.get("on_camera") is False:
        cuts_chars: list[str] = []
        for cut in shot.get("cuts") or []:
            cuts_chars.extend(str(c).strip() for c in (cut.get("characters") or []) if str(c).strip())
        extra = dict(cast or {})
        for cid in cuts_chars:
            extra.setdefault(cid, {})
        return _on_screen_standin(scene, speaker, extra)
    return speaker


def hydrate_shot_identity(shot: dict, *, scene: dict | None = None, cast: dict | None = None) -> dict:
    """Fill locked sheets for everyone pictured + scene plate. Empty establishing stays fl2va_first."""
    row = dict(shot)
    pictured_ids = pictured_character_ids(row, scene, cast)
    pictured = pictured_ids[0] if pictured_ids else pictured_character_id(row, scene, cast)
    existing = [str(r).strip() for r in (row.get("refs") or []) if str(r).strip()]
    plate_id = str(row.get("plate_id") or "").strip()
    last_frame = _continue_last_ref(row, existing)
    refs = ordered_shot_refs(pictured_ids, plate_id if pictured_ids else "", last_frame)
    for raw in existing:
        if raw not in refs and len(refs) < H3_MAX_REFS:
            refs.append(raw)
    costumes = dict(row.get("costume_ids") or {})
    for cid in pictured_ids:
        costumes.setdefault(cid, sheet_ref(cid))
    row["costume_ids"] = costumes
    row["refs"] = refs
    row["h3_mode"] = choose_h3_mode({**row, "refs": refs, "character_id": pictured or row.get("character_id")})
    if pictured:
        row["h3_mode"] = "ref2va"
    elif not _sheet_refs(row):
        row["h3_mode"] = "fl2va_first" if int(row.get("chain_index") or 0) == 0 else choose_h3_mode(row)
    return row


def choose_h3_mode(segment: dict) -> str:
    """Character shots are Hailuo ref2va. Empty establishing/off-camera plates are fl2va_first."""
    explicit = segment.get("h3_mode")
    if _sheet_refs(segment) or pictured_character_id(segment):
        return "ref2va"
    if int(segment.get("chain_index") or 0) > 0 or segment.get("match_cut"):
        return "fl2va_first_last" if explicit != "fl2va_first" else "fl2va_first"
    if explicit in {"ref2va", "fl2va_first", "fl2va_first_last"}:
        return explicit
    return "fl2va_first"


def line_text(line: dict, lang: str) -> str:
    value = line.get(lang)
    if not value and isinstance(line.get("text"), str):
        value = line["text"] if lang == "zh" else ""
    return str(value or "").strip()


def line_langs(line: dict, langs: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for lang in langs:
        text = line_text(line, lang)
        out[lang] = text or line_text(line, "zh") or str(line.get("text") or "").strip()
    return out


def line_speech_seconds(line: dict, langs: Sequence[str] | None = None) -> float:
    """Raw spoken length. Used to pack several lines into one ≤8s CineFlow take."""
    keys = list(langs) if langs else [k for k in ("zh", "en", "ja") if isinstance(line, dict) and line.get(k)]
    if not keys:
        keys = ["zh"]
    spoken = [estimate_speech_seconds(text, lang) for lang, text in line_langs(line, keys).items() if text]
    return round(max(spoken) if spoken else 0.0, 2)


def line_shot_seconds(line: dict, langs: Sequence[str], shot_seconds: float | None = None) -> float:
    """Shot duration follows speech. Do not clip to 8s — persist splits into segments."""
    longest = line_speech_seconds(line, langs)
    padded = longest + LEAD_IN_SECONDS + TAIL_SECONDS
    duration = round(max(padded, MIN_SHOT_SECONDS), 2)
    if shot_seconds is not None and shot_seconds > 0 and duration < MIN_SHOT_SECONDS:
        return round(float(shot_seconds), 2)
    return duration


def merge_line_langs(lines: Sequence[dict], langs: Sequence[str]) -> dict[str, str]:
    merged = {lang: "" for lang in langs}
    for line in lines:
        for lang, text in line_langs(line, langs).items():
            if not text:
                continue
            prev = merged.get(lang) or ""
            merged[lang] = f"{prev} {text}".strip() if prev else text
    return merged


def group_consecutive_speaker_lines(
    spoken: list[tuple[int, dict, dict, float]],
) -> list[list[tuple[int, dict, dict, float]]]:
    """Same speaker may chain for last-frame continue. A new dramatic beat starts a new shot."""
    groups: list[list[tuple[int, dict, dict, float]]] = []
    for item in spoken:
        cid = str(item[2].get("character_id") or "")
        beat = line_beat_id(item[2])
        if groups:
            prev = groups[-1][-1][2]
            same_speaker = str(prev.get("character_id") or "") == cid
            same_beat = line_beat_id(prev) == beat
            if same_speaker and same_beat:
                groups[-1].append(item)
                continue
        groups.append([item])
    return groups


def pack_consecutive_dialogue(
    scene_lines: Sequence[tuple[int, dict, dict, float]],
    max_s: float = H3_MAX_SECONDS,
) -> list[list[tuple[int, dict, dict, float]]]:
    """Pack consecutive in-scene lines into few long takes.

    H3 callers pass max_s=8. LongLive callers pass ~64s (NVlabs 384 latents).
    Split only when the take would exceed max_s, or when on-camera vs remote
    coverage cannot share a frame (phone/wechat vs store CU).
    Pack on speech length, not the 3s shot floor — otherwise one line fills an H3.
    """
    groups: list[list[tuple[int, dict, dict, float]]] = []
    current: list[tuple[int, dict, dict, float]] = []
    current_dur = 0.0
    current_remote: bool | None = None
    for item in scene_lines:
        dur = line_speech_seconds(item[2]) or float(item[3] or 0)
        remote = not line_on_camera(item[2])
        if current and current_remote is not None and remote != current_remote:
            groups.append(current)
            current = []
            current_dur = 0.0
            current_remote = None
        if current and current_dur + dur > max_s + 1e-9:
            groups.append(current)
            current = []
            current_dur = 0.0
            current_remote = None
        if not current:
            current_remote = remote
        current.append(item)
        current_dur += dur
    if current:
        groups.append(current)
    return groups


def scene_dialogue_chain_id(scene: dict, episode_code: str = "") -> str:
    sid = str(scene.get("id") or "").strip()
    if sid:
        return sid
    return f"{episode_code}-sc01" if episode_code else "sc01"


def ordered_shot_refs(
    pictured_ids: Sequence[str],
    plate_id: str = "",
    last_frame: str = "",
    max_refs: int = H3_MAX_REFS,
) -> list[str]:
    """Locked character sheets (everyone pictured) + locked scene plate + continue last frame."""
    refs: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        text = str(raw or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        refs.append(text)

    for cid in pictured_ids:
        add(sheet_ref(str(cid)))
    add(str(plate_id or "").strip())
    last = str(last_frame or "").strip()
    if last:
        room = max(1, int(max_refs) - 1)
        while len(refs) > room:
            refs.pop()
        add(last)
    return refs[: max(1, int(max_refs))]


def cite_locked_refs(prompt: str, refs: Sequence[str]) -> str:
    names = [str(r).strip() for r in refs if str(r).strip()]
    if not names:
        return scrub_copycat(prompt)
    tags = [f"<<<image_{i}>>>" for i in range(1, len(names) + 1)]
    head = (
        f"{', '.join(tags)} locked character-sheet and scene-plate refs, "
        "keep face/costume, no face drift, no costume swap, 禁止五官漂移/换装"
    )
    return scrub_copycat(f"{head}. {prompt}")


def cineflow_cut_script(cuts: Sequence[dict]) -> str:
    """First/Then / [动势匹配转场] for intra-shot cuts. Single cut keeps its staging."""
    rows = [c for c in cuts if isinstance(c, dict)]
    if not rows:
        return ""
    if len(rows) == 1:
        return str(rows[0].get("staging") or "").strip()
    parts: list[str] = []
    for i, cut in enumerate(rows):
        step = CINEFLOW_STEPS[i] if i < len(CINEFLOW_STEPS) else "Then"
        size = str(cut.get("size") or "MCU")
        zh_size = CUT_SIZE_ZH.get(size, SIZE_FRAMING.get(size, size))
        staging = str(cut.get("staging") or cut.get("camera") or "acting").strip()
        parts.append(f"[镜头{i + 1}：{zh_size}] {step}: {staging}")
    return f" {CINEFLOW_BRIDGE} ".join(parts)


def pictured_character_ids(shot: dict, scene: dict | None = None, cast: dict | None = None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        cid = str(raw or "").strip()
        if cid and cid not in seen:
            seen.add(cid)
            ids.append(cid)

    add(pictured_character_id(shot, scene, cast))
    for cut in shot.get("cuts") or []:
        if not isinstance(cut, dict):
            continue
        for other in cut.get("characters") or []:
            add(str(other))
    extra = shot.get("pictured_ids") or shot.get("on_screen") or []
    if isinstance(extra, (list, tuple)):
        for other in extra:
            add(str(other))
    return ids


def _continue_last_ref(shot: dict, existing: Sequence[str]) -> str:
    for raw in existing:
        low = str(raw).lower()
        if "last.png" in low or low.endswith("_last.png") or "chain_first" in low:
            return str(raw)
    if int(shot.get("chain_index") or 0) <= 0:
        return ""
    return str(shot.get("chain_source_last_frame") or shot.get("first_frame_path") or "").strip()


def _filler_plan(budget: float, shot_seconds: float) -> tuple[int, float]:
    if budget < MIN_SHOT_SECONDS:
        return 0, 0.0
    count = max(1, math.ceil(budget / shot_seconds))
    each = budget / count
    if each < MIN_SHOT_SECONDS:
        count = max(1, int(budget // MIN_SHOT_SECONDS))
        each = budget / count
    return count, round(each, 2)


def setting_text(scene: dict, locations: dict[str, dict], interiors: dict[str, dict]) -> str:
    interior = interiors.get(str(scene.get("interior_id") or ""))
    location = locations.get(str(scene.get("location_id") or ""))
    parts = [
        (interior or {}).get("plate_prompt") or (interior or {}).get("name"),
        (location or {}).get("plate_prompt") or (location or {}).get("name"),
        scene.get("time_of_day"),
        scene.get("mood"),
    ]
    return ", ".join(str(p).strip() for p in parts if str(p or "").strip())


def subject_text(character_id: str, cast: dict[str, dict]) -> str:
    char = cast.get(character_id) or {}
    return str(char.get("identity_prompt") or char.get("name") or character_id or "the lead character").strip()


def identity_text(character_id: str, cast: dict[str, dict] | None) -> str:
    """Canon's English visual lock for a character. Empty when canon has none."""
    char = (cast or {}).get(character_id) or {}
    return str(char.get("identity_prompt") or char.get("visual_lock_prompt") or "").strip()


def _scene_tag(setting: str, plate_id: str) -> str:
    head = str(setting or "").split(",")[0].strip() or str(plate_id or "scene").strip()
    return f"@{head}"


def _motion_text(camera: str) -> str:
    cam = str(camera or "").strip() or "Static Shot"
    if cam.lower() in {"static shot", "static", "locked-off"}:
        return "subtle acting motion, speaking gesture, do not freeze a still"
    return cam


# CJK, the Hailuo `@tag` / `<<<image_N>>>` reference syntax, camera direction and
# `shot N of M` are all video-side or bookkeeping terms. SDXL's CLIP reads English
# and treats every one of them as picture content.
CJK_RE = re.compile(r"[\u3000-\u303f\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]+")
BRACKET_TAG_RE = re.compile(r"[\[【][^\]】]*[\]】]")
IMAGE_TAG_RE = re.compile(r"<<<[^>]*>>>")
AT_TAG_RE = re.compile(r"@\s*")
SHOT_INDEX_RE = re.compile(r"\bshots?\s+\d+\s+of\s+\d+\b", re.I)
RESOLUTION_RE = re.compile(r"\b\d{3,4}\s*[x×]\s*\d{3,4}\b")
CINEFLOW_STEP_RE = re.compile(r"\b(?:first|then|finally)\s*:\s*", re.I)
# Motion verbs a still model cannot obey. "do not freeze a still" told the *still*
# model not to be a still.
CAMERA_TERM_RE = re.compile(
    r"\b(?:camera|dolly|dollies|zoom(?:s|ing)?|pan(?:s|ning)?|tilt(?:s|ing)?|truck(?:s|ing)?|"
    r"crane|handheld|steadicam|tracking\s+shot|push-?in|pull-?back|orbit(?:s|ing)?|"
    r"motion|movement|freeze|frame\s+rate|fps)\b",
    re.I,
)


def still_prompt_text(text: str) -> str:
    """Reduce any prompt fragment to English still description.

    Drops CJK, `@tag` / `<<<image_N>>>` reference syntax, CineFlow step labels,
    `shot N of M`, resolution strings and camera/motion clauses.
    """
    out = IMAGE_TAG_RE.sub(" ", str(text or ""))
    out = BRACKET_TAG_RE.sub(" ", out)
    out = CINEFLOW_STEP_RE.sub(" ", out)
    out = SHOT_INDEX_RE.sub(" ", out)
    out = RESOLUTION_RE.sub(" ", out)
    out = CJK_RE.sub(" ", out)
    out = AT_TAG_RE.sub("", out)
    clauses = []
    for clause in re.split(r"\s*[,;]\s*", out):
        body = clause.strip(" .·-—")
        if not body or CAMERA_TERM_RE.search(body):
            continue
        body = re.sub(r"\s{2,}", " ", body)
        if body and body not in clauses:
            clauses.append(body)
    return scrub_copycat(", ".join(clauses))


def prompt_pair(
    size: str,
    camera: str,
    subject: str,
    setting: str,
    staging: str,
    seq: int,
    total: int,
    character_id: str = "",
    plate_id: str = "",
    visual_en: str = "",
    motion_zh: str = "",
) -> tuple[str, str]:
    """Two prompts, two languages, two jobs.

    `first_frame_prompt` is English still description only: who, wearing what,
    doing what, where, in what light, at what framing. `h3_prompt` keeps the
    Chinese/`@tag`/camera/H3_TAIL video script. They are no longer the same string.
    """
    scene_tag = _scene_tag(setting, plate_id)
    framing = SIZE_FRAMING.get(size, "medium shot")
    motion = _motion_text(camera)
    if character_id:
        char_tag = f"@{subject}"
        h3_body = (
            f"{char_tag} 站在 {scene_tag} 中, {framing}, {staging}, "
            f"keep face/costume of {char_tag}"
        )
    else:
        h3_body = f"{scene_tag} anime location background, empty establishing shot, {framing}, {staging}, no extra people"
    zh_motion = str(motion_zh or "").strip()
    if zh_motion:
        h3_body = f"{h3_body}, {zh_motion}"
    still_parts = [framing]
    described = str(visual_en or "").strip()
    if described:
        still_parts.append(described)
    elif character_id:
        still_parts.append(subject)
    if not described:
        still_parts.append(staging)
        still_parts.append(setting)
        if not character_id:
            still_parts.append("anime location background, empty establishing shot, no people")
    first = still_prompt_text(", ".join(part for part in still_parts if str(part).strip()))
    h3 = scrub_copycat(f"{h3_body}, camera {camera}, {motion}, {H3_TAIL}")
    return first, h3


def scene_plate_id(scene: dict, interiors: dict[str, dict]) -> str:
    interior = interiors.get(str(scene.get("interior_id") or ""))
    scene_id = (interior or {}).get("scene_id") or scene.get("location_id") or scene.get("id")
    return f"plate_{scene_id}"


def opposite_eyeline(eyeline: str) -> str:
    return "left" if eyeline == "right" else "right"


def split_duration(seconds: float, max_s: float = H3_MAX_SECONDS) -> list[float]:
    """Split a shot longer than H3 max into even segments (10s → 5+5)."""
    seconds = max(0.1, float(seconds))
    if seconds <= max_s + 1e-9:
        return [round(seconds, 2)]
    n = max(2, math.ceil(seconds / max_s))
    parts = [round(seconds / n, 2)] * n
    parts[-1] = round(seconds - sum(parts[:-1]), 2)
    if any(p > max_s + 1e-9 or p < MIN_SEGMENT_SECONDS for p in parts):
        n = max(2, math.ceil(seconds / max(MIN_SEGMENT_SECONDS, max_s / 2)))
        while seconds / n > max_s:
            n += 1
        parts = [round(seconds / n, 2)] * n
        parts[-1] = round(seconds - sum(parts[:-1]), 2)
    return [p for p in parts if p > 0]


def is_chain_head(segment: dict) -> bool:
    return int(segment.get("chain_index") or 0) == 0


def needs_first_frame_still(segment: dict) -> bool:
    """Chain heads mint a composition f1 still. ref2va also needs the temporal keyframe."""
    if not is_chain_head(segment):
        return False
    mode = choose_h3_mode(segment)
    if mode in {"fl2va_first", "fl2va_first_last", "ref2va"}:
        return True
    # Character-bearing heads always need a composition anchor for H3 v2.
    return bool(str(segment.get("character_id") or "").strip()) and segment.get("on_camera") is not False


def expand_shots_to_segments(shots: list[dict], max_s: float | None = None) -> list[dict[str, Any]]:
    """Shot → one or more video segments. Same chain_id shares increasing chain_index."""
    limit = float(max_s) if max_s and max_s > 0 else H3_MAX_SECONDS
    segments: list[dict[str, Any]] = []
    global_seq = 0
    chain_cursor: dict[str, int] = {}
    for shot in shots:
        duration = float(shot.get("duration") or 0)
        already_split = "chain_index" in shot and duration <= limit + 1e-9
        parts = [duration] if already_split else split_duration(duration, max_s=limit)
        if not parts:
            continue
        shot_id = str(shot.get("shot_id") or shot.get("id") or f"shot-{len(segments) + 1}")
        chain_id = str(shot.get("chain_id") or f"{shot_id}-chain")
        n = len(parts)
        start_index = int(shot.get("chain_index") or 0) if already_split else chain_cursor.get(chain_id, 0)
        for i, dur in enumerate(parts):
            global_seq += 1
            seg = dict(shot)
            seg_id = shot_id if n == 1 else f"{shot_id}-s{i + 1:02d}"
            chain_index = start_index + i
            seg["id"] = seg_id
            seg["shot_id"] = shot_id
            seg["chain_id"] = chain_id
            seg["chain_index"] = chain_index
            seg["duration"] = round(float(dur), 2)
            seg["seq"] = global_seq
            if chain_index > 0:
                seg["status"] = "pending_chain"
                seg["h3_mode"] = choose_h3_mode({**shot, "chain_index": chain_index})
                seg.pop("keyframe_path", None)
                # Continuation first_frame is the previous last_frame, filled after render.
                if not seg.get("match_cut"):
                    seg["first_frame_path"] = None
            else:
                seg["h3_mode"] = choose_h3_mode({**shot, "chain_index": 0})
                seg.setdefault("status", "prepared")
            cuts = list(seg.get("cuts") or [])
            if cuts and n == 1:
                # Preserve multi-cut storyboard when the take already fits the backend cap.
                stored = []
                for j, cut in enumerate(cuts, start=1):
                    if not isinstance(cut, dict):
                        continue
                    row = dict(cut)
                    row["seq"] = int(row.get("seq") or j)
                    stored.append(row)
                if stored:
                    total = sum(float(c.get("seconds") or 0) for c in stored) or float(seg["duration"])
                    scaled = []
                    for row in stored:
                        share = float(row.get("seconds") or 0)
                        if total > 0 and abs(total - float(seg["duration"])) > 0.05:
                            share = float(seg["duration"]) * (share / total)
                        row = dict(row)
                        row["seconds"] = round(max(0.1, share), 2)
                        scaled.append(row)
                    if scaled:
                        head_sum = sum(c["seconds"] for c in scaled[:-1])
                        scaled[-1]["seconds"] = round(max(0.1, float(seg["duration"]) - head_sum), 2)
                    seg["cuts"] = scaled
            elif cuts:
                head = dict(cuts[0])
                head["seconds"] = seg["duration"]
                head["seq"] = 1
                seg["cuts"] = [head]
            segments.append(seg)
        chain_cursor[chain_id] = start_index + n
    return segments


def scale_shot_durations(shots: list[dict], target_seconds: float) -> None:
    natural = sum(float(s.get("duration") or 0) for s in shots)
    if natural <= 0 or target_seconds <= 0:
        return
    scale = target_seconds / natural
    if not (0.5 <= scale <= 2.0):
        return
    for shot in shots:
        shot["duration"] = round(float(shot["duration"]) * scale, 2)
        if shot.get("cuts"):
            shot["cuts"][0]["seconds"] = shot["duration"]


def emit_shot(
    shots: list[dict[str, Any]],
    *,
    episode_code: str,
    scene: dict,
    locations: dict[str, dict],
    interiors: dict[str, dict],
    cast: dict[str, dict],
    size: str,
    camera: str,
    character_id: str,
    staging: str,
    duration: float,
    line: dict | None,
    total_hint: int,
    purpose: str = "action",
    eyeline: str | None = None,
    match_cut: bool = False,
    on_camera: bool | None = None,
    on_screen_ids: Sequence[str] | None = None,
    chain_id: str | None = None,
    cut_list: Sequence[dict] | None = None,
    visual_en: str = "",
    motion_zh: str = "",
) -> dict[str, Any]:
    seq = len(shots) + 1
    setting = setting_text(scene, locations, interiors)
    plate_id = scene_plate_id(scene, interiors)
    if on_camera is None:
        on_camera = line_on_camera(line) if line else bool(character_id)
    pictured = bool(character_id) and bool(on_camera)
    standin = ""
    if purpose not in {"establish", "establishing"} and not pictured:
        standin = _on_screen_standin(scene, character_id, cast)
    pictured_id = character_id if pictured else standin
    ids: list[str] = []
    for cid in [pictured_id, *(on_screen_ids or [])]:
        text = str(cid or "").strip()
        if text and text not in ids:
            ids.append(text)
    subject = (
        subject_text(pictured_id or (ids[0] if ids else ""), cast)
        if (pictured_id or ids)
        else setting or "the empty set"
    )
    frame_staging = str(staging or "")
    if character_id and not on_camera:
        frame_staging = (
            f"{frame_staging}, speaker off-camera / remote, do not show {character_id} "
            "speaking in this location"
        ).strip(", ")
    cuts_in = [dict(c) for c in (cut_list or []) if isinstance(c, dict)]
    if len(cuts_in) > 1 and CINEFLOW_BRIDGE not in frame_staging:
        frame_staging = cineflow_cut_script(cuts_in) or frame_staging
    line_row = line if isinstance(line, dict) else {}
    described = str(visual_en or line_row.get("visual_en") or "").strip()
    motion = str(motion_zh or line_row.get("motion_zh") or "").strip()
    identity = identity_text(pictured_id or (ids[0] if ids else ""), cast)
    if described and identity:
        # Text-side identity lever: the locked sheet's own visual lock leads the still.
        described = f"{identity}, {described}"
    first, h3 = prompt_pair(
        size,
        camera,
        subject,
        setting,
        frame_staging,
        seq,
        max(total_hint, seq),
        character_id=pictured_id or (ids[0] if ids else ""),
        plate_id=plate_id,
        visual_en=described,
        motion_zh=motion,
    )
    refs = ordered_shot_refs(ids, plate_id if ids else "")
    # `<<<image_N>>>` is Hailuo multi-image syntax: video prompt only.
    h3 = cite_locked_refs(h3, refs)
    h3_mode = "ref2va" if ids else "fl2va_first"
    costumes = {cid: sheet_ref(cid) for cid in ids}
    if cuts_in:
        stored_cuts = []
        for j, cut in enumerate(cuts_in, start=1):
            row = dict(cut)
            row.setdefault("seq", j)
            row.setdefault("beats", [1, 1])
            row.setdefault("seconds", round(float(duration), 2))
            row.setdefault("size", size)
            row.setdefault("camera", camera)
            row.setdefault("characters", ids)
            stored_cuts.append(row)
        stored_cuts[0]["seconds"] = round(float(duration), 2)
    else:
        stored_cuts = [
            {
                "seq": 1,
                "beats": [1, 1],
                "seconds": round(float(duration), 2),
                "size": size,
                "camera": camera,
                "characters": ids,
                "staging": frame_staging,
            }
        ]
    shot = {
        "id": f"{episode_code}-{seq:02d}",
        "seq": seq,
        "duration": round(float(duration), 2),
        "scene_id": scene.get("id") or f"{episode_code}-sc01",
        "location_id": scene.get("location_id"),
        "interior_id": scene.get("interior_id"),
        "h3_mode": h3_mode,
        "refs": refs,
        "plate_id": plate_id,
        "costume_ids": costumes,
        "character_id": character_id or None,
        "on_camera": bool(on_camera),
        "first_frame_prompt": first,
        "h3_prompt": h3,
        "line": line,
        "purpose": purpose,
        "eyeline": eyeline,
        "match_cut": bool(match_cut),
        "chain_id": str(chain_id or f"{episode_code}-{seq:02d}-chain"),
        "shot_id": f"{episode_code}-{seq:02d}",
        "cuts": stored_cuts,
    }
    shots.append(shot)
    return shot


def world_maps(script: dict) -> tuple[list[dict], dict, dict, dict]:
    scenes = list(script.get("scenes") or [])
    cast = {str(c["id"]): c for c in script.get("cast") or [] if c.get("id")}
    locations = {str(loc["id"]): loc for loc in script.get("locations") or [] if loc.get("id")}
    interiors = {str(item["id"]): item for item in script.get("interiors") or [] if item.get("id")}
    return scenes, cast, locations, interiors


def spoken_items(
    scenes: list[dict], langs: Sequence[str], episode_code: str
) -> list[tuple[int, dict, dict, float]]:
    spoken: list[tuple[int, dict, dict, float]] = []
    for si, scene in enumerate(scenes):
        if not scene.get("id"):
            scene["id"] = f"{episode_code}-sc{si + 1:02d}"
        for line in scene.get("lines") or []:
            if not any(line_langs(line, langs).values()):
                continue
            spoken.append((si, scene, line, line_shot_seconds(line, langs)))
    return spoken

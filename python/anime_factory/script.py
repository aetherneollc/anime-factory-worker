"""Script stage: DeepSeek draft + gate2 rewrite loop. Blocked episodes never enter design.

`draft_episode_script` is the only place an episode's content comes from. It takes
the project title/logline plus the gate1 continuity injection and returns a draft in
the shape `run_script_gates` expects. With no director key it falls back to
`offline_draft`, which is for dry runs and CI, never for a delivered episode.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from typing import Any, Callable, Sequence

from anime_factory.continuity_gates import GateResult, run_script_gates
from anime_factory.db import utcnow
from anime_factory.directors.common import still_prompt_text
from anime_factory.langs import load_registry
from anime_factory.llm import LlmClient
from anime_factory.models import ROLE_DIRECTOR, scrub_copycat

DRAFT_SYSTEM = (
    "你是一部连载动画的分集编剧兼分镜。你只输出一个 JSON 对象，不要 markdown 代码块，不要解释。"
    "台词是漫剧对白：短句、口语、能演，不是旁白也不是散文。"
    "画面描述必须是这部片子自己的角色、道具、场景，不许写成任何知名电影剧照或某位导演的风格。"
    "每条台词的 staging 必须随场次、道具、电话/微信 vs 当面变化；禁止整集复制同一便利店对白。"
    "电话、微信、睡着、在外面的角色不要写成便利店特写，除非 on_camera 为 true。"
    "每条台词必须同时给两种画面描述："
    "visual_en 是**纯英文静帧**描述（人物外貌+服装+动作+表情+环境+光线+景别），"
    "里面禁止出现中文、@标记、运镜词（camera/dolly/pan/zoom/motion）、分辨率和 shot N of M；"
    "motion_zh 是**中文运镜与动势**描述（镜头怎么动、人物怎么动），只给视频模型用。"
)

DRAFT_SCHEMA = """{
  "synopsis": "本集一句话剧情",
  "recap": "给下一集用的结尾状态，一句话",
  "cast": [{"id": "ascii_snake_case", "name": "中文名", "age": 24, "gender": "male|female",
            "identity_prompt": "English visual lock: age, gender, build, hair, clothing, mood"}],
  "locations": [{"id": "ascii_snake_case", "name": "中文地名", "type": "settlement|landmark",
                 "plate_prompt": "English anime location background, empty establishing shot, no people, no text"}],
  "interiors": [{"id": "ascii_snake_case", "name": "中文室内名", "parent": "location_id",
                 "scene_id": "ascii_snake_case",
                 "plate_prompt": "English interior anime location background, empty establishing shot, no people, no text"}],
  "props": [{"id": "ascii_snake_case", "name": "中文道具名",
             "identity_prompt": "English prop design sheet, no people"}],
  "events": [{"id": "ev_ascii", "date": "Y1", "title": "本集发生的事"}],
  "scenes": [{"location_id": "...", "interior_id": "...", "interval_from_prev": "0h0m",
              "time_of_day": "night", "mood": "...", "synopsis": "这一场的画面基调",
              "lines": [{"character_id": "...", "LANGS": "每种语言一条同义台词",
                         "size": "CU|MCU|medium", "camera": "Static Shot",
                         "staging": "English staging note for this one shot — must change with the beat",
                         "visual_en": "English still description: appearance, clothing, action, expression, setting, light, framing. No Chinese, no @tags, no camera words",
                         "motion_zh": "中文运镜与动势：镜头如何移动、人物如何动作",
                         "on_camera": true,
                         "mentions_event_ids": []}]}]
}"""


def _lang_brief(langs: Sequence[str]) -> str:
    registry = load_registry()
    return "、".join(f"{code}（{registry[code].name}）" for code in langs if code in registry)


def build_draft_messages(
    *,
    episode_code: str,
    title: str,
    logline: str,
    langs: Sequence[str],
    target_seconds: float,
    shot_seconds: float,
    min_lines: int,
    max_lines: int,
    injected: dict,
    notes: str = "",
) -> list[dict[str, str]]:
    lang_keys = ", ".join(f'"{code}"' for code in langs)
    context = {
        "previous_recap": injected.get("previous_recap"),
        "previous_synopsis": injected.get("previous_synopsis"),
        "continuity_slice": injected.get("continuity_slice"),
        "open_threads": injected.get("open_threads"),
        "existing_places": injected.get("geo_places"),
        "known_events": injected.get("nearby_timeline"),
    }
    user = "\n".join(
        [
            f"片名：{title}",
            f"一句话故事：{logline}",
            f"本集：{episode_code}，成片目标 {target_seconds:.0f} 秒，单镜最长 {shot_seconds:.1f} 秒。",
            f"配音语种：{_lang_brief(langs)}。每条台词都要给全这些语种。",
            f"写 {min_lines} 到 {max_lines} 条台词，分成 3 到 5 场。",
            "沿用 existing_places 里已有的 id；只有故事真需要新地点时才新建。",
            "mentions_event_ids 只能填 continuity_slice.characters[].knows 里已有的 id，不确定就写 []。",
            "",
            "上下文（上一集结尾与已建立的世界，必须接得上）：",
            json.dumps(context, ensure_ascii=False),
            "",
            f"每条台词的语言字段用这些 key：{lang_keys}。按下面的结构输出：",
            DRAFT_SCHEMA,
        ]
    )
    if notes:
        user += f"\n\n上一版被连续性闸门打回，必须修掉这些问题：\n{notes}"
    return [{"role": "system", "content": DRAFT_SYSTEM}, {"role": "user", "content": user}]


def parse_draft_json(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("director returned no JSON object")
    return json.loads(text[start : end + 1])


def _ascii_id(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    return cleaned or fallback


SCHEMA_LABEL_RE = re.compile(
    r"^\s*(english\s+(visual\s+lock|staging\s+note|establishing\s+plate|interior\s+plate|prop\s+design\s+sheet)"
    r"[^:：]*|视觉锁定|画面说明)\s*[:：]\s*",
    re.I,
)


def _clean_prompt(text: str) -> str:
    """Drop the schema's own field label when the director echoes it back into the prompt."""
    return scrub_copycat(SCHEMA_LABEL_RE.sub("", str(text or "")).strip())


def normalize_draft(
    raw: dict,
    *,
    episode_code: str,
    title: str,
    langs: Sequence[str],
    known_event_ids: Sequence[str] = (),
    drafted_by: str = "director",
) -> dict:
    """Fill ids, mirror the primary language into `text`, and drop event mentions nobody knows."""
    primary = langs[0] if langs else "zh"
    known = set(known_event_ids)
    dropped: list[str] = []

    def _entities(key: str, id_fallback: str) -> list[dict]:
        out = []
        for i, item in enumerate(raw.get(key) or [], start=1):
            row = dict(item)
            row["id"] = _ascii_id(row.get("id") or row.get("name"), f"{id_fallback}{i}")
            for field in ("identity_prompt", "plate_prompt"):
                if row.get(field):
                    row[field] = _clean_prompt(row[field])
            out.append(row)
        return out

    locations = _entities("locations", "loc")
    interiors = _entities("interiors", "int")
    for interior in interiors:
        interior["scene_id"] = _ascii_id(interior.get("scene_id") or interior["id"], interior["id"])
        interior["asset_path"] = interior.get("asset_path") or f"assets/scenes/{interior['scene_id']}/plate_base.png"

    def _resolve_place(location_id: str) -> tuple[str, str]:
        """Gate 3 bounces a shot whose scene has no interior, so every scene gets one here."""
        if not any(l["id"] == location_id for l in locations):
            locations.append(
                {
                    "id": location_id,
                    "name": location_id,
                    "type": "settlement",
                    "plate_prompt": f"{location_id} anime location background, empty establishing shot, original production art, no people, no text",
                }
            )
        match = next((i for i in interiors if i.get("parent") == location_id), None)
        if match is None:
            match = {
                "id": f"int_{location_id}",
                "name": location_id,
                "parent": location_id,
                "scene_id": location_id,
                "plate_prompt": f"{location_id} interior, original production art, empty, no people, no text",
                "asset_path": f"assets/scenes/{location_id}/plate_base.png",
            }
            interiors.append(match)
        return location_id, match["id"]

    scenes: list[dict[str, Any]] = []
    line_no = 0
    for si, scene in enumerate(raw.get("scenes") or [], start=1):
        lines = []
        for line in scene.get("lines") or []:
            texts = {code: str(line.get(code) or "").strip() for code in langs}
            if not texts.get(primary):
                texts[primary] = str(line.get("text") or "").strip()
            if not texts.get(primary):
                continue
            for code in langs:
                if not texts[code]:
                    texts[code] = texts[primary]
            line_no += 1
            mentions = [str(e) for e in (line.get("mentions_event_ids") or [])]
            kept = [e for e in mentions if e in known]
            dropped.extend(e for e in mentions if e not in known)
            lines.append(
                {
                    "id": line.get("id") or f"{episode_code}-L{line_no:03d}",
                    "character_id": _ascii_id(line.get("character_id"), "lead"),
                    "text": texts[primary],
                    **texts,
                    "size": line.get("size"),
                    "camera": line.get("camera"),
                    "staging": _clean_prompt(line.get("staging") or ""),
                    # still_prompt_text strips anything the director slipped in that a
                    # still model cannot read (CJK, @tags, camera direction).
                    "visual_en": still_prompt_text(_clean_prompt(line.get("visual_en") or "")),
                    "motion_zh": _clean_prompt(line.get("motion_zh") or ""),
                    "on_camera": None if line.get("on_camera") is None else bool(line.get("on_camera")),
                    "mentions_event_ids": kept,
                }
            )
        if not lines:
            continue
        declared_interior = _ascii_id(scene.get("interior_id"), "")
        location_id, interior_id = _resolve_place(_ascii_id(scene.get("location_id"), "loc_main"))
        if declared_interior and any(i["id"] == declared_interior for i in interiors):
            interior_id = declared_interior
        scenes.append(
            {
                "id": scene.get("id") or f"{episode_code}-sc{si:02d}",
                "seq": si,
                "location_id": location_id,
                "interior_id": interior_id,
                "interval_from_prev": scene.get("interval_from_prev") or "0h0m",
                "time_of_day": scene.get("time_of_day"),
                "mood": scene.get("mood"),
                "synopsis": _clean_prompt(scene.get("synopsis") or ""),
                "characters": sorted({ln["character_id"] for ln in lines}),
                "lines": lines,
            }
        )
    if not scenes:
        raise ValueError(f"{episode_code}: draft has no usable lines")

    speakers = sorted({ln["character_id"] for sc in scenes for ln in sc["lines"]})
    cast = _entities("cast", "char")
    known_cast = {c["id"] for c in cast}
    for cid in speakers:
        if cid not in known_cast:
            # identity_prompt is a pure appearance lock: it lands inside scene stills,
            # so "design sheet" wording would turn every keyframe into a model sheet.
            cast.append(
                {
                    "id": cid,
                    "name": cid,
                    "gender": "male",
                    "identity_prompt": (
                        f"1boy, young adult, short dark hair, brown eyes, plain grey hoodie, lean build, quiet eyes"
                    ),
                }
            )
    return {
        "title": episode_code,
        "story_title": title,
        "synopsis": str(raw.get("synopsis") or logline_fallback(raw, title)),
        "recap": str(raw.get("recap") or ""),
        "drafted_by": drafted_by,
        "cast": cast,
        "locations": locations,
        "interiors": interiors,
        "props": _entities("props", "prop"),
        "events": _entities("events", "ev"),
        "scenes": scenes,
        "dropped_event_mentions": sorted(set(dropped)),
    }


def logline_fallback(raw: dict, title: str) -> str:
    first = (raw.get("scenes") or [{}])[0]
    return str(first.get("synopsis") or title)


def draft_episode_script(
    *,
    episode_code: str,
    title: str,
    logline: str,
    langs: Sequence[str],
    injected: dict,
    target_seconds: float,
    shot_seconds: float,
    min_lines: int = 48,
    max_lines: int = 88,
    llm: LlmClient | None = None,
    notes: str = "",
) -> dict:
    known_event_ids = [str(ev.get("id")) for ev in injected.get("nearby_timeline") or []]
    for ch in (injected.get("continuity_slice") or {}).get("characters") or []:
        known_event_ids.extend(str(e) for e in ch.get("knows") or [])
    if llm is None:
        raw = offline_draft(
            episode_code=episode_code,
            title=title,
            logline=logline,
            langs=langs,
            target_seconds=target_seconds,
            shot_seconds=shot_seconds,
            min_lines=min_lines,
            max_lines=max_lines,
        )
        drafted_by = "offline"
    else:
        messages = build_draft_messages(
            episode_code=episode_code,
            title=title,
            logline=logline,
            langs=langs,
            target_seconds=target_seconds,
            shot_seconds=shot_seconds,
            min_lines=min_lines,
            max_lines=max_lines,
            injected=injected,
            notes=notes,
        )
        raw = parse_draft_json(llm.chat(ROLE_DIRECTOR, messages, response_format={"type": "json_object"}))
        drafted_by = "director"
    return normalize_draft(
        raw,
        episode_code=episode_code,
        title=title,
        langs=langs,
        known_event_ids=known_event_ids,
        drafted_by=drafted_by,
    )


CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{2,4}")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]{2,}")
CLAUSE_RE = re.compile(r"[，。、；：！？,.;:!?\s]+")

OFFLINE_BEATS = (
    ("{a}还在那里吗。", "Is {a} still there.", "{a}はまだあるのか。"),
    ("你听见{b}了。", "You heard {b}.", "{b}が聞こえた。"),
    ("{c}这次不一样。", "{c} is different this time.", "{c}は今回違う。"),
    ("{b}不该在{c}。", "{b} should not be in {c}.", "{b}が{c}にあるはずない。"),
    ("我记得{a}的样子。", "I remember what {a} looked like.", "{a}の姿は覚えている。"),
    ("别碰{b}。", "Do not touch {b}.", "{b}に触るな。"),
    ("天亮之前离开{c}。", "Leave {c} before dawn.", "夜明け前に{c}を出る。"),
    ("{a}在等谁。", "Who is {a} waiting for.", "{a}は誰を待っている。"),
    ("关于{a}我只说一次。", "About {a}, I say it once.", "{a}のことは一度だけ言う。"),
    ("再说一遍{b}。", "Say {b} again.", "{b}をもう一度。"),
    ("{c}让你手在抖。", "{c} makes your hands shake.", "{c}で手が震えてる。"),
    ("因为{a}会记得。", "Because {a} will remember.", "{a}が覚えているから。"),
)


def _seed_words(title: str, logline: str) -> list[str]:
    words: list[str] = []
    for clause in CLAUSE_RE.split(f"{title} {logline}"):
        if not clause:
            continue
        words.extend(CJK_RE.findall(clause)[:1])
        words.extend(WORD_RE.findall(clause)[:2])
    unique = list(dict.fromkeys(w for w in words if w))
    return unique or [title.strip() or "这件事"]


def offline_draft(
    *,
    episode_code: str,
    title: str,
    logline: str,
    langs: Sequence[str],
    target_seconds: float,
    shot_seconds: float,
    min_lines: int = 48,
    max_lines: int = 88,
) -> dict:
    """Deterministic stand-in for the director. Same inputs in, same script out; a new title or episode, a new script."""
    digest = hashlib.blake2b(f"{title}\x00{logline}\x00{episode_code}".encode("utf-8"), digest_size=8).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    words = _seed_words(title, logline)
    tag = digest.hex()[:6]
    n_scenes = 3 + rng.randrange(3)
    ceiling = max(min_lines, min(max_lines, int(target_seconds / shot_seconds) + 12))
    n_lines = rng.randrange(min_lines, ceiling + 1)

    cast = []
    for i in range(2 + rng.randrange(2)):
        name = words[(i * 3 + rng.randrange(len(words))) % len(words)]
        cast.append(
            {
                "id": f"ch_{tag}_{i + 1}",
                "name": str(name)[:6],
                "age": 20 + rng.randrange(25),
                # English only, for the same reason as plate_prompt below.
                "gender": rng.choice(("male", "female")),
                "identity_prompt": (
                    f"{rng.choice(('1boy', '1girl'))}, "
                    f"{rng.choice(('young adult', 'adult'))}, "
                    f"{rng.choice(('short black hair', 'tied-back hair', 'cropped grey hair'))}, "
                    f"{rng.choice(('brown eyes', 'dark eyes', 'grey eyes'))}, "
                    f"{rng.choice(('dark work jacket', 'thin raincoat', 'wool coat'))}, "
                    f"{rng.choice(('lean build', 'stocky build', 'tall build'))}, "
                    f"{rng.choice(('scarred knuckles', 'wire-rim glasses', 'taped wristwatch', 'frayed collar'))}"
                ),
            }
        )
    locations = []
    interiors = []
    for i in range(n_scenes):
        place = str(words[(i * 5 + 1) % len(words)])[:6]
        lid = f"loc_{tag}_{i + 1}"
        # plate_prompt has to carry the story-specific look in **English**: the
        # still prompt strips CJK, so a Chinese place name leaves every project
        # with the same generic plate.
        shape = rng.choice(("low brick", "concrete stilt", "timber-frame", "tiled-roof", "corrugated iron"))
        feature = rng.choice(
            ("flooded corridor", "cable-strung yard", "half-shuttered arcade", "silted stairwell", "relay mast")
        )
        light = rng.choice(("night", "overcast morning", "blue hour"))
        locations.append(
            {
                "id": lid,
                "name": place,
                "type": "landmark" if i % 2 else "settlement",
                "plate_prompt": (
                    f"{shape} {feature}, {light}, anime location background, empty establishing shot, "
                    "original production location art, no people, no text"
                ),
            }
        )
        interiors.append(
            {
                "id": f"int_{tag}_{i + 1}",
                "name": f"{place}内",
                "parent": lid,
                "scene_id": f"sc_{tag}_{i + 1}",
                "plate_prompt": (
                    f"interior of the {shape} {feature}, {rng.choice(('bare bulb', 'desk lamp', 'window light'))}, "
                    "original production art, empty, no people, no text"
                ),
            }
        )
    props = [
        {
            "id": f"prop_{tag}_{i + 1}",
            "name": str(words[(i * 7 + 3) % len(words)] if len(words) > 1 else f"遗物{i + 1}")[:6],
            "identity_prompt": f"{rng.choice(('matte metal', 'worn paper', 'chipped enamel'))} "
            f"{rng.choice(('ledger', 'hand bell', 'signal lamp', 'mail crate'))}, prop design sheet, no people, no text",
        }
        for i in range(2)
    ]

    a = str(words[0])[:6]
    b = props[0]["name"]
    scenes: list[dict[str, Any]] = []
    per_scene = max(1, n_lines // n_scenes)
    made = 0
    beat = rng.randrange(len(OFFLINE_BEATS))
    for i in range(n_scenes):
        count = per_scene if i < n_scenes - 1 else n_lines - made
        lines = []
        c = locations[i]["name"]
        for j in range(count):
            beat = (beat + 1 + rng.randrange(len(OFFLINE_BEATS) - 1)) % len(OFFLINE_BEATS)
            zh, en, ja = OFFLINE_BEATS[beat]
            speaker = cast[(made + j) % len(cast)]["id"]
            texts = {
                "zh": zh.format(a=a, b=b, c=c),
                "en": en.format(a=a, b=b, c=c),
                "ja": ja.format(a=a, b=b, c=c),
            }
            expression = ("tight-jawed", "quiet", "restless")[(made + j) % 3]
            lines.append(
                {
                    "character_id": speaker,
                    "size": ("CU", "MCU", "medium")[(made + j) % 3],
                    "camera": "Static Shot",
                    "staging": f"{speaker} speaking in {locations[i]['name']}, eyeline level, no new subjects",
                    # Same two-prompt contract the live director must honour.
                    "visual_en": (
                        f"{cast[(made + j) % len(cast)]['identity_prompt']}, speaking, {expression} expression, "
                        f"{locations[i]['plate_prompt']}, eyeline level"
                    ),
                    "motion_zh": ("镜头缓慢推近，人物微微侧身", "镜头静止，人物换重心", "镜头轻微横移，人物抬头")[(made + j) % 3],
                    "mentions_event_ids": [],
                    **{code: texts.get(code, texts["zh"]) for code in langs},
                }
            )
        made += count
        scenes.append(
            {
                "location_id": locations[i]["id"],
                "interior_id": interiors[i]["id"],
                "interval_from_prev": "0h0m",
                "time_of_day": ("night", "dawn", "dusk")[i % 3],
                "mood": ("tight", "quiet", "restless")[i % 3],
                "synopsis": f"{locations[i]['name']}，{logline[:24]}",
                "lines": lines,
            }
        )
    return {
        "synopsis": f"{title}｜{episode_code}｜{logline}",
        "recap": f"{cast[0]['name']} 还守着 {b}，{episode_code} 到此为止。",
        "cast": cast,
        "locations": locations,
        "interiors": interiors,
        "props": props,
        "events": [{"id": f"ev_{tag}", "date": "Y1", "title": f"{title} 的开端"}],
        "scenes": scenes,
    }


def persist_script(conn: sqlite3.Connection, episode_code: str, script: dict) -> None:
    now = utcnow()
    conn.execute(
        """
        INSERT INTO episodes (episode_code, title, status, synopsis, recap, script_text,
                              duration_target_s, created_at, updated_at)
        VALUES (?, ?, 'running', ?, ?, ?, 600, ?, ?)
        ON CONFLICT(episode_code) DO UPDATE SET
            script_text = excluded.script_text,
            synopsis = excluded.synopsis,
            status = 'running',
            updated_at = excluded.updated_at
        """,
        (
            episode_code,
            script.get("title") or episode_code,
            script.get("synopsis"),
            script.get("recap"),
            json.dumps(script, ensure_ascii=False),
            now,
            now,
        ),
    )
    for i, scene in enumerate(script.get("scenes") or [], start=1):
        sid = scene.get("id") or f"{episode_code}-sc{i:02d}"
        conn.execute(
            """
            INSERT INTO scenes (id, episode_code, seq, location_id, interior_id, mood, time_of_day,
                                season, synopsis, beats_json, interval_from_prev)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET beats_json = excluded.beats_json
            """,
            (
                sid,
                episode_code,
                scene.get("seq", i),
                scene.get("location_id"),
                scene.get("interior_id"),
                scene.get("mood"),
                scene.get("time_of_day"),
                scene.get("season"),
                scene.get("synopsis"),
                json.dumps(scene.get("lines") or scene.get("beats") or [], ensure_ascii=False),
                scene.get("interval_from_prev"),
            ),
        )
    conn.commit()


def produce_script(
    conn: sqlite3.Connection,
    episode_code: str,
    drafts: list[dict],
    geo: dict,
    timeline: dict,
    continuity: dict,
    world_mode: str,
    media_hook: Callable[[str], None] | None = None,
) -> GateResult:
    result = run_script_gates(
        conn,
        episode_code,
        drafts,
        geo,
        timeline,
        continuity,
        world_mode,
        media_hook=media_hook,
    )
    if result.ok:
        persist_script(conn, episode_code, drafts[result.rewrites])
        conn.execute(
            "UPDATE episodes SET status = 'running', updated_at = ? WHERE episode_code = ?",
            (utcnow(), episode_code),
        )
        conn.commit()
    return result

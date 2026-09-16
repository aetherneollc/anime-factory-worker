"""Local/dev episode runner. Cloudflare Workflow maps onto these named stages."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from anime_factory.archive import ArchiveError, ColdStore, build_manifest, finish_and_archive, pack_story
from anime_factory.board import persist_board
from anime_factory.canon import export_continuity, export_geo, export_timeline
from anime_factory.compose import NON_GPU_STAGES
from anime_factory.config import load_settings
from anime_factory.db import checkpoint_and_upload, migrate, open_db, utcnow
from anime_factory.design import KolorsClient, generate_asset_library
from anime_factory.visual_qc import ClipScorer
from anime_factory.ids import new_story_id
from anime_factory.keyframe import assert_keyframe_files, ensure_keyframe
from anime_factory.langs import normalize_langs
from anime_factory.script import produce_script
from anime_factory.translate import Line, segment_duration, translate_lines
from anime_factory.tts import CosyVoiceClient, VoiceLockError, pcm16_wav_bytes, require_voice_uri
from anime_factory.world import init_fiction_world

STEP_NAMES = (
    "bible",
    "canon",
    "script",
    "tts",
    "board",
    "design",
    "keyframe",
    "anim",
    "qc",
    "compose",
    "master",
    "archive",
)

TTS_INTERNAL = ("tts_zh", "translate", "tts_en_ja", "timing")
GPU_STAGES = ("anim", "qc", "compose", "master")


def hard_order_ok(completed: list[str]) -> bool:
    """tts/timing before board/anim; non-GPU before GPU session."""
    if "board" in completed and "tts" not in completed:
        return False
    if "anim" in completed and "tts" not in completed:
        return False
    if "anim" in completed and "board" not in completed:
        return False
    return True


def _lock_voice(conn: sqlite3.Connection, character_id: str, langs: Sequence[str] | None = None) -> None:
    for lang in normalize_langs(langs):
        conn.execute(
            """
            INSERT INTO character_voice (character_id, lang, voice_uri, reference_audio, speed, emotion, version)
            VALUES (?, ?, ?, ?, 1.0, 'neutral', 'v1')
            ON CONFLICT(character_id, lang) DO UPDATE SET voice_uri = excluded.voice_uri
            """,
            (character_id, lang, f"voice://{character_id}/{lang}", f"assets/voice/{character_id}.{lang}.wav"),
        )
    conn.execute(
        """
        INSERT INTO characters (id, name, age, alive, current_location_id)
        VALUES (?, ?, 28, 1, ?)
        ON CONFLICT(id) DO UPDATE SET alive = 1
        """,
        (character_id, character_id, None),
    )
    conn.commit()


def run_local_episode(
    root: Path,
    title: str = "Demo Harbor",
    story_id: str | None = None,
    vast_dry_run: bool | None = None,
    clip_scorer: ClipScorer | None = None,
) -> dict[str, Any]:
    """Offline fixture through non-GPU stages. GPU steps skipped unless dry-run is off."""
    settings = load_settings()
    dry = settings.vast_dry_run if vast_dry_run is None else vast_dry_run
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sid = story_id or new_story_id(title)
    db_path = root / "story.sqlite"
    conn = open_db(db_path)
    migrate(conn)

    status: dict[str, str] = {name: "pending" for name in STEP_NAMES}
    episode = "EP001"

    world = init_fiction_world(
        conn,
        sid,
        title,
        locations=[
            {"id": "harbor", "name": "Harbor", "aka": [], "type": "settlement"},
            {"id": "ridge", "name": "Ridge", "aka": [], "type": "landmark", "parent": "harbor"},
        ],
        edges=[
            {
                "from": "harbor",
                "to": "ridge",
                "mode": "walk",
                "distance_km": 2,
                "travel_time": "0h24m",
            }
        ],
        interiors=[
            {
                "id": "int_dock",
                "name": "Dock",
                "parent": "harbor",
                "scene_id": "dock",
                "asset_path": "assets/scenes/dock/plate_base.png",
            }
        ],
        timeline=[
            {
                "id": "ev_pre",
                "date": "Y1",
                "title": "Founding",
                "kind": "fictional",
                "episode": None,
                "sources": [],
            }
        ],
        root=root,
    )
    status["bible"] = "passed"
    status["canon"] = "passed"

    _lock_voice(conn, "hero")
    conn.execute(
        """
        INSERT INTO continuity (episode_code, version, payload_json, created_at)
        VALUES ('EP000', 1, ?, ?)
        ON CONFLICT(episode_code) DO UPDATE SET payload_json = excluded.payload_json
        """,
        (
            json.dumps(
                {
                    "characters": [{"id": "hero", "alive": True, "knows": ["ev_pre"]}],
                    "positions": {"hero": "harbor"},
                    "props": [],
                    "locations": [],
                    "open_threads": [],
                }
            ),
            utcnow(),
        ),
    )
    conn.commit()

    geo = export_geo(conn, sid, invented=True, sources=[])
    timeline = export_timeline(conn, sid)
    continuity = export_continuity(conn, sid, "EP000")
    script = {
        "title": episode,
        "synopsis": "Hero waits at the dock.",
        "scenes": [
            {
                "id": f"{episode}-sc01",
                "location_id": "harbor",
                "interior_id": "int_dock",
                "interval_from_prev": "0h0m",
                "characters": ["hero"],
                "lines": [
                    {
                        "id": "L1",
                        "character_id": "hero",
                        "text": "潮水来了。",
                        "mentions_event_ids": ["ev_pre"],
                    }
                ],
            }
        ],
    }
    gated = produce_script(conn, episode, [script], geo, timeline, continuity, "fiction")
    if not gated.ok:
        status["script"] = "blocked"
        return {"story_id": sid, "status": status, "blocked": True, "violations": gated.violations}
    status["script"] = "passed"

    try:
        require_voice_uri(conn, "hero", "zh")
    except VoiceLockError:
        status["tts"] = "blocked"
        return {"story_id": sid, "status": status, "blocked": True}

    def opener(_req):
        return pcm16_wav_bytes(1.2)

    client = CosyVoiceClient(["test-key"], opener=opener, live=False)
    audio_dir = root / "episodes" / episode / "audio" / "lines"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_bytes = client.speech("潮水来了。", require_voice_uri(conn, "hero", "zh"))
    dur_zh = 1.2
    (audio_dir / "L1.zh.wav").write_bytes(wav_bytes)
    lines = [
        Line(id="L1", character_id="hero", zh="潮水来了。", dur_zh=dur_zh, en="The tide is in.", ja="潮が来た。")
    ]
    tr = translate_lines(lines, [])
    line = tr.lines[0]
    duration = segment_duration(line)
    status["tts"] = "passed"
    status["timing"] = "passed"

    persist_board(
        conn,
        episode,
        [
            {
                "id": "E01-01",
                "scene_id": f"{episode}-sc01",
                "duration": duration,
                "h3_mode": "ref2va",
                "refs": ["char_hero_sheet", "plate_dock"],
                "plate_id": "plate_dock",
                "interior_id": "int_dock",
                "first_frame_prompt": "medium shot, man on wooden dock, dusk",
                "costume_ids": {"hero": "char_hero_sheet"},
                "cuts": [
                    {
                        "seq": 1,
                        "beats": [1, 1],
                        "seconds": duration,
                        "size": "medium",
                        "camera": "Static Shot",
                        "characters": ["hero"],
                    }
                ],
            }
        ],
    )
    status["board"] = "passed"

    kolors = KolorsClient(["test-key"], live=False)
    chars, locs, props, interiors = (
        [
            {
                "id": "hero",
                "name": "hero",
                "identity_prompt": (
                    "1boy, adult, short black hair, dark brown eyes, dark wool coat, lean build, "
                    "clean cel-shaded edges, dusk rim light"
                ),
                "gender": "male",
                "seed": 1,
            }
        ],
        [
            {
                "id": "harbor",
                "name": "Harbor",
                "plate_prompt": "wooden dock at dusk",
                "seed": 2,
            }
        ],
        [
            {
                "id": "lantern",
                "name": "Lantern",
                "identity_prompt": "brass lantern",
                "seed": 3,
            }
        ],
        [
            {
                "id": "int_dock",
                "name": "Dock",
                "scene_id": "dock",
                "asset_path": "assets/scenes/dock/plate_base.png",
                "plate_prompt": "wooden dock at dusk",
            }
        ],
    )
    lib = generate_asset_library(
        conn,
        sid,
        chars,
        locs,
        props,
        kolors,
        None,
        "fiction",
        root,
        interiors=interiors,
        skip_existing=False,
        clip_scorer=clip_scorer,
    )
    specs = lib["specs"]
    status["design"] = "passed"

    segment = {
        "id": "E01-01",
        "scene_id": "dock",
        "interior_id": "int_dock",
        "refs": ["char_hero_sheet", "plate_dock"],
        "plate_id": "plate_dock",
        "costume_ids": {"hero": "char_hero_sheet"},
        "first_frame_prompt": "medium shot, man on wooden dock, dusk",
    }
    ensure_keyframe(
        conn,
        sid,
        episode,
        segment,
        geo,
        {"items": [{"id": k, **v} for k, v in specs.items()]},
        specs,
        kolors,
        None,
        "fiction",
        root,
    )
    assert_keyframe_files(
        root,
        episode,
        [segment],
        {"items": [{"id": k, **v} for k, v in specs.items()]},
    )
    status["keyframe"] = "passed"

    if dry:
        for name in GPU_STAGES:
            status[name] = "skipped_dry_run"
    else:
        for name in GPU_STAGES:
            status[name] = "pending"

    manifest = build_manifest(sid, "0.0.0-dev", "fiction", character_seeds={"hero": 1})
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pack = root / "pack.tar"
    pack_story(root, pack)
    store = ColdStore(backend="none")
    try:
        finish_and_archive(sid, "0.0.0-dev", "producing", [], pack.read_bytes(), store)
        status["archive"] = "passed"
    except ArchiveError:
        # none-backend must refuse purge; demo pack still written locally
        status["archive"] = "kept_heat_none_backend"

    checkpoint_and_upload(conn, db_path)
    if not hard_order_ok([s for s, v in status.items() if v == "passed"]):
        raise RuntimeError("hard order violated")
    if not all(status[s] == "passed" for s in NON_GPU_STAGES):
        raise RuntimeError(f"non-GPU stages incomplete: {status}")
    return {"story_id": sid, "status": status, "blocked": False, "root": str(root), "duration": duration}

"""CineFlow long takes, Toonflow last-frame tails, concat drop-frame-0."""

from __future__ import annotations

from pathlib import Path

from anime_factory.board import expand_shots_to_segments
from anime_factory.compose import chain_overlap_frames, drop_first_frames_cmd
from anime_factory.design import synthetic_still_png
from anime_factory.directors.common import (
    CINEFLOW_BRIDGE,
    cite_locked_refs,
    hydrate_shot_identity,
    ordered_shot_refs,
    pack_consecutive_dialogue,
)
from anime_factory.board import board_from_script
from anime_factory.directors.longlive import board_longlive, pack_longlive_takes
from anime_factory.directors.short import board_short
from anime_factory.models import H3_MAX_SECONDS, LONGLIVE_MAX_SECONDS
from gpu_worker.h3 import apply_chain_first_frame, native_h3_graph


def _art(tag: str) -> bytes:
    """A real PNG per tag; locked-sheet lookups reject stub bytes now."""
    return synthetic_still_png(1536, 512, tag=tag, placeholder=True)


def _line(cid: str, zh: str, **extra):
    row = {"character_id": cid, "zh": zh, "en": zh, "ja": zh}
    row.update(extra)
    return row


def _script(lines, *, scene_id="EP001-sc01", extra_scenes=None):
    scenes = [
        {
            "id": scene_id,
            "location_id": "store",
            "interior_id": "int_store",
            "characters": ["ke", "li"],
            "synopsis": "night counter",
            "lines": lines,
        }
    ]
    if extra_scenes:
        scenes.extend(extra_scenes)
    return {
        "synopsis": "A night convenience store.",
        "cast": [
            {"id": "ke", "name": "阿柯", "identity_prompt": "young clerk hoodie"},
            {"id": "li", "name": "阿李", "identity_prompt": "customer raincoat"},
        ],
        "locations": [{"id": "store", "name": "便利店", "plate_prompt": "small night convenience store"}],
        "interiors": [{"id": "int_store", "name": "柜台", "parent": "store", "scene_id": "store"}],
        "scenes": scenes,
    }


def test_short_board_packs_dialogue_h3_count_below_line_count():
    lines = [_line("ke", f"第{i}句今晚别赊账我再说一遍。") for i in range(1, 9)]
    shots = board_short(_script(lines), episode_code="EP001", langs=("zh", "en", "ja"))
    spoken = [s for s in shots if s.get("line")]
    assert len(spoken) < 8
    blob = " ".join(s.get("h3_prompt") or "" for s in spoken)
    assert "First" in blob or CINEFLOW_BRIDGE in blob
    assert "<<<image_1>>>" in blob
    assert "五官漂移" in blob or "face drift" in blob.lower()


def test_same_scene_shares_chain_id_new_scene_hard_cut():
    lines_a = [_line("ke", "又是你今晚别赊账。"), _line("li", "我只是买水别这样。")]
    lines_b = [_line("ke", "雨还在下我回去了。")]
    extra = [
        {
            "id": "EP001-sc02",
            "location_id": "store",
            "interior_id": "int_store",
            "characters": ["ke"],
            "synopsis": "street",
            "lines": lines_b,
        }
    ]
    shots = board_short(
        _script(lines_a, extra_scenes=extra),
        episode_code="EP001",
        langs=("zh", "en", "ja"),
    )
    dialogue = [s for s in shots if s.get("purpose") == "dialogue"]
    sc01 = [s for s in dialogue if s.get("scene_id") == "EP001-sc01"]
    sc02 = [s for s in dialogue if s.get("scene_id") == "EP001-sc02"]
    assert sc01
    assert sc02
    assert len({s["chain_id"] for s in sc01}) == 1
    assert sc01[0]["chain_id"] != sc02[0]["chain_id"]
    segs = expand_shots_to_segments(shots)
    sc01_segs = [s for s in segs if s.get("scene_id") == "EP001-sc01" and s.get("purpose") == "dialogue"]
    if len(sc01_segs) >= 2:
        assert sc01_segs[0]["chain_id"] == sc01_segs[1]["chain_id"]
        assert sc01_segs[0]["chain_index"] == 0
        assert sc01_segs[1]["chain_index"] == 1


def test_expand_accumulates_chain_index_across_same_chain_shots():
    shots = [
        {
            "id": "A",
            "shot_id": "A",
            "scene_id": "sc01",
            "chain_id": "sc01",
            "duration": 8.0,
            "h3_mode": "ref2va",
            "character_id": "ke",
            "refs": ["char_ke_sheet"],
        },
        {
            "id": "B",
            "shot_id": "B",
            "scene_id": "sc01",
            "chain_id": "sc01",
            "duration": 8.0,
            "h3_mode": "ref2va",
            "character_id": "ke",
            "refs": ["char_ke_sheet"],
        },
        {
            "id": "C",
            "shot_id": "C",
            "scene_id": "sc02",
            "chain_id": "sc02",
            "duration": 8.0,
            "h3_mode": "ref2va",
            "character_id": "ke",
            "refs": ["char_ke_sheet"],
        },
    ]
    segs = expand_shots_to_segments(shots)
    assert [s["chain_index"] for s in segs] == [0, 1, 0]
    assert segs[0]["chain_id"] == segs[1]["chain_id"] != segs[2]["chain_id"]


def test_hydrate_refs_everyone_pictured_plus_plate():
    shot = hydrate_shot_identity(
        {
            "id": "s001",
            "character_id": "ke",
            "plate_id": "plate_store",
            "h3_prompt": "two people at the counter",
            "line": {"zh": "又是你。"},
            "cuts": [{"characters": ["ke", "li"]}],
        },
        cast={"ke": {}, "li": {}},
    )
    assert shot["refs"][0] == "char_ke_sheet"
    assert "char_li_sheet" in shot["refs"]
    assert "plate_store" in shot["refs"]
    assert shot["refs"].index("char_ke_sheet") < shot["refs"].index("plate_store")


def test_cite_locked_refs_uses_cineflow_image_tags():
    text = cite_locked_refs("acting in the store", ["char_ke_sheet", "plate_store"])
    assert "<<<image_1>>>" in text
    assert "<<<image_2>>>" in text
    assert ordered_shot_refs(["ke", "li"], "plate_store") == [
        "char_ke_sheet",
        "char_li_sheet",
        "plate_store",
    ]


def test_pack_consecutive_splits_remote_from_store():
    spoken = [
        (0, {}, _line("ke", "又是你。"), 3.0),
        (0, {}, _line("ke", "今晚别赊账。"), 3.0),
        (0, {}, _line("li", "微信语音：在吗？", on_camera=False), 3.0),
    ]
    groups = pack_consecutive_dialogue(spoken)
    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert len(groups[1]) == 1


def test_chain_tail_refuses_submit_without_last_png():
    try:
        native_h3_graph(
            {
                "id": "E01-s02",
                "duration": 5,
                "chain_index": 1,
                "h3_mode": "ref2va",
                "character_id": "ke",
                "refs": ["char_ke_sheet.png"],
            },
            "ref2va",
        )
        raise AssertionError("chain tail must refuse without last.png")
    except RuntimeError as exc:
        assert "last.png" in str(exc)


def test_chain_tail_graph_has_addguide_same_refs_and_image_tags():
    graph = native_h3_graph(
        {
            "id": "E01-s02",
            "duration": 5,
            "chain_index": 1,
            "h3_mode": "ref2va",
            "character_id": "ke",
            "refs": ["char_ke_sheet.png", "plate_store.png"],
            "first_frame_path": "prev_last.png",
        },
        "ref2va",
    )
    assert graph["guide"]["class_type"] == "MiniMaxH3AddGuide"
    assert graph["5"]["inputs"]["image"] == "prev_last.png"
    assert graph["r1"]["inputs"]["image"] == "char_ke_sheet.png"
    assert graph["r2"]["inputs"]["image"] == "plate_store.png"
    prompt = graph["6"]["inputs"]["prompt"]
    assert "<<<image_1>>>" in prompt
    assert "延续上一镜头" in prompt
    linked = apply_chain_first_frame(
        {"id": "E01-s02", "chain_index": 1, "refs": ["char_ke_sheet.png"], "character_id": "ke"},
        Path("prev_last.png"),
    )
    assert linked["first_frame_path"].endswith("prev_last.png")
    assert "prev_last.png" not in (linked.get("refs") or [])


def test_concat_drops_frame_zero_same_chain_hard_cut_cross_scene():
    shots = [
        {"id": "a", "chain_id": "sc01", "chain_index": 0},
        {"id": "b", "chain_id": "sc01", "chain_index": 1},
        {"id": "c", "chain_id": "sc02", "chain_index": 0},
    ]
    assert chain_overlap_frames(shots) == [0, 1, 0]
    cmd = drop_first_frames_cmd(Path("in.mp4"), Path("out.mp4"), 1)
    blob = " ".join(cmd).lower()
    assert "select=gte(n" in blob
    assert "xfade" not in blob
    assert "fade=" not in blob
    from anime_factory.compose import build_compose_plan

    plan = build_compose_plan(Path("."), "EP001", [Path("a.mp4")], require_audio=False)
    encode = " ".join(str(x) for x in plan.encode).lower()
    assert "concat" in encode
    assert "xfade" not in encode
    assert "fade=" not in encode


def test_session_locked_sheet_not_all_siblings(tmp_path):
    from anime_factory.asset_lock import lock_after_qc
    from gpu_worker.session import _character_sheet_files
    from tests.clip_fakes import pass_qc

    cid_dir = tmp_path / "assets" / "characters" / "ke"
    cid_dir.mkdir(parents=True)
    (cid_dir / "sheet_front.png").write_bytes(_art("front"))
    (cid_dir / "sheet_side.png").write_bytes(_art("side"))
    (cid_dir / "sheet_turnaround.png").write_bytes(_art("turn"))
    lock_after_qc(tmp_path, character_id="ke", filename="sheet_front.png", qc=pass_qc())
    files = _character_sheet_files(tmp_path, "char_ke_sheet")
    assert len(files) == 1
    assert files[0].name == "sheet_front.png"


def test_stage_first_frame_copies_locked_sheet_not_every_png(tmp_path, monkeypatch):
    from anime_factory.asset_lock import lock_after_qc
    from gpu_worker import session as sess
    from tests.clip_fakes import pass_qc

    comfy_in = tmp_path / "comfy" / "input"
    monkeypatch.setattr(sess, "_comfy_input_dir", lambda: comfy_in)
    cid_dir = tmp_path / "assets" / "characters" / "ke"
    cid_dir.mkdir(parents=True)
    (cid_dir / "sheet_front.png").write_bytes(_art("front"))
    (cid_dir / "sheet_side.png").write_bytes(_art("side"))
    (cid_dir / "sheet_turnaround.png").write_bytes(_art("turn"))
    (tmp_path / "assets" / "scenes" / "store").mkdir(parents=True)
    (tmp_path / "assets" / "scenes" / "store" / "plate_base.png").write_bytes(_art("plate"))
    lock_after_qc(tmp_path, character_id="ke", filename="sheet_front.png", qc=pass_qc())
    lock_after_qc(tmp_path, scene_id="store", filename="plate_base.png", qc=pass_qc())
    last = tmp_path / "episodes" / "EP001" / "keyframes" / "s001" / "last.png"
    last.parent.mkdir(parents=True)
    last.write_bytes(_art("last"))
    staged = sess._stage_first_frame(
        {
            "id": "s002",
            "chain_index": 1,
            "h3_mode": "ref2va",
            "character_id": "ke",
            "refs": ["char_ke_sheet", "plate_store"],
            "first_frame_path": str(last),
        },
        tmp_path,
    )
    names = list(staged.get("refs") or [])
    assert "char_ke_sheet.png" in names
    assert "plate_store.png" in names
    assert not any("sheet_side" in n for n in names)
    assert not any("turnaround" in n for n in names)
    assert staged.get("first_frame_path") in {"last.png", str(last)}
    copied = {p.name for p in comfy_in.iterdir()} if comfy_in.is_dir() else set()
    assert "char_ke_sheet.png" in copied
    assert "plate_store.png" in copied
    assert "char_ke_sheet" in copied
    assert "plate_store" in copied
    assert "sheet_turnaround.png" not in copied
    assert "ke_sheet_turnaround.png" not in copied
    assert "sheet_side.png" not in copied
    assert "ke_sheet_side.png" not in copied


def test_stage_then_graph_loadimage_names_exist_in_comfy(tmp_path, monkeypatch):
    from anime_factory.asset_lock import lock_after_qc
    from gpu_worker.h3 import native_h3_graph
    from gpu_worker import session as sess
    from tests.clip_fakes import pass_qc

    comfy_in = tmp_path / "comfy" / "input"
    monkeypatch.setattr(sess, "_comfy_input_dir", lambda: comfy_in)
    cid_dir = tmp_path / "assets" / "characters" / "lin-xiao"
    cid_dir.mkdir(parents=True)
    (cid_dir / "sheet_front.png").write_bytes(_art("front"))
    scene = tmp_path / "assets" / "scenes" / "start-lot"
    scene.mkdir(parents=True)
    (scene / "plate_base.png").write_bytes(_art("plate"))
    lock_after_qc(tmp_path, character_id="lin-xiao", filename="sheet_front.png", qc=pass_qc())
    lock_after_qc(tmp_path, scene_id="start-lot", filename="plate_base.png", qc=pass_qc())
    f01 = tmp_path / "episodes" / "EP001" / "keyframes" / "s001" / "f01.png"
    f01.parent.mkdir(parents=True)
    f01.write_bytes(_art("f01"))
    staged = sess._stage_first_frame(
        {
            "id": "s001",
            "chain_index": 0,
            "h3_mode": "ref2va",
            "character_id": "lin-xiao",
            "plate_id": "start-lot",
            "refs": ["char_lin-xiao_sheet", "plate_start-lot"],
            "first_frame_path": str(f01),
        },
        tmp_path,
    )
    graph = native_h3_graph(staged, "ref2va")
    copied = {p.name for p in comfy_in.iterdir()} if comfy_in.is_dir() else set()
    for nid, node in graph.items():
        if not isinstance(node, dict) or node.get("class_type") != "LoadImage":
            continue
        if not str(nid).startswith("r"):
            continue
        assert node["inputs"]["image"] in copied, nid


def test_persist_board_writes_scene_chain_id(tmp_path):
    from anime_factory.board import persist_board
    from anime_factory.db import migrate, open_db

    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    shots = board_short(
        _script([_line("ke", "又是你今晚别赊账。"), _line("li", "我只是买水。")]),
        episode_code="EP001",
        langs=("zh", "en", "ja"),
    )
    persist_board(conn, "EP001", shots, kind="short")
    rows = conn.execute("SELECT chain_id, purpose FROM shots ORDER BY seq").fetchall()
    dialogue = [r["chain_id"] for r in rows if r["purpose"] == "dialogue"]
    assert dialogue
    assert len(set(dialogue)) == 1
    segs = conn.execute("SELECT chain_id, chain_index FROM segments WHERE chain_id = ? ORDER BY seq", (dialogue[0],)).fetchall()
    assert segs
    assert segs[0]["chain_index"] == 0


def test_longlive_board_packs_scene_into_few_long_takes():
    lines = [_line("ke", f"第{i}句今晚别赊账我再说一遍你听清楚。") for i in range(1, 9)]
    live = board_longlive(_script(lines), episode_code="EP001", langs=("zh", "en", "ja"))
    spoken = [s for s in live if s.get("line")]
    assert len(spoken) <= 2
    assert max(float(s["duration"]) for s in spoken) > H3_MAX_SECONDS
    assert max(float(s["duration"]) for s in spoken) <= LONGLIVE_MAX_SECONDS
    blob = " ".join(s.get("h3_prompt") or "" for s in spoken)
    assert "First" in blob and CINEFLOW_BRIDGE in blob
    h3 = board_short(_script(lines), episode_code="EP001", langs=("zh", "en", "ja"))
    h3_spoken = [s for s in h3 if s.get("line")]
    assert len(h3_spoken) >= len(spoken)
    assert max(float(s["duration"]) for s in h3_spoken) <= H3_MAX_SECONDS + 1e-9


def test_board_from_script_does_not_select_longlive():
    lines = [_line("ke", f"第{i}句今晚别赊账我再说一遍。") for i in range(1, 7)]
    h3 = board_from_script(
        _script(lines),
        episode_code="EP001",
        langs=("zh", "en", "ja"),
        kind="short",
        video_backend="h3",
    )
    live = board_from_script(
        _script(lines),
        episode_code="EP001",
        langs=("zh", "en", "ja"),
        kind="short",
        video_backend="longlive",
        shot_seconds=LONGLIVE_MAX_SECONDS,
    )
    h3_spoken = [s for s in h3 if s.get("line")]
    live_spoken = [s for s in live if s.get("line")]
    assert len(live_spoken) == len(h3_spoken)
    assert max(float(s["duration"]) for s in h3_spoken) <= H3_MAX_SECONDS + 1e-9
    assert max(float(s["duration"]) for s in live_spoken) <= H3_MAX_SECONDS + 1e-9


def test_pack_longlive_takes_keeps_one_scene_until_cap():
    spoken = [(0, {}, _line("ke", f"第{i}句今晚别赊账。"), 3.0) for i in range(1, 9)]
    groups = pack_longlive_takes(spoken)
    assert len(groups) == 1
    h3_groups = pack_consecutive_dialogue(spoken, max_s=H3_MAX_SECONDS)
    assert len(h3_groups) > 1

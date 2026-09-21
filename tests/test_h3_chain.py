"""Shot ≠ Segment, H3 chain, ref2va identity, empty last_frame prune."""

from __future__ import annotations

from pathlib import Path

from anime_factory.board import expand_shots_to_segments, persist_board, split_duration
from anime_factory.db import migrate, open_db
from anime_factory.design import synthetic_still_png
from anime_factory.keyframe import assert_keyframe_files, ensure_keyframe
from gpu_worker.h3 import apply_chain_first_frame, native_h3_graph, prepare_workflow, prune_empty_last_frame, select_mode


def test_split_10s_shot_is_two_even_segments():
    assert split_duration(10.0) == [5.0, 5.0]
    shots = [
        {
            "id": "EP001-01",
            "shot_id": "EP001-01",
            "scene_id": "EP001-sc01",
            "duration": 10.0,
            "h3_mode": "ref2va",
            "character_id": "ke",
            "refs": ["char_ke_sheet"],
            "cuts": [{"seq": 1, "seconds": 10.0, "size": "MCU", "camera": "Static Shot", "characters": ["ke"]}],
        }
    ]
    segs = expand_shots_to_segments(shots)
    assert len(segs) == 2
    assert segs[0]["duration"] == 5.0
    assert segs[1]["duration"] == 5.0
    assert segs[0]["shot_id"] == segs[1]["shot_id"] == "EP001-01"
    assert segs[0]["chain_id"] == segs[1]["chain_id"]
    assert segs[0]["chain_index"] == 0
    assert segs[1]["chain_index"] == 1
    assert segs[0]["h3_mode"] == "ref2va"
    assert segs[1]["h3_mode"] == "ref2va"
    assert segs[1]["refs"] == ["char_ke_sheet"]


def test_second_first_frame_equals_extracted_last_frame(tmp_path):
    last = tmp_path / "last.png"
    last.write_bytes(b"\x89PNG\r\n\x1a\n" + b"last-frame-bytes")
    segs = expand_shots_to_segments(
        [{"id": "S01", "duration": 10.0, "scene_id": "sc01", "h3_mode": "ref2va", "character_id": "ke"}]
    )
    linked = apply_chain_first_frame(segs[1], last)
    assert Path(linked["first_frame_path"]).read_bytes() == last.read_bytes()
    assert linked["first_frame_path"] == str(last)
    assert linked["h3_mode"] == "ref2va"
    assert linked.get("refs") == segs[0].get("refs") or linked.get("character_id") == "ke"


def test_character_shots_are_not_pure_i2v():
    seg = {
        "id": "E01-01",
        "duration": 8,
        "first_frame_path": "E01-01_f1.png",
        "h3_mode": "ref2va",
        "character_id": "ke",
        "refs": ["char_ke_sheet.png"],
    }
    assert select_mode(seg) != "fl2va_first"
    assert select_mode(seg) == "ref2va"
    types = {n["class_type"] for n in prepare_workflow(seg)["prompt"].values()}
    assert "MiniMaxH3ReferenceToVideo" in types
    assert "MiniMaxH3ImageToVideo" not in types


def test_empty_last_frame_still_pruned_from_workflow():
    wf = {
        "prompt": {
            "1": {"class_type": "LoadImage", "_meta": {"role": "first_frame"}, "inputs": {"image": "f1.png"}},
            "2": {"class_type": "LoadImage", "_meta": {"role": "last_frame"}, "inputs": {"image": "input/"}},
            "3": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"last_frame": ["2", 0], "first_frame": ["1", 0]}},
        }
    }
    pruned = prune_empty_last_frame(wf)
    prompt = pruned["prompt"]
    assert "2" not in prompt
    assert "last_frame" not in prompt["3"]["inputs"]
    native = prepare_workflow(
        {"id": "E01-01", "duration": 5, "first_frame_path": "f1.png", "h3_mode": "fl2va_first_last", "last_frame_path": "input/"}
    )
    for node in native["prompt"].values():
        if isinstance(node, dict) and node.get("class_type") == "LoadImage":
            role = (node.get("_meta") or {}).get("role")
            if role == "last_frame":
                raise AssertionError("empty last_frame LoadImage must be pruned")


def test_persist_board_splits_10s_shot(tmp_path):
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    persist_board(
        conn,
        "EP001",
        [
            {
                "id": "EP001-01",
                "scene_id": "EP001-sc01",
                "duration": 10.0,
                "h3_mode": "ref2va",
                "character_id": "ke",
                "refs": ["char_ke_sheet"],
                "cuts": [{"seq": 1, "seconds": 10.0, "size": "MCU", "camera": "Static Shot", "characters": ["ke"]}],
            }
        ],
        kind="short",
    )
    segs = conn.execute("SELECT id, shot_id, chain_id, chain_index, duration, h3_mode, status FROM segments ORDER BY seq").fetchall()
    assert len(segs) == 2
    assert segs[0]["shot_id"] == segs[1]["shot_id"]
    assert segs[0]["chain_id"] == segs[1]["chain_id"]
    assert [row["chain_index"] for row in segs] == [0, 1]
    assert segs[0]["h3_mode"] == "ref2va"
    assert segs[1]["h3_mode"] == "ref2va"
    assert segs[1]["status"] == "pending_chain"
    shots = conn.execute("SELECT id, duration FROM shots").fetchall()
    assert len(shots) == 1
    assert shots[0]["duration"] == 10.0


def test_persist_board_refuses_collapsed_store_board(tmp_path):
    from anime_factory.board import PromptCollapseError

    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    body = "@hero 站在 @night convenience store 中, medium close-up, speaking, same store, locked eyeline"
    items = []
    for i in range(15):
        items.append(
            {
                "id": f"s{i+1:03d}",
                "shot_id": f"s{i+1:03d}",
                "scene_id": "EP001-sc01",
                "duration": 3.0,
                "h3_prompt": f"{body}, camera Static Shot",
                "first_frame_prompt": f"{body}, shot {i+1} of 15",
                "line": {"zh": f"第{i+1}句。"},
                "character_id": "hero",
            }
        )
    try:
        persist_board(conn, "EP001", items, kind="short")
        raise AssertionError("collapsed store board must not persist")
    except PromptCollapseError:
        pass
    n = conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"]
    assert int(n) == 0


def test_chain_tail_skips_flux_keyframe(tmp_path):
    payloads = []

    def gpu_gen(payload):
        payloads.append(payload)
        return synthetic_still_png(1440, 800, tag=str(payload.get("seed")))

    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    from anime_factory.design import KolorsClient

    client = KolorsClient(["k"], live=True, gpu_generate=gpu_gen)
    key = ensure_keyframe(
        conn,
        "story-x",
        "EP001",
        {
            "id": "s001-s02",
            "chain_index": 1,
            "first_frame_path": str(tmp_path / "prev_last.png"),
        },
        {},
        {"items": []},
        {},
        client,
        None,
        "fiction",
        tmp_path,
    )
    assert payloads == []
    assert key.endswith("prev_last.png")


PNG_STUB = synthetic_still_png(832, 1216, tag="h3-chain-sheet", placeholder=True)


def test_ref2va_graph_loads_sheets_not_dummy_f1():
    graph = native_h3_graph(
        {
            "id": "E01-01",
            "duration": 8,
            "h3_mode": "ref2va",
            "character_id": "ke",
            "refs": ["char_ke_sheet.png", "plate_store.png"],
        },
        "ref2va",
    )
    loaders = [
        node
        for node in graph.values()
        if isinstance(node, dict) and node.get("class_type") == "LoadImage"
    ]
    images = [node["inputs"]["image"] for node in loaders]
    assert "f1.png" not in images
    assert "5" not in graph
    assert graph["r1"]["inputs"]["image"] == "char_ke_sheet.png"
    assert graph["r2"]["inputs"]["image"] == "plate_store.png"
    cond = graph["6"]["inputs"]
    assert "ref_image_1" not in cond
    assert "ref_images" not in cond
    assert cond["ref_images.ref_image_0"] == ["r1", 0]
    assert cond["ref_images.ref_image_1"] == ["r2", 0]
    assert "audio_vae" not in cond
    roles = [(node.get("_meta") or {}).get("role") for node in loaders]
    assert "first_frame" not in roles
    assert "ref_images.ref_image_0" in roles
    types = {node["class_type"] for node in graph.values() if isinstance(node, dict)}
    assert "VAEDecodeAudio" not in types
    assert "audio" not in graph["14"]["inputs"]


def test_ref2va_head_mints_composition_f1_and_keeps_sheet(tmp_path, monkeypatch):
    """H3 v2: ref2va chain heads mint composition f1 (+ cut anchors) while still requiring sheets."""
    from anime_factory.asset_lock import lock_after_qc
    from anime_factory.visual_qc import VisualQcResult
    from tests.clip_fakes import pass_qc

    sheet = tmp_path / "assets" / "characters" / "ke" / "sheet_front.png"
    sheet.parent.mkdir(parents=True)
    sheet.write_bytes(PNG_STUB)
    lock_after_qc(tmp_path, character_id="ke", filename="sheet_front.png", qc=pass_qc())
    payloads = []

    def gpu_gen(payload):
        payloads.append(payload)
        from anime_factory.models import STILL_HEIGHT, STILL_WIDTH

        return synthetic_still_png(STILL_WIDTH, STILL_HEIGHT, tag=str(payload.get("seed")))

    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    from anime_factory.design import KolorsClient
    from anime_factory import keyframe as kf

    monkeypatch.setattr(
        kf,
        "score_still",
        lambda *_a, **_k: VisualQcResult(verdict="pass", reasons=[], scores={}, kind="keyframe"),
    )
    client = KolorsClient(["k"], live=True, gpu_generate=gpu_gen)
    key = ensure_keyframe(
        conn,
        "story-x",
        "EP001",
        {
            "id": "s001",
            "chain_index": 0,
            "h3_mode": "ref2va",
            "character_id": "ke",
            "refs": ["char_ke_sheet"],
            "first_frame_prompt": "1girl, young adult, black hair, brown eyes, grey hoodie, medium shot in store",
            "cuts": [
                {
                    "seq": 1,
                    "seconds": 8,
                    "size": "MS",
                    "camera": "Static Shot",
                    "characters": ["ke"],
                    "frame_prompt": "1girl grey hoodie standing at store counter",
                }
            ],
        },
        {},
        {"items": [{"id": "char_ke_sheet", "path": "assets/characters/ke/sheet_front.png"}]},
        {},
        client,
        None,
        "fiction",
        tmp_path,
    )
    assert payloads, "ref2va head must mint composition stills in H3 v2"
    assert key.endswith("f1.png")
    assert (tmp_path / "episodes" / "EP001" / "keyframes" / "s001" / "f1.png").exists()
    assert (tmp_path / "episodes" / "EP001" / "keyframes" / "s001" / "f01.png").exists()
    assert sheet.is_file()


def test_keyframe_stage_fails_when_fl2va_files_missing(tmp_path):
    segs = [{"id": "s001", "chain_index": 0, "h3_mode": "fl2va_first"}]
    try:
        assert_keyframe_files(tmp_path, "EP001", segs)
    except RuntimeError as exc:
        assert "s001" in str(exc)
        assert "f1.png" in str(exc) or "missing" in str(exc)
    else:
        raise AssertionError("keyframe stage must fail with 0 files on disk")


def test_generate_missing_stills_fails_closed_without_f1(tmp_path, monkeypatch):
    import json

    from anime_factory.design import KolorsClient
    from gpu_worker import session as sess

    monkeypatch.setenv("ANIME_FACTORY_GPU_STILLS", "1")
    ep = tmp_path / "episodes" / "EP001"
    ep.mkdir(parents=True)
    (ep / "board.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "EP001-01",
                        "chain_index": 0,
                        "h3_mode": "fl2va_first",
                        "duration": 5.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    monkeypatch.setattr(
        "anime_factory.design.library_from_db",
        lambda *a, **k: {"created": [], "specs": {}},
    )
    monkeypatch.setattr(sess, "EP", "EP001")
    monkeypatch.setattr(
        sess,
        "KolorsClient",
        lambda *a, **k: KolorsClient(["k"], live=True, gpu_generate=lambda p: (_ for _ in ()).throw(RuntimeError("flux down"))),
    )
    try:
        sess.generate_missing_stills("story-x", tmp_path, conn)
    except RuntimeError as exc:
        assert "keyframe" in str(exc).lower() or "f1" in str(exc).lower() or "flux" in str(exc).lower()
    else:
        raise AssertionError("keyframe stage must not succeed with 0 files")
    assert not (ep / "keyframes" / "EP001-01" / "f1.png").exists()


def test_generate_missing_stills_still_mints_cuts_when_f1_exists(tmp_path, monkeypatch):
    """f1.png on disk must not skip ensure_keyframe; per-cut f01 is still required."""
    import json

    from gpu_worker import session as sess

    ep = tmp_path / "episodes" / "EP001"
    kf_dir = ep / "keyframes" / "s001"
    kf_dir.mkdir(parents=True)
    (kf_dir / "f1.png").write_bytes(PNG_STUB)
    (ep / "board.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "s001",
                        "chain_index": 0,
                        "h3_mode": "fl2va_first",
                        "duration": 5.0,
                        "first_frame_prompt": "wide establishing store interior",
                        "cuts": [{"seq": 1, "seconds": 5.0, "frame_prompt": "counter wide"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    called: list[str] = []
    monkeypatch.setattr(sess, "EP", "EP001")
    monkeypatch.setattr(sess, "_pull_studio_asset_index", lambda *_a, **_k: False)
    monkeypatch.setattr(sess, "upload_tree", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "anime_factory.design.library_from_db",
        lambda *_a, **_k: {"created": [], "specs": {}},
    )
    monkeypatch.setattr(sess, "KolorsClient", lambda *_a, **_k: type("C", (), {"live": True})())
    monkeypatch.setattr(
        sess,
        "ensure_keyframe",
        lambda *_a, **_k: called.append(str(_a[3].get("id"))) or "ok",
    )
    monkeypatch.setattr(sess, "assert_keyframe_files", lambda *_a, **_k: None)
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    sess.generate_missing_stills("story-x", tmp_path, conn)
    assert called == ["s001"]


def test_generate_missing_stills_uploads_before_needs_human(tmp_path, monkeypatch):
    from gpu_worker import session as sess

    uploaded: list[str] = []
    pulled: list[str] = []

    monkeypatch.setattr(
        sess,
        "_pull_studio_asset_index",
        lambda story_id, _root: pulled.append(story_id) or False,
    )
    monkeypatch.setattr(
        "anime_factory.design.library_from_db",
        lambda *_a, **_k: {"needs_human": ["char_a-kai_turnaround"], "created": [], "specs": {}},
    )
    monkeypatch.setattr(
        sess,
        "upload_tree",
        lambda story_id, root, rel_dir="assets": uploaded.append(rel_dir) or [{"ok": True, "key": f"{rel_dir}/x.png"}],
    )
    monkeypatch.setattr(
        sess,
        "KolorsClient",
        lambda *_a, **_k: type("C", (), {"live": True})(),
    )
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    try:
        sess.generate_missing_stills("story-x", tmp_path, conn)
    except RuntimeError as exc:
        assert "char_a-kai_turnaround" in str(exc)
        assert "needs_human" in str(exc)
    else:
        raise AssertionError("needs_human must still refuse H3")
    assert pulled == ["story-x"]
    assert uploaded == ["assets", "episodes", "canon"]


def test_pull_studio_asset_index_force_replaces_local_copy(tmp_path, monkeypatch):
    from gpu_worker import session as sess

    local = tmp_path / "assets" / "index.json"
    local.parent.mkdir(parents=True)
    local.write_text('{"source":"worker-cache"}', encoding="utf-8")
    requested: list[str] = []

    def fake_download(key, dest):
        requested.append(key)
        Path(dest).write_text('{"source":"studio"}', encoding="utf-8")
        return True

    monkeypatch.setattr(sess, "download_file", fake_download)
    assert sess._pull_studio_asset_index("story-x", tmp_path) is True
    assert requested == ["stories/story-x/assets/index.json"]
    assert local.read_text(encoding="utf-8") == '{"source":"studio"}'


def test_needs_human_error_surfaces_upload_failures(tmp_path, monkeypatch):
    from gpu_worker import session as sess

    monkeypatch.setattr(sess, "_pull_studio_asset_index", lambda *_a, **_k: False)
    monkeypatch.setattr(
        "anime_factory.design.library_from_db",
        lambda *_a, **_k: {"needs_human": ["char_a-kai_side"], "created": [], "specs": {}},
    )
    monkeypatch.setattr(
        sess,
        "upload_tree",
        lambda _story_id, _root, rel_dir="assets": (
            [{"ok": False, "key": "stories/story-x/assets/index.json", "error": "timeout"}]
            if rel_dir == "assets"
            else []
        ),
    )
    monkeypatch.setattr(
        sess,
        "KolorsClient",
        lambda *_a, **_k: type("C", (), {"live": True})(),
    )
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    try:
        sess.generate_missing_stills("story-x", tmp_path, conn)
    except RuntimeError as exc:
        message = str(exc)
        assert "needs_human" in message
        assert "char_a-kai_side" in message
        assert "r2_upload_failed" in message
        assert "assets/index.json" in message
        assert "timeout" in message
    else:
        raise AssertionError("needs_human must refuse H3 even when upload fails")


def test_chain_segment_graph_has_last_frame_guide_and_same_refs():
    graph = native_h3_graph(
        {
            "id": "E01-01-s02",
            "duration": 5,
            "chain_index": 1,
            "h3_mode": "ref2va",
            "character_id": "ke",
            "refs": ["char_ke_sheet.png", "plate_store.png"],
            "first_frame_path": "prev_last.png",
        },
        "ref2va",
    )
    assert graph["6"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert graph["r1"]["inputs"]["image"] == "char_ke_sheet.png"
    assert graph["r2"]["inputs"]["image"] == "plate_store.png"
    assert graph["5"]["inputs"]["image"] == "prev_last.png"
    assert graph["guide"]["class_type"] == "MiniMaxH3AddGuide"
    assert "f1.png" not in [n["inputs"]["image"] for n in graph.values() if n.get("class_type") == "LoadImage"]
    assert "VAEDecodeAudio" not in {n["class_type"] for n in graph.values()}
    prompt = graph["6"]["inputs"]["prompt"]
    assert "延续上一镜头" in prompt


def test_hosted_s001_hydrates_hero_sheet_not_empty_plate():
    from anime_factory.directors.common import choose_h3_mode, hydrate_shot_identity

    shot = hydrate_shot_identity(
        {
            "id": "s001",
            "character_id": "hero",
            "h3_prompt": "hero 站在 @loc_office 中，穿着灰色卫衣",
            "line": {"zh": "今天顺路，给你带了早餐。"},
        },
        cast={"hero": {}, "girl": {}},
    )
    assert shot["h3_mode"] == "ref2va"
    assert "char_hero_sheet" in shot["refs"]
    assert choose_h3_mode({"id": "wide", "purpose": "establish", "plate_id": "plate_office"}) == "fl2va_first"

from pathlib import Path
import json

from anime_factory.asset_lock import (
    FRONT_ALIAS_FILENAME,
    TURNAROUND_FILENAME,
    auto_lock_first_success,
    has_locked_identity,
    load_assets_index,
    lock_after_qc,
    locked_character_file,
    locked_character_relpath,
    locked_scene_file,
    next_history_filename,
    pop_regen_requests,
    queue_regen_request,
    save_assets_index,
    set_selected,
)
from anime_factory.continuity_gates import gate3_assets
from anime_factory.db import migrate, open_db
from anime_factory.design import (
    CHAR_BACK_FILENAME,
    CHAR_IMAGE_SIZE,
    CHAR_SIDE_FILENAME,
    CHAR_VIEW_HEIGHT,
    CHAR_VIEW_WIDTH,
    KolorsClient,
    compose_character_turnaround,
    generate_asset_library,
    harbor_mvp_cast,
    plan_derive_specs,
    plan_library_specs,
    png_dimensions,
    synthetic_still_png,
)
from anime_factory.r2_paths import character_asset_rel, scene_asset_rel
from tests.clip_fakes import pass_qc, passing_scorer, sized_placeholder_opener


def _art(marker: bytes = b"PNG") -> bytes:
    """A real PNG, distinct per marker. 3-byte stubs are rejected now (MIN_STILL_BYTES)."""
    return synthetic_still_png(1440, 800, tag=marker.decode("utf-8", "replace"), placeholder=True)


def _png(path: Path, marker: bytes = b"PNG") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_art(marker))


def test_frozen_index_contract_and_fallbacks(tmp_path):
    root = tmp_path / "story"
    cid = "hero"
    folder = root / "assets" / "characters" / cid
    _png(folder / TURNAROUND_FILENAME, b"TURN")
    _png(folder / FRONT_ALIAS_FILENAME, b"FRONT")
    lock_after_qc(root, character_id=cid, filename=FRONT_ALIAS_FILENAME, qc=pass_qc())
    index = load_assets_index(root)
    rec = index["characters"][cid]
    assert rec["selected"] == FRONT_ALIAS_FILENAME
    assert FRONT_ALIAS_FILENAME in rec["history"]
    assert rec["parent_id"] is None
    assert locked_character_relpath(cid) == "char_hero_sheet"
    assert locked_character_file(root, cid).name == FRONT_ALIAS_FILENAME
    assert has_locked_identity(root, cid)

    auto_lock_first_success(root, character_id=cid, filename="sheet_turnaround_2.png")
    rec = load_assets_index(root)["characters"][cid]
    assert rec["selected"] == FRONT_ALIAS_FILENAME
    assert "sheet_turnaround_2.png" in rec["history"]

    set_selected(root, character_id=cid, filename="sheet_turnaround_2.png")
    _png(folder / "sheet_turnaround_2.png", b"NEW")
    set_selected(root, character_id=cid, filename="sheet_turnaround_2.png")
    assert locked_character_file(root, cid).name == "sheet_turnaround_2.png"
    assert (folder / FRONT_ALIAS_FILENAME).read_bytes() == _art(b"NEW")


def test_old_story_front_fallback(tmp_path):
    root = tmp_path / "story"
    cid = "ke"
    _png(root / "assets" / "characters" / cid / FRONT_ALIAS_FILENAME, b"OLD")
    assert not has_locked_identity(root, cid)
    assert locked_character_file(root, cid).name == FRONT_ALIAS_FILENAME


def test_locked_scene_file(tmp_path):
    root = tmp_path / "story"
    _png(root / "assets" / "scenes" / "store" / "plate_base.png", b"PLATE")
    lock_after_qc(root, scene_id="store", filename="plate_base.png", qc=pass_qc())
    rec = load_assets_index(root)["scenes"]["store"]
    assert rec["selected"] == "plate_base.png"
    assert locked_scene_file(root, "store").read_bytes() == _art(b"PLATE")


def test_legacy_items_index_migrates(tmp_path):
    root = tmp_path / "story"
    save_assets_index(
        root,
        {
            "items": [
                {
                    "id": "char_hero_sheet",
                    "kind": "character_sheet",
                    "character_id": "hero",
                    "path": "assets/characters/hero/sheet_front.png",
                }
            ]
        },
    )
    rec = load_assets_index(root)["characters"]["hero"]
    assert rec["selected"] == "sheet_front.png"
    assert "sheet_front.png" in rec["history"]


def test_next_history_filename():
    assert next_history_filename([], "sheet_turnaround") == "sheet_turnaround.png"
    assert next_history_filename(["sheet_turnaround.png"], "sheet_turnaround") == "sheet_turnaround_2.png"
    assert next_history_filename(["sheet_turnaround.png", "sheet_turnaround_2.png"], "sheet_turnaround") == "sheet_turnaround_3.png"


def test_plan_library_generates_front_then_reference_views_and_composite():
    specs = plan_library_specs(
        [
            {
                "id": "hero",
                "name": "守潮人",
                "identity_prompt": "1boy, adult, short black hair, brown eyes, dark coat, lean build",
            }
        ],
        [{"id": "harbor", "name": "Harbor", "plate_prompt": "wooden harbor pier, anime location background"}],
        [],
    )
    front = specs["char_hero_sheet"]
    side = specs["char_hero_side"]
    back = specs["char_hero_back"]
    turnaround = specs["char_hero_turnaround"]
    assert front["path"] == character_asset_rel("hero", FRONT_ALIAS_FILENAME)
    assert front["view"] == "front"
    assert front["image_size"] == CHAR_IMAGE_SIZE == "832x1216"
    assert side["path"] == character_asset_rel("hero", CHAR_SIDE_FILENAME)
    assert back["path"] == character_asset_rel("hero", CHAR_BACK_FILENAME)
    assert side["parent_id"] == back["parent_id"] == "hero"
    assert side["kind"] == back["kind"] == "character_view_derive"
    assert len(turnaround["source_paths"]) == 3


def test_bind_or_generate_skips_locked_identity(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    client = KolorsClient(["k"], opener=sized_placeholder_opener("one"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    first = generate_asset_library(
        conn, "story-x", chars, locs, props, client, None, "fiction", tmp_path, interiors=interiors, skip_existing=False, clip_scorer=passing_scorer()
    )
    assert any(aid.startswith("char_hero") for aid in first["created"])
    turnaround = tmp_path / "assets/characters/hero/sheet_turnaround.png"
    front = tmp_path / "assets/characters/hero/sheet_front.png"
    side = tmp_path / "assets/characters/hero/sheet_side.png"
    back = tmp_path / "assets/characters/hero/sheet_back.png"
    assert turnaround.is_file() and front.is_file() and side.is_file() and back.is_file()
    assert png_dimensions(front.read_bytes()) == (CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT)
    assert png_dimensions(turnaround.read_bytes()) == (CHAR_VIEW_WIDTH * 3, CHAR_VIEW_HEIGHT)
    assert front.read_bytes() != turnaround.read_bytes()
    derive_payloads = [p for p in client.last_payloads if p.get("_kind") == "character_view_derive"]
    assert len(derive_payloads) == 2
    assert all(p.get("_parent_png") == front.read_bytes() for p in derive_payloads)
    first_blob = turnaround.read_bytes()
    index = json.loads((tmp_path / "assets/index.json").read_text(encoding="utf-8"))
    assert index["characters"]["hero"]["selected"] == FRONT_ALIAS_FILENAME
    assert TURNAROUND_FILENAME in index["characters"]["hero"]["history"]
    assert locked_character_file(tmp_path, "hero").name == FRONT_ALIAS_FILENAME
    items = {item["id"]: item for item in index["items"]}
    assert items["char_hero_turnaround"]["path"].endswith(TURNAROUND_FILENAME)
    assert "items" in index
    client2 = KolorsClient(["k"], opener=sized_placeholder_opener("two"), live=False)
    second = generate_asset_library(
        conn, "story-x", chars, locs, props, client2, None, "fiction", tmp_path, interiors=interiors, skip_existing=False, clip_scorer=passing_scorer()
    )
    assert "char_hero_sheet" not in second["created"]
    assert turnaround.read_bytes() == first_blob


def test_locked_identity_and_h3_session_bind_front_not_turnaround(tmp_path):
    from gpu_worker.session import _character_sheet_files

    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    client = KolorsClient(["k"], opener=sized_placeholder_opener("h3"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    generate_asset_library(
        conn, "story-x", chars, locs, props, client, None, "fiction", tmp_path, interiors=interiors, skip_existing=False, clip_scorer=passing_scorer()
    )
    files = _character_sheet_files(tmp_path, "char_hero_sheet")
    assert len(files) == 1
    assert files[0].name == FRONT_ALIAS_FILENAME


def test_compose_character_turnaround_is_deterministic_three_panel_strip(tmp_path):
    root = tmp_path / "story"
    hero = root / "assets" / "characters" / "hero"
    hero.mkdir(parents=True)
    front = synthetic_still_png(CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT, tag="front", placeholder=True)
    side = synthetic_still_png(CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT, tag="side", placeholder=True)
    back = synthetic_still_png(CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT, tag="back", placeholder=True)
    (hero / FRONT_ALIAS_FILENAME).write_bytes(front)
    (hero / CHAR_SIDE_FILENAME).write_bytes(side)
    (hero / CHAR_BACK_FILENAME).write_bytes(back)
    spec = {
        "id": "char_hero_turnaround",
        "source_paths": [
            character_asset_rel("hero", FRONT_ALIAS_FILENAME),
            character_asset_rel("hero", CHAR_SIDE_FILENAME),
            character_asset_rel("hero", CHAR_BACK_FILENAME),
        ],
    }
    composed = compose_character_turnaround(root, spec)
    assert png_dimensions(composed) == (CHAR_VIEW_WIDTH * 3, CHAR_VIEW_HEIGHT)
    assert composed != front
    assert composed.count(b"IHDR") == 1


def test_regen_appends_history_without_stealing_lock(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    client = KolorsClient(["k"], opener=sized_placeholder_opener("regen1"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    generate_asset_library(
        conn, "story-x", chars, locs, props, client, None, "fiction", tmp_path, interiors=interiors, skip_existing=False, clip_scorer=passing_scorer()
    )
    locked_turnaround = (tmp_path / "assets/characters/hero" / TURNAROUND_FILENAME).read_bytes()
    queue_regen_request(tmp_path, "character", "hero")
    client2 = KolorsClient(
        ["k"],
        opener=sized_placeholder_opener("regen2"),
        live=False,
    )
    out = generate_asset_library(
        conn, "story-x", chars, locs, props, client2, None, "fiction", tmp_path, interiors=interiors, skip_existing=True, clip_scorer=passing_scorer()
    )
    rec = load_assets_index(tmp_path)["characters"]["hero"]
    assert rec["selected"] == FRONT_ALIAS_FILENAME
    assert "sheet_front_2.png" in rec["history"]
    assert (tmp_path / "assets/characters/hero/sheet_front_2.png").is_file()
    assert (tmp_path / "assets/characters/hero" / TURNAROUND_FILENAME).read_bytes() == locked_turnaround
    assert "char_hero_sheet" in out["created"]
    assert pop_regen_requests(tmp_path) == []


def test_derive_costume_uses_parent_reference(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    ep = tmp_path / "episodes" / "EP001"
    ep.mkdir(parents=True)
    (ep / "script.json").write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "id": "sc1",
                        "location_id": "harbor",
                        "time_of_day": "night",
                        "lines": [{"character_id": "hero", "zh": "换上战斗服。"}],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    chars, locs, props, interiors = harbor_mvp_cast()
    derive = plan_derive_specs(chars, locs, tmp_path)
    assert any(s.get("costume_id") == "battle" for s in derive.values())
    assert any(s.get("scene_id") == "harbor_night" for s in derive.values())
    client = KolorsClient(["k"], opener=sized_placeholder_opener("derive"), live=False)
    generate_asset_library(
        conn, "story-x", chars, locs, props, client, None, "fiction", tmp_path, interiors=interiors, skip_existing=False, clip_scorer=passing_scorer()
    )
    parent_payloads = [p for p in client.last_payloads if p.get("reference_images") or p.get("image")]
    assert parent_payloads
    assert all("image" not in p or p.get("_kind") != "scene_plate" for p in client.last_payloads if p.get("_kind") == "scene_plate")
    rec = load_assets_index(tmp_path)
    assert rec["characters"]["hero"]["selected"]
    assert "hero_battle" in rec["characters"]
    assert rec["characters"]["hero_battle"]["parent_id"] == "hero"
    assert "harbor_night" in rec["scenes"]


def test_expressions_are_not_derived(tmp_path):
    ep = tmp_path / "episodes" / "EP001"
    ep.mkdir(parents=True)
    (ep / "script.json").write_text(
        json.dumps({"scenes": [{"lines": [{"zh": "她惊恐面部，眼眶泛红。"}]}]}),
        encoding="utf-8",
    )
    specs = plan_derive_specs([{"id": "hero", "name": "阿哲"}], [{"id": "store", "name": "店"}], tmp_path)
    assert specs == {}


def test_wet_location_wording_does_not_create_character_costumes(tmp_path):
    ep = tmp_path / "episodes" / "EP001"
    ep.mkdir(parents=True)
    (ep / "script.json").write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "id": "sc1",
                        "location_id": "harbor",
                        "lines": [
                            {
                                "character_id": "hero",
                                "prompt": "hero waits on a wet reflective platform after rain",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    chars, locs, _props, _interiors = harbor_mvp_cast()
    specs = plan_derive_specs(chars, locs, tmp_path)
    assert not any(spec.get("costume_id") == "wet" for spec in specs.values())


def test_gate3_fail_closed_without_locked_sheet(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    seg = {
        "id": "s001",
        "character_id": "hero",
        "refs": ["char_hero_sheet"],
        "cuts": [{"characters": ["hero"]}],
    }
    violations, missing = gate3_assets(conn, seg, {}, {"items": [{"id": "char_hero_sheet", "kind": "character_sheet"}]})
    assert "char_hero_sheet" not in missing
    assert any(v.code == "unlocked_identity" for v in violations)

    _png(tmp_path / "assets" / "characters" / "hero" / FRONT_ALIAS_FILENAME)
    lock_after_qc(tmp_path, character_id="hero", filename=FRONT_ALIAS_FILENAME, qc=pass_qc())
    violations, missing = gate3_assets(
        conn, seg, {}, {"items": [{"id": "char_hero_sheet", "kind": "character_sheet"}]}, story_root=tmp_path
    )
    assert not missing
    assert not any(v.code == "unlocked_identity" for v in violations)


def test_r2_path_helpers():
    assert character_asset_rel("hero") == "assets/characters/hero/sheet_turnaround.png"
    assert scene_asset_rel("dock") == "assets/scenes/dock/plate_base.png"

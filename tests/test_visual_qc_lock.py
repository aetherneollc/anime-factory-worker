from pathlib import Path
import json

from anime_factory.asset_lock import (
    FRONT_ALIAS_FILENAME,
    TURNAROUND_FILENAME,
    has_locked_identity,
    is_qc_locked,
    load_assets_index,
    lock_after_qc,
    mark_stale_visual_v1,
    record_qc_candidate,
)
from anime_factory.continuity_gates import gate3_assets
from anime_factory.db import migrate, open_db
from anime_factory.design import (
    CHAR_VIEW_HEIGHT,
    CHAR_VIEW_WIDTH,
    KolorsClient,
    compose_character_turnaround,
    derive_missing_assets,
    generate_asset_library,
    harbor_mvp_cast,
    synthetic_still_png,
)
from anime_factory.visual_qc import QC_VERSION, current_lineage
from tests.clip_fakes import dish_scorer, fail_qc, pass_qc, passing_scorer, sized_placeholder_opener, view_fail_scorer


def _front(path: Path, tag: str = "front") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(synthetic_still_png(CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT, tag=tag, placeholder=True))


def test_lock_after_qc_requires_pass(tmp_path):
    root = tmp_path / "story"
    _front(root / "assets" / "characters" / "hero" / FRONT_ALIAS_FILENAME)
    try:
        lock_after_qc(
            root,
            character_id="hero",
            filename=FRONT_ALIAS_FILENAME,
            qc=fail_qc("scene_classified_as_dinnerware"),
        )
        raise AssertionError("fail verdict must not lock")
    except ValueError:
        pass
    rec = load_assets_index(root)["characters"].get("hero") or {}
    assert rec.get("selected") not in {FRONT_ALIAS_FILENAME}
    assert not is_qc_locked(root, character_id="hero")
    lock_after_qc(root, character_id="hero", filename=FRONT_ALIAS_FILENAME, qc=pass_qc())
    rec = load_assets_index(root)["characters"]["hero"]
    assert rec["selected"] == FRONT_ALIAS_FILENAME
    assert rec["qc_verdict"] == "pass"
    assert rec["qc_version"] == QC_VERSION
    assert rec["model_version"] == current_lineage()["model_version"]
    assert rec["prompt_version"] == current_lineage()["prompt_version"]
    assert rec["workflow_version"] == current_lineage()["workflow_version"]
    assert has_locked_identity(root, "hero")


def test_three_deterministic_seeds_then_needs_human(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    client = KolorsClient(["k"], opener=sized_placeholder_opener("fail"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    out = generate_asset_library(
        conn,
        "story-x",
        chars,
        locs,
        props,
        client,
        None,
        "fiction",
        tmp_path,
        interiors=interiors,
        skip_existing=False,
        clip_scorer=dish_scorer(),
    )
    rec = load_assets_index(tmp_path)["scenes"]["harbor"]
    assert rec.get("selected") is None
    assert rec.get("qc_verdict") == "needs_human"
    assert int(rec.get("qc_attempts") or 0) >= 3
    assert len(rec.get("candidates") or []) >= 3
    assert "harbor" in str(out.get("needs_human") or rec.get("qc_verdict"))
    hero = load_assets_index(tmp_path)["characters"]["hero"]
    assert hero.get("selected") is None
    assert not has_locked_identity(tmp_path, "hero")
    assert "char_hero_side" in out["needs_human"]
    assert "char_hero_back" in out["needs_human"]
    assert "char_hero_turnaround" in out["needs_human"]
    assert not (tmp_path / "assets/characters/hero/sheet_side.png").is_file() or hero.get("qc_verdict") != "pass"


def test_skip_existing_does_not_lock_without_qc(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    _front(tmp_path / "assets" / "characters" / "hero" / FRONT_ALIAS_FILENAME, "old")
    plate = tmp_path / "assets" / "scenes" / "harbor" / "plate_base.png"
    plate.parent.mkdir(parents=True)
    plate.write_bytes(synthetic_still_png(1344, 768, tag="old-plate", placeholder=True))
    client = KolorsClient(["k"], opener=sized_placeholder_opener("skip"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    generate_asset_library(
        conn,
        "story-x",
        chars,
        locs,
        props,
        client,
        None,
        "fiction",
        tmp_path,
        interiors=interiors,
        skip_existing=True,
        clip_scorer=None,
    )
    rec = load_assets_index(tmp_path)["characters"].get("hero") or {}
    assert rec.get("qc_verdict") != "pass"
    assert not is_qc_locked(tmp_path, character_id="hero")


def test_has_locked_identity_false_for_stale_visual_v1(tmp_path):
    root = tmp_path / "story"
    _front(root / "assets" / "characters" / "hero" / FRONT_ALIAS_FILENAME)
    lock_after_qc(root, character_id="hero", filename=FRONT_ALIAS_FILENAME, qc=pass_qc())
    rec = load_assets_index(root)["characters"]["hero"]
    rec["qc_version"] = "visual_qc_v0"
    rec["qc_verdict"] = "pass"
    from anime_factory.asset_lock import save_assets_index

    save_assets_index(root, load_assets_index(root) | {"characters": {"hero": rec}})
    index = load_assets_index(root)
    index["characters"]["hero"]["qc_version"] = "visual_qc_v0"
    save_assets_index(root, index)
    mark_stale_visual_v1(root)
    rec = load_assets_index(root)["characters"]["hero"]
    assert rec["qc_verdict"] == "stale_visual_v1"
    assert rec["selected"] is None
    assert not has_locked_identity(root, "hero")
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    seg = {"id": "s001", "character_id": "hero", "refs": ["char_hero_sheet"]}
    violations, _missing = gate3_assets(
        conn,
        seg,
        {},
        {"items": [{"id": "char_hero_sheet", "kind": "character_sheet"}], "characters": {"hero": rec}},
        story_root=root,
    )
    assert any(v.code in {"stale_visual", "unlocked_identity"} for v in violations)


def test_mark_stale_keeps_files_and_history(tmp_path):
    root = tmp_path / "story"
    plate = root / "assets" / "scenes" / "harbor" / "plate_base.png"
    plate.parent.mkdir(parents=True)
    plate.write_bytes(synthetic_still_png(1344, 768, tag="dish", placeholder=True))
    from anime_factory.asset_lock import auto_lock_first_success

    auto_lock_first_success(root, scene_id="harbor", filename="plate_base.png")
    rec = load_assets_index(root)["scenes"]["harbor"]
    assert rec["selected"] == "plate_base.png"
    mark_stale_visual_v1(root)
    rec = load_assets_index(root)["scenes"]["harbor"]
    assert rec["selected"] is None
    assert rec["qc_verdict"] == "stale_visual_v1"
    assert "plate_base.png" in rec["history"]
    assert plate.is_file()


def test_derive_waits_for_parent_qc_pass(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    specs = {
        "char_hero_costume_wet": {
            "id": "char_hero_costume_wet",
            "kind": "costume_derive",
            "character_id": "hero_wet",
            "parent_id": "hero",
            "prompt": "1boy, adult, short black hair, brown eyes, wet cloak, lean build",
            "path": "assets/characters/hero_wet/sheet_turnaround.png",
            "seed": 1,
            "image_size": "768x1344",
        }
    }
    client = KolorsClient(["k"], opener=sized_placeholder_opener("derive"), live=False)
    created = derive_missing_assets(
        conn, "story-x", ["char_hero_costume_wet"], specs, client, None, "fiction", tmp_path, clip_scorer=passing_scorer()
    )
    assert created == []
    assert not (tmp_path / "assets/characters/hero_wet/sheet_turnaround.png").is_file()
    _front(tmp_path / "assets" / "characters" / "hero" / FRONT_ALIAS_FILENAME)
    lock_after_qc(tmp_path, character_id="hero", filename=FRONT_ALIAS_FILENAME, qc=pass_qc())
    created = derive_missing_assets(
        conn, "story-x", ["char_hero_costume_wet"], specs, client, None, "fiction", tmp_path, clip_scorer=passing_scorer()
    )
    assert "char_hero_costume_wet" in created


def test_pass_locks_and_round_trip_qc_fields(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    client = KolorsClient(["k"], opener=sized_placeholder_opener("pass"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    generate_asset_library(
        conn,
        "story-x",
        chars,
        locs,
        props,
        client,
        None,
        "fiction",
        tmp_path,
        interiors=interiors,
        skip_existing=False,
        clip_scorer=passing_scorer(),
    )
    rec = load_assets_index(tmp_path)["characters"]["hero"]
    assert rec["selected"] == FRONT_ALIAS_FILENAME
    assert rec["qc_verdict"] == "pass"
    assert rec["qc_version"] == QC_VERSION
    assert rec["qc_scorer"] == "scripted:pass"
    assert "qc_scores" in rec
    assert TURNAROUND_FILENAME in rec["history"]
    raw = json.loads((tmp_path / "assets" / "index.json").read_text(encoding="utf-8"))
    again = load_assets_index(tmp_path)
    assert again["characters"]["hero"]["qc_verdict"] == raw["characters"]["hero"]["qc_verdict"]
    assert again["characters"]["hero"]["qc_scores"] == raw["characters"]["hero"]["qc_scores"]
    assert again["characters"]["hero"].get("candidates") == raw["characters"]["hero"].get("candidates")
    assert is_qc_locked(tmp_path, character_id="hero")
    assert is_qc_locked(tmp_path, scene_id="harbor")


def test_session_blocks_unlocked_sheet(tmp_path):
    from gpu_worker.session import _character_sheet_files, _locked_character_file

    sheet = tmp_path / "assets" / "characters" / "hero" / FRONT_ALIAS_FILENAME
    _front(sheet)
    assert _locked_character_file(tmp_path, "hero") is None
    assert _character_sheet_files(tmp_path, "char_hero_sheet") == []
    lock_after_qc(tmp_path, character_id="hero", filename=FRONT_ALIAS_FILENAME, qc=pass_qc())
    files = _character_sheet_files(tmp_path, "char_hero_sheet")
    assert len(files) == 1
    assert files[0].name == FRONT_ALIAS_FILENAME


def test_record_qc_candidate_never_selects(tmp_path):
    root = tmp_path / "story"
    _front(root / "assets" / "characters" / "hero" / "sheet_front_2.png")
    record_qc_candidate(
        root,
        character_id="hero",
        filename="sheet_front_2.png",
        qc=fail_qc("prompt_mismatch", seed=9),
    )
    rec = load_assets_index(root)["characters"]["hero"]
    assert rec["selected"] is None
    assert rec["history"] == ["sheet_front_2.png"]
    assert rec["candidates"][0]["filename"] == "sheet_front_2.png"
    assert rec["qc_verdict"] in {"fail", "needs_human"}


def test_record_qc_candidate_preserves_current_lock(tmp_path):
    root = tmp_path / "story"
    _front(root / "assets" / "characters" / "hero" / FRONT_ALIAS_FILENAME)
    lock_after_qc(root, character_id="hero", filename=FRONT_ALIAS_FILENAME, qc=pass_qc())
    record_qc_candidate(
        root,
        character_id="hero",
        filename="sheet_side.png",
        qc=fail_qc("prompt_mismatch", seed=3),
    )
    rec = load_assets_index(root)["characters"]["hero"]
    assert rec["selected"] == FRONT_ALIAS_FILENAME
    assert rec["qc_verdict"] == "pass"
    assert rec["qc_version"] == QC_VERSION
    assert rec["candidates"][-1]["filename"] == "sheet_side.png"
    assert rec["candidates"][-1]["verdict"] == "fail"
    assert has_locked_identity(root, "hero")


def _seed_legacy_selected(root: Path) -> tuple[bytes, bytes]:
    from anime_factory.asset_lock import auto_lock_first_success

    front_bytes = synthetic_still_png(CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT, tag="old-front", placeholder=True)
    plate_bytes = synthetic_still_png(1344, 768, tag="old-plate", placeholder=True)
    front = root / "assets" / "characters" / "hero" / FRONT_ALIAS_FILENAME
    plate = root / "assets" / "scenes" / "harbor" / "plate_base.png"
    front.parent.mkdir(parents=True, exist_ok=True)
    plate.parent.mkdir(parents=True, exist_ok=True)
    front.write_bytes(front_bytes)
    plate.write_bytes(plate_bytes)
    auto_lock_first_success(root, character_id="hero", filename=FRONT_ALIAS_FILENAME)
    auto_lock_first_success(root, scene_id="harbor", filename="plate_base.png")
    return front_bytes, plate_bytes


def test_generate_does_not_overwrite_existing_and_keeps_stale_history(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    old_front, old_plate = _seed_legacy_selected(tmp_path)
    client = KolorsClient(["k"], opener=sized_placeholder_opener("fresh"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    out = generate_asset_library(
        conn,
        "story-x",
        chars,
        locs,
        props,
        client,
        None,
        "fiction",
        tmp_path,
        interiors=interiors,
        skip_existing=False,
        clip_scorer=passing_scorer(),
    )
    hero_dir = tmp_path / "assets" / "characters" / "hero"
    plate = tmp_path / "assets" / "scenes" / "harbor" / "plate_base.png"
    front_2 = hero_dir / "sheet_front_2.png"
    assert front_2.is_file()
    assert (hero_dir / FRONT_ALIAS_FILENAME).read_bytes() == front_2.read_bytes()
    assert plate.read_bytes() == old_plate
    assert (hero_dir / "sheet_front_2.png").is_file()
    assert (tmp_path / "assets" / "scenes" / "harbor" / "plate_base_2.png").is_file()
    rec = load_assets_index(tmp_path)["characters"]["hero"]
    assert rec["selected"] == "sheet_front_2.png"
    assert FRONT_ALIAS_FILENAME in rec["history"]
    assert "sheet_front_2.png" in rec["history"]
    harbor = load_assets_index(tmp_path)["scenes"]["harbor"]
    assert harbor["selected"] == "plate_base_2.png"
    assert "plate_base.png" in harbor["history"]
    assert rec["qc_verdict"] == "pass"
    assert "char_hero_sheet" in out["created"]


def test_side_back_reference_qc_front_2(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    old_front, _plate = _seed_legacy_selected(tmp_path)
    client = KolorsClient(["k"], opener=sized_placeholder_opener("views"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    generate_asset_library(
        conn,
        "story-x",
        chars,
        locs,
        props,
        client,
        None,
        "fiction",
        tmp_path,
        interiors=interiors,
        skip_existing=False,
        clip_scorer=passing_scorer(),
    )
    front_2 = tmp_path / "assets" / "characters" / "hero" / "sheet_front_2.png"
    assert front_2.is_file()
    assert (tmp_path / "assets" / "characters" / "hero" / FRONT_ALIAS_FILENAME).read_bytes() == front_2.read_bytes()
    derive_payloads = [p for p in client.last_payloads if p.get("_kind") == "character_view_derive"]
    hero_derives = [p for p in derive_payloads if p.get("_parent_id") == "hero"]
    assert hero_derives
    assert all(p.get("_parent_png") == front_2.read_bytes() for p in hero_derives)
    assert all(p.get("_parent_png") != old_front for p in hero_derives)


def test_side_fail_skips_turnaround_and_needs_human(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    client = KolorsClient(["k"], opener=sized_placeholder_opener("side-fail"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    out = generate_asset_library(
        conn,
        "story-x",
        chars,
        locs,
        props,
        client,
        None,
        "fiction",
        tmp_path,
        interiors=interiors,
        skip_existing=False,
        clip_scorer=view_fail_scorer("side"),
    )
    assert "char_hero_side" in out["needs_human"]
    assert "char_hero_turnaround" in out["needs_human"]
    assert "char_hero_side" not in out["created"]
    assert "char_hero_turnaround" not in out["created"]
    assert not (tmp_path / "assets" / "characters" / "hero" / TURNAROUND_FILENAME).is_file()
    rec = load_assets_index(tmp_path)["characters"]["hero"]
    assert rec["selected"] == FRONT_ALIAS_FILENAME
    assert rec["qc_verdict"] == "pass"
    assert rec.get("candidates")
    assert any(c.get("verdict") == "fail" for c in rec["candidates"])


def test_back_fail_skips_turnaround_and_needs_human(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    client = KolorsClient(["k"], opener=sized_placeholder_opener("back-fail"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    out = generate_asset_library(
        conn,
        "story-x",
        chars,
        locs,
        props,
        client,
        None,
        "fiction",
        tmp_path,
        interiors=interiors,
        skip_existing=False,
        clip_scorer=view_fail_scorer("back"),
    )
    assert "char_hero_back" in out["needs_human"]
    assert "char_hero_turnaround" in out["needs_human"]
    assert "char_hero_back" not in out["created"]
    assert "char_hero_turnaround" not in out["created"]
    assert not (tmp_path / "assets" / "characters" / "hero" / TURNAROUND_FILENAME).is_file()
    rec = load_assets_index(tmp_path)["characters"]["hero"]
    assert rec["selected"] == FRONT_ALIAS_FILENAME
    assert rec["qc_verdict"] == "pass"


def test_composite_uses_this_round_paths(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    hero = tmp_path / "assets" / "characters" / "hero"
    hero.mkdir(parents=True)
    old_front = synthetic_still_png(CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT, tag="old-front", placeholder=True)
    old_side = synthetic_still_png(CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT, tag="old-side", placeholder=True)
    old_back = synthetic_still_png(CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT, tag="old-back", placeholder=True)
    old_turn = synthetic_still_png(CHAR_VIEW_WIDTH * 3, CHAR_VIEW_HEIGHT, tag="old-turn", placeholder=True)
    (hero / FRONT_ALIAS_FILENAME).write_bytes(old_front)
    (hero / "sheet_side.png").write_bytes(old_side)
    (hero / "sheet_back.png").write_bytes(old_back)
    (hero / TURNAROUND_FILENAME).write_bytes(old_turn)
    from anime_factory.asset_lock import auto_lock_first_success

    auto_lock_first_success(tmp_path, character_id="hero", filename=FRONT_ALIAS_FILENAME)
    client = KolorsClient(["k"], opener=sized_placeholder_opener("round"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    out = generate_asset_library(
        conn,
        "story-x",
        chars,
        locs,
        props,
        client,
        None,
        "fiction",
        tmp_path,
        interiors=interiors,
        skip_existing=False,
        clip_scorer=passing_scorer(),
    )
    assert (hero / FRONT_ALIAS_FILENAME).read_bytes() == (hero / "sheet_front_2.png").read_bytes()
    assert (hero / "sheet_side.png").read_bytes() == old_side
    assert (hero / "sheet_back.png").read_bytes() == old_back
    assert (hero / TURNAROUND_FILENAME).read_bytes() == old_turn
    sources = out["specs"]["char_hero_turnaround"]["source_paths"]
    assert sources == [
        "assets/characters/hero/sheet_front_2.png",
        "assets/characters/hero/sheet_side_2.png",
        "assets/characters/hero/sheet_back_2.png",
    ]
    assert (hero / "sheet_turnaround_2.png").is_file()
    assert (hero / "sheet_turnaround_2.png").read_bytes() != old_turn
    assert "char_hero_turnaround" in out["created"]
    composed = compose_character_turnaround(tmp_path, {"source_paths": sources})
    assert (hero / "sheet_turnaround_2.png").read_bytes() == composed


def test_failed_views_are_regenerated_without_front_substitution(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    hero = tmp_path / "assets" / "characters" / "hero"
    hero.mkdir(parents=True)
    front = synthetic_still_png(CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT, tag="front", placeholder=True)
    failed_side = synthetic_still_png(
        CHAR_VIEW_WIDTH,
        CHAR_VIEW_HEIGHT,
        tag="failed-side",
        placeholder=True,
    )
    failed_back = synthetic_still_png(
        CHAR_VIEW_WIDTH,
        CHAR_VIEW_HEIGHT,
        tag="failed-back",
        placeholder=True,
    )
    failed_turnaround = synthetic_still_png(
        CHAR_VIEW_WIDTH * 3,
        CHAR_VIEW_HEIGHT,
        tag="failed-turnaround",
        placeholder=True,
    )
    (hero / FRONT_ALIAS_FILENAME).write_bytes(front)
    (hero / "sheet_side.png").write_bytes(failed_side)
    (hero / "sheet_back.png").write_bytes(failed_back)
    (hero / TURNAROUND_FILENAME).write_bytes(failed_turnaround)
    lock_after_qc(tmp_path, character_id="hero", filename=FRONT_ALIAS_FILENAME, qc=pass_qc())
    for filename in ("sheet_side.png", "sheet_back.png", TURNAROUND_FILENAME):
        record_qc_candidate(
            tmp_path,
            character_id="hero",
            filename=filename,
            qc=fail_qc("attr_pants_palette"),
        )

    client = KolorsClient(["k"], opener=sized_placeholder_opener("fresh-view"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    out = generate_asset_library(
        conn,
        "story-x",
        chars,
        locs,
        props,
        client,
        None,
        "fiction",
        tmp_path,
        interiors=interiors,
        skip_existing=True,
        clip_scorer=passing_scorer(),
    )

    assert out["specs"]["char_hero_side"]["path"].endswith("sheet_side_2.png")
    assert out["specs"]["char_hero_back"]["path"].endswith("sheet_back_2.png")
    assert out["specs"]["char_hero_turnaround"]["path"].endswith("sheet_turnaround_2.png")
    sources = out["specs"]["char_hero_turnaround"]["source_paths"]
    assert sources == [
        "assets/characters/hero/sheet_front.png",
        "assets/characters/hero/sheet_side_2.png",
        "assets/characters/hero/sheet_back_2.png",
    ]
    assert all(path != "assets/characters/hero/sheet_front.png" for path in sources[1:])
    assert (hero / "sheet_side.png").read_bytes() == failed_side
    assert (hero / "sheet_back.png").read_bytes() == failed_back
    assert (hero / TURNAROUND_FILENAME).read_bytes() == failed_turnaround


def test_locked_front_reuses_side_back_for_turnaround(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    hero = tmp_path / "assets" / "characters" / "hero"
    hero.mkdir(parents=True)
    front = synthetic_still_png(CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT, tag="locked-front", placeholder=True)
    side = synthetic_still_png(CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT, tag="side-2", placeholder=True)
    back = synthetic_still_png(CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT, tag="back-3", placeholder=True)
    (hero / FRONT_ALIAS_FILENAME).write_bytes(front)
    (hero / "sheet_side_2.png").write_bytes(side)
    (hero / "sheet_back_3.png").write_bytes(back)
    lock_after_qc(tmp_path, character_id="hero", filename=FRONT_ALIAS_FILENAME, qc=pass_qc())
    record_qc_candidate(
        tmp_path,
        character_id="hero",
        filename="sheet_side_2.png",
        qc=pass_qc(),
    )
    conn.execute(
        """
        INSERT INTO assets (id, kind, path, fingerprint, character_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "char_hero_back",
            "character_sheet",
            "https://r2.example/stories/story-x/assets/characters/hero/sheet_back_3.png?download=1",
            "sha256:passed-back",
            "hero",
        ),
    )
    client = KolorsClient(["k"], opener=sized_placeholder_opener("must-not-reroll"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    out = generate_asset_library(
        conn,
        "story-x",
        chars,
        locs,
        props,
        client,
        None,
        "fiction",
        tmp_path,
        interiors=interiors,
        skip_existing=True,
        clip_scorer=passing_scorer(),
    )
    derive_payloads = [p for p in client.last_payloads if p.get("_kind") == "character_view_derive"]
    assert derive_payloads == []
    assert "char_hero_turnaround" in out["created"]
    assert "char_hero_turnaround" not in out["needs_human"]
    sources = out["specs"]["char_hero_turnaround"]["source_paths"]
    assert sources[0].endswith("sheet_front.png")
    assert sources[1].endswith("sheet_side_2.png")
    assert sources[2].endswith("sheet_back_3.png")
    assert out["specs"]["char_hero_side"]["path"] == sources[1]
    assert out["specs"]["char_hero_back"]["path"] == sources[2]
    assert (hero / TURNAROUND_FILENAME).is_file()


def test_passed_turnaround_reuses_exact_evidenced_path(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    hero = tmp_path / "assets" / "characters" / "hero"
    hero.mkdir(parents=True)
    files = {
        FRONT_ALIAS_FILENAME: synthetic_still_png(
            CHAR_VIEW_WIDTH,
            CHAR_VIEW_HEIGHT,
            tag="front",
            placeholder=True,
        ),
        "sheet_side_2.png": synthetic_still_png(
            CHAR_VIEW_WIDTH,
            CHAR_VIEW_HEIGHT,
            tag="side",
            placeholder=True,
        ),
        "sheet_back_3.png": synthetic_still_png(
            CHAR_VIEW_WIDTH,
            CHAR_VIEW_HEIGHT,
            tag="back",
            placeholder=True,
        ),
        "sheet_turnaround_4.png": synthetic_still_png(
            CHAR_VIEW_WIDTH * 3,
            CHAR_VIEW_HEIGHT,
            tag="turnaround",
            placeholder=True,
        ),
    }
    for filename, blob in files.items():
        (hero / filename).write_bytes(blob)
    lock_after_qc(tmp_path, character_id="hero", filename=FRONT_ALIAS_FILENAME, qc=pass_qc())
    for filename in ("sheet_side_2.png", "sheet_back_3.png", "sheet_turnaround_4.png"):
        record_qc_candidate(
            tmp_path,
            character_id="hero",
            filename=filename,
            qc=pass_qc(),
        )

    client = KolorsClient(["k"], opener=sized_placeholder_opener("must-not-reroll"), live=False)
    chars, locs, props, interiors = harbor_mvp_cast()
    out = generate_asset_library(
        conn,
        "story-x",
        chars,
        locs,
        props,
        client,
        None,
        "fiction",
        tmp_path,
        interiors=interiors,
        skip_existing=True,
        clip_scorer=passing_scorer(),
    )

    assert out["specs"]["char_hero_turnaround"]["path"] == (
        "assets/characters/hero/sheet_turnaround_4.png"
    )
    assert (hero / "sheet_turnaround_4.png").read_bytes() == files["sheet_turnaround_4.png"]
    assert not (hero / TURNAROUND_FILENAME).exists()

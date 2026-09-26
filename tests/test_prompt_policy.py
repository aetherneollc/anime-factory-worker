import pytest

from anime_factory.board import PromptCollapseError, SpeakerFrameError, assert_shot_prompt_diversity, assert_speaker_matches_frame
from anime_factory.design import (
    _extract_wiki_identity,
    build_scene_plate_payload,
    CharacterIdentityError,
    FRONT_LOOK,
    LocationPromptError,
    PLATE_NEGATIVE,
    SHEET_NEGATIVE,
    identity_conditioned_negative,
    plan_library_specs,
    sanitize_location_prompt,
    seed_cast_from_story_root,
    style_prompt,
    synthetic_still_png,
    validate_character_identity,
)
from anime_factory.models import (
    FIXED_NEGATIVE,
    LUMINOUS_CINEMATIC_ANIME_PRESET,
    STYLE_PREFIX,
    STYLE_PREFIX_CHARACTER,
    STYLE_PREFIX_LOCATION,
    STYLE_PREFIX_LOCATION_LUMINOUS,
    scrub_copycat,
    style_prefix_for_kind,
)
from anime_factory.produce import TITLE, shot_plan


BANNED = (
    "makoto shinkai",
    "shinkai style",
    "新海诚",
    "kimi no na wa",
    "君の名は",
    "weathering with you",
    "天气之子",
    "suzume",
    "秒速5厘米",
    "5 centimeters per second",
)


def test_style_prefix_is_original_production_art_not_shinkai_stills():
    assert "original anime production still" in STYLE_PREFIX.lower()
    assert "makoto shinkai" not in STYLE_PREFIX.lower()
    assert "shinkai style" not in STYLE_PREFIX.lower()
    for term in ("Makoto Shinkai", "君の名は", "天气之子", "秒速5厘米"):
        assert term in FIXED_NEGATIVE
    assert "generic endless sunset" not in FIXED_NEGATIVE
    assert "clear luminous atmosphere" in STYLE_PREFIX_LOCATION
    assert "ghost film" in FIXED_NEGATIVE
    assert "corpse-pale skin" in FIXED_NEGATIVE


def test_style_prefix_has_no_negations():
    """Diffusion cannot read negation: "not a ghost film" *drew* ghost films.

    Everything the prefix used to forbid has to live in FIXED_NEGATIVE, where
    SDXL's real CFG actually applies it.
    """
    lowered = STYLE_PREFIX.lower()
    for negation in ("not ", "no ", "never", "without", "avoid", "don't", "free of", "非", "不要", "禁止"):
        assert negation not in lowered, f"STYLE_PREFIX still negates: {negation!r}"
    # The five clauses that used to be negated in the positive prompt.
    for concept in ("ghost film", "corpse-pale skin", "wet gloomy face", "horror", "movie screenshot"):
        assert concept in FIXED_NEGATIVE.lower(), f"{concept!r} dropped instead of moved to the negative"
    for concept in ("ceramic plate", "dinnerware", "nude", "nsfw"):
        assert concept in FIXED_NEGATIVE.lower(), f"{concept!r} missing from FIXED_NEGATIVE"


def test_scrub_copycat_strips_famous_stills():
    dirty = "Makoto Shinkai style anime still, 君の名は sky, Weathering With You clouds"
    clean = scrub_copycat(dirty).lower()
    for ban in BANNED:
        assert ban not in clean
    prompt = style_prompt("Makoto Shinkai lighthouse at dusk, 秒速5厘米 sky")
    lowered = prompt.lower()
    assert "makoto shinkai" not in lowered
    assert "秒速5厘米" not in prompt
    assert STYLE_PREFIX.split(",")[0] in prompt


def test_character_sheets_are_model_sheets():
    text = FRONT_LOOK
    assert "front view" in text.lower()
    assert "设定稿" not in text
    # "not a cinematic still" belongs in the paired negative, not in the look.
    assert "not a cinematic still" not in text.lower()
    assert "cinematic still" in SHEET_NEGATIVE
    assert "character turnaround" in SHEET_NEGATIVE
    identity = "1girl, young adult, short black hair, brown eyes, grey hoodie, lean build"
    specs = plan_library_specs(
        [{"id": "ke", "name": "阿柯", "gender": "female", "identity_prompt": identity}],
        [{"id": "loft", "name": "夜窗出租屋", "plate_prompt": "small night loft desk, anime location background"}],
        [{"id": "tablet", "name": "黑客平板", "identity_prompt": "matte tablet prop, product shot"}],
    )
    sheets = [s for s in specs.values() if s["kind"] == "character_sheet"]
    derives = [s for s in specs.values() if s["kind"] == "character_view_derive"]
    plates = [s for s in specs.values() if s["kind"] == "scene_plate"]
    props = [s for s in specs.values() if s["kind"] == "prop"]
    assert sheets and all("front view" in s["prompt"].lower() for s in sheets)
    assert derives and all(identity in s["prompt"] for s in derives)
    assert derives and all(s["parent_id"] == "ke" for s in derives)
    assert plates and all("empty establishing shot" not in s["prompt"] for s in plates)
    assert plates and all("establishing plate" not in s["prompt"] for s in plates)
    assert any("wide establishing shot" in s["prompt"] or "empty location shot" in s["prompt"] for s in plates)
    assert plates and all("famous movie still" not in s["prompt"] for s in plates)
    assert "famous movie still" in PLATE_NEGATIVE
    assert "ceramic plate" in PLATE_NEGATIVE
    assert props and all("prop design sheet" in s["prompt"] for s in props)
    blob = " ".join(s["prompt"] for s in specs.values()).lower()
    assert "makoto shinkai" not in blob
    assert "detailed background" not in sheets[0]["prompt"].lower()
    assert "cinematic composition" not in sheets[0]["prompt"].lower()


def test_style_prefixes_split_by_asset_kind():
    assert style_prefix_for_kind("character_sheet") == STYLE_PREFIX_CHARACTER
    assert style_prefix_for_kind("character_view_derive") == STYLE_PREFIX_CHARACTER
    assert style_prefix_for_kind("scene_plate") == STYLE_PREFIX_LOCATION
    assert "detailed background" in STYLE_PREFIX_LOCATION
    assert "detailed background" not in STYLE_PREFIX_CHARACTER


def test_luminous_preset_is_location_keyframe_only(monkeypatch):
    monkeypatch.setenv("STYLE_PRESET", LUMINOUS_CINEMATIC_ANIME_PRESET)
    assert style_prefix_for_kind("scene_plate") == STYLE_PREFIX_LOCATION_LUMINOUS
    assert style_prefix_for_kind("keyframe") == STYLE_PREFIX_LOCATION_LUMINOUS
    assert style_prefix_for_kind("character_sheet") == STYLE_PREFIX_CHARACTER

    payload = build_scene_plate_payload(
        "rainy canal shopping street, old stone bridge, anime location background",
        period_md="# Period\n\n## 正面清单\n- hand-painted wooden signs\n\n## 负面清单\n- modern neon signage\n",
        world_mode="history",
    )
    prompt = payload["prompt"].lower()
    for term in (
        "clear luminous atmosphere",
        "contextual layered clouds for open-sky scenes",
        "volumetric light",
        "rim light",
        "wet-surface reflections",
        "saturated blue and gold contrast",
        "detailed everyday urban and natural backgrounds",
        "atmospheric perspective",
        "cinematic depth",
        "hand-painted wooden signs",
        "rainy canal shopping street",
    ):
        assert term in prompt
    assert "sunset" not in prompt
    for ban in BANNED:
        assert ban not in prompt

    char_prompt = style_prompt(
        "1girl, young adult, short black hair, brown eyes, grey hoodie, lean build",
        kind="character_sheet",
        gender_tag="1girl",
    ).lower()
    assert "warm ivory studio background" in char_prompt
    for scenic in ("layered clouds", "wet-surface reflections", "detailed background", "cinematic depth"):
        assert scenic not in char_prompt


def test_location_prompt_rejects_standalone_plate():
    with pytest.raises(LocationPromptError):
        sanitize_location_prompt("wooden harbor street plate at dusk")
    clean = sanitize_location_prompt("wooden harbor establishing plate at dusk")
    assert "establishing plate" not in clean.lower()
    assert "wide establishing shot" in clean.lower()
    assert "empty establishing shot" not in clean.lower()


def test_plan_library_specs_uses_plate_prompt_when_name_is_cjk():
    plate = "small night convenience store, cold fluorescent light, anime location background"
    specs = plan_library_specs(
        [{"id": "hero", "name": "阿哲", "identity_prompt": "1boy, young adult, short black hair, brown eyes, grey hoodie, lean build"}],
        [{"id": "loc_store", "name": "便利店", "plate_prompt": plate}],
        [],
    )
    scene = next(v for v in specs.values() if v["kind"] == "scene_plate")
    assert plate in scene["prompt"]
    assert "便利店" not in scene["prompt"]


def test_extract_wiki_identity_from_chinese_markdown_labels():
    identity = (
        "1girl, young adult, 24 y/o, shoulder-length straight black hair, amber eyes, "
        "navy coat over white shirt, slim build, silver portable field recorder"
    )
    wiki = f"""# 林澄

- **编号**：`lin_cheng`
- **身份**：{identity}
"""
    extracted = _extract_wiki_identity(wiki)
    assert extracted == identity
    assert "编号" not in extracted
    assert "林澄" not in extracted
    validated = validate_character_identity(extracted, character_id="lin_cheng", name="林澄")
    assert validated.startswith("1girl")
    assert "navy coat" in validated.lower()


def test_extract_wiki_identity_strips_chinese_labels_from_r2_example():
    """longlive-3-1b9bd1 wiki: labels are CJK but the 身份 value is English."""
    wiki = """# 林澄

- **编号**：`lin_cheng`
- **身份**：1girl, young adult, 24 y/o, shoulder-length straight black hair, amber eyes, fixed ivory raincoat, navy scarf, slim build, silver portable field recorder
"""
    extracted = _extract_wiki_identity(wiki)
    assert extracted.startswith("1girl")
    assert "编号" not in extracted
    assert "身份" not in extracted
    assert "林澄" not in extracted
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in extracted)
    validated = validate_character_identity(extracted, character_id="lin_cheng", name="林澄")
    assert "ivory raincoat" in validated.lower()
    assert "navy scarf" in validated.lower()


def test_extract_wiki_identity_rejects_cjk_identity_value():
    wiki = """# 周野

- **编号**：`zhou_ye`
- **身份**：年轻男性，黑色短发，琥珀色眼睛，象牙色雨衣
"""
    extracted = _extract_wiki_identity(wiki)
    assert "年轻" in extracted
    with pytest.raises(CharacterIdentityError, match="English-only"):
        validate_character_identity(extracted, character_id="zhou_ye", name="周野")


def test_seed_cast_from_story_root_accepts_chinese_wiki_labels(tmp_path):
    from anime_factory.db import migrate, open_db

    wiki_dir = tmp_path / "canon" / "wiki" / "characters"
    wiki_dir.mkdir(parents=True)
    wiki_dir.joinpath("lin_cheng.md").write_text(
        """# 林澄

- **编号**：`lin_cheng`
- **身份**：1girl, young adult, 24 y/o, shoulder-length straight black hair, amber eyes, fixed ivory raincoat, navy scarf, slim build, silver portable field recorder
""",
        encoding="utf-8",
    )
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    result = seed_cast_from_story_root(conn, tmp_path)
    assert result["characters"] == 1
    row = conn.execute(
        "SELECT id, name, identity_prompt FROM characters WHERE id = ?",
        ("lin_cheng",),
    ).fetchone()
    assert row is not None
    assert row["name"] == "林澄"
    assert row["identity_prompt"].startswith("1girl")


def test_seed_cast_from_story_root_rejects_cjk_wiki_identity(tmp_path):
    from anime_factory.db import migrate, open_db

    wiki_dir = tmp_path / "canon" / "wiki" / "characters"
    wiki_dir.mkdir(parents=True)
    wiki_dir.joinpath("zhou_ye.md").write_text(
        """# 周野

- **编号**：`zhou_ye`
- **身份**：年轻男性，黑色短发，琥珀色眼睛，象牙色雨衣
""",
        encoding="utf-8",
    )
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    result = seed_cast_from_story_root(conn, tmp_path)
    assert result["characters"] == 0
    row = conn.execute("SELECT id FROM characters WHERE id = ?", ("zhou_ye",)).fetchone()
    assert row is None


def test_seed_cast_from_story_root_plain_english_wiki_fallback(tmp_path):
    from anime_factory.db import migrate, open_db

    wiki_dir = tmp_path / "canon" / "wiki" / "characters"
    wiki_dir.mkdir(parents=True)
    wiki_dir.joinpath("hero.md").write_text(
        "# Hero\n\n1boy, young adult, short black hair, brown eyes, grey hoodie, lean build",
        encoding="utf-8",
    )
    conn = open_db(tmp_path / "story.sqlite")
    migrate(conn)
    result = seed_cast_from_story_root(conn, tmp_path)
    assert result["characters"] == 1
    row = conn.execute("SELECT identity_prompt FROM characters WHERE id = ?", ("hero",)).fetchone()
    assert row is not None
    assert row["identity_prompt"].startswith("1boy")


def test_character_identity_fails_closed_without_structure():
    with pytest.raises(CharacterIdentityError):
        validate_character_identity("hero", character_id="hero")
    with pytest.raises(CharacterIdentityError):
        validate_character_identity("young hacker hoodie", character_id="ke")
    ok = validate_character_identity(
        "1boy, adult, short black hair, brown eyes, dark coat, lean build",
        character_id="hero",
    )
    assert ok.startswith("1boy")
    assert "male focus" in ok


def test_validate_character_identity_requires_color_and_layer_tokens():
    with pytest.raises(CharacterIdentityError, match="clothing color"):
        validate_character_identity(
            "1boy, adult, short black hair, brown eyes, jacket, lean build",
            character_id="hero",
        )
    with pytest.raises(CharacterIdentityError, match="jacket/shirt layer"):
        validate_character_identity(
            "1boy, adult, short black hair, brown eyes, navy jacket, lean build",
            character_id="hero",
        )
    hoodie = validate_character_identity(
        "1girl, young adult, short black hair, brown eyes, grey hoodie, lean build",
        character_id="ke",
    )
    assert "grey hoodie" in hoodie
    layered = validate_character_identity(
        "1boy, young adult, short black hair, brown eyes, navy jacket, white shirt, black pants",
        character_id="hero",
    )
    assert "navy jacket" in layered
    assert "white shirt" in layered
    hooded = validate_character_identity(
        "1boy, young adult college student, short black hair, brown eyes, "
        "navy hooded jacket over a plain white shirt, black straight-leg trousers",
        character_id="hero",
    )
    assert "navy jacket" in hooded.lower()
    assert "dark blue jacket" in hooded.lower()
    assert "black pants" in hooded.lower()
    assert "black trousers" in hooded.lower()


def test_identity_conditioned_negative_omits_hoodie_when_identity_is_hoodie():
    hoodie_neg = identity_conditioned_negative(
        "1girl, young adult, short black hair, brown eyes, grey hoodie, lean build"
    )
    assert "hoodie" not in hoodie_neg
    jacket_neg = identity_conditioned_negative(
        "1boy, male focus, young adult, short black hair, brown eyes, navy jacket, white shirt"
    )
    assert "hoodie" in jacket_neg
    assert "1girl" in jacket_neg
    black_pants_neg = identity_conditioned_negative(
        "1boy, male focus, young adult, short black hair, brown eyes, "
        "navy jacket, white shirt, black pants, black trousers"
    )
    assert "khaki pants" in black_pants_neg
    assert "olive pants" in black_pants_neg


def test_style_prompt_appends_animagine_quality_tags_at_end(monkeypatch):
    monkeypatch.setenv("STILL_BACKEND", "animagine")
    monkeypatch.setenv("CHARACTER_STILL_BACKEND", "animagine")
    prompt = style_prompt("night loft desk", kind="scene_plate")
    assert prompt.endswith("safe")
    assert "masterpiece" in prompt
    char_prompt = style_prompt(
        "1girl, young adult, short hair, brown eyes, hoodie, lean build",
        kind="character_sheet",
        gender_tag="1girl",
    )
    assert char_prompt.rstrip().endswith("safe")
    assert "1girl" in char_prompt


def test_keyframe_uses_board_prompt_and_unique_seed(tmp_path):
    from anime_factory.db import migrate, open_db
    from anime_factory.design import KolorsClient, locked_seed
    from anime_factory.keyframe import ensure_keyframe

    payloads = []
    art = synthetic_still_png(1344, 768, tag="keyframe-policy")

    def gpu_gen(payload):
        payloads.append(payload)
        return art

    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    client = KolorsClient(["k"], live=True, gpu_generate=gpu_gen)
    from anime_factory.visual_qc import set_clip_scorer
    from tests.clip_fakes import passing_scorer

    token = set_clip_scorer(passing_scorer())
    try:
        ensure_keyframe(
            conn,
            "story-x",
            "EP001",
            {"id": "s001", "prompt": "CU at a night convenience store counter, living skin"},
            {},
            {"items": []},
            {},
            client,
            None,
            "fiction",
            tmp_path,
        )
    finally:
        from anime_factory.visual_qc import reset_clip_scorer

        reset_clip_scorer(token)
    assert payloads
    assert "convenience store" in payloads[0]["prompt"]
    assert payloads[0]["seed"] == locked_seed("story-x:s001")
    assert payloads[0]["seed"] != 42
    # Stills are drawn at the 1.8 still aspect, not the old 1280x720.
    assert payloads[0]["image_size"] == "1344x768"
    assert (tmp_path / "episodes" / "EP001" / "keyframes" / "s001" / "f1.png").read_bytes() == art


def test_luminous_preset_reaches_keyframe_payload_and_period_terms(tmp_path, monkeypatch):
    from anime_factory.db import migrate, open_db
    from anime_factory.design import KolorsClient
    from anime_factory.keyframe import ensure_keyframe

    monkeypatch.setenv("STYLE_PRESET", LUMINOUS_CINEMATIC_ANIME_PRESET)
    (tmp_path / "bible").mkdir()
    (tmp_path / "bible" / "style.md").write_text(
        f"{STYLE_PREFIX}\n\nnegative: {FIXED_NEGATIVE}\n",
        encoding="utf-8",
    )
    period_md = "# Period\n\n## 正面清单\n- 1890s brick storefronts\n\n## 负面清单\n- glass skyscrapers\n"
    payloads = []
    art = synthetic_still_png(1344, 768, tag="luminous-keyframe")

    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    client = KolorsClient(["k"], live=True, gpu_generate=lambda p: payloads.append(p) or art)
    from anime_factory.visual_qc import reset_clip_scorer, set_clip_scorer
    from tests.clip_fakes import passing_scorer

    token = set_clip_scorer(passing_scorer())
    try:
        ensure_keyframe(
            conn,
            "story-x",
            "EP001",
            {"id": "s101", "prompt": "wide view of a rainy train station plaza, living skin"},
            {},
            {"items": []},
            {},
            client,
            period_md,
            "history",
            tmp_path,
        )
    finally:
        reset_clip_scorer(token)

    assert payloads
    prompt = payloads[0]["prompt"].lower()
    assert "clear luminous atmosphere" in prompt
    assert "rainy train station plaza" in prompt
    assert "1890s brick storefronts" in prompt
    assert "makoto shinkai" not in prompt
    assert "generic endless sunset" not in payloads[0]["negative_prompt"]
    assert "clear luminous atmosphere" in payloads[0]["prompt"]
    assert "glass skyscrapers" in payloads[0]["negative_prompt"]


def test_keyframe_refuses_to_draw_the_segment_id(tmp_path):
    """The old fallback drew a still of the literal string "s002"."""
    from anime_factory.db import migrate, open_db
    from anime_factory.design import KolorsClient
    from anime_factory.keyframe import ensure_keyframe

    calls = []
    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    client = KolorsClient(["k"], live=True, gpu_generate=lambda p: calls.append(p) or b"x")
    try:
        ensure_keyframe(
            conn,
            "story-x",
            "EP001",
            {"id": "s002", "h3_prompt": "@阿柯 站在 @便利店 中, camera dolly in"},
            {},
            {"items": []},
            {},
            client,
            None,
            "fiction",
            tmp_path,
        )
        raise AssertionError("a segment with no still prompt must fail, not draw its own id")
    except RuntimeError as exc:
        assert "first_frame_prompt" in str(exc)
    assert calls == []


def test_keyframe_refuses_stub_bytes(tmp_path):
    """A 3-byte b"PNG" used to ship as a keyframe."""
    from anime_factory.db import migrate, open_db
    from anime_factory.design import KolorsClient, StillBlobError
    from anime_factory.keyframe import ensure_keyframe

    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    client = KolorsClient(["k"], live=True, gpu_generate=lambda p: b"PNG")
    try:
        ensure_keyframe(
            conn,
            "story-x",
            "EP001",
            {"id": "s003", "prompt": "CU at a night counter, living skin"},
            {},
            {"items": []},
            {},
            client,
            None,
            "fiction",
            tmp_path,
        )
        raise AssertionError("stub bytes must not become a keyframe")
    except StillBlobError as exc:
        assert "bytes" in str(exc)
    assert not (tmp_path / "episodes" / "EP001" / "keyframes" / "s003" / "f1.png").exists()


def test_keyframe_refuses_wrong_size_and_placeholder(tmp_path):
    from anime_factory.db import migrate, open_db
    from anime_factory.design import KolorsClient, StillBlobError
    from anime_factory.keyframe import ensure_keyframe

    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    segment = {"id": "s004", "prompt": "MCU at a rainy bus stop, living skin"}

    def run(blob):
        client = KolorsClient(["k"], live=True, gpu_generate=lambda p: blob)
        ensure_keyframe(conn, "story-x", "EP001", segment, {}, {"items": []}, {}, client, None, "fiction", tmp_path)

    try:
        run(synthetic_still_png(1024, 1024, tag="square"))
        raise AssertionError("a square still must not pass as a 1344x768 keyframe")
    except StillBlobError as exc:
        assert "1024x1024" in str(exc)
    try:
        run(synthetic_still_png(1344, 768, tag="offline", placeholder=True))
        raise AssertionError("an offline placeholder must not persist on a live run")
    except StillBlobError as exc:
        assert "placeholder" in str(exc)


def test_tablet_episode_prompts_are_not_copycat_stills():
    assert TITLE == "黑客平板的故事"
    shots = shot_plan()
    blob = " ".join(
        (s.get("first_frame_prompt") or "") + " " + (s.get("h3_prompt") or "") for s in shots
    ).lower()
    assert "makoto shinkai" not in blob
    assert "lighthouse" not in blob
    assert "tablet" in blob
    assert shots[0]["location_id"] == shots[-1]["location_id"] == "loft"


def test_board_rejects_repeated_first_frame_prompts():
    shots = [
        {"id": f"s{i:03d}", "prompt": "empty horror corridor, corpse-pale skin"}
        for i in range(1, 16)
    ]
    try:
        assert_shot_prompt_diversity(shots)
        raise AssertionError("identical prompts must fail")
    except PromptCollapseError:
        pass


def test_board_rejects_fifteen_segs_two_store_prompts():
    """舔狗: 15 H3 segs / 2 unique store compositions, even if shot N of M differs."""
    a = "@hero 站在 @night convenience store 中, medium close-up, speaking, same store, locked eyeline"
    b = "@girl 站在 @night convenience store 中, close-up, speaking, same store, locked eyeline"
    shots = []
    for i in range(15):
        body = a if i < 8 else b
        shots.append(
            {
                "id": f"s{i+1:03d}",
                "shot_id": f"EP001-{(1 if i < 8 else 2):02d}",
                "chain_index": 0,
                "first_frame_prompt": f"{body}, shot {i+1} of 15",
                "h3_prompt": f"{body}, camera Static Shot, subtle acting motion, do not freeze a still",
                "line": {"zh": f"第{i+1}句。"},
            }
        )
    try:
        assert_shot_prompt_diversity(shots)
        raise AssertionError("15 segs / 2 unique store prompts must fail")
    except PromptCollapseError as exc:
        assert "unique" in str(exc).lower() or "collapse" in str(exc).lower() or "same visual" in str(exc).lower()


def test_chain_tails_may_repeat_last_frame_prompt():
    body = "@hero 站在 @loft 中, medium close-up, speaking at the desk, living skin"
    shots = [
        {
            "id": f"EP001-01-s{i:02d}",
            "shot_id": "EP001-01",
            "chain_id": "EP001-01-chain",
            "chain_index": i - 1,
            "h3_prompt": body,
            "line": {"zh": "灯亮了。"},
        }
        for i in range(1, 4)
    ]
    assert_shot_prompt_diversity(shots)


def test_xiaomei_phone_line_not_store_cu():
    shot = {
        "id": "s007",
        "character_id": "girl",
        "line": {"zh": "她可能睡着了。微信已读不回。"},
        "h3_prompt": "@小美 站在 @convenience store 中, close-up, speaking, same store",
        "on_camera": False,
    }
    try:
        assert_speaker_matches_frame(shot)
        raise AssertionError("remote 小美 must not be a store CU")
    except SpeakerFrameError:
        pass
    shot["on_camera"] = True
    assert_speaker_matches_frame(shot)
    off = dict(shot, on_camera=False, h3_prompt="phone UI, WeChat thread, speaker off-camera, do not show girl in the store")
    assert_speaker_matches_frame(off)


def test_establishing_is_fl2va_and_off_camera_keeps_onscreen_sheet():
    from anime_factory.directors.common import choose_h3_mode, emit_shot, hydrate_shot_identity

    establishing = []
    emit_shot(
        establishing,
        episode_code="EP001",
        scene={"id": "sc01", "location_id": "loc_office", "characters": ["hero", "girl"]},
        locations={"loc_office": {"name": "办公室"}},
        interiors={},
        cast={"hero": {"name": "阿哲"}, "girl": {"name": "小美"}},
        size="wide",
        camera="Static Shot",
        character_id="",
        staging="establishing only, no people entering frame",
        duration=5.0,
        line=None,
        total_hint=4,
        purpose="establish",
    )
    assert establishing[0]["h3_mode"] == "fl2va_first"
    assert choose_h3_mode(establishing[0]) == "fl2va_first"
    assert not any(str(r).startswith("char_") for r in establishing[0]["refs"])

    phone = []
    emit_shot(
        phone,
        episode_code="EP001",
        scene={"id": "sc01", "location_id": "loc_convenience", "characters": ["hero", "girl"]},
        locations={"loc_convenience": {"name": "便利店"}},
        interiors={},
        cast={"hero": {"name": "阿哲", "identity_prompt": "young man grey hoodie"}, "girl": {"name": "小美"}},
        size="CU",
        camera="Static Shot",
        character_id="girl",
        staging="speaking",
        duration=4.0,
        line={"zh": "微信语音：在吗？", "character_id": "girl"},
        total_hint=4,
        purpose="dialogue",
        on_camera=False,
    )
    assert phone[0]["character_id"] == "girl"
    assert "char_hero_sheet" in phone[0]["refs"]
    assert "char_girl_sheet" not in phone[0]["refs"]
    assert phone[0]["h3_mode"] == "ref2va"

    hosted = hydrate_shot_identity(
        {"id": "s001", "character_id": "hero", "h3_prompt": "hero 站在办公室", "line": {"zh": "今天顺路。"}},
        cast={"hero": {}, "girl": {}},
    )
    assert hosted["h3_mode"] == "ref2va"
    assert "char_hero_sheet" in hosted["refs"]

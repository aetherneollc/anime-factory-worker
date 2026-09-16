"""Still prompts and video prompts are two different strings.

The still model was being handed the Hailuo script: Chinese staging, `@tag`
references, `<<<image_1>>>` headers, `camera dolly in`, `do not freeze a still`
and `shot 7 of 15`. None of that is picture content for SDXL's CLIP.
"""

from __future__ import annotations

import pytest

from anime_factory.board import (
    StillPromptError,
    assert_shot_prompt_diversity,
    assert_still_prompt_clean,
    shot_motion_prompt,
    shot_visual_prompt,
)
from anime_factory.directors.common import H3_TAIL, cite_locked_refs, emit_shot, prompt_pair, still_prompt_text
from anime_factory.script import offline_draft


def test_shot_visual_prompt_never_falls_back_to_h3():
    shot = {
        "id": "s001",
        "h3_prompt": "@阿柯 站在 @便利店 中, camera dolly in, " + H3_TAIL,
        "first_frame_prompt": "close-up, young woman in a grey hoodie at a night counter",
    }
    assert shot_visual_prompt(shot) == "close-up, young woman in a grey hoodie at a night counter"
    assert shot_motion_prompt(shot).startswith("@阿柯")
    # No still prompt at all is an error upstream, not a reason to draw the video script.
    assert shot_visual_prompt({"id": "s002", "h3_prompt": "@阿柯 站在 @便利店 中"}) == ""


def test_prompt_pair_splits_languages_and_camera():
    first, h3 = prompt_pair(
        "CU",
        "Dolly In",
        "young woman in a grey hoodie",
        "night convenience store, cold fluorescent light",
        "leaning over the counter",
        7,
        15,
        character_id="ke",
        plate_id="plate_store",
        visual_en="young woman in a grey hoodie leaning over a night convenience store counter, cold fluorescent light",
        motion_zh="镜头缓慢推近，人物微微侧身",
    )
    assert_still_prompt_clean(first)
    assert "grey hoodie" in first
    # The video half keeps everything the still half is not allowed to carry.
    assert "@" in h3 and "镜头缓慢推近" in h3 and "Dolly In" in h3 and H3_TAIL in h3


@pytest.mark.parametrize(
    "prompt",
    [
        "@阿柯 站在 @便利店 中, medium close-up",
        "<<<image_1>>> locked character-sheet ref, medium close-up",
        "@ke at the counter, medium close-up",
        "young woman at a counter, shot 7 of 15",
        "young woman at a counter, 864x480",
        "young woman at a counter, camera dolly in",
        "young woman at a counter, subtle acting motion, do not freeze a still",
        "",
    ],
)
def test_assert_still_prompt_clean_rejects_video_syntax(prompt):
    with pytest.raises(StillPromptError):
        assert_still_prompt_clean(prompt, label="s001")


def test_assert_still_prompt_clean_accepts_a_still_description():
    prompt = (
        "medium close-up, young woman in a grey hoodie, tired eyes, leaning on a night "
        "convenience store counter, cold fluorescent light"
    )
    assert assert_still_prompt_clean(prompt) == prompt


def test_still_prompt_text_strips_cineflow_and_chinese_staging():
    dirty = "[镜头1：中近景] First: @阿柯 站在便利店中, camera pans left, Then: she looks up, shot 3 of 9, 1280x720"
    clean = still_prompt_text(dirty)
    assert_still_prompt_clean(clean)
    assert "she looks up" in clean


def test_cite_locked_refs_only_lands_on_the_video_prompt():
    shots: list[dict] = []
    emit_shot(
        shots,
        episode_code="EP001",
        scene={"id": "sc01", "location_id": "loc_store", "characters": ["ke"]},
        locations={"loc_store": {"name": "便利店", "plate_prompt": "night convenience store, cold fluorescent light"}},
        interiors={},
        cast={"ke": {"name": "阿柯", "identity_prompt": "young woman, grey hoodie, tired eyes"}},
        size="CU",
        camera="Dolly In",
        character_id="ke",
        staging="leaning over the counter",
        duration=4.0,
        line={"zh": "你还在吗？", "character_id": "ke"},
        total_hint=9,
        purpose="dialogue",
        visual_en="young woman in a grey hoodie leaning over the counter, tired eyes",
        motion_zh="镜头缓慢推近",
    )
    shot = shots[0]
    assert_still_prompt_clean(shot["first_frame_prompt"], label=shot["id"])
    assert "grey hoodie" in shot["first_frame_prompt"]
    assert "<<<image_1>>>" in shot["h3_prompt"]
    assert cite_locked_refs("x", shot["refs"]).startswith("<<<image_1>>>")


def test_director_visual_en_and_motion_zh_reach_the_board():
    from anime_factory.board import board_from_script

    draft = offline_draft(
        episode_code="E01",
        title="沉船邮局",
        logline="海底邮局每天寄出一封没有收件人的信。",
        langs=["zh"],
        target_seconds=180,
        shot_seconds=6,
    )
    line = draft["scenes"][0]["lines"][0]
    assert line["visual_en"] and line["motion_zh"]
    assert_still_prompt_clean(line["visual_en"])
    for kind in ("series", "short"):
        shots = board_from_script(draft, episode_code="E01", langs=["zh"], kind=kind)
        assert shots
        for shot in shots:
            assert_still_prompt_clean(shot["first_frame_prompt"], label=shot["id"])
            assert shot["h3_prompt"] != shot["first_frame_prompt"]
        # The split must not flatten the board into one repeated still.
        assert_shot_prompt_diversity(shots)


def test_board_refuses_shots_whose_still_prompt_is_the_video_script(monkeypatch):
    from anime_factory import board as board_mod

    draft = offline_draft(
        episode_code="E01",
        title="锈铃电台",
        logline="废弃电台在午夜自己播放点歌节目。",
        langs=["zh"],
        target_seconds=120,
        shot_seconds=6,
    )

    def leaky(*args, **kwargs):
        return [
            {
                "id": "E01-01",
                "shot_id": "E01-01",
                "chain_index": 0,
                "first_frame_prompt": "@阿柯 站在 @电台 中, camera dolly in, shot 1 of 2",
                "h3_prompt": "@阿柯 站在 @电台 中, camera dolly in",
                "line": {"zh": "在吗？"},
            }
        ]

    monkeypatch.setattr(board_mod, "director_board", leaky)
    with pytest.raises(StillPromptError):
        board_mod.board_from_script(draft, episode_code="E01", langs=["zh"])

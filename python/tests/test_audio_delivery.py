from anime_factory.directors.common import line_on_camera
from anime_factory.continuity_gates import repair_collapsed_staging
from anime_factory.script import normalize_draft


def test_thought_delivery_is_never_marked_as_on_camera():
    raw = {
        "cast": [{"id": "hero", "name": "主角", "gender": "male"}],
        "locations": [{"id": "room", "name": "宿舍", "plate_prompt": "anime dorm room"}],
        "scenes": [
            {
                "location_id": "room",
                "lines": [
                    {
                        "character_id": "hero",
                        "zh": "奖学金到账了。",
                        "en": "The scholarship came in.",
                        "ja": "奨学金が入った。",
                        "delivery_mode": "thought",
                        "on_camera": True,
                    }
                ],
            }
        ],
    }
    normalized = normalize_draft(raw, episode_code="EP001", title="短片", langs=("zh", "en", "ja"))
    line = normalized["scenes"][0]["lines"][0]
    assert line["delivery_mode"] == "thought"
    assert line["on_camera"] is False
    assert line_on_camera(line) is False
    repaired = repair_collapsed_staging(normalized)
    assert repaired["scenes"][0]["lines"][0]["on_camera"] is False

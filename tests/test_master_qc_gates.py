"""Master QC gates: back prompts, region CLIP, Qwen booleans, plate payload."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request

import pytest

from anime_factory.design import (
    BACK_LOOK,
    DERIVE_FACE_LOCK,
    DETAIL_PLATE_LOOK,
    ESTABLISHING_PLATE_LOOK,
    SHEET_NEGATIVE,
    KolorsClient,
    build_scene_plate_payload,
    plan_library_specs,
    synthetic_still_png,
)
from anime_factory.llm import (
    LlmClient,
    evaluate_master_vision_fields,
    master_vision_qc,
)
from anime_factory.visual_qc import (
    CHARACTER_SIZE,
    CLOTHING_IDENTITY_MIN,
    FOOTWEAR_IDENTITY_MIN,
    IDENTITY_MIN,
    ScriptedClipScorer,
    apply_region_identity_gates,
    score_still,
)
from tests.clip_fakes import POS, text_vec

# Orthogonal unit vectors so cosine(POS, ORTH) == 0.
ORTH = (0.0, 1.0)


def _char_png(tag: str) -> bytes:
    return synthetic_still_png(*CHARACTER_SIZE, tag=tag, placeholder=True)


def _footwear_mismatch_scorer(*, model_id: str = "test-clip") -> ScriptedClipScorer:
    """Whole-image + clothing crops match; footwear crop cosine is ~0."""
    foot_calls = {"n": 0}

    def image_embed(img):
        w, h = getattr(img, "size", (768, 1344))
        # FOOTWEAR_CROP is a short wide band near the bottom of the sheet.
        if h < 220 and w > 200:
            foot_calls["n"] += 1
            # Child footwear then parent footwear → cosine 0.
            return list(POS) if foot_calls["n"] % 2 == 1 else list(ORTH)
        return list(POS)

    return ScriptedClipScorer(image_embed=image_embed, text_embed=text_vec, model_id=model_id)


def test_back_prompt_hides_face_and_drops_same_face_lock():
    identity = (
        "1boy, male focus, young adult, short black hair, brown eyes, "
        "navy jacket, white shirt, black pants, white sneakers"
    )
    specs = plan_library_specs(
        [{"id": "hero", "name": "Hero", "gender": "male", "identity_prompt": identity}],
        [{"id": "dock", "name": "Dock", "plate_prompt": "wooden pier, anime location background"}],
        [],
    )
    back = specs["char_hero_back"]
    side = specs["char_hero_side"]
    assert "same face" not in back["prompt"].lower()
    assert DERIVE_FACE_LOCK not in back["prompt"]
    assert "face completely hidden" in back["prompt"]
    assert "face completely hidden" in BACK_LOOK
    assert "same face" not in BACK_LOOK.lower()
    assert DERIVE_FACE_LOCK in side["prompt"]
    for term in ("cropped feet", "half body", "bust"):
        assert term in SHEET_NEGATIVE


def test_establishing_plate_look_splits_from_detail():
    assert "wide establishing shot" in ESTABLISHING_PLATE_LOOK
    assert "architectural perspective" in ESTABLISHING_PLATE_LOOK
    assert ESTABLISHING_PLATE_LOOK != "anime location background, empty establishing shot"
    assert "empty establishing shot" not in DETAIL_PLATE_LOOK
    specs = plan_library_specs(
        [],
        [{"id": "harbor", "name": "Harbor", "plate_prompt": "wooden harbor pier at dusk"}],
        [],
        interiors=[
            {
                "id": "int_dock",
                "name": "Dock",
                "scene_id": "dock",
                "plate_prompt": "wooden dock interior close, wet boards",
            }
        ],
    )
    harbor = specs["plate_harbor"]["prompt"].lower()
    assert "wide establishing shot" in harbor
    assert "architectural perspective" in harbor
    assert "empty establishing shot" not in harbor
    assert "empty establishing shot" not in specs["plate_dock"]["prompt"].lower()


def test_scene_plate_payload_has_no_parent_or_reference():
    payload = build_scene_plate_payload("harbor pier, wide establishing shot, no characters")
    assert "_parent_png" not in payload
    assert "image" not in payload
    assert "reference_images" not in payload
    client = KolorsClient(
        ["k"],
        opener=lambda req: {"bytes": synthetic_still_png(1344, 768, tag="p", placeholder=True)},
        live=False,
    )
    built, _png = client.generate_scene_plate("harbor pier", 1, None, "fiction")
    assert "_parent_png" not in built or built.get("_parent_png") in (None, "")
    assert "image" not in built
    assert "reference_images" not in built
    # generate() must also reject a plate that somehow carries a parent blob.
    with pytest.raises(Exception):
        client.generate(
            {
                **build_scene_plate_payload("x"),
                "_kind": "scene_plate",
                "_parent_png": b"not-empty",
            }
        )


def test_apply_region_identity_gates_footwear_and_no_face():
    reasons: list[str] = []
    scores = {
        "torso": 0.9,
        "legs": 0.9,
        "clothing": 0.9,
        "footwear": 0.05,
    }
    apply_region_identity_gates(scores, view="side", reasons=reasons)
    assert "footwear_mismatch" in reasons
    assert "clothing_mismatch" not in reasons
    assert scores["footwear_identity"] < FOOTWEAR_IDENTITY_MIN
    assert scores["clothing_identity"] >= CLOTHING_IDENTITY_MIN

    back_reasons: list[str] = []
    # Face is intentionally absent; low face must not be consulted.
    back_scores = {"torso": 0.9, "legs": 0.9, "clothing": 0.9, "footwear": 0.9, "face": 0.01}
    apply_region_identity_gates(back_scores, view="back", reasons=back_reasons)
    assert back_reasons == []
    assert "face" not in "".join(back_reasons)


def test_region_clip_footwear_mismatch_fails_even_if_whole_image_matches():
    child = _char_png("child-side")
    parent = _char_png("parent-front")
    scorer = _footwear_mismatch_scorer(model_id="test-clip")
    result = score_still(
        child,
        kind="character_view_derive",
        prompt="clothed full-body anime character standing, from side, strict left side profile",
        parent_image=parent,
        scorer=scorer,
        allow_placeholder=True,
        view="side",
    )
    assert result.verdict == "fail"
    assert "footwear_mismatch" in result.reasons
    assert result.scores.get("identity", 0) >= IDENTITY_MIN
    assert result.scores.get("clothing_identity", 0) >= CLOTHING_IDENTITY_MIN
    assert result.scores.get("footwear_identity", 1.0) < FOOTWEAR_IDENTITY_MIN

    # Back view: same footwear gate.
    scorer_back = _footwear_mismatch_scorer(model_id="test-clip-back")
    back = score_still(
        _char_png("child-back"),
        kind="character_view_derive",
        prompt="clothed full-body anime character standing, from behind, strict rear view",
        parent_image=parent,
        scorer=scorer_back,
        allow_placeholder=True,
        view="back",
    )
    assert back.verdict == "fail"
    assert "footwear_mismatch" in back.reasons
    assert back.scores.get("identity", 0) >= IDENTITY_MIN


def test_back_does_not_fail_on_low_face_region_score():
    child = _char_png("child-back")
    parent = _char_png("parent-front")

    def image_embed(_img):
        return list(POS)

    scorer = ScriptedClipScorer(image_embed=image_embed, text_embed=text_vec, model_id="test-clip")
    result = score_still(
        child,
        kind="character_view_derive",
        prompt="clothed full-body anime character standing, from behind, strict rear view",
        parent_image=parent,
        scorer=scorer,
        allow_placeholder=True,
        view="back",
    )
    assert result.verdict == "pass"
    assert "identity_mismatch" not in result.reasons
    assert all("face" not in reason for reason in result.reasons)
    assert "region_face" not in result.scores


def test_qwen_face_visible_fails_back_view():
    result = evaluate_master_vision_fields(
        "back",
        {
            "rear_view": True,
            "face_visible": True,
            "head_rotated": False,
            "full_body": True,
            "both_feet_visible": True,
            "single_subject": True,
        },
    )
    assert result.passed is False
    assert "face_visible" in result.reasons


def test_qwen_transport_error_does_not_pass():
    blob = _char_png("vision-transport")

    def boom(_req: Request) -> dict:
        raise RuntimeError("siliconflow down")

    client = LlmClient(deepseek_key="", siliconflow_keys=["sf-key"], opener=boom)
    result = master_vision_qc(blob, view="back", kind="character_view_derive", client=client)
    assert result.passed is False
    assert "vision_transport_error" in result.reasons


def test_qwen_transport_error_when_no_client(monkeypatch):
    blob = _char_png("vision-no-client")
    monkeypatch.setattr("anime_factory.llm.client_from_env", lambda opener=None: None)
    result = master_vision_qc(blob, view="back", kind="character_view_derive", client=None)
    assert result.passed is False
    assert "vision_transport_error" in result.reasons


def test_qwen_http_mock_passes_back():
    blob = _char_png("vision-ok")

    def opener(req: Request) -> dict:
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "Qwen/Qwen3.5-4B"
        assert payload.get("temperature") == 0
        content = payload["messages"][1]["content"]
        assert any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)
        body = {
            "rear_view": True,
            "face_visible": False,
            "head_rotated": False,
            "full_body": True,
            "both_feet_visible": True,
            "single_subject": True,
        }
        return {"choices": [{"message": {"content": json.dumps(body)}}]}

    client = LlmClient(deepseek_key="", siliconflow_keys=["sf-key"], opener=opener)
    result = master_vision_qc(blob, view="back", kind="character_view_derive", client=client)
    assert result.passed is True


def test_keyframe_payload_carries_locked_plate(tmp_path, monkeypatch):
    from anime_factory import keyframe as kf
    from anime_factory.asset_lock import lock_after_qc, save_assets_index
    from anime_factory.db import migrate, open_db
    from anime_factory.visual_qc import VisualQcResult, current_lineage
    from tests.clip_fakes import sized_placeholder_opener

    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    plate_dir = tmp_path / "assets" / "scenes" / "harbor"
    plate_dir.mkdir(parents=True)
    plate_blob = synthetic_still_png(1344, 768, tag="locked-plate", placeholder=True)
    plate_path = plate_dir / "plate_base.png"
    plate_path.write_bytes(plate_blob)
    save_assets_index(tmp_path, {"scenes": {"harbor": {"selected": "plate_base.png", "history": ["plate_base.png"]}}})
    lock_after_qc(
        tmp_path,
        scene_id="harbor",
        filename="plate_base.png",
        qc=VisualQcResult(verdict="pass", lineage=current_lineage(), kind="scene_plate"),
        set_selected=True,
    )

    client = KolorsClient(["k"], opener=sized_placeholder_opener("kf"), live=False)
    monkeypatch.setattr(kf, "score_still", lambda *a, **k: VisualQcResult(verdict="pass", kind="keyframe"))
    monkeypatch.setattr(kf, "require_clip_for_client", lambda _c: False)

    segment = {
        "id": "s001",
        "chain_index": 0,
        "location_id": "harbor",
        "first_frame_prompt": "hero on the wet pier at dusk, medium shot",
        "h3_mode": "fl2va",
        "cuts": [{"seq": 1, "frame_prompt": "hero steps forward on wet pier"}],
    }
    kf.ensure_keyframe(
        conn,
        "story-x",
        "EP001",
        segment,
        {},
        {"items": [], "scenes": {"harbor": {"selected": "plate_base.png"}}},
        {},
        client,
        None,
        "fiction",
        tmp_path,
    )
    keyframe_payloads = [p for p in client.last_payloads if p.get("_kind") == "keyframe"]
    assert keyframe_payloads
    assert keyframe_payloads[0].get("_parent_png") == plate_blob
    assert keyframe_payloads[0].get("_parent_kind") == "scene_plate"


def test_lease_env_forwards_siliconflow_keys():
    src = Path(__file__).resolve().parents[1] / "python" / "anime_factory" / "produce.py"
    text = src.read_text(encoding="utf-8")
    assert "SILICONFLOW_API_KEY" in text
    assert "SILICONFLOW_API_KEYS" in text
    assert "lease_env=env" in text


def test_ipadapter_keyframe_weight_separate_from_view_derive():
    from gpu_worker.stills import (
        KEYFRAME_PLATE_IPADAPTER_WEIGHT,
        VIEW_DERIVE_IPADAPTER_WEIGHT,
        fill_still_workflow,
    )

    assert KEYFRAME_PLATE_IPADAPTER_WEIGHT == pytest.approx(0.35)
    assert VIEW_DERIVE_IPADAPTER_WEIGHT == pytest.approx(0.75)
    template = {
        "prompt": {
            "1": {"class_type": "IPAdapterAdvanced", "inputs": {"weight": 1.0, "end_at": 1.0, "model": ["2", 0]}},
            "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
            "3": {
                "class_type": "LoadImage",
                "inputs": {"image": ""},
                "_meta": {"role": "reference_image"},
            },
        }
    }
    graph = fill_still_workflow(
        template,
        {
            "prompt": "p",
            "negative": "n",
            "width": 1344,
            "height": 768,
            "seed": 1,
            "steps": 28,
            "cfg": 5,
            "sampler": "euler_ancestral",
            "scheduler": "normal",
            "ckpt": "x.safetensors",
            "reference_image": "af_ref.png",
            "kind": "keyframe",
            "parent_kind": "scene_plate",
        },
    )
    node = graph["prompt"]["1"]["inputs"]
    assert node["weight"] == pytest.approx(0.35)

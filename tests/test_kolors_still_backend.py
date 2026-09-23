"""STILL_BACKEND selects local Kolors or Animagine. Scene plates stay reference-free."""

from __future__ import annotations

import pytest

from anime_factory.design import (
    KOLORS_CHARACTER_FRAMING,
    KOLORS_CHARACTER_NEGATIVE_LEAD,
    _kind_negative,
    build_scene_plate_payload,
    style_negative,
    style_prompt,
)
from anime_factory.models import (
    IMAGE_CKPT,
    IMAGE_MODEL,
    KOLORS_MODEL,
    KOLORS_UNET_FILE,
    StillBackendError,
    still_backend,
    still_model_id,
    still_unet_file,
    still_workflow_name,
)
from gpu_worker.h3 import load_workflow
from gpu_worker.stills import fill_still_workflow, generate_still, still_payload_to_prompt
from gpu_worker.weights import still_weight_files


def test_default_backend_is_kolors(monkeypatch):
    monkeypatch.delenv("STILL_BACKEND", raising=False)
    assert still_backend() == "kolors"
    assert still_model_id() == KOLORS_MODEL
    assert still_workflow_name() == "kolors_t2i"
    assert still_unet_file() == KOLORS_UNET_FILE


def test_animagine_backend_keeps_sdxl_checkpoint(monkeypatch):
    monkeypatch.setenv("STILL_BACKEND", "animagine")
    assert still_model_id() == IMAGE_MODEL
    assert still_workflow_name() == "anime_t2i"
    assert still_unet_file() == IMAGE_CKPT
    dests = {item["dest"] for item in still_weight_files()}
    assert "models/checkpoints/animagine-xl-4.0.safetensors" in dests
    assert "models/diffusion_models/kolors_unet_fp16.safetensors" not in dests


def test_invalid_still_backend_fails_closed(monkeypatch):
    monkeypatch.setenv("STILL_BACKEND", "flux")
    with pytest.raises(StillBackendError):
        still_backend()


def test_kolors_prompt_skips_animagine_quality_tags(monkeypatch):
    monkeypatch.setenv("STILL_BACKEND", "kolors")
    prompt = style_prompt("night cliff road", kind="scene_plate")
    assert "masterpiece" not in prompt
    assert "absurdres" not in prompt
    assert "1boy" not in prompt
    assert "Makoto Shinkai" not in prompt


def test_scene_plate_payload_has_no_reference_on_either_backend(monkeypatch):
    for backend, model in (("kolors", KOLORS_MODEL), ("animagine", IMAGE_MODEL)):
        monkeypatch.setenv("STILL_BACKEND", backend)
        payload = build_scene_plate_payload("cliffside mountain road, empty")
        assert payload["model"] == model
        assert "reference_image" not in payload
        assert "image" not in payload
        assert "_parent_png" not in payload


def test_kolors_workflow_drops_ipadapter_without_a_reference(monkeypatch):
    monkeypatch.setenv("STILL_BACKEND", "kolors")
    spec = still_payload_to_prompt(
        {"prompt": "empty mountain road", "negative_prompt": "text", "image_size": "1344x768", "seed": 3}
    )
    prompt = fill_still_workflow(load_workflow("kolors_t2i"), spec)["prompt"]
    classes = {node.get("class_type") for node in prompt.values()}
    assert "MZ_KolorsUNETLoaderV2" in classes
    assert "MZ_ChatGLM3Loader" in classes
    assert "MZ_IPAdapterAdvancedKolors" not in classes
    assert "LoadImage" not in classes
    unet_id = next(nid for nid, node in prompt.items() if node.get("class_type") == "MZ_KolorsUNETLoaderV2")
    sampler = next(node for node in prompt.values() if node.get("class_type") == "KSampler")
    assert sampler["inputs"]["model"] == [unet_id, 0]
    assert prompt[unet_id]["inputs"]["unet_name"] == KOLORS_UNET_FILE
    texts = [
        node["inputs"].get("text")
        for node in prompt.values()
        if node.get("class_type") == "MZ_ChatGLM3_Advance_V2"
    ]
    assert any("mountain" in (text or "") for text in texts)
    assert any(text == "text" for text in texts)


def test_kolors_view_derive_keeps_ipadapter(monkeypatch):
    monkeypatch.setenv("STILL_BACKEND", "kolors")
    spec = still_payload_to_prompt(
        {
            "prompt": "side view",
            "image_size": "832x1216",
            "seed": 4,
            "_kind": "character_view_derive",
            "kind": "character_view_derive",
        }
    )
    spec["reference_image"] = "af_side.png"
    prompt = fill_still_workflow(load_workflow("kolors_t2i"), spec)["prompt"]
    ipadapter = next(node for node in prompt.values() if node.get("class_type") == "MZ_IPAdapterAdvancedKolors")
    assert ipadapter["inputs"]["weight"] == 0.75
    loader = next(node for node in prompt.values() if node.get("class_type") == "LoadImage")
    assert loader["inputs"]["image"] == "af_side.png"


def test_kolors_character_prompt_leads_with_full_length_framing(monkeypatch):
    monkeypatch.setenv("STILL_BACKEND", "kolors")
    identity = "1boy, young adult, short black hair, brown eyes, black leather jacket, red shirt"
    prompt = style_prompt(
        f"{identity}, solo, front view, looking at viewer, full body, standing",
        kind="character_sheet",
        gender_tag="1boy",
    )
    lowered = prompt.lower()
    assert lowered.startswith("full body shot")
    assert "both feet fully visible" in lowered
    assert "character design" not in lowered
    assert lowered.index("both feet fully visible") < lowered.index("black leather")
    assert KOLORS_CHARACTER_FRAMING.split(",")[0].lower() in lowered


def test_kolors_character_negative_puts_bust_bans_first(monkeypatch):
    monkeypatch.setenv("STILL_BACKEND", "kolors")
    negative = style_negative(
        extra=_kind_negative("character_sheet", "1boy, black pants"),
        kind="character_sheet",
    )
    lowered = negative.lower()
    assert lowered.startswith("close-up")
    assert "cropped feet" in lowered
    assert lowered.index("bust") < lowered.index("photorealistic")
    assert KOLORS_CHARACTER_NEGATIVE_LEAD.split(",")[0] in lowered


def test_animagine_character_prompt_keeps_booru_sheet(monkeypatch):
    monkeypatch.setenv("STILL_BACKEND", "animagine")
    prompt = style_prompt(
        "1boy, young adult, short black hair, brown eyes, grey hoodie",
        kind="character_sheet",
        gender_tag="1boy",
    ).lower()
    assert "both feet fully visible" not in prompt
    assert "original anime character design" in prompt
    assert "masterpiece" in prompt
    negative = style_negative(
        extra=_kind_negative("character_sheet", "1boy, grey hoodie"),
        kind="character_sheet",
    ).lower()
    assert negative.startswith("photorealistic")


def test_kolors_scene_plate_does_not_take_character_framing(monkeypatch):
    monkeypatch.setenv("STILL_BACKEND", "kolors")
    prompt = style_prompt("empty mountain road", kind="scene_plate").lower()
    assert "both feet fully visible" not in prompt
    assert "full body shot" not in prompt


def test_scene_plate_refuses_parent_bytes():
    with pytest.raises(RuntimeError, match="reference-image bleed"):
        generate_still(
            {"_kind": "scene_plate", "prompt": "road", "image_size": "1344x768", "_parent_png": b"not-a-plate"},
            router=object(),
        )

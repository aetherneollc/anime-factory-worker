"""fal stills: scene plates stay unreferenced, side views require master.png."""

from __future__ import annotations

import pytest

from anime_factory.design import KolorsClient, synthetic_still_png
from anime_factory.fal_models import edit_fal_image, generate_fal_image_t2i
from anime_factory.models import FIXED_NEGATIVE, STYLE_PREFIX_CHARACTER
from anime_factory.qwen_image_21 import MasterReferenceError
from gpu_worker.stills import generate_still


def test_scene_plate_refuses_a_parent_on_the_fal_backend(monkeypatch):
    monkeypatch.setenv("IMAGE_BACKEND", "fal")
    monkeypatch.setenv("FAL_DRY_RUN", "1")

    def _boom(*_a, **_k):
        raise AssertionError("scene plate must not call fal")

    monkeypatch.setattr("anime_factory.fal_models.generate_fal_image_t2i", _boom)
    with pytest.raises(RuntimeError, match="reference-image bleed"):
        generate_still(
            {
                "_kind": "scene_plate",
                "prompt": "empty night street",
                "image_size": "1344x768",
                "_parent_png": b"x" * 20_000,
            }
        )


def test_fal_scene_plate_payload_rejects_reference_keys():
    client = KolorsClient(["k"], live=True, min_interval_s=0)
    with pytest.raises(Exception, match="must not include"):
        client.generate(
            {
                "prompt": "empty alley",
                "image_size": "1344x768",
                "_kind": "scene_plate",
                "image": "data:image/png;base64,aaaa",
            }
        )


def test_side_view_requires_master_png(monkeypatch):
    monkeypatch.setenv("FAL_DRY_RUN", "1")
    parent = synthetic_still_png(64, 64, tag="parent", placeholder=False)
    with pytest.raises(MasterReferenceError):
        edit_fal_image(
            "side view",
            [parent],
            768,
            1344,
            master_path="assets/characters/hero/sheet_side.png",
            view="side",
            dry_run=True,
        )
    png = edit_fal_image(
        "side view",
        [parent],
        768,
        1344,
        master_path="assets/characters/hero/master.png",
        view="side",
        dry_run=True,
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_fal_still_route_enforces_side_master(monkeypatch):
    monkeypatch.setenv("IMAGE_BACKEND", "fal")
    monkeypatch.setenv("FAL_DRY_RUN", "1")
    parent = synthetic_still_png(768, 1344, tag="master", placeholder=False)
    with pytest.raises(MasterReferenceError):
        generate_still(
            {
                "_kind": "character_view_derive",
                "_view": "side",
                "_master_path": "assets/characters/hero/sheet_front.png",
                "_parent_png": parent,
                "prompt": "side profile",
                "image_size": "768x1344",
                "_styled": True,
            }
        )
    blob = generate_still(
        {
            "_kind": "character_view_derive",
            "_view": "side",
            "_master_path": "assets/characters/hero/master.png",
            "_parent_png": parent,
            "prompt": "side profile",
            "image_size": "768x1344",
            "_styled": True,
        }
    )
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"


def test_fal_character_keeps_style_and_negative_gates(monkeypatch):
    monkeypatch.setenv("CHARACTER_STILL_BACKEND", "fal")
    seen: dict[str, str] = {}

    def capture(prompt, width, height, **kwargs):
        seen["prompt"] = prompt
        seen["negative"] = kwargs.get("negative_prompt") or ""
        return synthetic_still_png(width, height, tag="fal-char", placeholder=False)

    monkeypatch.setattr("anime_factory.fal_models.generate_fal_image_t2i", capture)
    client = KolorsClient(["k"], live=True, min_interval_s=0)
    blob = client.generate(
        {
            "prompt": "1boy, short black hair",
            "negative_prompt": "",
            "image_size": "1344x768",
            "seed": 3,
            "_kind": "character_sheet",
            "_label": "char_hero",
        }
    )
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    assert STYLE_PREFIX_CHARACTER.lower() in seen["prompt"].lower()
    assert "photorealistic" in seen["negative"].lower()
    assert FIXED_NEGATIVE.split(",")[0] in seen["negative"]
    with pytest.raises(MasterReferenceError):
        client.generate(
            {
                "prompt": "side",
                "image_size": "768x1344",
                "_kind": "character_view_derive",
                "_view": "side",
                "_master_path": "assets/characters/hero/sheet_side.png",
                "_parent_png": synthetic_still_png(64, 64, tag="p", placeholder=False),
            }
        )


def test_qwen_image_backend_still_selects_local_qwen(monkeypatch):
    monkeypatch.setenv("IMAGE_BACKEND", "qwen_image_21")
    monkeypatch.setenv("QWEN_IMAGE_21_DRY_RUN", "1")
    from anime_factory.backends.image import ImageGenerateRequest, get_image_backend, select_image_backend

    assert select_image_backend() == "qwen_image_21"
    backend = get_image_backend()
    png = backend.generate(
        ImageGenerateRequest(
            prompt="sheet",
            width=64,
            height=64,
            seed=1,
        )
    )
    assert png.backend == "qwen_image_21"
    assert png.png[:8] == b"\x89PNG\r\n\x1a\n"
    dry = generate_fal_image_t2i("x", 64, 64, dry_run=True)
    assert dry[:8] == b"\x89PNG\r\n\x1a\n"

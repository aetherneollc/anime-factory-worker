"""Production stills: Qwen-Image-2.1, H3 video, shinkai lock."""

from __future__ import annotations

import pytest

from anime_factory.backends.image import select_image_backend
from anime_factory.design import (
    APPEARANCE_LOCK_SLOTS,
    appearance_lock_tokens,
    apply_appearance_lock,
)
from anime_factory.models import IMAGE_BACKENDS, VIDEO_BACKENDS
from anime_factory.qwen_image_21 import (
    MasterReferenceError,
    assert_side_master_reference,
    edit_qwen_image_21,
    generate_qwen_image_21,
)
from anime_factory.video_backend import ProductionBackendError, normalize_video_backend, select_video_backend


def test_production_enums_reject_skyreels_and_klein_as_defaults(monkeypatch):
    monkeypatch.delenv("IMAGE_BACKEND", raising=False)
    monkeypatch.delenv("VIDEO_BACKEND", raising=False)
    monkeypatch.delenv("AF_VIDEO_BACKEND", raising=False)
    assert IMAGE_BACKENDS == ("qwen_image_21",)
    assert VIDEO_BACKENDS == ("h3",)
    assert select_image_backend() == "qwen_image_21"
    assert select_video_backend() == "h3"
    assert normalize_video_backend(None) == "h3"
    for name in ("skyreels", "skyreels_v3_r2v", "longlive"):
        with pytest.raises(ProductionBackendError):
            normalize_video_backend(name)
    monkeypatch.setenv("IMAGE_BACKEND", "flux2_klein4b")
    with pytest.raises(Exception):
        select_image_backend()


def test_appearance_lock_includes_pants_and_shoes():
    identity = "1boy, adult, short black hair, brown eyes, navy jacket, black pants, white shoes"
    tokens = appearance_lock_tokens(identity)
    assert set(APPEARANCE_LOCK_SLOTS) <= set(tokens)
    assert "hair" in tokens["hair"]
    assert "jacket" in tokens["top"]
    assert "pants" in tokens["pants"]
    assert "shoes" in tokens["shoes"]
    retry = apply_appearance_lock("same face, new khaki pants, bare feet", tokens)
    assert tokens["pants"] in retry
    assert tokens["shoes"] in retry
    assert "keep the same pants" in retry
    assert "keep the same shoes" in retry


def test_side_view_requires_master_reference():
    with pytest.raises(MasterReferenceError):
        assert_side_master_reference("assets/characters/hero/sheet_front.png")
    with pytest.raises(MasterReferenceError):
        assert_side_master_reference("")
    assert assert_side_master_reference("assets/characters/hero/master.png").endswith("master.png")
    with pytest.raises(MasterReferenceError):
        edit_qwen_image_21(
            "side view",
            [b"\x89PNG\r\n\x1a\nref"],
            master_path="assets/characters/hero/sheet_side.png",
            view="side",
            dry_run=True,
        )


def test_dry_run_does_not_download_weights(monkeypatch):
    monkeypatch.setenv("QWEN_IMAGE_21_DRY_RUN", "1")

    def _boom(*_a, **_k):
        raise AssertionError("weights download")

    import anime_factory.qwen_image_21 as qwen

    monkeypatch.setattr(qwen, "_load_pipe", _boom)
    png = generate_qwen_image_21("sheet", 64, 64, seed=1)
    edited = edit_qwen_image_21(
        "side",
        [png],
        64,
        64,
        seed=2,
        master_path="assets/characters/hero/master.png",
        view="side",
    )
    assert png.startswith(b"\x89PNG")
    assert edited.startswith(b"\x89PNG")
    assert b"qwen21-dry" in png

"""CHARACTER_STILL_BACKEND=qwen routes character sheets to DashScope."""

from __future__ import annotations

import json
from urllib.request import Request

import pytest

from anime_factory.design import (
    QWEN_CHARACTER_FRAMING,
    KolorsClient,
    style_negative,
    style_prompt,
    synthetic_still_png,
)
from anime_factory.models import (
    CHARACTER_STILL_BACKENDS,
    DEFAULT_CHARACTER_STILL_BACKEND,
    QWEN_IMAGE_MODEL,
    CharacterStillBackendError,
    character_still_backend,
    qwen_image_model,
)
from anime_factory.qwen_image import image_size_for_dashscope


@pytest.mark.qwen_character_default
def test_default_character_backend_is_qwen(monkeypatch):
    monkeypatch.delenv("CHARACTER_STILL_BACKEND", raising=False)
    monkeypatch.setenv("CHARACTER_STILL_BACKEND", "")
    from anime_factory.models import (
        DEFAULT_CHARACTER_STILL_BACKEND,
        character_still_backend,
    )

    assert DEFAULT_CHARACTER_STILL_BACKEND == "qwen"
    assert character_still_backend() == "qwen"
    assert character_still_backend() != "hunyuan"


def test_invalid_character_backend_fails_closed(monkeypatch):
    monkeypatch.setenv("CHARACTER_STILL_BACKEND", "flux")
    with pytest.raises(CharacterStillBackendError):
        character_still_backend()
    assert "qwen" in CHARACTER_STILL_BACKENDS


def test_extract_image_bytes_from_https_image_field(monkeypatch):
    from anime_factory import qwen_image as qi

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return synthetic_still_png(64, 64, tag="url", placeholder=False)

    monkeypatch.setattr(qi.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    blob = qi.extract_image_bytes(
        {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [{"image": "https://example.com/a.png"}],
                        }
                    }
                ]
            }
        }
    )
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"


def test_dashscope_size_uses_star():
    assert image_size_for_dashscope("768x1344") == "768*1344"
    assert image_size_for_dashscope("1024*1536") == "1024*1536"


@pytest.mark.qwen_character_default
def test_qwen_character_prompt_uses_shinkai_sheet(monkeypatch):
    monkeypatch.setenv("CHARACTER_STILL_BACKEND", "qwen")
    text = style_prompt(
        "1boy, black leather jacket, red shirt",
        kind="character_sheet",
    )
    lowered = text.lower()
    assert "shinkai-like animated film frame" in lowered
    assert "even soft light" in lowered
    assert "16:9 character design sheet" in lowered
    assert "your name" not in lowered
    assert "weathering with you" not in lowered


@pytest.mark.qwen_character_default
def test_qwen_character_generate_uses_local_pipeline_not_dashscope(monkeypatch):
    monkeypatch.setenv("CHARACTER_STILL_BACKEND", "qwen")
    monkeypatch.setenv("QWEN_IMAGE_21_DRY_RUN", "1")
    calls: list[Request] = []

    def opener(req: Request) -> dict:
        calls.append(req)
        return {"bytes": synthetic_still_png(1344, 768, tag="dashscope", placeholder=False)}

    gpu_calls: list[dict] = []

    def gpu_generate(payload: dict) -> bytes:
        gpu_calls.append(payload)
        return synthetic_still_png(1344, 768, tag="gpu", placeholder=False)

    client = KolorsClient(
        ["k"],
        live=False,
        opener=opener,
        gpu_generate=gpu_generate,
        min_interval_s=0,
    )
    blob = client.generate(
        {
            "prompt": style_prompt("1boy, short black hair", kind="character_sheet"),
            "negative_prompt": style_negative(kind="character_sheet"),
            "image_size": "1344x768",
            "seed": 42,
            "_kind": "character_sheet",
            "_label": "char_test",
        }
    )
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    assert calls == []
    assert gpu_calls == []
    from anime_factory.models import QWEN_IMAGE_21_MODEL_ID

    assert client.last_payloads[-1]["model"] == QWEN_IMAGE_21_MODEL_ID
    assert "qwen-image-2.0-pro" not in str(client.last_payloads[-1]["model"])


@pytest.mark.qwen_character_default
def test_scene_plate_still_uses_gpu_when_qwen_is_character_default(monkeypatch):
    monkeypatch.delenv("CHARACTER_STILL_BACKEND", raising=False)
    gpu_calls: list[dict] = []

    def gpu_generate(payload: dict) -> bytes:
        gpu_calls.append(payload)
        return synthetic_still_png(1344, 768, tag="plate", placeholder=False)

    client = KolorsClient(["k"], live=True, gpu_generate=gpu_generate, min_interval_s=0)
    blob = client.generate(
        {
            "prompt": "empty alley at dusk",
            "negative_prompt": "people",
            "image_size": "1344x768",
            "seed": 7,
            "_kind": "scene_plate",
            "_label": "loc_alley",
        }
    )
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(gpu_calls) == 1
    from anime_factory.models import character_still_backend

    assert character_still_backend() == "qwen"
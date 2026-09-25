"""CHARACTER_STILL_BACKEND=hunyuan is the production character default."""

from __future__ import annotations

import json
from urllib.request import Request

import pytest

from anime_factory.design import (
    HUNYUAN_CHARACTER_FRAMING,
    KolorsClient,
    style_negative,
    style_prompt,
    synthetic_still_png,
)
from anime_factory.models import HUNYUAN_IMAGE_MODEL


@pytest.mark.qwen_character_default
def test_hunyuan_character_prompt_and_generate_with_reference(monkeypatch):
    monkeypatch.setenv("CHARACTER_STILL_BACKEND", "hunyuan")
    text = style_prompt("1boy, black leather jacket, red shirt, black pants", kind="character_sheet")
    assert "full body shot" in text.lower()
    assert HUNYUAN_CHARACTER_FRAMING.split(",")[0].lower() in text.lower()
    assert "missing pants" in style_negative(kind="character_sheet")

    calls: list[Request] = []

    def opener(req: Request) -> dict:
        calls.append(req)
        body = json.loads((req.data or b"{}").decode("utf-8"))
        assert body["model"] == HUNYUAN_IMAGE_MODEL
        assert body.get("size") == "768x1344"
        assert isinstance(body.get("images"), list) and body["images"][0].startswith("data:image/png")
        return {"bytes": synthetic_still_png(768, 1344, tag="hy-ref", placeholder=False)}

    parent = synthetic_still_png(768, 1344, tag="parent", placeholder=False)
    client = KolorsClient(["k"], live=True, opener=opener, min_interval_s=0)
    blob = client.generate(
        {
            "prompt": text,
            "negative_prompt": style_negative(kind="character_sheet"),
            "image_size": "768x1344",
            "seed": 7,
            "_kind": "character_view_derive",
            "_label": "char_side",
            "_parent_png": parent,
        }
    )
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(calls) == 1
    assert "tokenhub" in calls[0].full_url or "hunyuan" in calls[0].full_url

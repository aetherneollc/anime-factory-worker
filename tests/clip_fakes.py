"""Injectable CLIP doubles for tests. Not imported by production generate paths."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request

from anime_factory.design import synthetic_still_png
from anime_factory.visual_qc import (
    CHARACTER_NEGATIVE,
    CHARACTER_POSITIVE,
    SCENE_NEGATIVE,
    SCENE_POSITIVE,
    ScriptedClipScorer,
    VisualQcResult,
    current_lineage,
)

POS = (1.0, 0.0)
NEG = (0.0, 1.0)
_NEG_HINTS = (
    SCENE_NEGATIVE
    + CHARACTER_NEGATIVE
    + (
        "ceramic",
        "dinnerware",
        "dish",
        "bowl of food",
        "nude",
        "floating heads",
        "abstract costume",
        "beige canvas",
    )
)


def text_vec(text: str) -> list[float]:
    low = str(text or "").lower()
    if any(hint.lower() in low for hint in _NEG_HINTS):
        return list(NEG)
    return list(POS)


def passing_scorer() -> ScriptedClipScorer:
    return ScriptedClipScorer(image_embed=POS, text_embed=text_vec, model_id="scripted:pass")


def view_fail_scorer(view: str) -> ScriptedClipScorer:
    """Pass identity stills; fail the named character view via prompt mismatch."""
    token = "strict left side profile" if view == "side" else "strict rear view"

    def embed_text(text: str) -> list[float]:
        if token in str(text or "").lower():
            return list(NEG)
        return text_vec(text)

    return ScriptedClipScorer(image_embed=POS, text_embed=embed_text, model_id=f"scripted:fail-{view}")


def dish_scorer() -> ScriptedClipScorer:
    return ScriptedClipScorer(image_embed=NEG, text_embed=text_vec, model_id="scripted:dish")


def bad_character_scorer() -> ScriptedClipScorer:
    return ScriptedClipScorer(image_embed=NEG, text_embed=text_vec, model_id="scripted:bad-character")


def pass_qc(**kwargs: Any) -> VisualQcResult:
    return VisualQcResult(
        verdict="pass",
        reasons=[],
        scores={"entropy": 4.2, "clip_margin": 0.2, "alignment": 0.4},
        scorer="scripted:pass",
        lineage=current_lineage(),
        **kwargs,
    )


def fail_qc(*reasons: str, **kwargs: Any) -> VisualQcResult:
    return VisualQcResult(
        verdict="fail",
        reasons=list(reasons),
        scores={"entropy": 4.2, "clip_margin": 0.0},
        scorer="scripted:fail",
        lineage=current_lineage(),
        **kwargs,
    )


def sized_placeholder_opener(tag: str = "qc"):
    def opener(req: Request) -> dict:
        payload: dict[str, Any] = {}
        raw = getattr(req, "data", None) or b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        size = str(payload.get("image_size") or "1344x768")
        width, height = 1344, 768
        if "x" in size.lower():
            a, b = size.lower().split("x", 1)
            try:
                width, height = int(a), int(b)
            except ValueError:
                pass
        seed = payload.get("seed")
        return {"bytes": synthetic_still_png(width, height, tag=f"{tag}:{seed}", placeholder=True)}

    return opener

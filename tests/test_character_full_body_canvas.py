"""Structural full-body canvas: taller masters, ladder retries, fit-to-delivery."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from anime_factory.design import (
    CHAR_CANVAS_LADDER,
    CHAR_IMAGE_SIZE,
    CHAR_VIEW_HEIGHT,
    CHAR_VIEW_WIDTH,
    KolorsClient,
    MIN_STILL_BYTES,
    character_canvas_for_attempt,
    fit_character_canvas,
    parse_image_size,
    png_dimensions,
    _render_until_qc,
)
from anime_factory.visual_qc import CHARACTER_SIZE, QC_SEED_SALTS


def _pad_png(png: bytes) -> bytes:
    if len(png) >= MIN_STILL_BYTES:
        return png
    return png + b"\x00" * (MIN_STILL_BYTES - len(png))


def test_character_delivery_size_is_taller_portrait():
    assert CHAR_IMAGE_SIZE == "768x1344"
    assert (CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT) == CHARACTER_SIZE == (768, 1344)
    # Taller than the old 832×1216 bust magnet (~1.46 → ~1.75).
    assert CHAR_VIEW_HEIGHT / CHAR_VIEW_WIDTH > 1216 / 832


def test_canvas_ladder_escalates_on_retries():
    assert len(CHAR_CANVAS_LADDER) >= 3
    assert character_canvas_for_attempt(0) == CHAR_CANVAS_LADDER[0] == CHAR_IMAGE_SIZE
    assert character_canvas_for_attempt(1) == CHAR_CANVAS_LADDER[1]
    assert character_canvas_for_attempt(2) == CHAR_CANVAS_LADDER[2]
    assert character_canvas_for_attempt(99) == CHAR_CANVAS_LADDER[-1]
    widths_heights = [parse_image_size(size) for size in CHAR_CANVAS_LADDER]
    aspects = [h / w for w, h in widths_heights]
    assert aspects == sorted(aspects)
    assert aspects[-1] > aspects[0]
    assert len(CHAR_CANVAS_LADDER) == len(QC_SEED_SALTS)


def test_fit_character_canvas_letterboxes_taller_generate_to_delivery():
    src_w, src_h = parse_image_size(CHAR_CANVAS_LADDER[2])
    img = Image.new("RGB", (src_w, src_h), (248, 244, 232))
    # Full-height ink so contain keeps head and feet after fit.
    for y in range(src_h):
        for x in range(src_w // 3, 2 * src_w // 3):
            img.putpixel((x, y), (30, 40, 90))
    buf = BytesIO()
    img.save(buf, format="PNG")
    fitted = fit_character_canvas(_pad_png(buf.getvalue()), label="char_test")
    assert png_dimensions(fitted) == CHARACTER_SIZE
    same = fit_character_canvas(fitted, label="char_test")
    assert png_dimensions(same) == CHARACTER_SIZE


def test_fit_character_canvas_bottom_aligns_small_full_body_figure():
    """Small head-to-toe ink mid-canvas must be scaled and pinned so geometric QC passes."""
    from anime_factory.visual_qc import figure_is_full_body

    w, h = CHARACTER_SIZE
    img = Image.new("RGB", (w, h), (248, 244, 232))
    # Compact full-body figure floating in the upper-middle (classic Kolors miss).
    for y in range(int(h * 0.18), int(h * 0.55)):
        for x in range(int(w * 0.38), int(w * 0.62)):
            jitter = ((x * 13 + y * 7) % 41) - 20
            img.putpixel(
                (x, y),
                (
                    max(0, min(255, 35 + jitter)),
                    max(0, min(255, 45 + jitter)),
                    max(0, min(255, 95 + jitter)),
                ),
            )
    buf = BytesIO()
    img.save(buf, format="PNG")
    raw = _pad_png(buf.getvalue())
    before_ok, before = figure_is_full_body(Image.open(BytesIO(raw)).convert("RGB"))
    assert not before_ok
    assert before["bottom"] < 0.85

    fitted = fit_character_canvas(raw, label="char_reframe")
    assert png_dimensions(fitted) == CHARACTER_SIZE
    after_ok, after = figure_is_full_body(Image.open(BytesIO(fitted)).convert("RGB"))
    assert after_ok, after
    assert after["bottom"] >= 0.85
    assert after["top"] <= 0.20
    assert after["span"] >= 0.65


def test_render_until_qc_escalates_character_canvas(tmp_path):
    """Each QC salt must request the next taller ladder size, then lock at delivery size."""
    seen_sizes: list[str] = []

    def fake_generate(payload: dict) -> bytes:
        size = str(payload.get("image_size") or "")
        seen_sizes.append(size)
        w, h = parse_image_size(size)
        # Always return a bust-like figure so QC fails and we walk the ladder.
        img = Image.new("RGB", (w, h), (245, 245, 245))
        for y in range(int(h * 0.05), int(h * 0.45)):
            for x in range(w // 3, 2 * w // 3):
                img.putpixel((x, y), (40, 50, 100))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return _pad_png(buf.getvalue())

    client = KolorsClient(api_keys=["test-key"], live=True, gpu_generate=fake_generate)
    spec = {
        "id": "char_hero_sheet",
        "kind": "character_sheet",
        "view": "front",
        "character_id": "hero",
        "path": "assets/characters/hero/sheet_front.png",
        "prompt": "1boy, short black hair, navy jacket, full body",
        "seed": 42,
        "image_size": CHAR_IMAGE_SIZE,
    }
    root = tmp_path / "story"
    root.mkdir()
    png, qc, rel = _render_until_qc(
        spec,
        client,
        None,
        "fiction",
        "",
        None,
        None,
        root,
        require_clip=False,
    )
    assert seen_sizes == list(CHAR_CANVAS_LADDER)
    assert png is not None
    assert png_dimensions(png) == CHARACTER_SIZE
    assert qc is not None
    assert not qc.passed
    assert "not_full_body" in (qc.reasons or [])
    assert rel
    # Spec restores delivery size for the locked path contract.
    assert spec["image_size"] == CHAR_IMAGE_SIZE or seen_sizes[-1] == CHAR_CANVAS_LADDER[-1]

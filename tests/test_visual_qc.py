import sys
import types
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

import anime_factory.visual_qc as visual_qc
from anime_factory.design import MIN_STILL_BYTES, synthetic_still_png
from anime_factory.models import STILL_HEIGHT, STILL_WIDTH
from anime_factory.visual_qc import (
    ALIGNMENT_MIN,
    CHARACTER_NEGATIVE,
    CHARACTER_SIZE,
    CLASS_MARGIN_MIN,
    CLIP_ARCH,
    CLIP_MODEL_ID,
    CLIP_PRETRAINED,
    IDENTITY_MIN,
    SCENE_NEGATIVE,
    SCENE_POSITIVE,
    ClipScorerUnavailable,
    OpenClipScorer,
    ScriptedClipScorer,
    cosine,
    dominant_bin_ratio,
    figure_is_full_body,
    identity_attribute_probes,
    load_clip_scorer,
    score_still,
    set_clip_scorer,
    shannon_entropy,
    short_identity_text,
    structure_check,
    view_consistency_check,
)
from tests.clip_fakes import NEG, POS, dish_scorer, passing_scorer, text_vec


def _pad_png(png: bytes) -> bytes:
    if len(png) >= MIN_STILL_BYTES:
        return png
    iend = png.rfind(b"IEND")
    if iend < 8:
        return png + b"\x00" * (MIN_STILL_BYTES - len(png))
    # Insert a tEXt chunk before IEND (4-byte len prefix sits 4 bytes before type).
    insert_at = iend - 4
    pad = b"af-pad\x00" + (b"x" * (MIN_STILL_BYTES - len(png) + 32))
    import struct
    import zlib

    chunk = struct.pack(">I", len(pad)) + b"tEXt" + pad + struct.pack(">I", zlib.crc32(b"tEXt" + pad))
    return png[:insert_at] + chunk + png[insert_at:]


def solid_png(width: int, height: int, color=(248, 244, 232)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return _pad_png(buf.getvalue())


def test_entropy_rejects_flat_beige():
    blob = solid_png(STILL_WIDTH, STILL_HEIGHT)

    class Boom:
        model_id = "scripted:boom"

        def embed_images(self, images):
            raise AssertionError("CLIP must not run on a flat still")

        def embed_texts(self, texts):
            raise AssertionError("CLIP must not run on a flat still")

    result = score_still(blob, kind="scene_plate", prompt="harbor dock", scorer=Boom(), require_clip=False)
    assert result.verdict == "fail"
    assert "low_entropy" in result.reasons or "dominant_bin" in result.reasons
    img = Image.open(BytesIO(blob))
    assert shannon_entropy(img) < 2.5
    assert dominant_bin_ratio(img) > 0.55


def test_entropy_accepts_noisy_png():
    blob = synthetic_still_png(STILL_WIDTH, STILL_HEIGHT, tag="noisy-ok", placeholder=True)
    result = structure_check(blob, kind="scene_plate", allow_placeholder=True)
    assert result.verdict == "pass"
    assert result.scores["entropy"] >= 2.5
    assert result.scores["dominant_bin"] <= 0.55


def test_character_sheet_studio_backdrop_passes_structure():
    from PIL import ImageDraw

    img = Image.new("RGB", CHARACTER_SIZE, (242, 242, 242))
    draw = ImageDraw.Draw(img)
    # Head-to-toe figure on the tall canvas (fractions match FIGURE_* gates).
    w, h = CHARACTER_SIZE
    draw.rectangle(
        [int(w * 0.31), int(h * 0.08), int(w * 0.69), int(h * 0.92)],
        fill=(28, 36, 72),
    )
    draw.rectangle(
        [int(w * 0.36), int(h * 0.12), int(w * 0.64), int(h * 0.32)],
        fill=(210, 180, 140),
    )
    draw.rectangle(
        [int(w * 0.38), int(h * 0.40), int(w * 0.62), int(h * 0.72)],
        fill=(40, 50, 90),
    )
    pix = img.load()
    for x in range(int(w * 0.31), int(w * 0.69), 2):
        for y in range(int(h * 0.08), int(h * 0.92), 2):
            r, g, b = pix[x, y]
            jitter = ((x * 13 + y * 7) % 41) - 20
            pix[x, y] = (
                max(0, min(255, r + jitter)),
                max(0, min(255, g + jitter)),
                max(0, min(255, b + jitter)),
            )
    buf = BytesIO()
    img.save(buf, format="PNG")
    blob = _pad_png(buf.getvalue())
    assert dominant_bin_ratio(img) > 0.55
    result = structure_check(blob, kind="character_sheet", allow_placeholder=True)
    assert result.verdict == "pass"
    assert result.scores["dominant_bin"] <= 0.55
    assert result.scores["entropy"] >= 2.5
    assert result.scores.get("figure_span", 0) >= 0.65


def test_character_turnaround_studio_backdrop_passes_structure():
    from PIL import ImageDraw

    width, height = visual_qc.TURNAROUND_SIZE
    img = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    panel_w = CHARACTER_SIZE[0]
    for panel in range(3):
        left = panel * panel_w + int(panel_w * 0.35)
        head_top = int(height * 0.08)
        draw.ellipse(
            [left + 40, head_top, left + 160, head_top + 120],
            fill=(190, 150, 120),
        )
        for band in range(12):
            top = head_top + 120 + band * int(height * 0.06)
            color = (30 + band * 5, 42 + band * 4, 62 + band * 3)
            draw.rectangle(
                [left, top, left + 200, min(height - 8, top + int(height * 0.06) - 1)],
                fill=color,
            )
    buf = BytesIO()
    img.save(buf, format="PNG")
    blob = _pad_png(buf.getvalue())
    assert dominant_bin_ratio(img) > 0.55
    result = structure_check(blob, kind="character_turnaround", allow_placeholder=True)
    assert result.verdict == "pass"
    assert result.scores["dominant_bin"] <= 0.55
    assert result.scores["entropy"] >= 2.5


def test_structure_rejects_3to1_character_canvas():
    blob = synthetic_still_png(1536, 512, tag="three-to-one", placeholder=True)
    result = structure_check(blob, kind="character_sheet", allow_placeholder=True)
    assert result.verdict == "fail"
    blob_text = " ".join(result.reasons)
    assert "1536x512" in blob_text or "character_canvas_3to1" in result.reasons


def _drawn_char_png(*, top_frac: float, bottom_frac: float) -> bytes:
    """Non-placeholder RGB sheet with a solid figure band for geometric span tests."""
    from PIL import ImageDraw

    img = Image.new("RGB", CHARACTER_SIZE, (245, 245, 245))
    draw = ImageDraw.Draw(img)
    w, h = CHARACTER_SIZE
    draw.rectangle(
        [int(w * 0.30), int(h * top_frac), int(w * 0.70), int(h * bottom_frac)],
        fill=(40, 50, 100),
    )
    buf = BytesIO()
    img.save(buf, format="PNG")
    return _pad_png(buf.getvalue())


def test_figure_is_full_body_accepts_head_to_toe():
    blob = _drawn_char_png(top_frac=0.06, bottom_frac=0.94)
    img = Image.open(BytesIO(blob)).convert("RGB")
    ok, metrics = figure_is_full_body(img)
    assert ok
    assert metrics["span"] >= 0.65
    assert metrics["bottom"] >= 0.85


def test_structure_rejects_bust_crop_as_not_full_body():
    # Upper-body poster: figure ends mid-canvas, lower band is empty studio.
    blob = _drawn_char_png(top_frac=0.05, bottom_frac=0.48)
    result = structure_check(blob, kind="character_sheet", allow_placeholder=True)
    assert result.verdict == "fail"
    assert "not_full_body" in result.reasons
    assert result.scores["figure_bottom"] < 0.85


def test_structure_rejects_cowboy_shot_missing_feet():
    blob = _drawn_char_png(top_frac=0.04, bottom_frac=0.72)
    result = structure_check(blob, kind="character_view_derive", allow_placeholder=True)
    assert result.verdict == "fail"
    assert "not_full_body" in result.reasons


def test_placeholder_noise_skips_geometric_full_body_gate():
    # Offline placeholders are full-frame noise; geometric gate must not block dry-runs.
    blob = synthetic_still_png(*CHARACTER_SIZE, tag="placeholder-skip", placeholder=True)
    result = structure_check(blob, kind="character_sheet", allow_placeholder=True)
    assert "not_full_body" not in result.reasons


def test_clip_unavailable_fail_closed_when_required():
    blob = synthetic_still_png(STILL_WIDTH, STILL_HEIGHT, tag="need-clip", placeholder=True)
    token = set_clip_scorer(None)
    try:
        with pytest.raises(ClipScorerUnavailable):
            score_still(
                blob,
                kind="scene_plate",
                prompt="empty anime location background",
                scorer=None,
                require_clip=True,
                allow_placeholder=True,
            )
    finally:
        from anime_factory.visual_qc import reset_clip_scorer

        reset_clip_scorer(token)


def test_no_hash_scorer_in_module():
    src = Path(__file__).resolve().parents[1] / "python" / "anime_factory" / "visual_qc.py"
    text = src.read_text(encoding="utf-8")
    assert "sha256(" not in text
    assert "hashlib" not in text
    assert CLIP_ARCH == "ViT-B-32"
    assert CLIP_PRETRAINED == "openai"
    assert OpenClipScorer.model_id == CLIP_MODEL_ID
    with pytest.raises(ClipScorerUnavailable):
        ScriptedClipScorer(image_embed=POS, text_embed=text_vec, model_id="sha256:fake")
    with pytest.raises(ClipScorerUnavailable):
        ScriptedClipScorer(image_embed=POS, text_embed=text_vec, model_id="rng")


def test_zero_shot_scene_prefers_interior_over_dish():
    blob = synthetic_still_png(STILL_WIDTH, STILL_HEIGHT, tag="scene-ok", placeholder=True)
    ok = score_still(
        blob,
        kind="scene_plate",
        prompt="anime interior room, empty establishing shot",
        scorer=passing_scorer(),
        allow_placeholder=True,
    )
    assert ok.verdict == "pass"
    assert ok.scores["clip_margin"] >= CLASS_MARGIN_MIN
    bad = score_still(
        blob,
        kind="scene_plate",
        prompt="anime interior room",
        scorer=dish_scorer(),
        allow_placeholder=True,
    )
    assert bad.verdict == "fail"
    assert "scene_classified_as_dinnerware" in bad.reasons
    # Cliff/hairpin plates land ~0.02 above dinnerware. That is a location, not a dish.
    weak = score_still(
        blob,
        kind="scene_plate",
        prompt="anime mountain road, cliff overlook, empty anime location background",
        scorer=ScriptedClipScorer(
            image_embed=(1.0, 0.96),
            text_embed=text_vec,
            model_id="scripted:weak-location",
        ),
        allow_placeholder=True,
    )
    assert weak.scores["clip_margin"] < CLASS_MARGIN_MIN
    assert weak.scores["clip_margin"] > 0
    assert weak.verdict == "pass"
    assert "scene_classified_as_dinnerware" not in weak.reasons


def test_zero_shot_character_rejects_heads_and_nude():
    blob = synthetic_still_png(*CHARACTER_SIZE, tag="char-ok", placeholder=True)
    ok = score_still(
        blob,
        kind="character_sheet",
        prompt="clothed full-body anime character standing",
        scorer=passing_scorer(),
        allow_placeholder=True,
    )
    assert ok.verdict == "pass"
    bad = score_still(
        blob,
        kind="character_sheet",
        prompt="clothed full-body anime character standing",
        scorer=ScriptedClipScorer(image_embed=NEG, text_embed=text_vec, model_id="scripted:heads"),
        allow_placeholder=True,
    )
    assert bad.verdict == "fail"
    assert "character_classified_as_bad" in bad.reasons


def test_identity_similarity_rejects_face_swap():
    child = synthetic_still_png(*CHARACTER_SIZE, tag="child", placeholder=True)
    parent = synthetic_still_png(*CHARACTER_SIZE, tag="parent", placeholder=True)
    n = {"i": 0}

    def image_embed(_img):
        n["i"] += 1
        return list(NEG) if n["i"] == 1 else list(POS)

    scorer = ScriptedClipScorer(image_embed=image_embed, text_embed=text_vec, model_id="scripted:identity")
    result = score_still(
        child,
        kind="character_view_derive",
        prompt="clothed full-body anime character standing, side view",
        parent_image=parent,
        scorer=scorer,
        allow_placeholder=True,
    )
    assert result.verdict == "fail"
    assert "identity_mismatch" in result.reasons
    assert result.scores["identity"] < IDENTITY_MIN


def test_prompt_alignment_uses_english_still_text():
    blob = synthetic_still_png(STILL_WIDTH, STILL_HEIGHT, tag="align", placeholder=True)
    prompt = "wooden harbor dock at madder dusk"

    def texts(text: str):
        if text == prompt:
            return list(NEG)
        return text_vec(text)

    result = score_still(
        blob,
        kind="scene_plate",
        prompt=prompt,
        scorer=ScriptedClipScorer(image_embed=POS, text_embed=texts, model_id="scripted:prompt"),
        allow_placeholder=True,
    )
    assert result.verdict == "fail"
    assert "prompt_mismatch" in result.reasons
    assert result.scores["alignment"] < ALIGNMENT_MIN or result.scores["alignment"] <= result.scores.get("alignment_neg", 1)


def test_scripted_scorer_is_cosine_not_rng():
    left = [0.6, 0.8]
    right = [0.6, 0.8]
    assert cosine(left, right) == pytest.approx(1.0)
    orthogonal = cosine([1.0, 0.0], [0.0, 1.0])
    assert orthogonal == pytest.approx(0.0)
    scorer = ScriptedClipScorer(image_embed=POS, text_embed={SCENE_POSITIVE[0]: POS, SCENE_NEGATIVE[0]: NEG}, model_id="scripted:cosine")
    img_vec = scorer.embed_images([object()])[0]
    pos_vec = scorer.embed_texts([SCENE_POSITIVE[0]])[0]
    neg_vec = scorer.embed_texts([SCENE_NEGATIVE[0]])[0]
    assert cosine(img_vec, pos_vec) == pytest.approx(1.0)
    assert cosine(img_vec, neg_vec) == pytest.approx(0.0)
    assert load_clip_scorer(require=False, scorer=scorer) is scorer


@pytest.fixture(autouse=True)
def _reset_open_clip_singleton():
    visual_qc._loaded_open_clip = None
    yield
    visual_qc._loaded_open_clip = None


def _install_fake_open_clip(monkeypatch, *, calls: dict):
    class FakeModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    def fake_create(arch, pretrained=None, **kwargs):
        calls["create"] = {"arch": arch, "pretrained": pretrained, "kwargs": kwargs}
        return FakeModel(), None, lambda image: image

    def fake_load_checkpoint(model, path):
        calls["load_checkpoint"] = {"model": model, "path": path}

    def fake_get_tokenizer(arch):
        calls["tokenizer_arch"] = arch
        return lambda texts: texts

    fake_open_clip = types.SimpleNamespace(
        create_model_and_transforms=fake_create,
        load_checkpoint=fake_load_checkpoint,
        get_tokenizer=fake_get_tokenizer,
    )
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_open_clip


def test_open_clip_from_pretrained_uses_local_checkpoint(tmp_path, monkeypatch):
    weight = tmp_path / "open_clip_model.safetensors"
    weight.write_bytes(b"fake-weights")
    monkeypatch.setenv("AF_VISUAL_QC_CLIP_WEIGHTS", str(weight))
    monkeypatch.setenv("AF_VISUAL_QC_CLIP_DIR", str(tmp_path))
    calls: dict = {}
    _install_fake_open_clip(monkeypatch, calls=calls)

    scorer = OpenClipScorer.from_pretrained()

    assert scorer.model_id == CLIP_MODEL_ID
    assert calls["create"]["arch"] == CLIP_ARCH
    assert calls["create"]["pretrained"] is None
    assert calls["load_checkpoint"]["path"] == str(weight)
    assert calls["tokenizer_arch"] == CLIP_ARCH


def test_open_clip_from_pretrained_rejects_missing_weights(tmp_path, monkeypatch):
    missing = tmp_path / "missing.safetensors"
    monkeypatch.setenv("AF_VISUAL_QC_CLIP_WEIGHTS", str(missing))
    with pytest.raises(ClipScorerUnavailable, match="weights missing locally"):
        OpenClipScorer.from_pretrained()


def test_open_clip_from_pretrained_rejects_unsupported_model(tmp_path, monkeypatch):
    weight = tmp_path / "open_clip_model.safetensors"
    weight.write_bytes(b"fake-weights")
    monkeypatch.setenv("AF_VISUAL_QC_CLIP_WEIGHTS", str(weight))
    monkeypatch.setenv("AF_OPEN_CLIP_MODEL", "ViT-H-14")
    with pytest.raises(ClipScorerUnavailable, match="unsupported OpenCLIP variant"):
        OpenClipScorer.from_pretrained()


def test_clip_env_prefers_visual_qc_vars_and_legacy_aliases(tmp_path, monkeypatch):
    primary = tmp_path / "primary.safetensors"
    primary.write_bytes(b"p")
    legacy = tmp_path / "legacy.pt"
    legacy.write_bytes(b"l")
    cache = tmp_path / "cache"
    cache.mkdir()

    monkeypatch.setenv("AF_VISUAL_QC_CLIP_WEIGHTS", str(primary))
    monkeypatch.setenv("AF_CLIP_WEIGHTS", str(legacy))
    assert visual_qc._clip_weight_path() == primary

    monkeypatch.delenv("AF_VISUAL_QC_CLIP_WEIGHTS", raising=False)
    monkeypatch.setenv("AF_CLIP_WEIGHTS", str(legacy))
    assert visual_qc._clip_weight_path() == legacy

    monkeypatch.delenv("AF_CLIP_WEIGHTS", raising=False)
    monkeypatch.setenv("AF_VISUAL_QC_CLIP_DIR", str(cache))
    (cache / "open_clip_model.safetensors").write_bytes(b"c")
    assert visual_qc._clip_cache_dir() == cache
    assert visual_qc._clip_weight_path() == cache / "open_clip_model.safetensors"


def test_visual_qc_env_contract_with_weights_module(tmp_path, monkeypatch):
    from gpu_worker import weights

    for name in (
        "AF_VISUAL_QC_CLIP_WEIGHTS",
        "AF_VISUAL_QC_CLIP_CONFIG",
        "AF_VISUAL_QC_CLIP_DIR",
        "AF_CLIP_WEIGHTS",
        "AF_CLIP_CACHE",
        "AF_OPEN_CLIP_MODEL",
        "AF_OPEN_CLIP_PRETRAINED",
    ):
        monkeypatch.delenv(name, raising=False)

    env = weights.visual_qc_clip_env(tmp_path)
    weight = Path(env["AF_VISUAL_QC_CLIP_WEIGHTS"])
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"fake")
    Path(env["AF_VISUAL_QC_CLIP_CONFIG"]).write_bytes(b"{}")

    for key, value in env.items():
        if key == "OPEN_CLIP_TORCH_VERSION":
            continue
        monkeypatch.setenv(key, value)

    assert visual_qc._clip_cache_dir() == Path(env["AF_VISUAL_QC_CLIP_DIR"])
    assert visual_qc._clip_weight_path() == weight
    assert visual_qc._clip_config_path() == Path(env["AF_VISUAL_QC_CLIP_CONFIG"])
    assert visual_qc._clip_model_env() == (CLIP_ARCH, CLIP_PRETRAINED)

    calls: dict = {}
    _install_fake_open_clip(monkeypatch, calls=calls)
    OpenClipScorer.from_pretrained()
    assert calls["create"]["pretrained"] is None
    assert calls["load_checkpoint"]["path"] == str(weight)


def test_attribute_margin_rejects_white_hoodie_for_navy_jacket():
    blob = synthetic_still_png(*CHARACTER_SIZE, tag="hoodie-miss", placeholder=True)
    identity = (
        "1boy, male focus, young adult, short black hair, brown eyes, "
        "navy jacket, white shirt, black pants, white sneakers"
    )

    def embed_text(text: str):
        low = str(text or "").lower()
        if "white hoodie" in low or "grey hoodie" in low:
            return list(POS)
        if "navy jacket" in low or "dark blue jacket" in low:
            return list(NEG)
        return text_vec(text)

    result = score_still(
        blob,
        kind="character_sheet",
        prompt=identity,
        scorer=ScriptedClipScorer(image_embed=POS, text_embed=embed_text, model_id="scripted:hoodie"),
        allow_placeholder=True,
    )
    assert result.verdict == "fail"
    assert any("top" in reason or "hoodie" in reason for reason in result.reasons)


def test_alignment_uses_short_identity_not_look_prefix():
    probes = identity_attribute_probes(
        "1boy, male focus, navy jacket, white shirt, black pants, looking at viewer, front view",
        view="front",
    )
    assert probes
    top = next(item for item in probes if item["id"] == "top")
    assert "navy jacket" in top["pos"]
    assert "looking at viewer" not in top["pos"]
    short = short_identity_text(
        "original anime character design, cel shaded, 1boy, navy jacket, white shirt, front view"
    )
    assert "navy jacket" in short
    assert "original anime character design" not in short


def test_hooded_jacket_and_straight_leg_trousers_parse_colors():
    wiki = (
        "1boy, young adult college student, 20 years old, slim build, "
        "short straight black hair with neat fringe, warm brown eyes, light tan skin, "
        "navy hooded jacket over a plain white shirt, black straight-leg trousers, white canvas sneakers"
    )
    probes = identity_attribute_probes(wiki, view="front")
    ids = {item["id"] for item in probes}
    assert "top" in ids
    assert "pants" in ids
    top = next(item for item in probes if item["id"] == "top")
    assert "navy jacket" in top["pos"]
    pants = next(item for item in probes if item["id"] == "pants")
    assert "black pants" in pants["pos"]
    assert "khaki pants" in pants["neg"]
    assert "olive pants" in pants["neg"]


def test_ivory_raincoat_and_safety_harness_parse_live_identity():
    identity = (
        "1girl, young adult, shoulder-length black hair, amber eyes, "
        "fixed ivory raincoat, navy scarf, orange safety harness, slim build"
    )
    probes = identity_attribute_probes(identity, view="front")
    top = next(item for item in probes if item["id"] == "top")
    assert "ivory raincoat" in top["pos"]


def test_navy_torso_ok_rejects_cream_hoodie_luma():
    from anime_factory.visual_qc import navy_torso_ok

    assert navy_torso_ok(195.2, 192.9, 179.2) is False
    assert navy_torso_ok(28.0, 36.0, 72.0) is True


def test_black_pants_ok_rejects_live_khaki_joggers():
    from anime_factory.visual_qc import black_pants_ok

    # Live 6454ab7 sheet_front / sheet_front_2 lower-body means.
    assert black_pants_ok(132.5, 134.3, 129.4) is False
    assert black_pants_ok(189.7, 181.1, 169.5) is False
    # A brighter warm candidate remains khaki-like and rejectable.
    assert black_pants_ok(140.0, 130.0, 110.0) is False
    assert black_pants_ok(40.0, 42.0, 48.0) is True


@pytest.mark.parametrize(
    "rgb",
    [
        (84.0, 74.6, 62.5),
        (94.5, 84.7, 68.7),
        (87.7, 74.4, 64.2),
        (82.8, 67.8, 56.1),
        (83.2, 71.3, 59.7),
    ],
)
def test_black_pants_ok_accepts_live_warm_shadow_samples(rgb):
    from anime_factory.visual_qc import black_pants_ok

    assert black_pants_ok(*rgb) is True


def _navy_khaki_sheet_png(*, pants=(160, 150, 110)) -> bytes:
    from PIL import ImageDraw

    w, h = CHARACTER_SIZE
    img = Image.new("RGB", CHARACTER_SIZE, (242, 242, 242))
    draw = ImageDraw.Draw(img)
    # Head-to-toe figure so structure clears FIGURE_* before palette probes run.
    body = [int(w * 0.31), int(h * 0.08), int(w * 0.69), int(h * 0.92)]
    head = [int(w * 0.36), int(h * 0.10), int(w * 0.64), int(h * 0.28)]
    legs = [int(w * 0.38), int(h * 0.48), int(w * 0.62), int(h * 0.90)]
    draw.rectangle(body, fill=(28, 36, 72))
    draw.rectangle(head, fill=(210, 180, 140))
    draw.rectangle(legs, fill=pants)
    pix = img.load()
    for x in range(body[0], body[2], 2):
        for y in range(body[1], body[3], 2):
            r, g, b = pix[x, y]
            jitter = ((x * 13 + y * 7) % 41) - 20
            pix[x, y] = (
                max(0, min(255, r + jitter)),
                max(0, min(255, g + jitter)),
                max(0, min(255, b + jitter)),
            )
    buf = BytesIO()
    img.save(buf, format="PNG")
    return _pad_png(buf.getvalue())


def test_black_pants_palette_rejects_khaki_even_when_clip_passes():
    identity = (
        "1boy, male focus, young adult, short black hair, brown eyes, "
        "navy hooded jacket over a plain white shirt, black straight-leg trousers, "
        "white canvas sneakers"
    )
    blob = _navy_khaki_sheet_png()
    result = score_still(
        blob,
        kind="character_sheet",
        prompt=identity,
        scorer=passing_scorer(),
        allow_placeholder=True,
        require_clip=True,
    )
    assert result.verdict == "fail"
    assert "attr_pants_palette" in result.reasons


def test_black_pants_palette_accepts_dark_trousers():
    identity = (
        "1boy, male focus, young adult, short black hair, brown eyes, "
        "navy hooded jacket over a plain white shirt, black straight-leg trousers, "
        "white canvas sneakers"
    )
    blob = _navy_khaki_sheet_png(pants=(32, 34, 40))
    result = score_still(
        blob,
        kind="character_sheet",
        prompt=identity,
        scorer=passing_scorer(),
        allow_placeholder=True,
        require_clip=True,
    )
    assert result.verdict == "pass"
    assert "attr_pants_palette" not in result.reasons
    assert "attr_top_palette" not in result.reasons
    assert result.scores["legs_luma"] < 110


def test_attribute_margin_ignores_tiny_pants_clip_noise():
    identity = (
        "1boy, male focus, young adult, short black hair, brown eyes, "
        "navy hooded jacket over a plain white shirt, black straight-leg trousers, "
        "white canvas sneakers"
    )
    blob = _navy_khaki_sheet_png(pants=(32, 34, 40))

    def embed_text(text: str):
        low = str(text or "").lower()
        if "khaki pants" in low or "olive pants" in low:
            return [0.99, 0.01]
        if "black pants" in low:
            return [0.98, 0.02]
        return text_vec(text)

    result = score_still(
        blob,
        kind="character_sheet",
        prompt=identity,
        scorer=ScriptedClipScorer(image_embed=POS, text_embed=embed_text, model_id="scripted:pants-noise"),
        allow_placeholder=True,
        require_clip=True,
    )
    assert result.verdict == "pass"
    assert "attr_pants" not in result.reasons


def test_gender_sign_flip_rejects_1girl_for_1boy():
    blob = synthetic_still_png(*CHARACTER_SIZE, tag="gender", placeholder=True)
    identity = "1boy, male focus, young adult, short black hair, brown eyes, grey hoodie"

    def embed_text(text: str):
        low = str(text or "").lower()
        if "female anime character" in low or low.strip().startswith("1girl"):
            return list(POS)
        if "male anime character" in low:
            return list(NEG)
        return text_vec(text)

    bad = score_still(
        blob,
        kind="character_sheet",
        prompt=identity,
        scorer=ScriptedClipScorer(image_embed=POS, text_embed=embed_text, model_id="scripted:girl"),
        allow_placeholder=True,
    )
    assert bad.verdict == "fail"
    assert any("gender" in reason for reason in bad.reasons)

    ok = score_still(
        blob,
        kind="character_sheet",
        prompt=identity,
        scorer=passing_scorer(),
        allow_placeholder=True,
    )
    assert ok.verdict == "pass"


def test_view_consistency_rejects_palette_drift():
    white = Image.new("RGB", CHARACTER_SIZE, (250, 250, 250))
    blue = Image.new("RGB", CHARACTER_SIZE, (40, 80, 180))
    navy = Image.new("RGB", CHARACTER_SIZE, (20, 40, 90))
    drifted = view_consistency_check([white, blue])
    assert drifted["ok"] is False
    same = view_consistency_check([navy, navy, navy])
    assert same["ok"] is True


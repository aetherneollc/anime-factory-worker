"""Still-image visual QC. Structure is offline (Pillow); CLIP is injectable.

Production scoring uses real OpenCLIP ViT-B-32/openai. Hash/rng/sha256
embeddings are forbidden. Dry-run and tests must inject a ClipScorer;
missing CLIP on a live/GPU path raises ClipScorerUnavailable (fail closed).
"""

from __future__ import annotations

import math
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from anime_factory.models import (
    IMAGE_MODEL,
    PROMPT_TEMPLATE_VERSION,
    STILL_HEIGHT,
    STILL_WIDTH,
)

QC_VERSION = "visual_qc_v1"
WORKFLOW_VERSION = "anime_t2i"
CLIP_ARCH = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_MODEL_ID = f"open_clip:{CLIP_ARCH}:{CLIP_PRETRAINED}"
QC_SEED_SALTS = (0, 1, 2)
MAX_QC_ATTEMPTS = len(QC_SEED_SALTS)

CHARACTER_SIZE = (832, 1216)
SCENE_SIZE = (STILL_WIDTH, STILL_HEIGHT)  # 1344x768
TURNAROUND_SIZE = (CHARACTER_SIZE[0] * 3, CHARACTER_SIZE[1])
PROP_SIZE = (768, 768)

ENTROPY_MIN = 2.5
DOMINANT_BIN_MAX = 0.55
# Character sheets sit on a near-white studio drop. Count figure pixels only.
STUDIO_BACKDROP_LUMA = 235
NAVY_TORSO_LUMA_MAX = 150.0
# Khaki/olive joggers on live 6454ab7 sheets sat at luma ~133–182 in this crop.
BLACK_PANTS_LUMA_MAX = 110.0
# Warm reflected light can make genuinely dark trousers brown. Only use hue
# skew as a rejection signal above the darkest observed live shadow samples.
BLACK_PANTS_WARM_LUMA_MIN = 90.0
_GARMENT_MODIFIERS = (
    r"(?:(?:rescue|safety|utility|hooded|wool|zip-up|zipped|denim|leather|canvas|straight-leg|"
    r"cropped|oversized|padded|plain|quilted)\s+)*"
)
_GARMENT_COLOR_RE = re.compile(
    rf"\b(navy|black|white|ivory|charcoal|orange|grey|gray|brown|red|blue|green|pink|purple|gold|silver|tan|beige|khaki|dark|light)\s+"
    rf"{_GARMENT_MODIFIERS}"
    rf"(raincoat|jacket|hoodie|coat|shirt|pants|trousers|scarf|harness|boots|sneakers|shoes)\b",
    re.I,
)
CLASS_MARGIN_MIN = 0.04
IDENTITY_MIN = 0.28
ALIGNMENT_MIN = 0.18

SCENE_POSITIVE = (
    "anime interior room",
    "anime city street",
    "anime building exterior",
    "empty anime location background",
)
SCENE_NEGATIVE = (
    "ceramic dinner plate",
    "bowl of food on a table",
    "dinnerware",
    "empty beige canvas",
)
CHARACTER_POSITIVE = (
    "clothed full-body anime character standing",
    "single anime character design",
)
CHARACTER_NEGATIVE = (
    "collage of floating heads",
    "nude person",
    "abstract costume scraps",
    "three-panel turnaround strip",
)

_FAKE_SCORER_TOKENS = ("hash", "rng", "sha256")
_scorer_var: ContextVar["ClipScorer | None"] = ContextVar("anime_factory_clip_scorer", default=None)
_loaded_open_clip: "OpenClipScorer | None" = None
_STYLE_SKIP_PREFIXES = (
    "original anime",
    "cel shaded",
    "clean lineart",
    "warm ivory",
    "flat shading",
    "2d anime",
    "keep the same face",
    "masterpiece",
    "high score",
    "great score",
    "absurdres",
)


class ClipScorerUnavailable(RuntimeError):
    """Live/GPU still QC required a real CLIP scorer and none was available."""


class ClipScorer(Protocol):
    model_id: str

    def embed_images(self, images: Sequence[Any]) -> list[list[float]]:
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...


@dataclass
class VisualQcResult:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    scorer: str | None = None
    lineage: dict[str, str] = field(default_factory=dict)
    kind: str = ""
    seed: int | None = None

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def current_lineage() -> dict[str, str]:
    return {
        "model_version": IMAGE_MODEL,
        "prompt_version": PROMPT_TEMPLATE_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "qc_version": QC_VERSION,
    }


def is_current_pass(rec: dict[str, Any] | None) -> bool:
    if not isinstance(rec, dict):
        return False
    return (
        rec.get("qc_verdict") == "pass"
        and rec.get("qc_version") == QC_VERSION
        and bool(str(rec.get("selected") or "").strip())
    )


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Dot-product cosine. Pure Python; no numpy."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(left, right):
        xa = float(a)
        xb = float(b)
        dot += xa * xb
        norm_a += xa * xa
        norm_b += xb * xb
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _refuse_fake_scorer(scorer: ClipScorer) -> ClipScorer:
    model_id = str(getattr(scorer, "model_id", "") or "").strip().lower()
    if not model_id:
        raise ClipScorerUnavailable("ClipScorer.model_id is required")
    if any(token in model_id for token in _FAKE_SCORER_TOKENS):
        raise ClipScorerUnavailable(
            f"refusing hash/rng/sha256 scorer {getattr(scorer, 'model_id', None)!r}"
        )
    return scorer


def set_clip_scorer(scorer: ClipScorer | None):
    """Inject a scorer for this context (tests). Production GPU loads OpenCLIP."""
    if scorer is not None:
        _refuse_fake_scorer(scorer)
    return _scorer_var.set(scorer)


def reset_clip_scorer(token) -> None:
    _scorer_var.reset(token)


def _env_path(*names: str) -> Path | None:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if raw:
            return Path(raw)
    return None


def _clip_cache_dir() -> Path | None:
    return _env_path(
        "AF_VISUAL_QC_CLIP_DIR",
        "AF_CLIP_CACHE",
        "AF_OPENCLIP_CACHE",
        "OPENCLIP_CACHE_DIR",
    )


def _clip_config_path() -> Path | None:
    return _env_path("AF_VISUAL_QC_CLIP_CONFIG")


def _clip_weight_path() -> Path | None:
    explicit = _env_path(
        "AF_VISUAL_QC_CLIP_WEIGHTS",
        "AF_CLIP_WEIGHTS",
        "AF_OPENCLIP_CHECKPOINT",
        "OPENCLIP_CHECKPOINT",
    )
    if explicit is not None:
        return explicit
    cache = _clip_cache_dir()
    if cache is None:
        return None
    for candidate in (
        cache / "open_clip_model.safetensors",
        cache / "ViT-B-32.pt",
        cache / "openai" / "ViT-B-32.pt",
        cache / CLIP_ARCH / f"{CLIP_PRETRAINED}.pt",
    ):
        if candidate.is_file():
            return candidate
    return None


def _clip_model_env() -> tuple[str, str]:
    model = (os.environ.get("AF_OPEN_CLIP_MODEL") or CLIP_ARCH).strip() or CLIP_ARCH
    pretrained = (os.environ.get("AF_OPEN_CLIP_PRETRAINED") or CLIP_PRETRAINED).strip() or CLIP_PRETRAINED
    if model != CLIP_ARCH or pretrained != CLIP_PRETRAINED:
        raise ClipScorerUnavailable(
            f"unsupported OpenCLIP variant {model}/{pretrained!r}; "
            f"only {CLIP_ARCH}/{CLIP_PRETRAINED} is allowed"
        )
    return model, pretrained


class OpenClipScorer:
    """Production CLIP. Only ViT-B-32/openai; never IP-Adapter's vision-only ViT-H."""

    model_id = CLIP_MODEL_ID

    def __init__(self, model: Any, preprocess: Any, tokenizer: Any, device: str = "cpu"):
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(cls) -> "OpenClipScorer":
        arch, _pretrained_tag = _clip_model_env()
        weight = _clip_weight_path()
        if weight is None or not weight.is_file():
            raise ClipScorerUnavailable(
                f"{CLIP_MODEL_ID} weights missing locally "
                "(set AF_VISUAL_QC_CLIP_WEIGHTS or AF_VISUAL_QC_CLIP_DIR; runtime download is forbidden)"
            )
        from gpu_worker.weights import safetensors_mmap_unsafe

        unsafe = safetensors_mmap_unsafe(weight)
        if unsafe:
            raise ClipScorerUnavailable(unsafe)
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise ClipScorerUnavailable("open_clip_torch is not installed") from exc
        # Architecture only — load local safetensors/pt via load_checkpoint (never Hub tags).
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=None)
        open_clip.load_checkpoint(model, str(weight))
        tokenizer = open_clip.get_tokenizer(arch)
        device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        return cls(model, preprocess, tokenizer, device=device)

    def embed_images(self, images: Sequence[Any]) -> list[list[float]]:
        import torch

        if not images:
            return []
        tensors = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.detach().cpu().tolist()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        import torch

        if not texts:
            return []
        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.detach().cpu().tolist()


class ScriptedClipScorer:
    """Test double: caller supplies vectors; scores are still cosine. Not a production default."""

    def __init__(
        self,
        *,
        image_embed: Sequence[float] | Callable[[Any], Sequence[float]] | None = None,
        text_embed: dict[str, Sequence[float]] | Callable[[str], Sequence[float]] | None = None,
        model_id: str = "scripted:test",
    ):
        if any(token in str(model_id).lower() for token in _FAKE_SCORER_TOKENS):
            raise ClipScorerUnavailable(f"refusing hash/rng/sha256 scorer {model_id!r}")
        self.model_id = model_id
        self._image_embed = image_embed
        self._text_embed = text_embed

    def embed_images(self, images: Sequence[Any]) -> list[list[float]]:
        out: list[list[float]] = []
        for image in images:
            if callable(self._image_embed):
                vec = self._image_embed(image)
            elif self._image_embed is not None:
                vec = self._image_embed
            else:
                raise ClipScorerUnavailable("ScriptedClipScorer has no image_embed")
            out.append([float(x) for x in vec])
        return out

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        mapping = self._text_embed if isinstance(self._text_embed, dict) else None
        for text in texts:
            if callable(self._text_embed):
                vec = self._text_embed(text)
            elif mapping is not None and text in mapping:
                vec = mapping[text]
            elif mapping is not None:
                vec = _scripted_text_fallback(text, mapping)
            else:
                raise ClipScorerUnavailable("ScriptedClipScorer has no text_embed")
            out.append([float(x) for x in vec])
        return out


def _scripted_text_fallback(text: str, mapping: dict[str, Sequence[float]]) -> Sequence[float]:
    lowered = text.lower()
    for key, vec in mapping.items():
        if key.lower() == lowered or key.lower() in lowered or lowered in key.lower():
            return vec
    for probe in SCENE_NEGATIVE + CHARACTER_NEGATIVE:
        if probe in mapping and (probe in lowered or any(part in lowered for part in probe.split()[:2])):
            return mapping[probe]
    if mapping:
        return next(iter(mapping.values()))
    raise ClipScorerUnavailable(f"no scripted text vector for {text!r}")


def load_clip_scorer(
    *,
    require: bool = False,
    scorer: ClipScorer | None = None,
) -> ClipScorer | None:
    """Resolve an injectable scorer. Never synthesizes embeddings from hashes or RNG."""
    global _loaded_open_clip
    if scorer is not None:
        return _refuse_fake_scorer(scorer)
    injected = _scorer_var.get()
    if injected is not None:
        return _refuse_fake_scorer(injected)
    if _loaded_open_clip is not None:
        return _loaded_open_clip
    try:
        _loaded_open_clip = OpenClipScorer.from_pretrained()
        return _loaded_open_clip
    except ClipScorerUnavailable:
        if require:
            raise
        return None
    except Exception as exc:  # noqa: BLE001 — fail closed; never fall back to hash embeds
        if require:
            raise ClipScorerUnavailable(f"{CLIP_MODEL_ID} failed to load: {exc}") from exc
        return None


def _open_gray(image: Any):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("L")
    raise TypeError("expected a PIL Image")


def _gray_hist(image: Any, *, ignore_near_white: bool = False) -> list[int]:
    hist = list(_open_gray(image).histogram())
    if ignore_near_white:
        for luma in range(STUDIO_BACKDROP_LUMA, 256):
            hist[luma] = 0
    return hist


def shannon_entropy(image: Any, *, ignore_near_white: bool = False) -> float:
    """Grayscale Shannon entropy from a 256-bin Pillow histogram. No numpy."""
    hist = _gray_hist(image, ignore_near_white=ignore_near_white)
    total = float(sum(hist)) or 1.0
    entropy = 0.0
    for count in hist:
        if not count:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def dominant_bin_ratio(image: Any, *, ignore_near_white: bool = False) -> float:
    hist = _gray_hist(image, ignore_near_white=ignore_near_white)
    total = float(sum(hist)) or 1.0
    return max(hist) / total if hist else 1.0


def _load_rgb(blob: bytes | bytearray | Path | str):
    from PIL import Image

    if isinstance(blob, (Path, str)):
        image = Image.open(blob)
    else:
        image = Image.open(BytesIO(bytes(blob)))
    return image.convert("RGB")


def _expected_size(kind: str) -> tuple[int, int] | None:
    key = str(kind or "").strip().lower()
    if key in {"character_sheet", "character_view_derive", "costume_derive"}:
        return CHARACTER_SIZE
    if key == "character_turnaround":
        return TURNAROUND_SIZE
    if key in {"scene_plate", "scene_derive", "keyframe"}:
        return SCENE_SIZE
    if key == "prop":
        return PROP_SIZE
    return None


def _result(
    verdict: str,
    *,
    reasons: list[str] | None = None,
    scores: dict[str, float] | None = None,
    scorer: str | None = None,
    kind: str = "",
    seed: int | None = None,
) -> VisualQcResult:
    return VisualQcResult(
        verdict=verdict,
        reasons=list(reasons or []),
        scores=dict(scores or {}),
        scorer=scorer,
        lineage=current_lineage(),
        kind=kind,
        seed=seed,
    )


def structure_check(
    blob: bytes | bytearray | Path | str,
    *,
    kind: str,
    allow_placeholder: bool = False,
    label: str = "still",
) -> VisualQcResult:
    """PNG/size via assert_still_blob, then grayscale entropy / dominant-bin."""
    from anime_factory.design import StillBlobError, assert_still_blob, png_dimensions

    if isinstance(blob, (Path, str)):
        data = Path(blob).read_bytes()
    else:
        data = bytes(blob or b"")
    expected = _expected_size(kind)
    reasons: list[str] = []
    scores: dict[str, float] = {"bytes": float(len(data))}
    try:
        assert_still_blob(
            data,
            width=expected[0] if expected else None,
            height=expected[1] if expected else None,
            label=label,
            allow_placeholder=allow_placeholder,
        )
    except StillBlobError as exc:
        return _result("fail", reasons=[str(exc)], scores=scores, kind=kind)

    size = png_dimensions(data)
    if size:
        scores["width"] = float(size[0])
        scores["height"] = float(size[1])
        if kind in {"character_sheet", "character_view_derive", "costume_derive"} and size[1]:
            if abs(size[0] / size[1] - 3.0) < 0.2:
                reasons.append("character_canvas_3to1")

    image = _load_rgb(data)
    char_kind = kind in {
        "character_sheet",
        "character_view_derive",
        "character_turnaround",
        "costume_derive",
    }
    entropy = shannon_entropy(image, ignore_near_white=char_kind)
    dominant = dominant_bin_ratio(image, ignore_near_white=char_kind)
    scores["entropy"] = float(entropy)
    scores["dominant_bin"] = float(dominant)
    if entropy < ENTROPY_MIN:
        reasons.append("low_entropy")
    if dominant > DOMINANT_BIN_MAX:
        reasons.append("dominant_bin")
    if reasons:
        return _result("fail", reasons=reasons, scores=scores, kind=kind)
    return _result("pass", scores=scores, kind=kind)


def zero_shot_margin(
    image_vec: Sequence[float],
    pos_vecs: Sequence[Sequence[float]],
    neg_vecs: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    pos = max((cosine(image_vec, vec) for vec in pos_vecs), default=0.0)
    neg = max((cosine(image_vec, vec) for vec in neg_vecs), default=0.0)
    return pos - neg, pos, neg


def identity_similarity(image_vec: Sequence[float], parent_vec: Sequence[float]) -> float:
    return cosine(image_vec, parent_vec)


def short_identity_text(prompt: str) -> str:
    """Drop look/style prefixes so clothing tokens stay inside CLIP's 77-token budget."""
    parts = [part.strip() for part in str(prompt or "").split(",") if part.strip()]
    keep: list[str] = []
    for part in parts:
        low = part.lower()
        if any(low.startswith(prefix) for prefix in _STYLE_SKIP_PREFIXES):
            continue
        keep.append(part)
        if len(keep) >= 16:
            break
    return ", ".join(keep) or str(prompt or "").strip()


def parse_garment_colors(identity: str) -> dict[str, str]:
    """Map garment → color. Allows adjectives: 'navy hooded jacket', 'black straight-leg trousers'."""
    return {
        match.group(2).lower(): match.group(1).lower()
        for match in _GARMENT_COLOR_RE.finditer(str(identity or "").lower())
    }


def navy_torso_ok(red: float, green: float, blue: float) -> bool:
    """Reject cream/white hoodies when the lock is a navy/dark jacket."""
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    if luma > NAVY_TORSO_LUMA_MAX:
        return False
    if red > blue + 15:
        return False
    return True


def black_pants_ok(red: float, green: float, blue: float) -> bool:
    """Reject khaki/olive joggers when the lock is black trousers."""
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    if luma > BLACK_PANTS_LUMA_MAX:
        return False
    if green > blue + 12 and luma > BLACK_PANTS_WARM_LUMA_MIN:
        return False
    if (
        red > blue + 18
        and green > blue + 8
        and luma > BLACK_PANTS_WARM_LUMA_MIN
    ):
        return False
    return True


def identity_attribute_probes(identity: str, *, view: str = "front") -> list[dict[str, str]]:
    """Short pos/neg CLIP probes parsed from Danbooru-style identity tokens."""
    text = str(identity or "")
    low = text.lower()
    probes: list[dict[str, str]] = []
    if "1boy" in low:
        probes.append(
            {
                "id": "gender",
                "pos": "1boy, male anime character",
                "neg": "1girl, female anime character",
            }
        )
    elif "1girl" in low:
        probes.append(
            {
                "id": "gender",
                "pos": "1girl, female anime character",
                "neg": "1boy, male anime character",
            }
        )
    garments = parse_garment_colors(low)
    if "jacket" in garments:
        color = garments["jacket"]
        pos = "navy jacket, dark blue jacket" if color == "navy" else f"{color} jacket"
        probes.append({"id": "top", "pos": pos, "neg": "white hoodie, grey hoodie, beige hoodie"})
    elif "hoodie" in garments:
        color = garments["hoodie"]
        probes.append(
            {"id": "top", "pos": f"{color} hoodie", "neg": "navy jacket, white jacket"}
        )
    elif "coat" in garments:
        color = garments["coat"]
        probes.append({"id": "top", "pos": f"{color} coat", "neg": "white hoodie, grey hoodie"})
    elif "raincoat" in garments:
        color = garments["raincoat"]
        probes.append({"id": "top", "pos": f"{color} raincoat", "neg": "dark jacket, grey hoodie"})
    pants_color = garments.get("pants") or garments.get("trousers")
    if pants_color:
        if pants_color == "black":
            pos = "black pants, black trousers"
            neg = "khaki pants, olive pants, beige pants, grey sweatpants"
        else:
            pos = f"{pants_color} pants"
            neg = "black pants, white pants"
        probes.append({"id": "pants", "pos": pos, "neg": neg})
    if view != "back" and "short" in low and "hair" in low:
        hair_pos = "short black hair, bangs" if "black hair" in low else "short hair, bangs"
        probes.append({"id": "hair", "pos": hair_pos, "neg": "long hair, blonde hair"})
    return probes


def _mean_rgb(image: Any, box: tuple[int, int, int, int]) -> tuple[float, float, float]:
    """Mean RGB of a crop. Skip near-white studio drop and white sneakers."""
    crop = image.crop(box)
    pixels = list(crop.getdata())
    figure = [
        pixel
        for pixel in pixels
        if (0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]) < STUDIO_BACKDROP_LUMA
    ]
    use = figure or pixels
    count = float(len(use) or 1)
    red = sum(pixel[0] for pixel in use) / count
    green = sum(pixel[1] for pixel in use) / count
    blue = sum(pixel[2] for pixel in use) / count
    return red, green, blue


def _torso_mean_rgb(image: Any) -> tuple[float, float, float]:
    width, height = image.size
    return _mean_rgb(
        image,
        (
            int(width * 0.3),
            int(height * 0.28),
            int(width * 0.7),
            int(height * 0.72),
        ),
    )


def _legs_mean_rgb(image: Any) -> tuple[float, float, float]:
    """Lower-body crop that skips white sneakers at the ankles."""
    width, height = image.size
    return _mean_rgb(
        image,
        (
            int(width * 0.34),
            int(height * 0.55),
            int(width * 0.66),
            int(height * 0.82),
        ),
    )


def view_consistency_check(images: Sequence[Any], *, max_delta: float = 48.0) -> dict[str, Any]:
    """Reject turnaround panels whose torso palette drifted (white hoodie vs blue sleeve)."""
    if len(images) < 2:
        return {"ok": True, "delta": 0.0}
    means = [_torso_mean_rgb(image) for image in images]
    delta = 0.0
    for left in means:
        for right in means:
            delta = max(delta, abs(left[0] - right[0]), abs(left[1] - right[1]), abs(left[2] - right[2]))
    return {"ok": delta <= max_delta, "delta": float(delta)}


def prompt_alignment(image_vec: Sequence[float], prompt_vec: Sequence[float]) -> float:
    return cosine(image_vec, prompt_vec)


def _semantic_kind(kind: str) -> str:
    key = str(kind or "").strip().lower()
    if key in {"character_sheet", "character_view_derive", "costume_derive"}:
        return "character"
    if key in {"scene_plate", "scene_derive", "keyframe"}:
        return "scene"
    return key


def score_still(
    blob: bytes | bytearray | Path | str,
    *,
    kind: str,
    prompt: str = "",
    parent_image: bytes | bytearray | Path | str | None = None,
    scorer: ClipScorer | None = None,
    require_clip: bool = False,
    allow_placeholder: bool | None = None,
    seed: int | None = None,
    label: str = "still",
) -> VisualQcResult:
    """Structure first; CLIP only after the blob is a real still of the right size."""
    placeholder = (not require_clip) if allow_placeholder is None else allow_placeholder
    struct = structure_check(blob, kind=kind, allow_placeholder=placeholder, label=label)
    struct.seed = seed
    if not struct.passed:
        return struct
    if str(kind or "") == "character_turnaround":
        # Deterministic composite of already-QC'd views: structure only.
        return _result("pass", scores=struct.scores, kind=kind, seed=seed)

    resolved = load_clip_scorer(require=require_clip, scorer=scorer)
    if resolved is None:
        if require_clip:
            raise ClipScorerUnavailable(
                f"CLIP scorer required for {kind} lock (live/GPU); hash/rng substitutes are forbidden"
            )
        struct.verdict = "fail"
        struct.reasons.append("clip_scorer_unavailable")
        return struct

    resolved = _refuse_fake_scorer(resolved)
    image = _load_rgb(blob if not isinstance(blob, (Path, str)) else Path(blob))
    semantic = _semantic_kind(kind)
    texts: list[str] = []
    pos_prompts: tuple[str, ...] = ()
    neg_prompts: tuple[str, ...] = ()
    if semantic == "scene":
        pos_prompts, neg_prompts = SCENE_POSITIVE, SCENE_NEGATIVE
    elif semantic == "character":
        pos_prompts, neg_prompts = CHARACTER_POSITIVE, CHARACTER_NEGATIVE
    texts.extend(pos_prompts)
    texts.extend(neg_prompts)
    prompt_text = str(prompt or "").strip()
    alignment_text = short_identity_text(prompt_text) if semantic == "character" else prompt_text
    prompt_index = None
    if alignment_text:
        prompt_index = len(texts)
        texts.append(alignment_text)
    view = "side" if "side" in str(kind or "") or "from side" in prompt_text.lower() else (
        "back" if "back" in str(kind or "") or "from behind" in prompt_text.lower() else "front"
    )
    attr_probes = identity_attribute_probes(alignment_text or prompt_text, view=view) if semantic == "character" else []
    attr_start = len(texts)
    for probe in attr_probes:
        texts.append(probe["pos"])
        texts.append(probe["neg"])

    image_vec = resolved.embed_images([image])[0]
    parent_vec = None
    if parent_image is not None:
        parent_pil = _load_rgb(parent_image)
        parent_vec = resolved.embed_images([parent_pil])[0]
    text_vecs = resolved.embed_texts(texts) if texts else []

    scores = dict(struct.scores)
    reasons: list[str] = []
    if pos_prompts and neg_prompts:
        pos_vecs = text_vecs[: len(pos_prompts)]
        neg_vecs = text_vecs[len(pos_prompts) : len(pos_prompts) + len(neg_prompts)]
        margin, pos, neg = zero_shot_margin(image_vec, pos_vecs, neg_vecs)
        scores["clip_margin"] = float(margin)
        scores["clip_pos"] = float(pos)
        scores["clip_neg"] = float(neg)
        if margin < CLASS_MARGIN_MIN:
            if semantic == "scene":
                reasons.append("scene_classified_as_dinnerware")
            elif semantic == "character":
                reasons.append("character_classified_as_bad")
            else:
                reasons.append("zero_shot_margin")

    if parent_vec is not None:
        ident = identity_similarity(image_vec, parent_vec)
        scores["identity"] = float(ident)
        if ident < IDENTITY_MIN:
            reasons.append("identity_mismatch")

    if prompt_index is not None:
        prompt_vec = text_vecs[prompt_index]
        align = prompt_alignment(image_vec, prompt_vec)
        scores["alignment"] = float(align)
        neg_end = len(pos_prompts) + len(neg_prompts)
        max_neg = max((cosine(image_vec, vec) for vec in text_vecs[len(pos_prompts) : neg_end]), default=0.0)
        scores["alignment_neg"] = float(max_neg)
        if align < ALIGNMENT_MIN or (neg_prompts and align <= max_neg):
            reasons.append("prompt_mismatch")

    cursor = attr_start
    for probe in attr_probes:
        pos_vec = text_vecs[cursor]
        neg_vec = text_vecs[cursor + 1]
        cursor += 2
        pos = cosine(image_vec, pos_vec)
        neg = cosine(image_vec, neg_vec)
        margin = pos - neg
        scores[f"attr_{probe['id']}_margin"] = float(margin)
        scores[f"attr_{probe['id']}_pos"] = float(pos)
        scores[f"attr_{probe['id']}_neg"] = float(neg)
        # Sign-flip of 0.006 is CLIP noise on dark anime trousers vs khaki probes.
        if margin < -CLASS_MARGIN_MIN:
            reasons.append(f"attr_{probe['id']}")

    if require_clip and semantic == "character":
        garments = parse_garment_colors(alignment_text or prompt_text)
        jacket_color = garments.get("jacket") or garments.get("coat") or garments.get("raincoat")
        if jacket_color in {"navy", "dark", "blue"}:
            red, green, blue = _torso_mean_rgb(image)
            luma = 0.299 * red + 0.587 * green + 0.114 * blue
            scores["torso_luma"] = float(luma)
            scores["torso_r"] = float(red)
            scores["torso_b"] = float(blue)
            if not navy_torso_ok(red, green, blue):
                reasons.append("attr_top_palette")
        pants_color = garments.get("pants") or garments.get("trousers")
        if pants_color == "black":
            red, green, blue = _legs_mean_rgb(image)
            luma = 0.299 * red + 0.587 * green + 0.114 * blue
            scores["legs_luma"] = float(luma)
            scores["legs_r"] = float(red)
            scores["legs_g"] = float(green)
            scores["legs_b"] = float(blue)
            if not black_pants_ok(red, green, blue):
                reasons.append("attr_pants_palette")

    verdict = "fail" if reasons else "pass"
    return _result(
        verdict,
        reasons=reasons,
        scores=scores,
        scorer=str(resolved.model_id),
        kind=kind,
        seed=seed,
    )


def require_clip_for_client(client: Any) -> bool:
    if bool(getattr(client, "live", False)):
        return True
    flag = os.environ.get("ANIME_FACTORY_GPU_STILLS", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}

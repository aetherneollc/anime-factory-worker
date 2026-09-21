"""Stills assets. Scene plates MUST NOT send image / reference_images.

Identity is ONE four-view turnaround (`sheet_turnaround.png`), not three
independent rolls. `sheet_front.png` is a copy alias for old lookups.
Locked seed + assets/index.json selected pointer. GPU stills use anime SDXL
(Animagine XL 4.0) via Comfy. SiliconFlow Kolors remains a hosted fallback.
TTS never uses Vast.

Every blob that reaches disk goes through `assert_still_blob`: PNG magic, the
requested pixel size, and `MIN_STILL_BYTES`. The 3-byte `f1.png` files on disk
were written by the old `return b"PNG"` stub and passed a `>= 1` byte check.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import struct
import time
import zlib
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from anime_factory.asset_lock import (
    FRONT_ALIAS_FILENAME,
    PLATE_FILENAME,
    TURNAROUND_FILENAME,
    has_locked_identity,
    has_locked_scene,
    is_qc_locked,
    load_assets_index,
    lock_after_qc,
    locked_character_file,
    locked_character_relpath,
    locked_scene_file,
    mark_stale_visual_v1,
    next_history_filename,
    pop_regen_requests,
    record_qc_candidate,
    save_assets_index,
    write_front_alias,
)
from anime_factory.config import load_settings
from anime_factory.db import utcnow
from anime_factory.instrument import Counters
from anime_factory.models import (
    DEFAULT_STYLE_PRESET,
    FIXED_NEGATIVE,
    IMAGE_MODEL,
    IMAGE_STEPS,
    KOLORS_MODEL,
    MIN_STILL_BYTES,
    PROMPT_TEMPLATE_VERSION,
    SILICONFLOW_BASE_URL,
    STILL_HEIGHT,
    STILL_IMAGE_SIZE,
    STILL_WIDTH,
    STYLE_PREFIX,
    animagine_quality_suffix,
    normalize_style_preset,
    scrub_copycat,
    style_prefix_for_kind,
)
from anime_factory.r2_paths import character_asset_rel, join_story, scene_asset_rel
from anime_factory.visual_qc import (
    MAX_QC_ATTEMPTS,
    QC_SEED_SALTS,
    QC_VERSION,
    WORKFLOW_VERSION,
    ClipScorer,
    score_still,
    require_clip_for_client,
)
from anime_factory.world import parse_period_md

FORBIDDEN_PLATE_KEYS = frozenset({"image", "reference_images"})
# Draw one portrait full-body view at a time. Side/back views inherit the approved
# front through IP-Adapter; only then are the three panels composed for legacy readers.
CHAR_VIEWS = ("front", "side", "back", "turnaround")
CHAR_IMAGE_SIZE = "832x1216"
CHAR_VIEW_WIDTH = 832
CHAR_VIEW_HEIGHT = 1216
CHAR_SIDE_FILENAME = "sheet_side.png"
CHAR_BACK_FILENAME = "sheet_back.png"
SCENE_IMAGE_SIZE = STILL_IMAGE_SIZE
PROP_IMAGE_SIZE = "768x768"
KOLORS_STEPS = IMAGE_STEPS
# Affirmative only, same reason as STYLE_PREFIX. What these looks must NOT be
# lives in the paired *_NEGATIVE below, which real CFG applies.
FRONT_LOOK = (
    "solo, front view, looking at viewer, full body, standing, simple background"
)
SIDE_LOOK = (
    "solo, from side, strict left side profile, full body, standing, simple background"
)
BACK_LOOK = (
    "solo, from behind, facing away, strict rear view, full body, standing, simple background"
)
TURNAROUND_LOOK = FRONT_LOOK
SHEET_LOOK = FRONT_LOOK
SHEET_NEGATIVE = (
    "character turnaround, multiple views, split panel, collage, contact sheet, floating heads, "
    "cropped body, cinematic still, movie screenshot, dramatic rim light, scenery background, "
    "text on the sheet, caption"
)
PLATE_LOOK = "anime location background, empty establishing shot"
PLATE_NEGATIVE = "famous movie still, generic sunset cloud plate, people, crowd, text, ceramic plate, dish, dinnerware, bowl"
PROP_LOOK = "prop design sheet, product turnaround, plain studio background"
PROP_NEGATIVE = "cinematic still, scenery background, people, text"
IDENTITY_PROMPT_MAX_CHARS = 1000
DERIVE_FACE_LOCK = "keep the same face body and hairline as the parent sheet, do not redesign identity"
_STANDALONE_PLATE_RE = re.compile(r"(?<![\w-])\bplate\b(?![\w-])", re.I)
_CJK_STILL_RE = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]+"
)
_GENDER_TAG_RE = re.compile(r"\b(1boy|1girl)\b", re.I)
_HAIR_RE = re.compile(
    r"\b(hair|ponytail|bangs|bob|braid|crew cut|buzz cut|blonde|black hair|brown hair|silver hair|red hair)\b",
    re.I,
)
_EYES_RE = re.compile(r"\b(eyes?|iris|heterochromia|eye color)\b", re.I)
_CLOTHING_RE = re.compile(
    r"\b(raincoat|jacket|coat|uniform|dress|shirt|hoodie|skirt|armor|suit|clothes|clothing|"
    r"robe|vest|sweater|blouse|trousers|pants|scarf|harness|boots|shoes|overalls|apron|"
    r"coveralls|smock|workwear|build|frame|physique|slim|athletic)\b",
    re.I,
)
_CLOTHING_COLOR_RE = re.compile(
    r"\b("
    r"navy(?:\s+blue)?|dark\s+blue|light\s+blue|black|white|ivory|charcoal|orange|grey|gray|"
    r"brown|red|blue|green|pink|purple|gold|silver|tan|beige|khaki|dark|light"
    r")\s+"
    r"(?:(?:rescue|safety|utility|worker|janitor|cleaning|maintenance|hooded|wool|zip-up|zipped|"
    r"denim|leather|canvas|straight-leg|cropped|oversized|padded|plain|quilted|work)\s+)*"
    r"(?:(?:worker|janitor|cleaning|maintenance|school|rescue|safety|utility)\s+)*"
    r"(raincoat|jacket|coat|hoodie|shirt|dress|uniform|armor|suit|robe|pants|trousers|"
    r"sweater|blouse|vest|scarf|harness|boots|shoes|overalls|apron|coveralls|smock|workwear)\b",
    re.I,
)
_JACKET_RE = re.compile(r"\b(jacket)\b", re.I)
_SHIRT_RE = re.compile(r"\b(shirt|blouse|sweater)\b", re.I)
_SELF_LAYER_RE = re.compile(r"\b(hoodie|coat|dress|uniform|armor|suit|robe|overalls|apron|coveralls|smock)\b", re.I)
_AGE_RE = re.compile(
    r"\b(\d{1,2}\s*y/o|\d{1,2}\s*years?\s*old|teen|young adult|adult|elderly|child|adolescent|"
    r"middle.?aged|mid-?\d{2}s|senior)\b",
    re.I,
)
_AGE_NUMBER_RE = re.compile(r"\b(\d{1,2})\s*(?:y/o|years?\s*old)\b", re.I)
_EXCLUSION_RE = re.compile(
    r"\b(?:no|without|not)\s+([a-z0-9][\w\s\-/]{0,40}?)(?=,|$|\.)",
    re.I,
)
_ID_ONLY_RE = re.compile(r"^[a-z0-9_]+$", re.I)
DERIVE_SKIP_MARKERS = (
    "惊恐面部",
    "眼眶泛红",
    "瞬时表情",
    "特写眼睛",
    "close-up of eyes",
    "closeup of lips",
    "hand close-up",
    "手背",
)
_COSTUME_PATTERNS = (
    (
        re.compile(
            r"rain[- ]soaked|soaked\s+(?:clothes|coat|jacket|shirt|hair)|"
            r"wet\s+(?:clothes|coat|jacket|shirt|hair)|湿透|淋湿",
            re.I,
        ),
        "wet",
        "rain-soaked clothes, wet cloak and hair, same face",
    ),
    (re.compile(r"校服|制服|school uniform", re.I), "uniform", "school uniform, same face"),
    (re.compile(r"礼服|婚纱|formal dress|evening gown", re.I), "formal", "formal dress, same face"),
    (re.compile(r"战斗服|盔甲|armor|battle outfit", re.I), "battle", "battle outfit, same face"),
    (re.compile(r"变身|henshin|transform|兽化|巨大化", re.I), "transform", "transformed appearance, same face and body proportions"),
)
_TOD_PATTERNS = (
    (re.compile(r"\bnight\b|夜晚|夜里|深夜|nocturnal", re.I), "night", "night time, moonlight, street lamps, same architecture and camera"),
    (re.compile(r"\bdusk\b|黄昏|傍晚|sunset", re.I), "dusk", "dusk madder light, same architecture and camera"),
    (re.compile(r"\bdawn\b|清晨|黎明|sunrise", re.I), "dawn", "dawn light, same architecture and camera"),
)


class KolorsPayloadError(ValueError):
    pass


class LocationPromptError(KolorsPayloadError):
    """Location still prompt contains forbidden wording such as standalone `plate`."""


class CharacterIdentityError(KolorsPayloadError):
    """Character still lacks structured English visual identity."""


def sanitize_location_prompt(text: str) -> str:
    """Location positives must never carry standalone `plate` (Animagine reads it as dinnerware)."""
    out = scrub_copycat(str(text or ""))
    out = re.sub(r"\bestablishing\s+plate\b", "empty establishing shot", out, flags=re.I)
    out = re.sub(r"\bbackground\s+plate\b", "anime location background", out, flags=re.I)
    out = re.sub(r"\blocation\s+plate\b", "anime location background", out, flags=re.I)
    if _STANDALONE_PLATE_RE.search(out):
        raise LocationPromptError(f"location still prompt contains forbidden standalone 'plate': {out[:160]}")
    if _CJK_STILL_RE.search(out):
        raise LocationPromptError(f"location still prompt must be English-only: {out[:160]}")
    return out.strip(" ,")


def _gender_tag_from_fields(identity: str, gender: str | None = None) -> str:
    if _GENDER_TAG_RE.search(identity):
        return _GENDER_TAG_RE.search(identity).group(1).lower()  # type: ignore[union-attr]
    g = str(gender or "").strip().lower()
    if g in {"male", "m", "boy", "man"}:
        return "1boy"
    if g in {"female", "f", "girl", "woman"}:
        return "1girl"
    return ""


def parse_identity_age(identity: str | None, default: int | None = None) -> int | None:
    """Extract numeric age from identity text. Prefer explicit N y/o over band defaults."""
    text = str(identity or "")
    hit = _AGE_NUMBER_RE.search(text)
    if hit:
        return int(hit.group(1))
    low = text.lower()
    if "elderly" in low or "senior" in low:
        return 70
    if "child" in low:
        return 10
    if "teen" in low or "adolescent" in low:
        return 16
    if "young adult" in low:
        return 22
    if "middle" in low and "aged" in low:
        return 45
    if "adult" in low:
        return 30
    return default


def identity_exclusions(identity: str | None) -> list[str]:
    """Preserve negative wardrobe constraints such as 'no military uniform/epaulettes'."""
    out: list[str] = []
    for match in _EXCLUSION_RE.finditer(str(identity or "")):
        phrase = match.group(0).strip().rstrip(".,;")
        if phrase and phrase.lower() not in {x.lower() for x in out}:
            out.append(phrase)
    return out


def validate_character_identity(
    identity: str | None,
    *,
    character_id: str = "",
    gender: str | None = None,
    name: str | None = None,
) -> str:
    """Require structured English visual identity before any character still is drawn."""
    raw = scrub_copycat(str(identity or "").strip())
    cid = str(character_id or "").strip()
    label = cid or str(name or "character")
    if not raw:
        raise CharacterIdentityError(f"{label}: missing English visual identity (identity_prompt)")
    if _CJK_STILL_RE.search(raw):
        raise CharacterIdentityError(f"{label}: character identity must be English-only, not CJK wiki text")
    lowered = raw.lower()
    if _ID_ONLY_RE.match(raw) or lowered in {cid.lower(), str(name or "").strip().lower(), "hero", "lead"}:
        raise CharacterIdentityError(f"{label}: character identity cannot be an id or placeholder name")
    if "design sheet" in lowered and not (_HAIR_RE.search(raw) and _EYES_RE.search(raw) and _CLOTHING_RE.search(raw)):
        raise CharacterIdentityError(f"{label}: generic design-sheet wording is not a visual lock")
    missing: list[str] = []
    gender_tag = _gender_tag_from_fields(raw, gender)
    if not gender_tag:
        missing.append("1boy/1girl")
    if not _AGE_RE.search(raw):
        missing.append("age band")
    if not _HAIR_RE.search(raw):
        missing.append("hair")
    if not _EYES_RE.search(raw):
        missing.append("eyes")
    if not _CLOTHING_RE.search(raw):
        missing.append("fixed clothing/body")
    has_named_uniform = bool(
        re.search(
            r"\b(school|worker|janitor|cleaning|maintenance|utility|rescue|safety)\s+uniform\b",
            raw,
            re.I,
        )
    )
    if not _CLOTHING_COLOR_RE.search(raw) and not has_named_uniform:
        missing.append("clothing color")
    if _JACKET_RE.search(raw) and not _SHIRT_RE.search(raw) and not _SELF_LAYER_RE.search(raw):
        # Occupational "worker uniform jacket" is a self-layer; don't demand a shirt under it.
        if not re.search(r"\b(worker|janitor|cleaning|maintenance|utility)\s+uniform\b", raw, re.I):
            missing.append("jacket/shirt layer")
    if missing:
        raise CharacterIdentityError(
            f"{label}: character identity missing {', '.join(missing)}: {raw[:160]}"
        )
    if not _GENDER_TAG_RE.search(raw):
        raw = f"{gender_tag}, {raw}"
    lowered_out = raw.lower()
    if gender_tag == "1boy" and "male focus" not in lowered_out:
        raw = re.sub(r"\b1boy\b", "1boy, male focus", raw, count=1, flags=re.I)
    if re.search(r"\bnavy\b", raw, re.I) and re.search(r"\bjacket\b", raw, re.I):
        if "navy jacket" not in raw.lower() and "navy blue" not in raw.lower():
            raw = f"{raw}, navy jacket, dark blue jacket"
    if re.search(r"\bblack\b", raw, re.I) and re.search(r"\b(trousers|pants)\b", raw, re.I):
        if "black pants" not in raw.lower():
            raw = f"{raw}, black pants, black trousers"
    # Keep exclusion clauses intact (military uniform bans, etc.).
    return raw[:IDENTITY_PROMPT_MAX_CHARS]


def identity_conditioned_negative(identity: str | None) -> str:
    """CFG negatives that depend on the locked identity. Hoodie identities must not ban hoodie."""
    text = str(identity or "").lower()
    extra: list[str] = []
    if "1boy" in text and "1girl" not in text:
        extra.extend(["1girl", "feminine"])
    if "1girl" in text and "1boy" not in text:
        extra.extend(["1boy", "masculine"])
    if "hoodie" not in text:
        extra.extend(["hoodie", "oversized hoodie"])
    if "black pants" in text or "black trousers" in text:
        extra.extend(["khaki pants", "olive pants", "beige pants", "grey sweatpants"])
    if "long hair" not in text:
        extra.append("long hair")
    extra.extend(["child", "loli", "old man", "mismatched footwear"])
    return ", ".join(extra)


def _kind_negative(kind: str | None, identity: str | None = None) -> str:
    """Per-asset-kind negatives. A turnaround must not be cinematic; a plate must not be famous."""
    base = {
        "character_sheet": SHEET_NEGATIVE,
        "character_view_derive": SHEET_NEGATIVE,
        "costume_derive": SHEET_NEGATIVE,
        "scene_plate": PLATE_NEGATIVE,
        "prop": PROP_NEGATIVE,
    }.get(str(kind or ""), "")
    if str(kind or "") in {"character_sheet", "character_view_derive", "costume_derive"}:
        extra = identity_conditioned_negative(identity)
        return ", ".join(part for part in (base, extra) if part)
    return base


def style_prompt(
    user_prompt: str,
    period_positive: list[str] | None = None,
    prefix: str | None = None,
    *,
    kind: str | None = None,
    gender_tag: str | None = None,
) -> str:
    extra = ""
    if period_positive:
        extra = ", " + ", ".join(period_positive)
    head = scrub_copycat(prefix if prefix is not None else style_prefix_for_kind(kind))
    body = scrub_copycat(user_prompt)
    if kind in {"scene_plate", "scene_derive", "keyframe"}:
        body = sanitize_location_prompt(body)
    if body.startswith(head):
        composed = f"{body}{extra}"
    else:
        composed = f"{head}, {body}{extra}"
    tail = animagine_quality_suffix(kind, gender_tag=gender_tag)
    if tail.lower() not in composed.lower():
        composed = f"{composed}, {tail}"
    return composed


def style_negative(
    period_negative: list[str] | None = None,
    *,
    base: str | None = None,
    extra: str | None = None,
) -> str:
    """FIXED_NEGATIVE (or the bible's `negative:` line) plus per-kind and period terms."""
    terms = [scrub_copycat(base) if base else FIXED_NEGATIVE]
    if extra:
        terms.append(extra)
    terms.extend(period_negative or [])
    return ", ".join(term for term in (str(t).strip() for t in terms) if term)


def build_scene_plate_payload(
    prompt: str,
    seed: int | None = None,
    period_md: str | None = None,
    world_mode: str = "fiction",
    prefix: str | None = None,
    negative_base: str | None = None,
) -> dict[str, Any]:
    positives = negatives = []
    if world_mode != "fiction" and period_md:
        lists = parse_period_md(period_md)
        positives, negatives = lists.positive, lists.negative
    payload: dict[str, Any] = {
        "model": IMAGE_MODEL,
        "prompt": style_prompt(
            prompt,
            positives if world_mode != "fiction" else None,
            prefix=prefix,
            kind="scene_plate",
        ),
        "negative_prompt": style_negative(
            negatives if world_mode != "fiction" else None,
            base=negative_base,
            extra=PLATE_NEGATIVE,
        ),
        "image_size": SCENE_IMAGE_SIZE,
        "batch_size": 1,
        "num_inference_steps": KOLORS_STEPS,
        "_styled": True,
    }
    if seed is not None:
        payload["seed"] = seed
    banned = FORBIDDEN_PLATE_KEYS.intersection(payload)
    if banned:
        raise KolorsPayloadError(f"scene plate payload contains forbidden keys {banned}")
    return payload


def assert_no_reference_images(payload: dict) -> None:
    for key in FORBIDDEN_PLATE_KEYS:
        if key in payload:
            raise KolorsPayloadError(f"scene plate must not include {key!r}")


def _png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _with_parent_reference(payload: dict[str, Any], parent_png: bytes | None) -> dict[str, Any]:
    """Parent locked sheet is referenceList[0]. Scene plates must never call this."""
    if not parent_png:
        return payload
    url = _png_data_url(parent_png)
    payload["image"] = url
    payload["reference_images"] = [url]
    return payload


def _image_url_from_kolors(parsed: dict) -> str | None:
    if parsed.get("images"):
        return parsed["images"][0].get("url")
    if parsed.get("data"):
        first = parsed["data"][0]
        if isinstance(first, dict):
            return first.get("url")
    return parsed.get("url")


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# Offline/dev placeholders carry this tEXt marker so a live run can refuse to
# persist one as if it were art.
PLACEHOLDER_KEYWORD = b"af-placeholder"


class StillBlobError(KolorsPayloadError):
    """A still blob is not a usable image (stub bytes, wrong size, placeholder on a live run)."""


def parse_image_size(
    image_size: str | None,
    default: tuple[int, int] = (STILL_WIDTH, STILL_HEIGHT),
) -> tuple[int, int]:
    raw = str(image_size or "").lower().replace(" ", "")
    if "x" in raw:
        a, b = raw.split("x", 1)
        try:
            return max(int(a), 64), max(int(b), 64)
        except ValueError:
            pass
    return default


def png_dimensions(blob: bytes | bytearray | None) -> tuple[int, int] | None:
    """Width/height straight out of IHDR, so a 3-byte `b"PNG"` cannot claim a size."""
    data = bytes(blob or b"")
    if len(data) < 24 or data[:8] != PNG_MAGIC or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def synthetic_still_png(width: int, height: int, *, tag: str = "", placeholder: bool = False) -> bytes:
    """A real PNG of the requested size for offline paths — never a 3-byte stub.

    Content is deterministic noise derived from `tag`, so two different offline
    rolls differ and the blob is large enough to clear MIN_STILL_BYTES.
    """
    w, h = max(int(width), 64), max(int(height), 64)
    seed = hashlib.sha256(f"{tag}:{w}x{h}".encode("utf-8")).digest()
    block = b"".join(hashlib.sha256(seed + bytes([i])).digest() for i in range(64))
    raw = bytearray()
    for y in range(h):
        offset = (y * 37) % len(block)
        row = (block[offset:] + block[:offset]) * (w // len(block) + 1)
        raw.append(0)
        raw.extend(row[:w])
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    chunks = [PNG_MAGIC, _png_chunk(b"IHDR", ihdr)]
    if placeholder:
        chunks.append(_png_chunk(b"tEXt", PLACEHOLDER_KEYWORD + b"\x00" + (tag or "offline").encode("utf-8")))
    chunks.append(_png_chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
    chunks.append(_png_chunk(b"IEND", b""))
    png = b"".join(chunks)
    if len(png) < MIN_STILL_BYTES:
        # Small canvases compress under the fail-closed floor; pad in a chunk
        # decoders ignore so the blob still validates like a real still.
        pad = _png_chunk(b"tEXt", b"af-pad\x00" + block * (MIN_STILL_BYTES // len(block) + 1))
        png = png[: -len(chunks[-1])] + pad + chunks[-1]
    return png


def is_placeholder_still(blob: bytes | bytearray | None) -> bool:
    return PLACEHOLDER_KEYWORD in bytes(blob or b"")[:4096]


def assert_still_blob(
    blob: bytes | bytearray | None,
    *,
    width: int | None = None,
    height: int | None = None,
    label: str = "still",
    allow_placeholder: bool = False,
) -> bytes:
    """Fail closed on anything that is not a real image of the requested size."""
    data = bytes(blob or b"")
    if len(data) < MIN_STILL_BYTES:
        raise StillBlobError(f"{label}: still is {len(data)} bytes, need >= {MIN_STILL_BYTES}")
    if data[:8] != PNG_MAGIC:
        raise StillBlobError(f"{label}: still is not a PNG (magic {data[:8]!r})")
    size = png_dimensions(data)
    if size is None:
        raise StillBlobError(f"{label}: PNG has no readable IHDR")
    if width and height and size != (int(width), int(height)):
        raise StillBlobError(f"{label}: still is {size[0]}x{size[1]}, requested {width}x{height}")
    if not allow_placeholder and is_placeholder_still(data):
        raise StillBlobError(f"{label}: refusing to persist an offline placeholder still")
    return data


def still_file_ok(path: Path | None, *, allow_placeholder: bool = True) -> bool:
    """An existing still on disk counts as rendered only if it is a real PNG.

    `allow_placeholder=False` also rejects dry-run placeholders, so a live run
    redraws them instead of treating them as finished art.
    """
    if path is None:
        return False
    try:
        if not path.is_file() or path.stat().st_size < MIN_STILL_BYTES:
            return False
        with path.open("rb") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    if head[:8] != PNG_MAGIC:
        return False
    return allow_placeholder or PLACEHOLDER_KEYWORD not in head


class KolorsClient:
    def __init__(
        self,
        api_keys: list[str],
        opener: Callable[[Request], dict] | None = None,
        live: bool | None = None,
        min_interval_s: float | None = None,
        gpu_generate: Callable[[dict], bytes] | None = None,
    ):
        self.api_keys = api_keys
        self.opener = opener
        self.live = load_settings().live_kolors if live is None else live
        self.gpu_generate = gpu_generate
        self._i = 0
        self.last_payloads: list[dict] = []
        self._last_call = 0.0
        if min_interval_s is None:
            min_interval_s = float(os.environ.get("SILICONFLOW_IMAGE_MIN_INTERVAL_SECONDS") or 0)
        self.min_interval_s = min_interval_s
        if gpu_generate is None and os.environ.get("ANIME_FACTORY_GPU_STILLS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            from gpu_worker.stills import generate_still

            self.gpu_generate = generate_still

    def _key(self) -> str:
        if not self.api_keys:
            return ""
        k = self.api_keys[self._i % len(self.api_keys)]
        self._i += 1
        return k

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        wait = self.min_interval_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def generate(self, payload: dict) -> bytes:
        if payload.get("_kind") in {"scene_plate", "scene_derive"}:
            # POSTMORTEM_KOLORS_REFERENCE_IMAGE_BLEED: a plate never binds a parent image,
            # not even a time-of-day derive of another plate.
            assert_no_reference_images(payload)
            if payload.get("_parent_png"):
                raise KolorsPayloadError("scene plate must not bind a parent sheet as a reference image")
        parent_png = payload.get("_parent_png")
        if parent_png and payload.get("_kind") not in {"scene_plate", "scene_derive"}:
            blob = parent_png if isinstance(parent_png, (bytes, bytearray)) else None
            if blob:
                payload = _with_parent_reference(dict(payload), bytes(blob))
        self.last_payloads.append(payload)
        Counters.kolors_requests += 1
        width, height = parse_image_size(payload.get("image_size"))
        label = str(payload.get("_label") or payload.get("_kind") or "still")
        if self.gpu_generate is not None:
            gpu_payload = dict(payload)
            gpu_payload["model"] = IMAGE_MODEL
            return self._checked(self.gpu_generate(gpu_payload), width, height, label)
        # One size, one aspect. The old ladder retried 1024x1024 / 768x1024 and is
        # why square and portrait sheets are sitting in finished stories.
        attempt = dict(payload)
        attempt["model"] = KOLORS_MODEL
        attempt["image_size"] = f"{width}x{height}"
        body = json.dumps({k: v for k, v in attempt.items() if not str(k).startswith("_")}).encode()
        req = Request(
            f"{SILICONFLOW_BASE_URL}/v1/images/generations",
            data=body,
            headers={"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"},
            method="POST",
        )
        if self.opener:
            data = self.opener(req)
            blob = data.get("bytes") if isinstance(data, dict) else None
            if blob is None:
                blob = self._offline_still(attempt, width, height)
            return self._checked(blob, width, height, label)
        if not self.live:
            return self._checked(self._offline_still(attempt, width, height), width, height, label)
        self._throttle()
        try:
            with urlopen(req, timeout=180) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — fail closed; retrying used to change the aspect
            self._last_call = time.time()
            raise KolorsPayloadError(f"kolors request failed: {exc}") from exc
        self._last_call = time.time()
        if isinstance(parsed, dict) and parsed.get("bytes"):
            return self._checked(parsed["bytes"], width, height, label)
        url = _image_url_from_kolors(parsed) if isinstance(parsed, dict) else None
        if not url:
            keys = list(parsed)[:8] if isinstance(parsed, dict) else type(parsed)
            raise KolorsPayloadError(f"kolors returned no image url: keys={keys}")
        with urlopen(url, timeout=120) as img:
            blob = img.read()
        return self._checked(blob, width, height, label)

    def _offline_still(self, payload: dict, width: int, height: int) -> bytes:
        """Dry-run stand-in. A real PNG of the requested size, tagged as a placeholder."""
        tag = f"{payload.get('prompt') or ''}|{payload.get('seed')}"
        return synthetic_still_png(width, height, tag=tag, placeholder=True)

    def _checked(self, blob: Any, width: int, height: int, label: str) -> bytes:
        # Dimensions are only a promise the live generator can break; a dry run
        # stand-in is allowed to be any real PNG.
        return assert_still_blob(
            blob if isinstance(blob, (bytes, bytearray)) else b"",
            width=width if self.live else None,
            height=height if self.live else None,
            label=label,
            allow_placeholder=not self.live,
        )

    def generate_scene_plate(
        self,
        prompt: str,
        seed: int | None,
        period_md: str | None,
        world_mode: str,
        prefix: str | None = None,
        negative_base: str | None = None,
    ) -> tuple[dict, bytes]:
        payload = build_scene_plate_payload(
            prompt, seed, period_md, world_mode, prefix=prefix, negative_base=negative_base
        )
        payload["_kind"] = "scene_plate"
        assert_no_reference_images(payload)
        png = self.generate(payload)
        return payload, png


def _parent_png_bytes(story_root: Path | None, spec: dict) -> bytes | None:
    if story_root is None:
        return None
    cid = str(spec.get("character_id") or "").strip()
    if spec.get("kind") == "character_view_derive" and cid:
        if not is_qc_locked(story_root, character_id=cid):
            return None
        locked = locked_character_file(story_root, cid)
        if still_file_ok(locked):
            return locked.read_bytes()
        return None
    parent_id = str(spec.get("parent_id") or "").strip()
    lid = str(spec.get("scene_id") or "").strip()
    path: Path | None = None
    if parent_id:
        if parent_id.startswith("char_") and parent_id.endswith("_sheet"):
            pid = parent_id[len("char_") : -len("_sheet")]
            if not is_qc_locked(story_root, character_id=pid):
                return None
            path = locked_character_file(story_root, pid)
        elif parent_id.startswith("plate_"):
            pid = parent_id[len("plate_") :]
            if not is_qc_locked(story_root, scene_id=pid):
                return None
            path = locked_scene_file(story_root, pid)
        else:
            if is_qc_locked(story_root, character_id=parent_id):
                path = locked_character_file(story_root, parent_id)
            elif is_qc_locked(story_root, scene_id=parent_id):
                path = locked_scene_file(story_root, parent_id)
            else:
                return None
    elif spec.get("kind") in {"costume_derive", "character_sheet"} and cid:
        parent_cid = str(spec.get("parent_id") or cid)
        if parent_cid != cid or spec.get("kind") == "costume_derive":
            if not is_qc_locked(story_root, character_id=parent_cid):
                return None
            path = locked_character_file(story_root, parent_cid)
    if path is None or not path.is_file() or path.stat().st_size < 1:
        return None
    return path.read_bytes()


def _script_blob(story_root: Path | None) -> str:
    if story_root is None:
        return ""
    parts: list[str] = []
    for rel in (
        "episodes/EP001/script.json",
        "episodes/EP001/board.json",
        "bible.json",
        "bible/bible.json",
    ):
        path = Path(story_root) / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            parts.append(json.dumps(data, ensure_ascii=False))
        except json.JSONDecodeError:
            parts.append(text)
    return "\n".join(parts)


def _skip_expression_blob(blob: str) -> bool:
    low = blob.lower()
    return any(marker.lower() in low for marker in DERIVE_SKIP_MARKERS)


def _scene_rows(story_root: Path | None) -> list[dict]:
    rows: list[dict] = []
    if story_root is None:
        return rows
    for rel in ("episodes/EP001/script.json", "episodes/EP001/board.json"):
        path = Path(story_root) / rel
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        for scene in data.get("scenes") or []:
            if isinstance(scene, dict):
                rows.append(scene)
    return rows


def plan_derive_specs(
    characters: list[dict],
    locations: list[dict],
    story_root: Path | None = None,
    period_md: str | None = None,
    world_mode: str = "fiction",
) -> dict[str, dict]:
    """Costume/transform for characters, time-of-day for scenes. Instant expressions are not assets."""
    blob = _script_blob(story_root)
    for row in _board_payloads(story_root):
        blob += " " + json.dumps(row, ensure_ascii=False)
    if _skip_expression_blob(blob) and not any(p[0].search(blob) for p in _COSTUME_PATTERNS + _TOD_PATTERNS):
        return {}
    pos, _neg = period_lists(period_md, world_mode)
    extra = (", " + ", ".join(pos)) if pos else ""
    specs: dict[str, dict] = {}
    for char in characters:
        cid = str(char.get("id") or "").strip()
        if not cid:
            continue
        identity = validate_character_identity(
            char.get("identity_prompt") or char.get("visual_lock_prompt"),
            character_id=cid,
            gender=char.get("gender"),
            name=char.get("name"),
        )
        seed = char.get("seed") if char.get("seed") is not None else locked_seed(cid)
        costumes: dict[str, str] = {}
        current = str(char.get("current_costume_id") or "").strip()
        if current and current not in {cid, locked_character_relpath(cid), "default"}:
            costumes[current] = f"{current} costume, {DERIVE_FACE_LOCK}"
        for rx, slug, desc in _COSTUME_PATTERNS:
            if rx.search(blob):
                costumes.setdefault(slug, f"{desc}, {DERIVE_FACE_LOCK}")
        for slug, desc in list(costumes.items())[:5]:
            child = f"{cid}_{slug}"
            aid = locked_character_relpath(child)
            specs[aid] = {
                "id": aid,
                "kind": "costume_derive",
                "view": "turnaround",
                "character_id": child,
                "parent_id": cid,
                "costume_id": slug,
                "path": character_asset_rel(child, TURNAROUND_FILENAME),
                "alias_path": character_asset_rel(child, FRONT_ALIAS_FILENAME),
                "prompt": f"{identity}, {TURNAROUND_LOOK}, {desc}{extra}",
                "seed": int(seed),
                "image_size": CHAR_IMAGE_SIZE,
            }
    loc_times: dict[str, dict[str, str]] = {}
    for row in list(_board_payloads(story_root)) + _scene_rows(story_root):
        lid = str(row.get("location_id") or "").strip()
        tod = str(row.get("time_of_day") or "").strip().lower()
        if not lid or not tod:
            continue
        for rx, slug, desc in _TOD_PATTERNS:
            if slug == tod or rx.search(tod):
                loc_times.setdefault(lid, {})[slug] = desc
    for loc in locations:
        lid = str(loc.get("id") or "").strip()
        if not lid:
            continue
        prompt = sanitize_location_prompt(loc.get("plate_prompt") or loc.get("identity_prompt") or lid)
        seed = loc.get("seed") if loc.get("seed") is not None else locked_seed(f"loc:{lid}")
        for slug, desc in list((loc_times.get(lid) or {}).items())[:5]:
            if re.search(slug, str(prompt), re.I):
                continue
            if any(rx.search(str(prompt)) for rx, s, _d in _TOD_PATTERNS if s == slug):
                continue
            child = f"{lid}_{slug}"
            aid = f"plate_{child}"
            specs[aid] = {
                "id": aid,
                "kind": "scene_plate",
                "scene_id": child,
                "parent_id": lid,
                "path": scene_asset_rel(child, PLATE_FILENAME),
                "prompt": f"{prompt}, {desc}, {PLATE_LOOK}{extra}",
                "seed": int(seed) + abs(locked_seed(slug)) % 997,
                "image_size": SCENE_IMAGE_SIZE,
                "_kind": "scene_derive",
            }
    return specs


def asset_path(story_id: str, rel: str) -> str:
    return join_story(story_id, rel)


def derive_missing_assets(
    conn: sqlite3.Connection,
    story_id: str,
    missing_ids: list[str],
    specs: dict[str, dict],
    client: KolorsClient,
    period_md: str | None,
    world_mode: str,
    story_root: Path | None = None,
    clip_scorer: ClipScorer | None = None,
) -> list[str]:
    """Missing sheet/plate → derive first, no freehand Kolors. Lock only after QC pass."""
    created = []
    require_clip = require_clip_for_client(client)
    prefix, bible_negative = style_md_prefix(story_root)
    pos, neg = period_lists(period_md, world_mode)
    for aid in missing_ids:
        spec = specs.get(aid)
        if spec is None:
            raise KolorsPayloadError(f"cannot freehand Kolors for unknown asset {aid}")
        if story_root is not None and not _parent_ready(story_root, spec):
            continue
        if story_root is not None and _bind_locked(story_root, spec):
            continue
        png, qc, rel = _render_until_qc(
            spec,
            client,
            period_md,
            world_mode,
            prefix,
            pos,
            neg,
            story_root,
            negative_base=bible_negative,
            clip_scorer=clip_scorer,
            require_clip=require_clip,
        )
        if png is None or rel is None:
            continue
        key = asset_path(story_id, rel)
        _upsert_asset_row(
            conn,
            aid,
            spec,
            key,
            spec.get("prompt") or aid,
            spec.get("seed"),
            png,
            selected_image_id=Path(rel).name,
        )
        created.append(aid)
    conn.commit()
    return created


def costume_parent_chain(conn: sqlite3.Connection, asset_id: str) -> list[str]:
    chain = []
    current = asset_id
    seen = set()
    while current:
        if current in seen:
            break
        seen.add(current)
        chain.append(current)
        row = conn.execute("SELECT parent_id FROM assets WHERE id = ?", (current,)).fetchone()
        current = row["parent_id"] if row else None
    return chain


def locked_seed(stable_id: str, salt: int = 0) -> int:
    digest = hashlib.sha256(f"{stable_id}:{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def attempt_seed(spec: dict, salt: int) -> int:
    if salt == 0 and spec.get("seed") is not None:
        return int(spec["seed"])
    kind = str(spec.get("kind") or "")
    cid = str(spec.get("character_id") or "").strip()
    lid = str(spec.get("scene_id") or "").strip()
    if kind == "costume_derive":
        return locked_seed(str(spec.get("id") or cid), salt)
    if cid and kind != "scene_plate":
        seed = locked_seed(cid, salt)
        if kind == "character_view_derive":
            extra = 101 if spec.get("view") == "side" else 211 if spec.get("view") == "back" else 0
            return seed + extra
        return seed
    if lid:
        return locked_seed(f"loc:{lid}", salt)
    return locked_seed(str(spec.get("id") or "asset"), salt)


def _history_stem(spec: dict) -> str:
    kind = str(spec.get("kind") or "")
    if kind == "character_sheet":
        return "sheet_front"
    if kind == "character_view_derive":
        return "sheet_side" if spec.get("view") == "side" else "sheet_back"
    if kind in {"costume_derive", "character_turnaround"}:
        return "sheet_turnaround"
    if kind == "prop":
        return "turnaround"
    return "plate_base"


def _parent_ready(story_root: Path | None, spec: dict) -> bool:
    if story_root is None:
        return True
    kind = spec.get("kind")
    if kind in {"character_view_derive", "character_turnaround"}:
        cid = str(spec.get("parent_id") or spec.get("character_id") or "")
        return bool(cid) and is_qc_locked(story_root, character_id=cid)
    if kind == "costume_derive":
        parent = str(spec.get("parent_id") or "")
        return bool(parent) and is_qc_locked(story_root, character_id=parent)
    if kind in {"scene_plate", "scene_derive"} and spec.get("parent_id"):
        return is_qc_locked(story_root, scene_id=str(spec.get("parent_id")))
    return True


def _qc_kind(spec: dict) -> str:
    kind = str(spec.get("kind") or "")
    if kind == "scene_plate" and spec.get("_kind") == "scene_derive":
        return "scene_derive"
    return kind


def _candidate_rel(story_root: Path, spec: dict, salt: int) -> str:
    """Pick a writable candidate path. Never overwrite an existing file, including salt 0."""
    _ = salt
    rel = str(spec.get("path") or "")
    kind = spec.get("kind")
    history: list[str] = []
    if kind in {"character_sheet", "character_view_derive", "costume_derive", "character_turnaround"}:
        cid = str(spec.get("character_id") or "")
        rec = (load_assets_index(story_root).get("characters") or {}).get(cid) or {}
        history = list(rec.get("history") or [])
        name = Path(rel).name if rel and not (Path(story_root) / rel).is_file() else ""
        if name and name not in history:
            return rel
        name = next_history_filename(history, _history_stem(spec))
        folder = Path(story_root) / "assets" / "characters" / cid
        while (folder / name).is_file():
            history.append(name)
            name = next_history_filename(history, _history_stem(spec))
        return character_asset_rel(cid, name)
    if kind == "scene_plate":
        lid = str(spec.get("scene_id") or "")
        rec = (load_assets_index(story_root).get("scenes") or {}).get(lid) or {}
        history = list(rec.get("history") or [])
        name = Path(rel).name if rel and not (Path(story_root) / rel).is_file() else ""
        if name and name not in history:
            return rel
        name = next_history_filename(history, "plate_base")
        folder = Path(story_root) / "assets" / "scenes" / lid
        while (folder / name).is_file():
            history.append(name)
            name = next_history_filename(history, "plate_base")
        return scene_asset_rel(lid, name)
    return rel


def _store_qc_outcome(
    story_root: Path | None,
    spec: dict,
    filename: str,
    qc: Any,
    *,
    lock: bool,
) -> None:
    if story_root is None:
        return
    kind = spec.get("kind")
    if kind == "prop":
        return
    if kind == "character_turnaround":
        record_qc_candidate(
            story_root,
            character_id=spec.get("character_id"),
            filename=filename,
            qc=qc,
            parent_id=spec.get("parent_id"),
            max_attempts=MAX_QC_ATTEMPTS,
        )
        return
    cid = spec.get("character_id") if kind != "scene_plate" else None
    lid = spec.get("scene_id") if kind == "scene_plate" else None
    if not cid and not lid:
        return
    owns_lock = kind in {"character_sheet", "costume_derive", "scene_plate"}
    if lock and getattr(qc, "verdict", None) == "pass" and owns_lock:
        lock_after_qc(
            story_root,
            character_id=cid,
            scene_id=lid,
            filename=filename,
            qc=qc,
            parent_id=spec.get("parent_id"),
            set_selected=True,
        )
        return
    record_qc_candidate(
        story_root,
        character_id=cid,
        scene_id=lid,
        filename=filename,
        qc=qc,
        parent_id=spec.get("parent_id"),
        max_attempts=MAX_QC_ATTEMPTS,
    )


def style_md_prefix(story_root: Path | None) -> tuple[str, str]:
    """bible/style.md prefix + fixed negative on every call."""
    selected_prefix = scrub_copycat(style_prefix_for_kind("scene_plate"))
    prefix, negative = selected_prefix, FIXED_NEGATIVE
    if story_root is None:
        return prefix, negative
    style_path = story_root / "bible" / "style.md"
    if not style_path.is_file():
        return prefix, negative
    text = style_path.read_text(encoding="utf-8").strip()
    if not text:
        return prefix, negative
    neg_line = ""
    pos_lines = []
    for line in text.splitlines():
        if line.lower().startswith("negative:"):
            neg_line = line.split(":", 1)[1].strip()
        else:
            pos_lines.append(line)
    bible_prefix = scrub_copycat(" ".join(p.strip() for p in pos_lines if p.strip()) or prefix)
    if normalize_style_preset() == DEFAULT_STYLE_PRESET:
        prefix = bible_prefix
    elif bible_prefix and bible_prefix not in {STYLE_PREFIX, selected_prefix}:
        prefix = f"{selected_prefix}, {bible_prefix}"
    if neg_line:
        negative = scrub_copycat(neg_line)
    return prefix, negative


def period_lists(period_md: str | None, world_mode: str) -> tuple[list[str] | None, list[str] | None]:
    if world_mode == "fiction" or not period_md:
        return None, None
    lists = parse_period_md(period_md)
    return lists.positive, lists.negative


def plan_library_specs(
    characters: list[dict],
    locations: list[dict],
    props: list[dict],
    interiors: list[dict] | None = None,
    period_md: str | None = None,
    world_mode: str = "fiction",
    style_prefix: str | None = None,
) -> dict[str, dict]:
    """Build the standard library: one four-view turnaround / character, one plate / location, props."""
    pos, _neg = period_lists(period_md, world_mode)
    extra = (", " + ", ".join(pos)) if pos else ""
    specs: dict[str, dict] = {}
    for char in characters:
        cid = char["id"]
        name = char.get("name") or cid
        identity = validate_character_identity(
            char.get("identity_prompt") or char.get("visual_lock_prompt"),
            character_id=cid,
            gender=char.get("gender"),
            name=name,
        )
        gender_tag = _gender_tag_from_fields(identity, char.get("gender"))
        seed = char.get("seed") if char.get("seed") is not None else locked_seed(cid)
        aid = locked_character_relpath(cid)
        specs[aid] = {
            "id": aid,
            "kind": "character_sheet",
            "view": "front",
            "character_id": cid,
            "path": character_asset_rel(cid, FRONT_ALIAS_FILENAME),
            "prompt": f"{identity}, {FRONT_LOOK}{extra}",
            "gender_tag": gender_tag,
            "seed": int(seed),
            "image_size": CHAR_IMAGE_SIZE,
            "parent_id": char.get("parent_id"),
        }
        for view, filename, look, seed_offset in (
            ("side", CHAR_SIDE_FILENAME, SIDE_LOOK, 101),
            ("back", CHAR_BACK_FILENAME, BACK_LOOK, 211),
        ):
            view_aid = f"char_{cid}_{view}"
            specs[view_aid] = {
                "id": view_aid,
                "kind": "character_view_derive",
                "view": view,
                "character_id": cid,
                "path": character_asset_rel(cid, filename),
                "prompt": f"{identity}, {look}, {DERIVE_FACE_LOCK}{extra}",
                "gender_tag": gender_tag,
                "seed": int(seed) + seed_offset,
                "image_size": CHAR_IMAGE_SIZE,
                "parent_id": cid,
            }
        turnaround_aid = f"char_{cid}_turnaround"
        specs[turnaround_aid] = {
            "id": turnaround_aid,
            "kind": "character_turnaround",
            "view": "turnaround",
            "character_id": cid,
            "path": character_asset_rel(cid, TURNAROUND_FILENAME),
            "source_paths": [
                character_asset_rel(cid, FRONT_ALIAS_FILENAME),
                character_asset_rel(cid, CHAR_SIDE_FILENAME),
                character_asset_rel(cid, CHAR_BACK_FILENAME),
            ],
            "prompt": f"deterministic three-view composite, {identity}",
            "seed": int(seed),
        }
    seen_scenes: set[str] = set()
    for loc in locations:
        lid = loc["id"]
        name = loc.get("name") or lid
        raw_prompt = loc.get("plate_prompt") or loc.get("location_prompt") or loc.get("identity_prompt")
        if not raw_prompt and _CJK_STILL_RE.search(str(name or "")):
            raw_prompt = f"{lid} anime location background, empty establishing shot, no people, no text"
        prompt = sanitize_location_prompt(
            raw_prompt or f"{name} anime location background, empty establishing shot, no people, no text"
        )
        seed = loc.get("seed") if loc.get("seed") is not None else locked_seed(f"loc:{lid}")
        aid = f"plate_{lid}"
        specs[aid] = {
            "id": aid,
            "kind": "scene_plate",
            "scene_id": lid,
            "path": f"assets/scenes/{lid}/plate_base.png",
            "prompt": f"{prompt}, {PLATE_LOOK}{extra}",
            "seed": int(seed),
            "image_size": SCENE_IMAGE_SIZE,
        }
        seen_scenes.add(lid)
    for interior in interiors or []:
        scene_id = interior.get("scene_id") or interior["id"]
        if scene_id in seen_scenes:
            continue
        name = interior.get("name") or scene_id
        prompt = sanitize_location_prompt(
            interior.get("plate_prompt") or f"{name} interior, anime location background, empty, no people, no text"
        )
        seed = locked_seed(f"int:{scene_id}")
        aid = f"plate_{scene_id}"
        specs[aid] = {
            "id": aid,
            "kind": "scene_plate",
            "scene_id": scene_id,
            "path": interior.get("asset_path") or f"assets/scenes/{scene_id}/plate_base.png",
            "prompt": f"{prompt}, {PLATE_LOOK}{extra}",
            "seed": int(seed),
            "image_size": SCENE_IMAGE_SIZE,
        }
        seen_scenes.add(scene_id)
    for prop in props:
        pid = prop["id"]
        name = prop.get("name") or pid
        prompt = prop.get("identity_prompt") or f"key prop {name}, product shot, no people, no text"
        seed = prop.get("seed") if prop.get("seed") is not None else locked_seed(f"prop:{pid}")
        aid = f"prop_{pid}"
        specs[aid] = {
            "id": aid,
            "kind": "prop",
            "prop_id": pid,
            "path": f"assets/props/{pid}/turnaround.png",
            "prompt": f"{prompt}, {PROP_LOOK}{extra}",
            "seed": int(seed),
            "image_size": PROP_IMAGE_SIZE,
        }
    return specs


def _asset_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(assets)")}


def _upsert_asset_row(
    conn: sqlite3.Connection,
    aid: str,
    spec: dict,
    key: str,
    prompt: str,
    seed: int | None,
    png: bytes,
    *,
    selected_image_id: str | None = None,
    history: list[str] | None = None,
) -> None:
    fingerprint = "sha256:" + hashlib.sha256(png).hexdigest()
    cols = _asset_columns(conn)
    # Keep the frozen sqlite enum compatible with existing story databases.
    # The exact stage remains available in spec["kind"] and assets/index.json.
    db_kind = (
        "character_sheet"
        if spec.get("kind") in {"character_view_derive", "character_turnaround"}
        else spec["kind"]
    )
    fields = [
        "id",
        "kind",
        "path",
        "prompt",
        "seed",
        "fingerprint",
        "parent_id",
        "costume_id",
        "character_id",
        "scene_id",
        "prop_id",
        "created_at",
    ]
    values: list[Any] = [
        aid,
        db_kind,
        key,
        prompt,
        seed,
        fingerprint,
        spec.get("parent_id"),
        spec.get("costume_id"),
        spec.get("character_id"),
        spec.get("scene_id"),
        spec.get("prop_id"),
        utcnow(),
    ]
    updates = [
        "fingerprint = excluded.fingerprint",
        "path = excluded.path",
        "prompt = excluded.prompt",
        "seed = excluded.seed",
    ]
    if "selected_image_id" in cols:
        fields.append("selected_image_id")
        values.append(selected_image_id or spec.get("selected_image_id") or Path(spec.get("path") or "").name)
        updates.append("selected_image_id = excluded.selected_image_id")
    if "history_json" in cols:
        hist = history if history is not None else spec.get("history")
        fields.append("history_json")
        values.append(json.dumps(hist or [], ensure_ascii=False))
        updates.append("history_json = excluded.history_json")
    placeholders = ", ".join("?" for _ in fields)
    conn.execute(
        f"INSERT INTO assets ({', '.join(fields)}) VALUES ({placeholders}) "
        f"ON CONFLICT(id) DO UPDATE SET {', '.join(updates)}",
        values,
    )


def write_assets_index(story_root: Path, story_id: str, specs: dict[str, dict], created: list[str]) -> Path:
    items = []
    extra_index = load_assets_index(story_root)
    for aid, spec in specs.items():
        rel = spec.get("path") or f"assets/{aid}.png"
        items.append(
            {
                "id": aid,
                "kind": spec["kind"],
                "view": spec.get("view"),
                "path": rel,
                "r2": join_story(story_id, rel),
                "seed": spec.get("seed"),
                "character_id": spec.get("character_id"),
                "scene_id": spec.get("scene_id"),
                "prop_id": spec.get("prop_id"),
                "parent_id": spec.get("parent_id"),
                "generated": aid in created,
            }
        )
        cid = spec.get("character_id")
        lid = spec.get("scene_id")
        filename = Path(rel).name
        if spec.get("kind") in {"character_sheet", "character_view_derive", "character_turnaround", "costume_derive"} and cid:
            rec = extra_index.setdefault("characters", {}).setdefault(
                cid, {"selected": None, "history": [], "parent_id": spec.get("parent_id")}
            )
            if filename not in rec["history"] and (Path(story_root) / rel).is_file():
                rec["history"].append(filename)
            rec.setdefault("parent_id", spec.get("parent_id"))
        elif spec.get("kind") == "scene_plate" and lid:
            rec = extra_index.setdefault("scenes", {}).setdefault(lid, {"selected": None, "history": []})
            if filename not in rec["history"] and (Path(story_root) / rel).is_file():
                rec["history"].append(filename)
    extra = {
        "story_id": story_id,
        "style_prefix": style_prefix_for_kind("scene_plate"),
        "negative": FIXED_NEGATIVE,
        "vast_for_stills": bool(os.environ.get("ANIME_FACTORY_GPU_STILLS", "").strip().lower() in {"1", "true", "yes", "on"}),
        "model": IMAGE_MODEL,
        "model_version": IMAGE_MODEL,
        "prompt_version": PROMPT_TEMPLATE_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "qc_version": QC_VERSION,
        "items": items,
    }
    return save_assets_index(story_root, extra_index, extra=extra)


def _regen_set(story_root: Path | None) -> set[tuple[str, str]]:
    return {(row["kind"], row["id"]) for row in pop_regen_requests(story_root)}


def _force_regen(regen: set[tuple[str, str]], spec: dict) -> bool:
    kind = spec.get("kind")
    if kind in {"character_sheet", "costume_derive"}:
        return ("character", str(spec.get("character_id") or "")) in regen
    if kind == "scene_plate":
        return ("scene", str(spec.get("scene_id") or "")) in regen
    if kind == "prop":
        return ("prop", str(spec.get("prop_id") or "")) in regen
    return False


def _bind_locked(
    story_root: Path | None,
    spec: dict,
    *,
    locked_idents: set[str] | None = None,
    locked_scenes: set[str] | None = None,
) -> bool:
    if story_root is None:
        return False
    kind = spec.get("kind")
    if kind in {"character_sheet", "costume_derive"}:
        cid = str(spec.get("character_id") or "")
        if locked_idents is not None:
            return bool(cid) and cid in locked_idents
        return bool(cid) and is_qc_locked(story_root, character_id=cid)
    if kind in {"character_view_derive", "character_turnaround"}:
        # The character lock points at the identity/front sheet. Derived views
        # are reusable only through their own QC evidence, handled below.
        return False
    if kind == "scene_plate":
        lid = str(spec.get("scene_id") or "")
        if locked_scenes is not None:
            return bool(lid) and lid in locked_scenes
        return bool(lid) and is_qc_locked(story_root, scene_id=lid)
    rel = spec.get("path")
    return bool(rel) and still_file_ok(Path(story_root) / rel)


def _assign_history_path(story_root: Path, spec: dict) -> str:
    """Studio regen writes a new history candidate; it does not overwrite the locked file."""
    kind = spec.get("kind")
    if kind in {"character_sheet", "costume_derive"}:
        cid = str(spec.get("character_id") or "")
        rec = (load_assets_index(story_root).get("characters") or {}).get(cid) or {}
        history = list(rec.get("history") or [])
        folder = Path(story_root) / "assets" / "characters" / cid
        stem = "sheet_front" if kind == "character_sheet" else "sheet_turnaround"
        name = next_history_filename(history, stem)
        while (folder / name).is_file():
            history.append(name)
            name = next_history_filename(history, stem)
        spec["path"] = character_asset_rel(cid, name)
        spec["alias_path"] = character_asset_rel(cid, FRONT_ALIAS_FILENAME)
        return spec["path"]
    if kind == "scene_plate":
        lid = str(spec.get("scene_id") or "")
        rec = (load_assets_index(story_root).get("scenes") or {}).get(lid) or {}
        history = list(rec.get("history") or [])
        folder = Path(story_root) / "assets" / "scenes" / lid
        name = next_history_filename(history, "plate_base")
        while (folder / name).is_file():
            history.append(name)
            name = next_history_filename(history, "plate_base")
        spec["path"] = scene_asset_rel(lid, name)
        return spec["path"]
    return spec["path"]


def compose_character_turnaround(story_root: Path | None, spec: dict) -> bytes:
    """Compose approved front/side/back portrait renders into the legacy turnaround file."""
    if story_root is None:
        raise KolorsPayloadError("character turnaround composition requires a story_root")
    source_paths = [str(path or "") for path in spec.get("source_paths") or []]
    if len(source_paths) != 3:
        raise KolorsPayloadError("character turnaround requires front, side, and back source paths")
    try:
        from io import BytesIO

        from PIL import Image, ImageOps
    except ImportError as exc:
        raise KolorsPayloadError("Pillow is required to compose character turnaround images") from exc

    panels = []
    for rel in source_paths:
        path = Path(story_root) / rel
        if not still_file_ok(path):
            raise KolorsPayloadError(f"character turnaround source is missing or invalid: {rel}")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            panel = ImageOps.pad(
                image,
                (CHAR_VIEW_WIDTH, CHAR_VIEW_HEIGHT),
                method=Image.Resampling.LANCZOS,
                color=(248, 244, 232),
                centering=(0.5, 0.5),
            )
            panels.append(panel)

    canvas = Image.new(
        "RGB",
        (CHAR_VIEW_WIDTH * len(panels), CHAR_VIEW_HEIGHT),
        color=(248, 244, 232),
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * CHAR_VIEW_WIDTH, 0))
    out = BytesIO()
    canvas.save(out, format="PNG", optimize=False, compress_level=6)
    return assert_still_blob(
        out.getvalue(),
        width=CHAR_VIEW_WIDTH * len(panels),
        height=CHAR_VIEW_HEIGHT,
        label=str(spec.get("id") or "character_turnaround"),
    )


def _render_until_qc(
    spec: dict,
    client: KolorsClient,
    period_md: str | None,
    world_mode: str,
    prefix: str,
    pos: list[str] | None,
    neg: list[str] | None,
    story_root: Path | None,
    *,
    negative_base: str | None = None,
    clip_scorer: ClipScorer | None = None,
    require_clip: bool = False,
) -> tuple[bytes | None, Any, str | None]:
    """Up to 3 deterministic seeds. History always records; selected only on QC pass."""
    kind = str(spec.get("kind") or "")
    salts = QC_SEED_SALTS
    last_png: bytes | None = None
    last_rel = str(spec.get("path") or "")
    last_qc = None
    preserve_selected = False
    owns_lock = kind in {"character_sheet", "costume_derive", "scene_plate"}
    if story_root is not None and owns_lock:
        if spec.get("kind") != "scene_plate" and spec.get("character_id"):
            preserve_selected = is_qc_locked(story_root, character_id=str(spec.get("character_id")))
        elif spec.get("scene_id"):
            preserve_selected = is_qc_locked(story_root, scene_id=str(spec.get("scene_id")))
    for salt in salts:
        work = dict(spec)
        seed = attempt_seed(spec, salt)
        work["seed"] = seed
        rel = str(work.get("path") or "")
        if story_root is not None:
            rel = _candidate_rel(story_root, work, salt)
            work["path"] = rel
        if kind == "character_turnaround":
            try:
                png = compose_character_turnaround(story_root, work)
            except KolorsPayloadError:
                return None, last_qc, rel
        else:
            png = _render_spec(
                work,
                client,
                period_md,
                world_mode,
                prefix,
                pos,
                neg,
                story_root,
                negative_base=negative_base,
            )
        parent_png = None if kind in {"scene_plate", "character_sheet", "character_turnaround", "prop", "keyframe"} else _parent_png_bytes(story_root, work)
        if kind == "character_view_derive" or kind == "costume_derive":
            parent_png = _parent_png_bytes(story_root, work)
        qc = score_still(
            png,
            kind=_qc_kind(work),
            prompt=str(work.get("prompt") or ""),
            parent_image=parent_png,
            scorer=clip_scorer,
            require_clip=require_clip,
            allow_placeholder=not require_clip,
            seed=seed,
            label=str(work.get("id") or kind),
        )
        last_png, last_rel, last_qc = png, rel, qc
        if story_root is not None:
            dest = Path(story_root) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(png)
            if qc.passed and preserve_selected:
                record_qc_candidate(
                    story_root,
                    character_id=work.get("character_id") if work.get("kind") != "scene_plate" else None,
                    scene_id=work.get("scene_id") if work.get("kind") == "scene_plate" else None,
                    filename=Path(rel).name,
                    qc=qc,
                    parent_id=work.get("parent_id"),
                    max_attempts=MAX_QC_ATTEMPTS,
                )
            else:
                _store_qc_outcome(story_root, work, Path(rel).name, qc, lock=qc.passed)
            if qc.passed and kind == "costume_derive" and not preserve_selected:
                alias = dest.parent / FRONT_ALIAS_FILENAME
                if not still_file_ok(alias):
                    write_front_alias(dest if dest.name == TURNAROUND_FILENAME else dest)
        if qc.passed:
            spec["path"] = rel
            spec["seed"] = seed
            return png, qc, rel
        if kind == "character_turnaround":
            break
    return last_png, last_qc, last_rel


def _render_spec(
    spec: dict,
    client: KolorsClient,
    period_md: str | None,
    world_mode: str,
    prefix: str,
    pos: list[str] | None,
    neg: list[str] | None,
    story_root: Path | None,
    negative_base: str | None = None,
) -> bytes:
    prompt = spec["prompt"]
    seed = spec.get("seed")
    kind = spec["kind"]
    # Plates are text-conditioned only, parent or not (reference-image bleed postmortem).
    parent_png = None if kind == "scene_plate" else _parent_png_bytes(story_root, spec)
    if kind == "scene_plate":
        payload, png = client.generate_scene_plate(
            prompt, seed, period_md, world_mode, prefix=prefix, negative_base=negative_base
        )
        assert_no_reference_images({k: v for k, v in payload.items() if not str(k).startswith("_")})
        return png
    prompt_prefix = (
        style_prefix_for_kind(kind)
        if kind in {"character_sheet", "character_view_derive", "costume_derive"}
        else prefix
    )
    payload: dict[str, Any] = {
        "model": IMAGE_MODEL,
        "prompt": style_prompt(
            prompt,
            pos,
            prefix=prompt_prefix,
            kind=kind,
            gender_tag=spec.get("gender_tag") or _gender_tag_from_fields(prompt, None),
        ),
        "negative_prompt": style_negative(
            neg, base=negative_base, extra=_kind_negative(kind, prompt)
        ),
        "image_size": spec.get("image_size") or CHAR_IMAGE_SIZE,
        "batch_size": 1,
        "num_inference_steps": KOLORS_STEPS,
        "_styled": True,
        "_label": str(spec.get("id") or kind),
    }
    if seed is not None:
        payload["seed"] = seed
    if parent_png:
        payload["_parent_png"] = parent_png
        payload["_parent_id"] = str(spec.get("parent_id") or "")
        payload["_kind"] = kind if kind in {"character_view_derive", "costume_derive"} else "costume_derive"
        payload["prompt"] = style_prompt(
            f"{prompt}, {DERIVE_FACE_LOCK}",
            pos,
            prefix=prompt_prefix,
            kind=payload["_kind"],
            gender_tag=spec.get("gender_tag") or _gender_tag_from_fields(prompt, None),
        )
    png = client.generate(payload)
    return png


def _snapshot_qc_locks(story_root: Path | None) -> tuple[set[str], set[str]]:
    if story_root is None:
        return set(), set()
    index = load_assets_index(story_root)
    idents = {
        cid
        for cid in (index.get("characters") or {})
        if is_qc_locked(story_root, character_id=str(cid))
    }
    scenes = {
        lid
        for lid in (index.get("scenes") or {})
        if is_qc_locked(story_root, scene_id=str(lid))
    }
    return idents, scenes


def _record_round_view(round_views: dict[str, dict[str, str]], spec: dict, rel: str) -> None:
    cid = str(spec.get("character_id") or "")
    if not cid or not rel:
        return
    view = str(spec.get("view") or "")
    if spec.get("kind") == "character_sheet":
        view = "front"
    elif spec.get("kind") != "character_view_derive":
        return
    if view:
        round_views.setdefault(cid, {})[view] = rel


def _normalized_stored_asset_relpath(value: Any) -> str | None:
    """Map legacy relative/R2/URL paths back to the local assets/ subtree."""
    text = str(value or "").replace("\\", "/").split("?", 1)[0].split("#", 1)[0]
    parts = [part for part in text.strip().split("/") if part not in {"", "."}]
    if ".." in parts or "assets" not in parts:
        return None
    return "/".join(parts[parts.index("assets") :])


def _specific_character_view_rel(
    story_root: Path,
    cid: str,
    stem: str,
    stored_path: Any,
) -> str | None:
    normalized = _normalized_stored_asset_relpath(stored_path)
    name = Path(normalized or str(stored_path or "")).name
    if not re.fullmatch(rf"{re.escape(stem)}(?:_\d+)?\.png", name):
        return None
    expected = character_asset_rel(cid, name)
    if normalized is not None and normalized != expected:
        return None
    return expected if still_file_ok(Path(story_root) / expected) else None


def _reuse_existing_view(
    conn: sqlite3.Connection,
    story_root: Path | None,
    spec: dict,
) -> str | None:
    """Reuse an exact side/back/turnaround only when successful QC left evidence."""
    if story_root is None:
        return None
    kind = str(spec.get("kind") or "")
    cid = str(spec.get("character_id") or "")
    view = str(spec.get("view") or "")
    if kind == "character_view_derive" and view in {"side", "back"}:
        stem = f"sheet_{view}"
    elif kind == "character_turnaround":
        stem = "sheet_turnaround"
    else:
        return None
    if not cid:
        return None

    rec = (load_assets_index(story_root).get("characters") or {}).get(cid) or {}
    for candidate in reversed(rec.get("candidates") or []):
        if not isinstance(candidate, dict) or candidate.get("verdict") != "pass":
            continue
        rel = _specific_character_view_rel(
            Path(story_root),
            cid,
            stem,
            candidate.get("filename"),
        )
        if rel:
            return rel

    try:
        cols = _asset_columns(conn)
        selected = ", selected_image_id" if "selected_image_id" in cols else ""
        row = conn.execute(
            f"SELECT path{selected} FROM assets WHERE id = ?",
            (str(spec.get("id") or ""),),
        ).fetchone()
    except sqlite3.Error:
        row = None
    if row is None:
        return None
    stored_values = [row["path"]]
    if "selected_image_id" in row.keys():
        stored_values.append(row["selected_image_id"])
    for stored in stored_values:
        rel = _specific_character_view_rel(Path(story_root), cid, stem, stored)
        if rel:
            return rel
    return None


def generate_asset_library(
    conn: sqlite3.Connection,
    story_id: str,
    characters: list[dict],
    locations: list[dict],
    props: list[dict],
    client: KolorsClient,
    period_md: str | None,
    world_mode: str,
    story_root: Path | None = None,
    interiors: list[dict] | None = None,
    skip_existing: bool = True,
    clip_scorer: ClipScorer | None = None,
) -> dict[str, Any]:
    """Fill stories/<id>/assets/. Bind a QC-passed identity; never re-roll it unless studio queued regen."""
    prefix, bible_negative = style_md_prefix(story_root)
    specs = plan_library_specs(
        characters, locations, props, interiors, period_md, world_mode, style_prefix=prefix
    )
    created: list[str] = []
    needs_human: list[str] = []
    round_views: dict[str, dict[str, str]] = {}
    pos, neg = period_lists(period_md, world_mode)
    require_clip = require_clip_for_client(client)
    if require_clip:
        from anime_factory.visual_qc import load_clip_scorer

        load_clip_scorer(require=True, scorer=clip_scorer)
    if story_root is not None:
        mark_stale_visual_v1(story_root)
    locked_idents, locked_scenes = _snapshot_qc_locks(story_root)
    regen = _regen_set(story_root)

    def emit(batch: dict[str, dict]) -> None:
        def block(asset_id: str) -> None:
            if asset_id and asset_id not in needs_human:
                needs_human.append(asset_id)

        for aid, spec in batch.items():
            rel = spec["path"]
            kind = str(spec.get("kind") or "")
            force = _force_regen(regen, spec)
            if story_root is not None and not _parent_ready(story_root, spec) and not force:
                parent = str(spec.get("parent_id") or spec.get("character_id") or "")
                if kind in {"character_view_derive", "character_turnaround", "costume_derive"}:
                    block(f"char_{parent}_sheet")
                elif kind in {"scene_plate", "scene_derive"} and parent:
                    block(f"plate_{parent}")
                block(aid)
                continue
            if (
                story_root is not None
                and _bind_locked(story_root, spec, locked_idents=locked_idents, locked_scenes=locked_scenes)
                and not force
            ):
                locked = (
                    locked_character_file(story_root, spec["character_id"])
                    if spec.get("character_id")
                    else locked_scene_file(story_root, spec["scene_id"]) if spec.get("scene_id") else Path(story_root) / rel
                )
                if locked and still_file_ok(locked):
                    try:
                        rel_locked = str(Path(locked).resolve().relative_to(Path(story_root).resolve()))
                    except ValueError:
                        rel_locked = rel
                    spec["path"] = rel_locked
                    _record_round_view(round_views, spec, rel_locked)
                if kind in {"character_sheet", "costume_derive"} and spec.get("character_id"):
                    alias = Path(story_root) / character_asset_rel(str(spec["character_id"]), FRONT_ALIAS_FILENAME)
                    if locked and locked.suffix.lower() == ".png" and not still_file_ok(alias):
                        write_front_alias(locked)
                continue
            reused = (
                _reuse_existing_view(conn, story_root, spec)
                if story_root is not None and not force
                else None
            )
            if reused:
                spec["path"] = reused
                _record_round_view(round_views, spec, reused)
                created.append(aid)
                continue
            existing = story_root / rel if story_root is not None else None
            if (
                story_root is not None
                and skip_existing
                and not force
                and kind == "prop"
                and existing is not None
                and still_file_ok(existing, allow_placeholder=not client.live)
            ):
                continue
            if kind == "character_turnaround":
                cid = str(spec.get("character_id") or "")
                views = round_views.get(cid) or {}
                if not (views.get("front") and views.get("side") and views.get("back")):
                    for view in ("front", "side", "back"):
                        if views.get(view):
                            continue
                        block(f"char_{cid}_{'sheet' if view == 'front' else view}")
                    block(aid)
                    continue
                spec["source_paths"] = [views["front"], views["side"], views["back"]]
            if force and story_root is not None:
                rel = _assign_history_path(story_root, spec)
            png, qc, written = _render_until_qc(
                spec,
                client,
                period_md,
                world_mode,
                prefix,
                pos,
                neg,
                story_root,
                negative_base=bible_negative,
                clip_scorer=clip_scorer,
                require_clip=require_clip,
            )
            passed = (
                png is not None
                and written is not None
                and qc is not None
                and getattr(qc, "verdict", None) == "pass"
            )
            if not passed:
                if kind in {
                    "character_sheet",
                    "scene_plate",
                    "costume_derive",
                    "character_view_derive",
                    "character_turnaround",
                }:
                    block(aid)
                continue
            rel = written
            _record_round_view(round_views, spec, rel)
            key = asset_path(story_id, rel)
            _upsert_asset_row(
                conn,
                aid,
                spec,
                key,
                spec["prompt"],
                spec.get("seed"),
                png,
                selected_image_id=Path(rel).name,
            )
            created.append(aid)

    emit(specs)
    derive_specs = plan_derive_specs(characters, locations, story_root, period_md, world_mode)
    for aid, spec in derive_specs.items():
        parent = str(spec.get("parent_id") or "")
        if spec.get("kind") == "costume_derive" and parent and story_root is not None:
            if not has_locked_identity(story_root, parent):
                continue
        if spec.get("kind") == "scene_plate" and parent and story_root is not None:
            if not has_locked_scene(story_root, parent):
                continue
        specs[aid] = spec
    emit({aid: spec for aid, spec in derive_specs.items() if aid in specs})
    conn.commit()
    index_path = None
    if story_root is not None:
        index_path = str(write_assets_index(story_root, story_id, specs, created))
    return {
        "story_id": story_id,
        "specs": specs,
        "created": created,
        "needs_human": needs_human,
        "count": len(specs),
        "index": index_path,
        "vast_for_stills": bool(os.environ.get("ANIME_FACTORY_GPU_STILLS", "").strip().lower() in {"1", "true", "yes", "on"}),
    }


_WIKI_ID_RE = re.compile(r"编号[`*：:\s]*`+([^`]+)`+", re.I)
_AT_LOC_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_]*)")
_WIKI_IDENTITY_LABELS = (
    "身份",
    "identity_prompt",
    "identity prompt",
    "visual_lock_prompt",
    "visual lock prompt",
    "identity",
    "visual_lock",
    "visual lock",
    "appearance",
)
_WIKI_ID_LABELS = ("编号", "character_id", "character id", "id")


def _unwrap_wiki_inline(value: str) -> str:
    out = str(value or "").strip()
    if out.startswith("`") and out.endswith("`"):
        return out.strip("`").strip()
    return out


def _normalize_wiki_bullet(line: str) -> str:
    stripped = str(line or "").strip()
    stripped = re.sub(r"^[-*]\s+", "", stripped)
    stripped = re.sub(r"^\d+\.\s+", "", stripped)
    return stripped.replace("**", "").strip()


def _wiki_field_value(normalized: str) -> tuple[str, str] | None:
    lowered = normalized.lower()
    for label in _WIKI_IDENTITY_LABELS:
        for sep in ("：", ":"):
            prefix = f"{label}{sep}"
            if lowered.startswith(prefix.lower()):
                return ("identity", _unwrap_wiki_inline(normalized[len(prefix) :].strip()))
    for label in _WIKI_ID_LABELS:
        for sep in ("：", ":"):
            prefix = f"{label}{sep}"
            if lowered.startswith(prefix.lower()):
                return ("id", "")
    return None


def _extract_wiki_identity(text: str) -> str:
    """Pull English visual lock from labeled wiki bullets; legacy body fallback stays English-only."""
    labeled = ""
    for line in str(text or "").splitlines():
        norm = _normalize_wiki_bullet(line)
        if not norm or norm.startswith("#"):
            continue
        field = _wiki_field_value(norm)
        if field and field[0] == "identity" and field[1]:
            labeled = field[1]
            break
    if labeled:
        return labeled
    parts: list[str] = []
    for line in str(text or "").splitlines():
        norm = _normalize_wiki_bullet(line)
        if not norm or norm.startswith("#"):
            continue
        field = _wiki_field_value(norm)
        if field and field[0] == "id":
            continue
        parts.append(norm)
    return " ".join(parts).strip()


def _wiki_title(text: str, fallback: str) -> str:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def _board_payloads(story_root: Path | None) -> list[dict]:
    if story_root is None:
        return []
    out: list[dict] = []
    for rel in ("episodes/EP001/board.json", "episodes/EP001/script.json"):
        path = Path(story_root) / rel
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        out.extend(data.get("segments") or [])
        out.extend(data.get("shots") or [])
        for extra in data.get("cast") or []:
            if isinstance(extra, dict) and extra.get("id"):
                out.append(
                    {
                        "character_id": extra.get("id"),
                        "name": extra.get("name"),
                        "identity_prompt": extra.get("identity_prompt"),
                        "age": extra.get("age"),
                        "gender": extra.get("gender"),
                        "exclusions": extra.get("exclusions"),
                    }
                )
        for extra in data.get("locations") or []:
            if isinstance(extra, dict) and extra.get("id"):
                out.append(
                    {
                        "location_id": extra.get("id"),
                        "name": extra.get("name"),
                        "display_name": extra.get("display_name"),
                        "plate_prompt": extra.get("plate_prompt") or extra.get("location_prompt"),
                    }
                )
    return out


def _story_location_rows(story_root: Path | None) -> dict[str, dict[str, Any]]:
    """Merge hosted script/board location rows keyed by id."""
    rows: dict[str, dict[str, Any]] = {}
    if story_root is None:
        return rows
    for rel in ("episodes/EP001/board.json", "episodes/EP001/script.json"):
        path = Path(story_root) / rel
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        for extra in data.get("locations") or []:
            if not isinstance(extra, dict) or not extra.get("id"):
                continue
            lid = str(extra["id"]).strip()
            if not lid:
                continue
            merged = dict(rows.get(lid) or {})
            for key in ("name", "display_name", "plate_prompt", "location_prompt", "type"):
                if extra.get(key) not in (None, ""):
                    merged[key] = extra.get(key)
            rows[lid] = merged
    return rows


def character_ids_needed(story_root: Path | None) -> set[str]:
    needed: set[str] = set()
    for row in _board_payloads(story_root):
        cid = str(row.get("character_id") or "").strip()
        if cid and str(row.get("purpose") or "").lower() not in {"establish", "establishing"}:
            if row.get("on_camera") is not False:
                needed.add(cid)
        for ref in row.get("refs") or []:
            text = str(ref or "")
            if text.startswith("char_") and text.endswith("_sheet"):
                needed.add(text[len("char_") : -len("_sheet")])
        for cut in row.get("cuts") or []:
            for other in cut.get("characters") or []:
                if str(other).strip():
                    needed.add(str(other).strip())
    return {cid for cid in needed if cid}


def seed_cast_from_story_root(conn: sqlite3.Connection, story_root: Path | None) -> dict[str, int]:
    """Hosted board/wiki often never land in sqlite. Design must still draw sheets.

    script.json.cast / approved bible is authoritative until the first QC lock.
    Existing sqlite rows are updated when the source identity/age hash changes;
    locked assets are marked stale via asset_lock instead of silent skip.
    """
    if story_root is None:
        return {"characters": 0, "locations": 0}
    root = Path(story_root)
    names: dict[str, str] = {}
    identities: dict[str, str] = {}
    ages: dict[str, int] = {}
    exclusions_map: dict[str, list[str]] = {}
    loc_names: dict[str, str] = {}
    loc_plates: dict[str, str] = {}
    loc_meta: dict[str, dict[str, str]] = {}
    wiki_chars = root / "canon" / "wiki" / "characters"
    if wiki_chars.is_dir():
        for path in sorted(wiki_chars.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            match = _WIKI_ID_RE.search(text)
            cid = (match.group(1) if match else path.stem).strip()
            if not cid:
                continue
            names[cid] = _wiki_title(text, cid)
            identity_raw = _extract_wiki_identity(text)
            if identity_raw:
                # 240 chars cut the wiki description off mid-sentence, so the sheet
                # prompt lost hair/clothing detail. SDXL still reads ~1000.
                try:
                    identities[cid] = validate_character_identity(
                        identity_raw[:IDENTITY_PROMPT_MAX_CHARS],
                        character_id=cid,
                        name=names[cid],
                    )
                except CharacterIdentityError:
                    # A title plus `编号: hero` is metadata, not a visual lock.
                    # Leave the slot open so the hosted board can supply one.
                    pass
    wiki_locs = root / "canon" / "wiki" / "locations"
    if wiki_locs.is_dir():
        for path in sorted(wiki_locs.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            match = _WIKI_ID_RE.search(text)
            lid = (match.group(1) if match else path.stem).strip()
            if lid:
                loc_names[lid] = _wiki_title(text, lid)
    for row in _board_payloads(root):
        cid = str(row.get("character_id") or "").strip()
        if cid:
            names.setdefault(cid, str(row.get("name") or cid))
            # identity_prompt / first_frame_prompt only: h3_prompt is Chinese camera
            # direction and would become the sheet's visual lock.
            visual = str(row.get("identity_prompt") or row.get("first_frame_prompt") or "").strip()
            if visual:
                try:
                    identities[cid] = validate_character_identity(
                        visual[:IDENTITY_PROMPT_MAX_CHARS],
                        character_id=cid,
                        name=names[cid],
                        gender=str(row.get("gender") or "") or None,
                    )
                except CharacterIdentityError:
                    if cid not in identities:
                        pass
            declared_age = row.get("age")
            try:
                if declared_age is not None and str(declared_age).strip() != "":
                    ages[cid] = int(declared_age)
            except (TypeError, ValueError):
                pass
            parsed = parse_identity_age(identities.get(cid) or visual)
            if parsed is not None:
                ages.setdefault(cid, parsed)
            excl = row.get("exclusions")
            if isinstance(excl, list):
                exclusions_map[cid] = [str(x).strip() for x in excl if str(x).strip()]
            else:
                exclusions_map.setdefault(cid, identity_exclusions(identities.get(cid) or visual))
        lid = str(row.get("location_id") or "").strip()
        if lid:
            display = str(row.get("display_name") or "").strip()
            english = str(row.get("name") or "").strip()
            loc_names.setdefault(lid, display or english or lid)
            meta = loc_meta.setdefault(lid, {})
            if display:
                meta["display_name"] = display
            if english:
                meta["english_name"] = english
            plate = str(row.get("plate_prompt") or "").strip()
            if plate:
                loc_plates[lid] = plate
        blob = " ".join(str(row.get(k) or "") for k in ("h3_prompt", "first_frame_prompt", "prompt"))
        for hit in _AT_LOC_RE.findall(blob):
            loc_names.setdefault(hit, hit)
    for lid, row in _story_location_rows(root).items():
        display = str(row.get("display_name") or "").strip()
        english = str(row.get("name") or "").strip()
        loc_names.setdefault(lid, display or english or lid)
        meta = loc_meta.setdefault(lid, {})
        if display:
            meta["display_name"] = display
        if english:
            meta["english_name"] = english
        plate = str(row.get("plate_prompt") or row.get("location_prompt") or "").strip()
        if plate:
            loc_plates[lid] = plate
    inserted_chars = 0
    updated_chars = 0
    try:
        from anime_factory.asset_lock import is_qc_locked, sync_character_source_hash
        from anime_factory.h3_storyboard import character_source_hash
    except ImportError:
        is_qc_locked = None  # type: ignore[assignment]
        sync_character_source_hash = None  # type: ignore[assignment]
        character_source_hash = None  # type: ignore[assignment]
    for cid, name in names.items():
        raw_identity = identities.get(cid) or ""
        if not raw_identity:
            continue
        try:
            prompt = validate_character_identity(raw_identity, character_id=cid, name=name)
        except CharacterIdentityError:
            continue
        age = ages.get(cid)
        if age is None:
            age = parse_identity_age(prompt, default=None)
        if age is None:
            age = 30  # band default only when no numeric/band cue exists
        existing = conn.execute(
            "SELECT id, identity_prompt, age, name FROM characters WHERE id = ?",
            (cid,),
        ).fetchone()
        if existing:
            old_prompt = str(existing["identity_prompt"] or "")
            old_age = int(existing["age"] or 0)
            if old_prompt == prompt and old_age == int(age) and str(existing["name"] or "") == name:
                if sync_character_source_hash is not None and character_source_hash is not None:
                    sync_character_source_hash(
                        root,
                        character_id=cid,
                        source_hash=character_source_hash(
                            identity_prompt=prompt,
                            age=age,
                            name=name,
                            exclusions=exclusions_map.get(cid) or identity_exclusions(prompt),
                        ),
                        fail_closed_if_locked=False,
                    )
                continue
            locked = bool(is_qc_locked(root, character_id=cid)) if is_qc_locked else False
            if locked and sync_character_source_hash is not None and character_source_hash is not None:
                sync_character_source_hash(
                    root,
                    character_id=cid,
                    source_hash=character_source_hash(
                        identity_prompt=prompt,
                        age=age,
                        name=name,
                        exclusions=exclusions_map.get(cid) or identity_exclusions(prompt),
                    ),
                    fail_closed_if_locked=True,
                )
                continue
            conn.execute(
                "UPDATE characters SET name = ?, identity_prompt = ?, age = ? WHERE id = ?",
                (name, prompt, int(age), cid),
            )
            updated_chars += 1
            if sync_character_source_hash is not None and character_source_hash is not None:
                sync_character_source_hash(
                    root,
                    character_id=cid,
                    source_hash=character_source_hash(
                        identity_prompt=prompt,
                        age=age,
                        name=name,
                        exclusions=exclusions_map.get(cid) or identity_exclusions(prompt),
                    ),
                    fail_closed_if_locked=False,
                )
            continue
        conn.execute(
            """
            INSERT INTO characters (id, name, identity_prompt, age, alive, seed)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (cid, name, prompt, int(age), locked_seed(cid)),
        )
        inserted_chars += 1
        if sync_character_source_hash is not None and character_source_hash is not None:
            sync_character_source_hash(
                root,
                character_id=cid,
                source_hash=character_source_hash(
                    identity_prompt=prompt,
                    age=age,
                    name=name,
                    exclusions=exclusions_map.get(cid) or identity_exclusions(prompt),
                ),
                fail_closed_if_locked=False,
            )
    inserted_locs = 0
    for lid, name in loc_names.items():
        existing = conn.execute("SELECT id, plate_prompt FROM locations WHERE id = ?", (lid,)).fetchone()
        meta = loc_meta.get(lid) or {}
        display = str(meta.get("display_name") or "").strip()
        english = str(meta.get("english_name") or name or lid).strip()
        stored_name = display or english or lid
        aka = "[]"
        if display and english and display != english:
            aka = json.dumps({"display_name": display, "english_name": english}, ensure_ascii=False)
        plate = str(loc_plates.get(lid) or "").strip() or None
        if existing:
            if plate and str(existing["plate_prompt"] or "") != plate:
                conn.execute(
                    "UPDATE locations SET name = ?, aka_json = ?, plate_prompt = ? WHERE id = ?",
                    (stored_name, aka, plate, lid),
                )
            continue
        conn.execute(
            """
            INSERT INTO locations (id, name, aka_json, type, plate_prompt)
            VALUES (?, ?, ?, 'settlement', ?)
            """,
            (lid, stored_name, aka, plate),
        )
        inserted_locs += 1
    if inserted_chars or updated_chars or inserted_locs:
        conn.commit()
    return {"characters": inserted_chars + updated_chars, "locations": inserted_locs}


def library_from_db(
    conn: sqlite3.Connection,
    story_id: str,
    client: KolorsClient,
    story_root: Path | None = None,
    period_md: str | None = None,
    world_mode: str = "fiction",
    skip_existing: bool = True,
    clip_scorer: ClipScorer | None = None,
) -> dict[str, Any]:
    seed_cast_from_story_root(conn, story_root)
    characters = [
        dict(r)
        for r in conn.execute(
            "SELECT id, name, identity_prompt, visual_lock_prompt, seed FROM characters"
        ).fetchall()
    ]
    locations = [dict(r) for r in conn.execute("SELECT id, name, type, plate_prompt FROM locations").fetchall()]
    story_locs = _story_location_rows(story_root)
    for loc in locations:
        if loc.get("plate_prompt"):
            continue
        src = story_locs.get(str(loc.get("id") or "").strip())
        if not src:
            continue
        plate = str(src.get("plate_prompt") or src.get("location_prompt") or "").strip()
        if plate:
            loc["plate_prompt"] = plate
    props = [dict(r) for r in conn.execute("SELECT id, name, identity_prompt FROM props").fetchall()]
    interiors = [
        dict(r)
        for r in conn.execute("SELECT id, name, parent, scene_id, asset_path FROM geo_interiors").fetchall()
    ]
    if story_root and (story_root / "bible" / "period.md").is_file() and period_md is None:
        period_md = (story_root / "bible" / "period.md").read_text(encoding="utf-8")
    row = conn.execute("SELECT world_mode FROM story WHERE id = ?", (story_id,)).fetchone()
    if row and row["world_mode"]:
        world_mode = row["world_mode"]
    needed = character_ids_needed(story_root)
    have = {str(ch.get("id") or "").strip() for ch in characters}
    missing_cast = sorted(cid for cid in needed if cid not in have)
    if missing_cast:
        raise RuntimeError(f"design has no character rows for {missing_cast}; refusing empty library")
    result = generate_asset_library(
        conn,
        story_id,
        characters,
        locations,
        props,
        client,
        period_md,
        world_mode,
        story_root,
        interiors=interiors,
        skip_existing=skip_existing,
        clip_scorer=clip_scorer,
    )
    sheet_chars = {
        str(spec.get("character_id") or "")
        for spec in (result.get("specs") or {}).values()
        if spec.get("kind") == "character_sheet"
    }
    missing_sheets = sorted(cid for cid in needed if cid not in sheet_chars)
    if missing_sheets:
        raise RuntimeError(f"design produced no character sheets for {missing_sheets}")
    return result


def harbor_mvp_cast() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Default library for story-dcc831dfc9bc-3e6e67 / Demo Harbor."""
    characters = [
        {
            "id": "hero",
            "name": "守潮人",
            "identity_prompt": (
                "1boy, adult, short black hair, dark brown eyes, dark wool coat, lean build, "
                "clean cel-shaded edges, dusk rim light"
            ),
            "gender": "male",
            "seed": locked_seed("hero"),
        }
    ]
    locations = [
        {
            "id": "harbor",
            "name": "暮潮码头",
            "plate_prompt": "wooden harbor dock at madder dusk, wet planks, lanterns, empty boats, anime location background, empty establishing shot, no people, no text",
        }
    ]
    interiors = [
        {
            "id": "int_dock",
            "name": "Dock",
            "parent": "harbor",
            "scene_id": "dock",
            "asset_path": "assets/scenes/dock/plate_base.png",
            "plate_prompt": "wooden dock interior close, wet boards reflecting lanterns, anime location background, empty, no people, no text",
        }
    ]
    props = [
        {
            "id": "lantern",
            "name": "煤油灯",
            "identity_prompt": "old brass kerosene lantern, warm glow, product turnaround, no people, no text",
        },
        {
            "id": "hawser",
            "name": "缆绳",
            "identity_prompt": "thick harbor hawser rope coiled on wet wood, product shot, no people, no text",
        },
    ]
    return characters, locations, props, interiors

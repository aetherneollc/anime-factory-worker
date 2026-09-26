"""Locked model names from ANIME_FACTORY.md chapter 2. Roles must not be swapped."""

import os
import re

DEEPSEEK_MODEL_FLASH = "deepseek-v4-flash"
DEEPSEEK_BACKUP_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SIMPLE_MODEL = "Qwen/Qwen3-8B"
QC_MODEL = "Qwen/Qwen3.5-4B"
# GPU stills: anime-native SDXL (one ungated ~6.9GB checkpoint). Flux schnell is
# guidance-distilled, so cfg=1 silently dropped FIXED_NEGATIVE; real CFG needs SDXL.
IMAGE_MODEL = "cagliostrolab/animagine-xl-4.0"
IMAGE_CKPT = "animagine-xl-4.0.safetensors"
# GPU stills: scene plates / keyframes stay on Kwai Kolors (backup for characters too).
# Character sheets default to HunyuanImage (commercial open-weight family; TokenHub interim,
# self-host on leased GPU is the target). Qwen open-weight 2.1 is research-licensed — not default.
# Kolors is not an SDXL checkpoint: ChatGLM3 encodes text, so it cannot reuse anime_t2i.
STILL_BACKENDS = ("kolors", "animagine")
DEFAULT_STILL_BACKEND = "kolors"
CHARACTER_STILL_BACKENDS = ("hunyuan", "qwen", "kolors", "animagine")
DEFAULT_CHARACTER_STILL_BACKEND = "hunyuan"
HUNYUAN_IMAGE_MODEL = "hy-image-v3"
QWEN_IMAGE_MODEL = "qwen-image-2.0-pro"
DASHSCOPE_MULTIMODAL_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)
TOKENHUB_HUNYUAN_URL = (
    "https://tokenhub.tencentmaas.com/v1/wand/hunyuan-image/v3-generation"
)
KOLORS_UNET_FILE = "kolors_unet_fp16.safetensors"
KOLORS_VAE_FILE = "kolors_vae_fp16.safetensors"
KOLORS_CHATGLM_FILE = "chatglm3-fp16.safetensors"
KOLORS_IPADAPTER_FILE = "kolors_ip_adapter_plus_general.bin"
KOLORS_CLIP_VISION_FILE = "kolors_ip_image_encoder.bin"
KOLORS_WORKFLOW = "kolors_t2i"
ANIMAGINE_WORKFLOW = "anime_t2i"
IMAGE_STEPS = 28
IMAGE_CFG = 5.0
IMAGE_SAMPLER = "euler_ancestral"
IMAGE_SCHEDULER = "normal"
# Location/keyframe masters use the Animagine XL SDXL bucket 1344×768, then crop/scale
# downstream to the 864×480 H3 plate. Turnaround strips and props keep their own aspects.
STILL_WIDTH = 1344
STILL_HEIGHT = 768
STILL_IMAGE_SIZE = f"{STILL_WIDTH}x{STILL_HEIGHT}"
STILL_ASPECT = STILL_WIDTH / STILL_HEIGHT
# A 3-byte "PNG" passed the old `>= 1` check and shipped as a keyframe.
MIN_STILL_BYTES = 10_000
# SiliconFlow hosted fallback only (no local weights). GPU path never downloads this.
KOLORS_MODEL = "Kwai-Kolors/Kolors"
TTS_MODEL = "FunAudioLLM/CosyVoice2-0.5B"
CONTROL_PLANE_DEFAULT = ""
H3_MODEL_NAME = "MiniMax-H3"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EMBEDDING_DIMENSION = 1024

ROLE_DIRECTOR = "director"
ROLE_ASSISTANT = "assistant"
ROLE_QC = "qc"

ROLE_MODELS = {
    ROLE_DIRECTOR: DEEPSEEK_MODEL_FLASH,
    ROLE_ASSISTANT: SIMPLE_MODEL,
    ROLE_QC: QC_MODEL,
}

DEFAULT_STYLE_PRESET = "cinematic"
LUMINOUS_CINEMATIC_ANIME_PRESET = "luminous-cinematic-anime"
STYLE_PRESET_ALIASES = {
    DEFAULT_STYLE_PRESET: DEFAULT_STYLE_PRESET,
    "default": DEFAULT_STYLE_PRESET,
    LUMINOUS_CINEMATIC_ANIME_PRESET: LUMINOUS_CINEMATIC_ANIME_PRESET,
    "luminous": LUMINOUS_CINEMATIC_ANIME_PRESET,
}
STYLE_PRESET_VERSION = "cinematic-v3"
PROMPT_TEMPLATE_VERSION = "v4"
SCHEMA_VERSION = "v1"
CHUNKING_VERSION = "v3"
VECTOR_SCHEMA_VERSION = "v1"

# Spec 1.2 — per-asset-kind affirmative prefixes. Diffusion cannot process negation;
# everything we do not want lives in FIXED_NEGATIVE, which SDXL CFG applies.
# Character sheets must not inherit cinematic / detailed-background wording.
# Default location/keyframe look is luminous cinematic anime (Shinkai-like light and sky)
# described in paint terms. Director names and film titles stay out of the positive
# prompt; Animagine copies those famous stills instead of this story's places.
STYLE_PREFIX_LOCATION_LUMINOUS = (
    "original anime production still for this story, anime production still, cel shaded, "
    "clean lineart, flat shading, 2d anime illustration, clear luminous atmosphere, "
    "hand-painted background, detailed background, "
    "location-specific weather and time of day, contextual layered clouds for open-sky scenes, "
    "volumetric light, god rays, rim light, rain or water wet-surface reflections, "
    "saturated blue and gold contrast, natural sky gradient, "
    "detailed everyday urban and natural backgrounds, atmospheric perspective, cinematic depth, "
    "cinematic composition, story-specific props and locations from the bible, natural living skin tone"
)
STYLE_PREFIX_LOCATION = STYLE_PREFIX_LOCATION_LUMINOUS
STYLE_PREFIX_CHARACTER = (
    "original anime character design, solo, cel shaded, clean lineart, "
    "warm ivory studio background"
)
# Backward-compatible default for location/keyframe paths.
STYLE_PREFIX = STYLE_PREFIX_LOCATION

ANIMAGINE_QUALITY_TAGS = ("masterpiece", "high score", "great score", "absurdres", "safe")

FIXED_NEGATIVE = (
    "photorealistic, 3d render, cgi, western cartoon, chibi, extra fingers, extra limbs, "
    "deformed face, bad anatomy, lowres, worst quality, jpeg artifacts, blurry, "
    "watermark, subtitle, text overlay, logo, signature, username, "
    "Makoto Shinkai, Your Name, Kimi no Na wa, Weathering With You, Suzume, "
    "5 centimeters per second, 君の名は, 天气之子, 秒速5厘米, "
    "movie screenshot, famous movie still, screenshot, famous still composition, "
    "copycat composition, exaggerated moe eyes, "
    "ceramic plate, dish, dinnerware, bowl, "
    "nude, nsfw, "
    "ghost film, horror lighting, corpse-pale skin, wet gloomy face, empty horror corridor, "
    "undead eyes, found footage, dead white skin, haunted hallway"
)


class StillBackendError(ValueError):
    """STILL_BACKEND is not kolors or animagine. Never silently pick one."""


class CharacterStillBackendError(ValueError):
    """CHARACTER_STILL_BACKEND is not hunyuan, qwen, kolors, or animagine."""


class QwenImageError(ValueError):
    """DashScope Qwen-Image call failed or returned no art."""


def still_backend() -> str:
    """Active GPU still backend (plates / keyframes). Invalid values fail closed."""
    raw = (os.environ.get("STILL_BACKEND") or DEFAULT_STILL_BACKEND).strip().lower()
    if raw not in STILL_BACKENDS:
        raise StillBackendError(
            f"STILL_BACKEND must be one of {STILL_BACKENDS}, got {raw!r}"
        )
    return raw


def character_still_backend() -> str:
    """Backend for character_sheet / view_derive / costume_derive. Default: hunyuan."""
    raw = (
        os.environ.get("CHARACTER_STILL_BACKEND") or DEFAULT_CHARACTER_STILL_BACKEND
    ).strip().lower()
    if raw not in CHARACTER_STILL_BACKENDS:
        raise CharacterStillBackendError(
            f"CHARACTER_STILL_BACKEND must be one of {CHARACTER_STILL_BACKENDS}, got {raw!r}"
        )
    return raw


def qwen_image_model() -> str:
    return (os.environ.get("QWEN_IMAGE_MODEL") or QWEN_IMAGE_MODEL).strip() or QWEN_IMAGE_MODEL


def hunyuan_image_model() -> str:
    return (
        os.environ.get("HUNYUAN_IMAGE_MODEL") or HUNYUAN_IMAGE_MODEL
    ).strip() or HUNYUAN_IMAGE_MODEL


def still_model_id() -> str:
    return KOLORS_MODEL if still_backend() == "kolors" else IMAGE_MODEL


def still_workflow_name() -> str:
    return KOLORS_WORKFLOW if still_backend() == "kolors" else ANIMAGINE_WORKFLOW


def still_unet_file() -> str:
    return KOLORS_UNET_FILE if still_backend() == "kolors" else IMAGE_CKPT


def normalize_style_preset(preset: str | None = None) -> str:
    raw = str(preset if preset is not None else os.environ.get("STYLE_PRESET", DEFAULT_STYLE_PRESET)).strip().lower()
    return STYLE_PRESET_ALIASES.get(raw, DEFAULT_STYLE_PRESET)


def style_prefix_for_kind(kind: str | None, preset: str | None = None) -> str:
    """Return the style prefix appropriate for a still asset kind."""
    k = str(kind or "").strip().lower()
    if k in {"character_sheet", "character_view_derive", "costume_derive"}:
        return STYLE_PREFIX_CHARACTER
    if k in {"scene_plate", "scene_derive", "keyframe"} and normalize_style_preset(preset) == LUMINOUS_CINEMATIC_ANIME_PRESET:
        return STYLE_PREFIX_LOCATION_LUMINOUS
    return STYLE_PREFIX_LOCATION


def animagine_quality_suffix(kind: str | None, *, gender_tag: str | None = None) -> str:
    """Animagine XL 4.0 expects quality tags and `safe` at the prompt tail."""
    tags = list(ANIMAGINE_QUALITY_TAGS)
    k = str(kind or "").strip().lower()
    if k in {"character_sheet", "character_view_derive", "costume_derive"}:
        tag = str(gender_tag or "").strip().lower()
        if tag in {"1boy", "1girl"} and tag not in tags:
            tags.insert(0, tag)
    return ", ".join(tags)

COPYCAT_RE = re.compile(
    r"makoto\s*shinkai|shinkai\s*style|新海诚|"
    r"kimi\s*no\s*na\s*wa\.?|君の名は|your\s*name\s*still|"
    r"weathering\s*with\s*you|tenki\s*no\s*ko|天气之子|天気の子|"
    r"suzume(?:'s)?(?:\s+no\s+tojimari)?|铃芽之旅|すずめの戸締まり|"
    r"5\s*centimeters\s*per\s*second|秒速\s*5\s*センチメートル|秒速5厘米|秒速5センチ",
    re.I,
)


def scrub_copycat(text: str) -> str:
    """Strip famous-still copycat tokens so the still model cannot overfit 新海诚剧照."""
    cleaned = COPYCAT_RE.sub("", text or "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    return cleaned.strip(" ,")


# H3 generates 1024×576 (primary), 864×480 (first OOM fallback), or 640×352
# (chain-tail / last-resort canvas). All scale to delivery 1280×720 at session
# normalize. Odd sizes break the 32px latent grid; 640×360 is not aligned.
H3_GEN_WIDTH = 1024
H3_GEN_HEIGHT = 576
H3_OOM_FALLBACK_WIDTH = 864
H3_OOM_FALLBACK_HEIGHT = 480
H3_OOM_SAFE_WIDTH = 640
H3_OOM_SAFE_HEIGHT = 352
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 24
TARGET_EPISODE_SECONDS = 600
DEFAULT_SHORT_SECONDS = 120.0
H3_MAX_SECONDS = 8.0
# SkyReels V3 R2V recommended shot unit.
SKYREELS_MAX_SECONDS = 5.0
# NVlabs LongLive inference.yaml default num_output_frames=384 latent frames.
# (384-1)*4+1 = 1533 video frames ≈ 63.9s @ 24fps; 64s snaps exactly to 384 latents.
# 5090 + local_attn_size=32 is the AR window, not an 8s H3 slice.
LONGLIVE_MAX_SECONDS = 64.0
# ~120s shorts pack into 2–3 AR takes; each take is ≤60s (compose scales 1280×704).
LONGLIVE_SHORT_MAX_SECONDS = 60.0
LONGLIVE_SHORT_TARGET_SECONDS = 120.0
LONGLIVE_SHORT_MIN_TAKES = 2
LONGLIVE_SHORT_MAX_TAKES = 3
VIDEO_BACKENDS = ("skyreels_v3_r2v",)
DEFAULT_VIDEO_BACKEND = "skyreels_v3_r2v"
# Production stills backend. Kolors is not selectable here.
IMAGE_BACKENDS = ("flux2_klein4b",)
DEFAULT_IMAGE_BACKEND = "flux2_klein4b"
CONTROL_BACKENDS = ("flux2_klein_ref", "composite", "kolors_ipadapter", "none")
DEFAULT_CONTROL_BACKEND = "flux2_klein_ref"
H3_MAX_REFS = 9
H3_MAX_RETRIES = 2
STORY_KINDS = ("film", "series", "short")


def normalize_story_kind(kind: str | None) -> str:
    """Canonical story kind. Legacy oneshot / pv / mv map to short."""
    raw = str(kind or "series").strip().lower()
    if raw in {"oneshot", "short", "pv", "mv"}:
        return "short"
    if raw == "film":
        return "film"
    return "series"


def is_short_kind(kind: str | None) -> bool:
    return normalize_story_kind(kind) == "short"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn"
METASO_DEFAULT_BASE_URL = "https://metaso.cn/api/v1/search"
VAST_API_BASE = "https://console.vast.ai/api/v0"
COMFY_ROUTER_PORT = 8199

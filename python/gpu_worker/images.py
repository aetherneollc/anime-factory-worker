"""Which Docker/Vast image is on the box — and whether Comfy / H3 / 生图 exist.

`docker.io/aetherneo/anime-factory-gpu` is the **scheduled** worker:
ComfyUI + MiniMax H3 + Flux 生图 + agent + ffmpeg + cloudflared.
Labels and this catalog must stay `comfy/h3/kolors/image_gen = true` so
the scheduler will PUT /asks/. Never lease a comfy:false image.

Weights are not baked (too large). Boot pulls HuggingFace onto instance-local disk. Never R2.
A legacy agent-only name is kept only so tests still encode the historical
"do not probe :8199 on an image that cannot run Comfy" failure.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


# Content-digest pin fields. Leave unset until a real sha256 is known — never invent one,
# and never treat floating tags (:main, :latest, :sha-<git>) as a digest.
IMAGE_DIGEST_ENV = "AF_IMAGE_DIGEST"
EXPECTED_IMAGE_DIGEST_ENV = "AF_EXPECTED_IMAGE_DIGEST"
REPORTED_IMAGE_DIGEST_ENV = "AF_REPORTED_IMAGE_DIGEST"
IMAGE_DIGEST_CONFIG_FIELDS = (
    "AF_EXPECTED_IMAGE_DIGEST",
    "AF_IMAGE_DIGEST",
    "AF_REPORTED_IMAGE_DIGEST",
    "CapabilityProfile.expected_image_digest",
    "GpuImage.digest",
)

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def parse_content_digest(raw: str | None) -> str | None:
    """Return ``sha256:<64 hex>`` or None. Image tags are never digests."""
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    lower = text.lower()
    at = lower.rfind("@sha256:")
    if at >= 0:
        text = text[at + 1 :]
        lower = text.lower()
    if lower.startswith("sha256:"):
        hexpart = lower[7:]
        return f"sha256:{hexpart}" if _SHA256_HEX.fullmatch(hexpart) else None
    if _SHA256_HEX.fullmatch(lower):
        return f"sha256:{lower}"
    return None


@dataclass(frozen=True)
class GpuImage:
    name: str
    comfy: bool
    h3: bool
    kolors: bool
    image_gen: bool
    agent: bool
    install_comfy: bool
    note: str
    digest: str | None = None

    def capabilities(self) -> dict:
        return {
            "comfy": self.comfy,
            "h3": self.h3,
            "kolors": self.kolors,
            "image_gen": self.image_gen,
            "agent": self.agent,
            "install_comfy": self.install_comfy,
        }


AGENT_IMAGE = "docker.io/aetherneo/anime-factory-gpu"
LEGACY_AGENT_ONLY_IMAGE = "docker.io/aetherneo/anime-factory-gpu-agent"
PYTORCH_IMAGE = "pytorch/pytorch"
COMFY_TEMPLATE_IMAGE = "vast-template/comfy-h3"

# Canonical catalog. Keys are normalized (no tag, no docker.io prefix variants).
IMAGE_CATALOG: dict[str, GpuImage] = {
    AGENT_IMAGE: GpuImage(
        name=AGENT_IMAGE,
        comfy=True,
        h3=True,
        kolors=True,
        image_gen=True,
        agent=True,
        install_comfy=False,
        note="Hub image: ComfyUI + MiniMax H3 + Flux 生图 + agent. Probe :8199. One card does stills and video.",
    ),
    LEGACY_AGENT_ONLY_IMAGE: GpuImage(
        name=LEGACY_AGENT_ONLY_IMAGE,
        comfy=False,
        h3=False,
        kolors=False,
        image_gen=False,
        agent=True,
        install_comfy=False,
        note="Historical agent-only. Do not probe :8199. Do not lease — cannot 生图 or H3.",
    ),
    PYTORCH_IMAGE: GpuImage(
        name=PYTORCH_IMAGE,
        comfy=False,
        h3=False,
        kolors=False,
        image_gen=False,
        agent=False,
        install_comfy=True,
        note="Base CUDA/pytorch. May install Comfy/H3/Flux on the same instance; do not lease until flags flip true.",
    ),
    COMFY_TEMPLATE_IMAGE: GpuImage(
        name=COMFY_TEMPLATE_IMAGE,
        comfy=True,
        h3=True,
        kolors=True,
        image_gen=True,
        agent=True,
        install_comfy=False,
        note="Vast template with Comfy/H3/Flux. Probe :8199.",
    ),
}

_ALIASES = {
    "aetherneo/anime-factory-gpu": AGENT_IMAGE,
    "docker.io/aetherneo/anime-factory-gpu": AGENT_IMAGE,
    "docker.io/aetherneo/anime-factory-gpu:main": AGENT_IMAGE,
    "anime-factory-gpu": AGENT_IMAGE,
    "aetherneo/anime-factory-gpu-agent": LEGACY_AGENT_ONLY_IMAGE,
    "docker.io/aetherneo/anime-factory-gpu-agent": LEGACY_AGENT_ONLY_IMAGE,
    "pytorch/pytorch": PYTORCH_IMAGE,
    "nvcr.io/nvidia/pytorch": PYTORCH_IMAGE,
}


def normalize_image(image: str | None) -> str:
    raw = (image or AGENT_IMAGE).strip()
    if raw in _ALIASES:
        return _ALIASES[raw]
    no_tag = raw.rsplit(":", 1)[0] if ":" in raw.rsplit("/", 1)[-1] else raw
    return _ALIASES.get(no_tag, no_tag if no_tag in IMAGE_CATALOG else AGENT_IMAGE)


def resolve_image(image: str | None) -> GpuImage:
    key = normalize_image(image)
    return IMAGE_CATALOG.get(key, IMAGE_CATALOG[AGENT_IMAGE])


def image_capabilities(image: str | None) -> dict:
    img = resolve_image(image)
    caps = img.capabilities()
    caps["image"] = img.name
    caps["note"] = img.note
    return caps


def stills_capable(capabilities: dict | None) -> bool:
    if not capabilities:
        return False
    return bool(capabilities.get("kolors") or capabilities.get("image_gen") or capabilities.get("image-gen"))


def image_can_lease(capabilities: dict | None) -> bool:
    """Lease only when the image can run Comfy, H3, and 生图 on the same card."""
    if not capabilities:
        return False
    return bool(capabilities.get("comfy") and capabilities.get("h3") and stills_capable(capabilities))


def should_probe_comfy(image: str | None = None, capabilities: dict | None = None) -> bool:
    """Never probe :8199 on an agent-only / comfy:false image."""
    if capabilities is not None:
        return bool(capabilities.get("comfy"))
    return resolve_image(image).comfy


def default_register_capabilities() -> dict:
    caps = image_capabilities(AGENT_IMAGE)
    caps["profile_id"] = default_profile_id()
    caps["image_digest"] = handshake_image_digest()
    return caps


@dataclass(frozen=True)
class CapabilityProfile:
    profile_id: str
    supported_arch: tuple[str, ...] = ("x86_64", "AMD64")
    supported_sm: tuple[str, ...] = ("sm_120",)
    min_vram_mb: int = 32_000
    min_disk_gb: float = 200.0
    min_mem_gb: float = 16.0
    require_torch: bool = True
    require_flash_attn: bool = False
    require_fouroversix: bool = False
    expected_image_digest: str | None = None
    note: str = ""


CAPABILITY_PROFILES: dict[str, CapabilityProfile] = {
    "h3-comfy-cu128-sm120": CapabilityProfile(
        profile_id="h3-comfy-cu128-sm120",
        supported_sm=("sm_120",),
        require_torch=True,
        require_flash_attn=False,
        require_fouroversix=False,
        note="Default RTX 5090 Comfy+H3+SDXL stills (CUDA 12.8, sm_120).",
    ),
    "longlive-nvfp4-sm120": CapabilityProfile(
        profile_id="longlive-nvfp4-sm120",
        supported_sm=("sm_120",),
        require_torch=True,
        require_flash_attn=True,
        require_fouroversix=True,
        note="LongLive NVFP4 on Blackwell sm_120; wheels must be baked.",
    ),
    # 4090 sm_89 is intentionally absent until a validated sm_89 image exists.
}


def default_profile_id() -> str:
    raw = (os.environ.get("AF_GPU_PROFILE") or "").strip()
    if raw in CAPABILITY_PROFILES:
        return raw
    backend = (os.environ.get("AF_VIDEO_BACKEND") or "h3").strip().lower()
    if backend == "longlive":
        return "longlive-nvfp4-sm120"
    return "h3-comfy-cu128-sm120"


def resolve_capability_profile(profile_id: str | None = None) -> CapabilityProfile:
    key = (profile_id or default_profile_id()).strip()
    return CAPABILITY_PROFILES.get(key, CAPABILITY_PROFILES["h3-comfy-cu128-sm120"])


def expected_image_digest(profile_id: str | None = None) -> str | None:
    """Expected content digest from env or image/profile config. Not a tag."""
    for key in (EXPECTED_IMAGE_DIGEST_ENV, IMAGE_DIGEST_ENV):
        parsed = parse_content_digest(os.environ.get(key))
        if parsed:
            return parsed
    profile = resolve_capability_profile(profile_id)
    parsed = parse_content_digest(profile.expected_image_digest)
    if parsed:
        return parsed
    image = IMAGE_CATALOG.get(AGENT_IMAGE)
    return parse_content_digest(image.digest if image else None)


def reported_image_digest() -> str | None:
    """Digest of the running image, if a real sha256 was injected."""
    for key in (REPORTED_IMAGE_DIGEST_ENV, IMAGE_DIGEST_ENV, "IMAGE_DIGEST"):
        parsed = parse_content_digest(os.environ.get(key))
        if parsed:
            return parsed
    return None


def handshake_image_digest() -> str | None:
    return reported_image_digest() or expected_image_digest()

"""Stack version contracts shared by capability profiles and Docker pins.

Keep ``deploy/gpu-worker/pins.env`` and Dockerfile ARG defaults aligned with these
values. Tests assert pins/Dockerfile/profile consistency — do not drift silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StackContract:
    torch: str
    torchvision: str
    torchaudio: str
    flash_attn: str | None = None
    torchao: str | None = None


H3_STACK = StackContract(
    torch="2.13.0+cu130",
    torchvision="0.28.0+cu130",
    torchaudio="2.11.0+cu130",
)

LONGLIVE_STACK = StackContract(
    torch="2.10.0+cu128",
    torchvision="0.25.0+cu128",
    torchaudio="2.10.0+cu128",
    flash_attn="2.8.3",
    torchao="0.16.0",
)

PROFILE_STACK: dict[str, StackContract] = {
    "h3-comfy-cu130-sm120": H3_STACK,
    "longlive-nvfp4-sm120": LONGLIVE_STACK,
}


def repo_pins_env_path() -> Path:
    return Path(__file__).resolve().parents[2] / "deploy" / "gpu-worker" / "pins.env"


def parse_pins_env(path: Path | None = None) -> dict[str, str]:
    src = path or repo_pins_env_path()
    out: dict[str, str] = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        key, _, value = text.partition("=")
        out[key.strip()] = value.strip()
    return out

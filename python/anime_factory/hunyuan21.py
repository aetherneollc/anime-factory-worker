"""HunyuanImage-2.1 T2I helpers (Phase A primary image backend).

Extracted patterns from ``tools/hunyuan21_gpu_canary.py``:
  - distilled FP8 pipeline name
  - inference hyper-params (steps / guidance / shift)
  - optional SDPA / dry-run paths for unit tests without GPU weights

Weights are never baked into Docker; ``generate_hunyuan21_t2i`` loads the
official Diffusers pipeline only when ``dry_run=False`` and CUDA is available.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL_NAME = "hunyuanimage-v2.1-distilled"
# Official 9:16 2K bucket used by the GPU canary for character masters.
DEFAULT_MASTER_WIDTH = 1536
DEFAULT_MASTER_HEIGHT = 2560
# Keyframe / plate bucket aligns with STILL_WIDTH×STILL_HEIGHT when not master.
DEFAULT_KEYFRAME_WIDTH = 1344
DEFAULT_KEYFRAME_HEIGHT = 768


class Hunyuan21Error(RuntimeError):
    """HunyuanImage-2.1 load or generate failed."""


def hunyuan21_model_name() -> str:
    return (os.environ.get("HY21_MODEL") or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME


def hunyuan21_use_refiner() -> bool:
    return str(os.environ.get("HY21_REFINER") or "0").strip() in {"1", "true", "yes"}


def inference_params(model_name: str | None = None) -> dict[str, Any]:
    """Match the canary distilled vs base hyper-parameters."""
    name = (model_name or hunyuan21_model_name()).lower()
    distilled = "distilled" in name
    return {
        "num_inference_steps": 8 if distilled else 28,
        "guidance_scale": 3.25 if distilled else 3.5,
        "shift": 4 if distilled else 5,
        "use_reprompt": False,
        "use_refiner": hunyuan21_use_refiner(),
    }


def _dry_run_png(width: int, height: int, *, prompt: str, seed: int | None) -> bytes:
    """Deterministic placeholder for unit tests (not a shippable still)."""
    tag = f"hunyuan21-dry:{width}x{height}:seed={seed or 0}:{prompt[:48]}".encode("utf-8", errors="replace")
    header = b"\x89PNG\r\n\x1a\n"
    pad = max(0, 128 - len(header) - len(tag))
    return header + tag + (b"\x00" * pad)


def generate_hunyuan21_t2i(
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = DEFAULT_KEYFRAME_WIDTH,
    height: int = DEFAULT_KEYFRAME_HEIGHT,
    seed: int | None = None,
    dry_run: bool | None = None,
    model_name: str | None = None,
) -> bytes:
    """Text-to-image via HunyuanImage-2.1.

    When ``dry_run`` is True (or HUNYUAN21_DRY_RUN=1), returns a stub PNG and
    never touches GPU / Hub weights.
    """
    if dry_run is None:
        dry_run = str(os.environ.get("HUNYUAN21_DRY_RUN") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
    if dry_run:
        return _dry_run_png(width, height, prompt=prompt, seed=seed)

    name = model_name or hunyuan21_model_name()
    params = inference_params(name)
    try:
        import torch
        from hyimage.diffusion.pipelines.hunyuanimage_pipeline import (  # type: ignore[import-not-found]
            HunyuanImagePipeline,
        )
    except Exception as exc:  # noqa: BLE001
        raise Hunyuan21Error(
            f"HunyuanImage-2.1 pipeline unavailable ({exc}); "
            "set HUNYUAN21_DRY_RUN=1 for stub, or install hyimage + CUDA weights"
        ) from exc

    if not torch.cuda.is_available():
        raise Hunyuan21Error("CUDA required for HunyuanImage-2.1 (or set HUNYUAN21_DRY_RUN=1)")

    pipe = HunyuanImagePipeline.from_pretrained(model_name=name, use_fp8=True)
    pipe = pipe.to("cuda")
    try:
        image = pipe(
            prompt=prompt,
            width=int(width),
            height=int(height),
            use_reprompt=params["use_reprompt"],
            use_refiner=params["use_refiner"],
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
            shift=params["shift"],
            seed=int(seed) if seed is not None else None,
        )
    except Exception as exc:  # noqa: BLE001
        raise Hunyuan21Error(f"HunyuanImage-2.1 generate failed: {exc}") from exc

    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def master_prompt(identity: str, *, framing: str | None = None) -> str:
    """Compose a Master full-body T2I prompt (canary framing + identity)."""
    frame = framing or (
        "original anime character reference sheet, solo, cel shaded, clean lineart, "
        "warm ivory studio background, full body shot, long shot, zoomed out, head to toe, "
        "the whole figure from the top of the head to both shoes fits inside the frame, "
        "both feet and shoes fully visible, standing with empty space below the shoes, "
        "front view, looking at viewer"
    )
    ident = str(identity or "").strip()
    return f"{frame}, {ident}" if ident else frame

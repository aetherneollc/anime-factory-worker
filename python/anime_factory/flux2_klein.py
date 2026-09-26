"""FLUX.2 Klein 4B distilled T2I and multi-reference edit.

Hugging Face model id (Apache-2.0): ``black-forest-labs/FLUX.2-klein-4B``.
The 9B checkpoint is non-commercial and must never be loaded.

Diffusers class is ``Flux2KleinPipeline`` (diffusers >= 0.37). The model card
calls the pipeline with ``torch.bfloat16``, ``guidance_scale=1.0``, and
``num_inference_steps=4``. Multi-reference editing passes
``image=[pil, ...]`` (at most 4). Weights are not baked into the image.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

# Distilled 4B only. Never FLUX.2-klein-9B.
KLEIN_4B_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
MODEL_ID = KLEIN_4B_MODEL_ID
MAX_REFS = 4
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 1.0
DEFAULT_MASTER_WIDTH = 768
DEFAULT_MASTER_HEIGHT = 1344
DEFAULT_KEYFRAME_WIDTH = 1280
DEFAULT_KEYFRAME_HEIGHT = 720
DEFAULT_WIDTH = DEFAULT_KEYFRAME_WIDTH
DEFAULT_HEIGHT = DEFAULT_KEYFRAME_HEIGHT

_PIPE: Any = None


class Flux2KleinError(RuntimeError):
    """Klein 4B load, ref-count, or generate failed."""


def klein_model_id() -> str:
    """Return the only allowed model id. Rejects any 9B override."""
    raw = (os.environ.get("FLUX2_KLEIN_MODEL") or KLEIN_4B_MODEL_ID).strip()
    lowered = raw.lower()
    if "9b" in lowered or "klein-9" in lowered:
        raise Flux2KleinError(
            f"refusing non-commercial FLUX.2 Klein 9B ({raw!r}); "
            f"only {KLEIN_4B_MODEL_ID} is allowed"
        )
    if lowered != KLEIN_4B_MODEL_ID.lower():
        raise Flux2KleinError(
            f"FLUX2_KLEIN_MODEL must be {KLEIN_4B_MODEL_ID}, got {raw!r}"
        )
    return KLEIN_4B_MODEL_ID


def inference_params() -> dict[str, Any]:
    """Match the HF model card distilled Klein 4B call."""
    return {
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "torch_dtype": "bfloat16",
        "model_id": klein_model_id(),
    }


def klein_dry_run(explicit: bool | None = None, env: Any | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    mapping = env if env is not None else os.environ
    return str(mapping.get("FLUX2_KLEIN_DRY_RUN") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _dry_png(width: int, height: int, *, tag: str) -> bytes:
    body = f"klein4b-dry:{tag}:{width}x{height}".encode("utf-8", errors="replace")
    header = b"\x89PNG\r\n\x1a\n"
    pad = max(0, 128 - len(header) - len(body))
    return header + body + (b"\x00" * pad)


def release_klein_vram() -> None:
    """Drop the resident Klein pipeline before SkyReels loads.

    The two models run serially and must not both occupy the GPU.
    """
    global _PIPE
    pipe = _PIPE
    _PIPE = None
    if pipe is not None:
        try:
            del pipe
        except Exception:  # noqa: BLE001
            pass
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — dry-run hosts have no torch
        pass


def _load_pipe() -> Any:
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    model_id = klein_model_id()
    try:
        import torch
        from diffusers import Flux2KleinPipeline
    except Exception as exc:  # noqa: BLE001
        raise Flux2KleinError(
            f"Flux2KleinPipeline unavailable ({exc}); "
            "set FLUX2_KLEIN_DRY_RUN=1 or install diffusers>=0.37"
        ) from exc
    if not torch.cuda.is_available():
        raise Flux2KleinError("CUDA required for FLUX.2 Klein 4B (or set FLUX2_KLEIN_DRY_RUN=1)")
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    _PIPE = pipe
    return pipe


def _save_png(image: Any) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _call_pipe(
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int | None,
    images: list[Any] | None = None,
) -> bytes:
    import torch

    pipe = _load_pipe()
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
        "guidance_scale": GUIDANCE_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": generator,
    }
    # HF / diffusers Flux2KleinPipeline.__call__ image: PIL or list[PIL].
    if images:
        kwargs["image"] = images
    try:
        result = pipe(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise Flux2KleinError(f"FLUX.2 Klein 4B generate failed: {exc}") from exc
    frames = getattr(result, "images", None) or []
    if not frames:
        raise Flux2KleinError("FLUX.2 Klein 4B returned no images")
    return _save_png(frames[0])


def generate_klein_t2i(
    prompt: str,
    width: int = DEFAULT_KEYFRAME_WIDTH,
    height: int = DEFAULT_KEYFRAME_HEIGHT,
    *,
    negative_prompt: str = "",
    seed: int | None = None,
    dry_run: bool | None = None,
) -> bytes:
    """Text-to-image. ``negative_prompt`` is accepted and unused (distilled cfg=1)."""
    del negative_prompt
    if klein_dry_run(dry_run):
        return _dry_png(width, height, tag=f"t2i:seed={seed or 0}:{prompt[:48]}")
    return _call_pipe(prompt=prompt, width=width, height=height, seed=seed, images=None)


def _as_pil(ref: str | bytes | Path) -> Any:
    from PIL import Image

    if isinstance(ref, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(ref))).convert("RGB")
    return Image.open(ref).convert("RGB")


def edit_klein_refs(
    prompt: str,
    refs: Sequence[str | bytes | Path],
    width: int = DEFAULT_KEYFRAME_WIDTH,
    height: int = DEFAULT_KEYFRAME_HEIGHT,
    *,
    negative_prompt: str = "",
    seed: int | None = None,
    dry_run: bool | None = None,
) -> bytes:
    """Edit with 1–4 reference images via ``image=[...]``."""
    del negative_prompt
    items = list(refs or [])
    if not items:
        raise Flux2KleinError("edit_klein_refs requires at least one reference")
    if len(items) > MAX_REFS:
        raise Flux2KleinError(f"FLUX.2 Klein accepts at most {MAX_REFS} refs, got {len(items)}")
    if klein_dry_run(dry_run):
        return _dry_png(width, height, tag=f"edit:n={len(items)}:seed={seed or 0}:{prompt[:48]}")
    images = [_as_pil(ref) for ref in items]
    return _call_pipe(prompt=prompt, width=width, height=height, seed=seed, images=images)


def master_prompt(identity: str, *, framing: str | None = None) -> str:
    """Compose a full-body master T2I prompt."""
    frame = framing or (
        "original anime character reference sheet, solo, cel shaded, clean lineart, "
        "warm ivory studio background, full body shot, long shot, zoomed out, head to toe, "
        "the whole figure from the top of the head to both shoes fits inside the frame, "
        "both feet and shoes fully visible, standing with empty space below the shoes, "
        "front view, looking at viewer"
    )
    ident = str(identity or "").strip()
    return f"{frame}, {ident}" if ident else frame

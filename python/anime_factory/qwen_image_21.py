"""Local Qwen-Image-2.1 stills: one pipeline for text-to-image and reference edit.

Model id ``Qwen/Qwen-Image-2.1``. Diffusers class ``QwenImage21Pipeline``
(text-to-image and up to about 10 reference images on the same pipe).

This is the production still path. Do not route it through DashScope
``qwen-image-2.0-pro`` in ``qwen_image.py``.

Weights stay off the image. ``QWEN_IMAGE_21_DRY_RUN=1`` returns a stub PNG and
does not import the pipeline or download weights. Drop the pipe with
``release_qwen_image_21_vram`` before MiniMax H3 anim so the two models do not
share the card.

The weight files are Qwen Research License. Generated pictures are not the
licensed weights; commercial use of the model weights themselves is a separate
grant. H3 video uses the commercial license already held for MiniMax H3.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

QWEN_IMAGE_21_MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_ID = QWEN_IMAGE_21_MODEL_ID
MAX_REFS = 10
NUM_INFERENCE_STEPS = 40
TRUE_CFG_SCALE = 4.0
# Generic still defaults are landscape; character masters pass portrait sizes explicitly.
DEFAULT_SHEET_WIDTH = 1344
DEFAULT_SHEET_HEIGHT = 768
DEFAULT_WIDTH = DEFAULT_SHEET_WIDTH
DEFAULT_HEIGHT = DEFAULT_SHEET_HEIGHT

_PIPE: Any = None


class QwenImage21Error(RuntimeError):
    """Qwen-Image-2.1 load, ref-count, or generate failed."""


class MasterReferenceError(QwenImage21Error):
    """A derived view was asked to run without master.png."""


def qwen_image_21_model_id() -> str:
    raw = (os.environ.get("QWEN_IMAGE_21_MODEL") or QWEN_IMAGE_21_MODEL_ID).strip()
    if raw.lower() != QWEN_IMAGE_21_MODEL_ID.lower():
        raise QwenImage21Error(
            f"QWEN_IMAGE_21_MODEL must be {QWEN_IMAGE_21_MODEL_ID}, got {raw!r}"
        )
    return QWEN_IMAGE_21_MODEL_ID


def qwen_image_21_dry_run(explicit: bool | None = None, env: Any | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    mapping = env if env is not None else os.environ
    return str(mapping.get("QWEN_IMAGE_21_DRY_RUN") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _dry_png(width: int, height: int, *, tag: str) -> bytes:
    """Real PNG of the requested size. Marker text proves no weights were fetched."""
    from anime_factory.design import synthetic_still_png

    return synthetic_still_png(
        int(width),
        int(height),
        tag=f"qwen21-dry:{tag}:{int(width)}x{int(height)}",
        placeholder=True,
    )


def release_qwen_image_21_vram() -> None:
    """Drop the resident Qwen pipeline before H3 loads.

    Stills and H3 anim run one after the other. Both weight sets must not
    stay resident on the same 5090.
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
    model_id = qwen_image_21_model_id()
    try:
        import torch
        from diffusers import QwenImage21Pipeline
    except Exception as exc:  # noqa: BLE001
        raise QwenImage21Error(
            f"QwenImage21Pipeline unavailable ({exc}); "
            "set QWEN_IMAGE_21_DRY_RUN=1 or install a diffusers build that includes it"
        ) from exc
    if not torch.cuda.is_available():
        raise QwenImage21Error("CUDA required for Qwen-Image-2.1 (or set QWEN_IMAGE_21_DRY_RUN=1)")
    pipe = QwenImage21Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
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
    negative_prompt: str = "",
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
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": generator,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
        kwargs["true_cfg_scale"] = TRUE_CFG_SCALE
    # QwenImage21Pipeline.__call__ image: one PIL or a list of up to MAX_REFS.
    if images:
        kwargs["image"] = images[0] if len(images) == 1 else images
    try:
        result = pipe(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise QwenImage21Error(f"Qwen-Image-2.1 generate failed: {exc}") from exc
    frames = getattr(result, "images", None) or []
    if not frames:
        raise QwenImage21Error("Qwen-Image-2.1 returned no images")
    return _save_png(frames[0])


def generate_qwen_image_21(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    *,
    negative_prompt: str = "",
    seed: int | None = None,
    dry_run: bool | None = None,
) -> bytes:
    """Text-to-image. One call per character master. No reference images."""
    if qwen_image_21_dry_run(dry_run):
        return _dry_png(width, height, tag=f"t2i:seed={seed or 0}:{prompt[:48]}")
    return _call_pipe(
        prompt=prompt,
        width=width,
        height=height,
        seed=seed,
        negative_prompt=negative_prompt,
        images=None,
    )


def _as_pil(ref: str | bytes | Path) -> Any:
    from PIL import Image

    if isinstance(ref, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(ref))).convert("RGB")
    return Image.open(ref).convert("RGB")


def _is_master_path(path: str) -> bool:
    text = str(path or "").replace("\\", "/").split("?", 1)[0].strip()
    return text.endswith("/master.png") or text.endswith("master.png")


def assert_side_master_reference(master_path: str | None) -> str:
    """Side (and any derive) must name master.png. A fresh T2I is refused."""
    path = str(master_path or "").replace("\\", "/").strip()
    if not _is_master_path(path):
        raise MasterReferenceError(
            f"side view requires master.png reference, got {master_path!r}"
        )
    return path


def edit_qwen_image_21(
    prompt: str,
    refs: Sequence[str | bytes | Path],
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    *,
    negative_prompt: str = "",
    seed: int | None = None,
    dry_run: bool | None = None,
    master_path: str | None = None,
    view: str | None = None,
) -> bytes:
    """Edit with 1–10 reference images via ``image=``.

    Side view refuses to run unless ``master_path`` is ``master.png``.
    """
    if str(view or "").strip().lower() == "side":
        assert_side_master_reference(master_path)
    items = list(refs or [])
    if not items:
        raise QwenImage21Error("edit_qwen_image_21 requires at least one reference")
    if len(items) > MAX_REFS:
        raise QwenImage21Error(
            f"Qwen-Image-2.1 accepts at most {MAX_REFS} refs, got {len(items)}"
        )
    if qwen_image_21_dry_run(dry_run):
        return _dry_png(width, height, tag=f"edit:n={len(items)}:seed={seed or 0}:{prompt[:48]}")
    images = [_as_pil(ref) for ref in items]
    return _call_pipe(
        prompt=prompt,
        width=width,
        height=height,
        seed=seed,
        negative_prompt=negative_prompt,
        images=images,
    )

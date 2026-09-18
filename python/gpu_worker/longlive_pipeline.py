"""In-process NVFP4 sampling against the official NVlabs LongLive 2.0 API.

Source of truth is NVlabs/LongLive at ``LONGLIVE_REF``:

* ``README.md`` "Quick Start -> NVFP4": build ``CausalDiffusionInferencePipeline(config,
  device=device)``, then ``setup_nvfp4_pipeline(pipe, config, device)``, then
  ``pipe.generator.model.eval().requires_grad_(False)``. The docs call out that the
  bf16 ``pipe.to(...)`` shortcut is unsafe because it casts the quantized buffers.
* ``utils/inference_utils.py``: ``setup_nvfp4_pipeline``, ``prepare_single_prompt_inputs``,
  ``save_video``.
* ``inference.py``: the i2v branch encodes one still to a clean first latent via
  ``pipeline.vae.encode_to_latent`` on a ``[B, C, 1, H, W]`` tensor, and clears the VAE
  cache between samples.

The pipeline is constructed and set up once per batch and then reused for every take.
Nothing in this module shells out to ``inference.py``.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_worker.longlive import (
    FAIL_CLOSED,
    LONGLIVE_REF,
    NUM_FRAME_PER_BLOCK,
    SETUP_HINT,
    WAN_SPATIAL_COMPRESSION,
    longlive_root,
    nvfp4_sampling_steps,
)

# inference.py: fps = 24 if '5B' in config.model_kwargs.model_name else 16.
LONGLIVE_OUTPUT_FPS = 24
# inference.py: low_memory = get_cuda_free_memory_gb(device) < 40.
LOW_VRAM_GB = 40.0
# utils/dataset.py DEFAULT_SCENE_CUT_PREFIX, mirrored by inference.py's getattr default.
DEFAULT_SCENE_CUT_PREFIX = "The scene transitions. "

# Every symbol the production path calls. Absent at the pinned ref -> fail closed;
# importing a class is not a model load and must never be treated as one.
REQUIRED_OFFICIAL_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("pipeline", "CausalDiffusionInferencePipeline"),
    ("utils.config", "normalize_config"),
    ("utils.inference_utils", "setup_nvfp4_pipeline"),
    ("utils.inference_utils", "prepare_single_prompt_inputs"),
    ("utils.inference_utils", "save_video"),
    ("utils.dataset", "ImagePromptDataset"),
    ("utils.misc", "set_seed"),
    ("utils.memory", "get_cuda_free_memory_gb"),
    ("utils.memory", "DynamicSwapInstaller"),
)


@dataclass(frozen=True)
class OfficialApi:
    """Resolved handles into the pinned NVlabs tree."""

    torch: Any
    OmegaConf: Any
    CausalDiffusionInferencePipeline: Any
    normalize_config: Any
    setup_nvfp4_pipeline: Any
    prepare_single_prompt_inputs: Any
    save_video: Any
    ImagePromptDataset: Any
    set_seed: Any
    get_cuda_free_memory_gb: Any
    DynamicSwapInstaller: Any


def import_official_api() -> OfficialApi:
    """Resolve the official inference helpers. Missing helpers fail closed."""
    root = longlive_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    missing: list[str] = []
    resolved: dict[str, Any] = {}
    for module_name, symbol in REQUIRED_OFFICIAL_SYMBOLS:
        try:
            resolved[symbol] = getattr(importlib.import_module(module_name), symbol)
        except Exception as exc:  # noqa: BLE001 — any import failure is fatal here
            missing.append(f"{module_name}.{symbol} ({type(exc).__name__}: {exc})")
    for module_name, symbol in (("torch", "torch"), ("omegaconf", "OmegaConf")):
        try:
            module = importlib.import_module(module_name)
            resolved[symbol] = module if symbol == module_name else getattr(module, symbol)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{module_name}.{symbol} ({type(exc).__name__}: {exc})")
    if missing:
        raise RuntimeError(
            f"{FAIL_CLOSED}: official LongLive inference API unavailable at {root} "
            f"(pinned {LONGLIVE_REF}): " + "; ".join(missing) + ". " + SETUP_HINT
        )
    return OfficialApi(**resolved)


def load_official_config(api: OfficialApi, yaml_path: Path) -> Any:
    """``normalize_config(OmegaConf.load(...))``, then assert the NVFP4 S2 contract."""
    config = api.normalize_config(api.OmegaConf.load(str(yaml_path)))
    steps = int(getattr(config, "sampling_steps", 0) or 0)
    expected = nvfp4_sampling_steps()
    if steps != expected:
        raise RuntimeError(
            f"{FAIL_CLOSED}: NVFP4 S2 needs sampling_steps={expected}, config has {steps}"
        )
    if not bool(getattr(config, "model_quant", False)):
        raise RuntimeError(f"{FAIL_CLOSED}: model_quant must be true for the NVFP4 checkpoint")
    if bool(getattr(config, "model_quant_use_transformer_engine", False)):
        raise RuntimeError(
            f"{FAIL_CLOSED}: model_4o6.pt is a FourOverSix checkpoint; "
            "model_quant_use_transformer_engine must be false"
        )
    return config


@contextmanager
def _official_working_directory():
    """Resolve NVlabs' cwd-relative ``wan_models/...`` sidecar paths."""
    root = longlive_root()
    if not root.is_dir():
        raise RuntimeError(f"{FAIL_CLOSED}: LongLive root is missing: {root}. {SETUP_HINT}")
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def _park_text_encoder_on_cpu(pipe: Any, torch: Any) -> None:
    """Keep UMT5 entirely in RAM on 32GB cards.

    Official ``inference.py`` DynamicSwap-pages T5 onto CUDA during encode. On a
    5090 that overlaps the NVFP4 sampler (observed 30.04 GiB in use, 1.30 GiB
    alloc fail). Encode on CPU instead; sampling then owns the whole card.
    """
    enc = getattr(pipe, "text_encoder", None)
    if enc is None:
        return
    to = getattr(enc, "to", None)
    if callable(to):
        to("cpu")
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
        try:
            if not callable(getattr(cuda, "is_available", None)) or cuda.is_available():
                cuda.empty_cache()
        except Exception:  # noqa: BLE001 — CPU unit tests have a stub cuda
            pass


def build_pipeline(api: OfficialApi, config: Any, *, device: Any = None) -> tuple[Any, Any]:
    """Construct and set up the NVFP4 pipeline exactly once, per the official README."""
    torch = api.torch
    dev = torch.device(device) if device is not None else torch.device("cuda")
    torch.set_grad_enabled(False)
    # Measured before the generator lands on the card, as inference.py does.
    low_memory = float(api.get_cuda_free_memory_gb(dev)) < LOW_VRAM_GB
    # The pinned wrapper opens config, UMT5, tokenizer and VAE files as
    # ``wan_models/Wan2.2-TI2V-5B/...``. The container starts in /app, so building
    # without the official cwd makes diffusers misread that path as a Hub repo.
    with _official_working_directory():
        pipe = api.CausalDiffusionInferencePipeline(config, device=dev)
        api.setup_nvfp4_pipeline(pipe, config, dev)
    if low_memory:
        # Do not DynamicSwap T5 onto CUDA: that is the 32GB NVFP4 OOM.
        _park_text_encoder_on_cpu(pipe, torch)
    pipe.generator.model.eval().requires_grad_(False)
    return pipe, dev


def _latent_shape(config: Any) -> list[int]:
    shape = [int(v) for v in list(config.image_or_video_shape)]
    if len(shape) != 5:
        raise RuntimeError(
            f"{FAIL_CLOSED}: image_or_video_shape must be [B, F, C, H, W], got {shape}"
        )
    return shape


def prepare_i2v_inputs(
    api: OfficialApi,
    pipeline: Any,
    config: Any,
    device: Any,
    *,
    data_path: Path,
    frames: int,
) -> tuple[Any, list[list[str]], Any]:
    """Encode the staged still to a clean first latent, exactly as inference.py does.

    The conditioning image is read through the official ``ImagePromptDataset`` so the
    ``/255 -> resize(antialias) -> normalize to [-1, 1] -> fp16`` preprocessing and the
    per-block prompt spreading match training and the upstream i2v branch.
    """
    torch = api.torch
    shape = _latent_shape(config)
    frame_height = shape[3] * WAN_SPATIAL_COMPRESSION
    frame_width = shape[4] * WAN_SPATIAL_COMPRESSION
    frames_per_block = int(getattr(config, "num_frame_per_block", NUM_FRAME_PER_BLOCK))
    if frames_per_block < 1 or int(frames) % frames_per_block != 0:
        raise RuntimeError(
            f"{FAIL_CLOSED}: num_output_frames={frames} must be a multiple of "
            f"num_frame_per_block={frames_per_block}"
        )
    dataset = api.ImagePromptDataset(
        data_path=str(data_path),
        image_size=(frame_height, frame_width),
        num_blocks=int(frames) // frames_per_block,
        scene_cut_prefix=str(getattr(config, "scene_cut_prefix", DEFAULT_SCENE_CUT_PREFIX)),
    )
    if len(dataset) != 1:
        raise RuntimeError(
            f"{FAIL_CLOSED}: i2v take must stage exactly one conditioning still under "
            f"{data_path}, ImagePromptDataset found {len(dataset)}"
        )
    item = dataset[0]
    num_samples = int(getattr(config, "num_samples", 1) or 1)
    # Dataset yields [C, H, W]; image_prompt_collate_fn stacks a batch axis.
    image = item["image"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    if image.ndim == 4:
        image = image.unsqueeze(2)
    elif image.ndim != 5:
        raise RuntimeError(
            f"{FAIL_CLOSED}: expected i2v image [B,C,H,W] or [B,C,T,H,W], got {tuple(image.shape)}"
        )
    initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
    if int(initial_latent.shape[0]) != num_samples:
        initial_latent = initial_latent.repeat(num_samples, 1, 1, 1, 1)
    if int(frames) <= int(initial_latent.shape[1]):
        raise RuntimeError(
            f"{FAIL_CLOSED}: num_output_frames must exceed the i2v conditioning frames; "
            f"got {frames} and {int(initial_latent.shape[1])}"
        )
    block_prompts = list(item["prompts"])
    noise = torch.randn(
        [num_samples, int(frames), shape[2], shape[3], shape[4]],
        device=device,
        dtype=torch.bfloat16,
    )
    return noise, [block_prompts] * num_samples, initial_latent


def sample_take(
    api: OfficialApi,
    pipeline: Any,
    config: Any,
    device: Any,
    *,
    dest: Path,
    frames: int,
    seed: int,
    i2v: bool,
    caption: str = "",
    data_path: Path | None = None,
) -> Path:
    """Sample one take with the resident pipeline and write it to ``dest``.

    ``dest`` is the caller's explicit destination, so output mapping never depends on
    scanning an output folder for the newest mp4.
    """
    torch = api.torch
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Takes differ only in length: the resident pipeline pins latent H/W, block size
    # and attention window, all of which are identical across the batch.
    config.num_output_frames = int(frames)
    config.image_or_video_shape[1] = int(frames)
    api.set_seed(int(seed))
    if i2v:
        if data_path is None:
            raise RuntimeError(f"{FAIL_CLOSED}: i2v take has no staged data_path")
        noise, prompts, initial_latent = prepare_i2v_inputs(
            api, pipeline, config, device, data_path=data_path, frames=frames
        )
        kwargs: dict[str, Any] = {
            "noise": noise,
            "text_prompts": prompts,
            "initial_latent": initial_latent,
        }
    else:
        if not caption.strip():
            raise RuntimeError(f"{FAIL_CLOSED}: t2v take has an empty prompt")
        noise, prompts = api.prepare_single_prompt_inputs(config, caption, device)
        kwargs = {"noise": noise, "text_prompts": prompts}
    try:
        with torch.inference_mode():
            generated = pipeline.inference(**kwargs)
    except Exception as exc:  # noqa: BLE001 — wrap CUDA OOM as fail-closed
        compact = str(exc).lower().replace(" ", "")
        if "outofmemory" in compact or "out of memory" in str(exc).lower():
            raise RuntimeError(f"{FAIL_CLOSED}: CUDA OOM during LongLive sample: {exc}") from exc
        raise
    # write_video is video-only; LongLive output stays silent.
    api.save_video(generated[0], str(dest), fps=LONGLIVE_OUTPUT_FPS)
    pipeline.vae.model.clear_cache()
    if not dest.is_file() or dest.stat().st_size < 32:
        raise RuntimeError(f"{FAIL_CLOSED}: pipeline wrote no video to {dest}. {SETUP_HINT}")
    return dest

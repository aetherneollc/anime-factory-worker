"""LongLive 2.0 (NVlabs) anim path — silent video, first-frame I2V continue.

Not a Comfy graph and not a MiniMax/Hailuo API. Official inference is
`inference.py` from https://github.com/NVlabs/LongLive (Apache-2.0) plus the
NVFP4 FourOverSix checkpoint `model_4o6.pt` from
Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2 (~3GB packed generator vs
~10GB BF16). Sidecar Wan files stay under `wan_models/Wan2.2-TI2V-5B/`
(config.json, VAE, UMT5, tokenizer) — never snapshot the full Wan training repo.

Missing repo/weights/extensions raise. Never write a black/placeholder mp4.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from gpu_worker.h3 import extract_last_frame, is_directory_like, segment_prompt
from gpu_worker.preflight import production_stack_locked

ProgressCallback = Callable[[str], None]

LONGLIVE_ROOT_DEFAULT = "/opt/LongLive"
WAN_ARCH_NAME = "Wan2.2-TI2V-5B"
WAN_DIR_DEFAULT = f"/opt/LongLive/wan_models/{WAN_ARCH_NAME}"
GENERATOR_DEFAULT = "/opt/LongLive/checkpoints/longlive2_5b_nvfp4_s2/model_4o6.pt"
LONGLIVE_REPO_URL = "https://github.com/NVlabs/LongLive.git"
LONGLIVE_CKPT_REPO = "Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2"
LONGLIVE_CKPT_FILE = "model_4o6.pt"
LONGLIVE_NVFP4_SAMPLING_STEPS = 2  # S2 distillation; S4 would use 4
# Sidecar-only Hub repo. Transformer weights come from LONGLIVE_CKPT_REPO.
WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B"
# Files NVlabs inference actually opens. Not the 20GB diffusion_pytorch_model shards.
WAN_INFERENCE_FILES = (
    "config.json",
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)
# Only if the cloned wrapper still CausalWanModel.from_pretrained's original Wan.
WAN_TRANSFORMER_SHARDS = (
    "diffusion_pytorch_model.safetensors.index.json",
    "diffusion_pytorch_model-00001-of-00003.safetensors",
    "diffusion_pytorch_model-00002-of-00003.safetensors",
    "diffusion_pytorch_model-00003-of-00003.safetensors",
)
# Wan2.2-TI2V-5B native latent grid 80×44 @ 16× → 1280×704. Delivery scales to 864×480.
LONGLIVE_NATIVE_WIDTH = 1280
LONGLIVE_NATIVE_HEIGHT = 704
TEMPORAL_COMPRESSION = 4
NUM_FRAME_PER_BLOCK = 8
# Official configs/inference.yaml: num_output_frames=384 latent frames ≈ 64s @ 24fps.
LONGLIVE_DEFAULT_LATENT_FRAMES = 384
# NVlabs ImagePromptDataset (inference.py i2v branch when data_path has no video/).
I2V_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
MIN_I2V_STILL_BYTES = 32
# Inference extras only. Do not pip the full LongLive requirements (CLIP git, wandb, flask).
LONGLIVE_PIP_PACKAGES = (
    "omegaconf",
    "einops",
    "easydict",
    "imageio",
    "imageio-ffmpeg",
    "av",
    "opencv-python-headless",
    "ftfy",
    "sentencepiece",
    "peft",
    "transformers>=4.49.0,<5",
    "accelerate>=1.1.1",
    "diffusers==0.31.0",
    "tqdm",
    "open_clip_torch",
    "torchao==0.16.0",
)
LONGLIVE_NVFP4_BUILD_PACKAGES = (
    "ninja",
    "packaging",
    "psutil",
    "setuptools>=77.0.3",
)
SETUP_HINT = (
    "LongLive 2.0 NVFP4 inference loads LONGLIVE_GENERATOR_CKPT "
    "(Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2 model_4o6.pt, FourOverSix "
    "materialized NVFP4) plus fouroversix, flash-attn, and utils/kernel KV dequant. "
    "Sidecar-only Wan files under LONGLIVE_WAN_DIR: config.json, Wan2.2_VAE.pth, "
    "models_t5_umt5-xxl-enc-bf16.pth, google/umt5-xxl/. "
    "Do not snapshot the full Wan-AI/Wan2.2-TI2V-5B training repo. "
    "Do not load the BF16 generator. FourOverSix is baked as an SM 12.0 wheel in the Hub "
    "image; first boot must not pip install -e (that needs nvcc and fails on slim). "
    "First boot pulls Hub onto this card when AF_VIDEO_BACKEND=longlive. "
    "No Comfy node and no commercial API — do not fake an mp4."
)
FAIL_CLOSED = "longlive_fail_closed"
WHEELS_DIR_DEFAULT = "/opt/longlive-wheels"


def longlive_root() -> Path:
    return Path(os.environ.get("LONGLIVE_ROOT") or LONGLIVE_ROOT_DEFAULT)


def wan_dir() -> Path:
    return Path(os.environ.get("LONGLIVE_WAN_DIR") or WAN_DIR_DEFAULT)


def generator_ckpt() -> Path:
    return Path(os.environ.get("LONGLIVE_GENERATOR_CKPT") or GENERATOR_DEFAULT)


def longlive_wheels_dir() -> Path:
    return Path(os.environ.get("LONGLIVE_WHEELS") or WHEELS_DIR_DEFAULT)


def _assert_nvfp4_generator(ckpt: Path | None = None) -> Path:
    """Refuse the ~10GB BF16 generator that SIGKILLs a 32GB cgroup."""
    path = Path(ckpt or generator_ckpt())
    name = path.name.lower()
    if "bf16" in name or "model_bf16" in name:
        raise RuntimeError(
            f"{FAIL_CLOSED}: refuse BF16 generator {path}. "
            f"Use {LONGLIVE_CKPT_FILE} from {LONGLIVE_CKPT_REPO}. {SETUP_HINT}"
        )
    return path


def lora_ckpt() -> Path | None:
    raw = (os.environ.get("LONGLIVE_LORA_CKPT") or "").strip()
    return Path(raw) if raw else None


def python_bin() -> str:
    return (os.environ.get("LONGLIVE_PYTHON") or "").strip() or sys.executable


def _wan_wrapper_path() -> Path:
    return longlive_root() / "utils" / "wan_5b_wrapper.py"


def _wrapper_loads_original_transformer() -> bool:
    """True when NVlabs still from_pretrained's the original Wan transformer."""
    path = _wan_wrapper_path()
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return "CausalWanModel.from_pretrained" in text


def _patch_wrapper_skip_wan_init() -> bool:
    """Load architecture from config.json, then overwrite with LongLive generator_ckpt.

    Official inference does not need the 20GB Wan transformer shards; from_pretrained
    is the training/init path. Returns True when the wrapper no longer from_pretrained's.
    """
    path = _wan_wrapper_path()
    if not path.is_file():
        return not _wrapper_loads_original_transformer()
    text = path.read_text(encoding="utf-8")
    if "CausalWanModel.from_pretrained" not in text:
        return True
    patched = re.sub(
        r"CausalWanModel\.from_pretrained\(\s*"
        r"f[\"']wan_models/\{model_name\}/[\"']\s*,\s*"
        r"local_attn_size=local_attn_size,\s*"
        r"sink_size=sink_size,\s*"
        r"num_frame_per_block=num_frame_per_block\s*\)",
        "CausalWanModel.from_config(\n"
        "            CausalWanModel.load_config(f\"wan_models/{model_name}/\"),\n"
        "            local_attn_size=local_attn_size, sink_size=sink_size,\n"
        "            num_frame_per_block=num_frame_per_block)",
        text,
        count=1,
    )
    patched = re.sub(
        r"WanModel\.from_pretrained\(\s*f[\"']wan_models/\{model_name\}/[\"']\s*\)",
        'WanModel.from_config(WanModel.load_config(f"wan_models/{model_name}/"))',
        patched,
        count=1,
    )
    if patched == text:
        return False
    path.write_text(patched, encoding="utf-8")
    return "CausalWanModel.from_pretrained" not in patched


def _patch_wrapper_t5_lowmem() -> bool:
    """UMT5 as float32 on CPU is ~22GB and SIGKILLs a 32GB cgroup. Use bf16 on CUDA."""
    path = _wan_wrapper_path()
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    patched = text.replace("dtype=torch.float32,", "dtype=torch.bfloat16,")
    patched = patched.replace(
        "device=torch.device('cpu')",
        "device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        1,
    )
    patched = patched.replace(
        'map_location=\'cpu\', weights_only=False)',
        'map_location="cpu", weights_only=False, mmap=True)',
        1,
    )
    patched = patched.replace(
        'map_location="cpu", weights_only=False)',
        'map_location="cpu", weights_only=False, mmap=True)',
        1,
    )
    if patched == text:
        return "bfloat16" in text and "mmap=True" in text
    path.write_text(patched, encoding="utf-8")
    return "bfloat16" in patched


def _patch_inference_mmap_load() -> bool:
    """Avoid a 10GB+ CPU copy of model_bf16.pt on 32GB cgroup boxes (SIGKILL / exit -9)."""
    path = longlive_root() / "inference.py"
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    needles = (
        "generator_checkpoint = torch.load(generator_ckpt_path, map_location=\"cpu\")",
        "generator_checkpoint = torch.load(generator_ckpt_path, map_location='cpu')",
    )
    repl = "generator_checkpoint = torch.load(generator_ckpt_path, map_location=\"cpu\", mmap=True)"
    if "mmap=True" in text and "generator_ckpt_path" in text:
        return True
    for needle in needles:
        if needle in text:
            path.write_text(text.replace(needle, repl, 1), encoding="utf-8")
            return "mmap=True" in path.read_text(encoding="utf-8")
    return False


def _install_torch_mmap_sitecustomize() -> Path:
    """Force mmap on CPU torch.load (UMT5 / VAE as well as the generator ckpt)."""
    path = longlive_root() / "sitecustomize.py"
    path.write_text(
        "try:\n"
        "    import torch\n"
        "    _orig = torch.load\n"
        "    def _load(*args, **kwargs):\n"
        "        loc = kwargs.get('map_location', 'cpu')\n"
        "        if loc in (None, 'cpu') or str(loc) == 'cpu':\n"
        "            kwargs.setdefault('mmap', True)\n"
        "        return _orig(*args, **kwargs)\n"
        "    torch.load = _load\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8",
    )
    return path


def _longlive_inference_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(longlive_root())
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def nvfp4_sampling_steps() -> int:
    raw = (os.environ.get("LONGLIVE_NVFP4_SAMPLING_STEPS") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return LONGLIVE_NVFP4_SAMPLING_STEPS


def _nvfp4_cuda_archs() -> str:
    explicit = (os.environ.get("CUDA_ARCHS") or "").strip()
    if explicit:
        return explicit
    try:
        from gpu_worker.stack import cuda_sm

        sm = cuda_sm() or ""
    except Exception:  # noqa: BLE001 — optional probe
        sm = ""
    if sm.startswith("sm_12"):
        return "120"
    return "100"


def is_longlive_fatal_error(exc: BaseException | str | None) -> bool:
    """Install/compile/OOM failures must fail closed — do not retry or re-lease."""
    if exc is None:
        return False
    code = getattr(exc, "returncode", None)
    if code in (-9, 9, 137):
        return True
    compact = str(exc).lower().replace(" ", "_").replace("-", "_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    if FAIL_CLOSED in compact or "sigkill" in compact or "signal_9" in compact:
        return True
    if "nvcc_is_not" in compact or "cuda_extension_missing" in compact:
        return True
    return bool(re.search(r"exit_9([^0-9]|$)", compact))


def _nvcc_available() -> bool:
    if shutil.which("nvcc"):
        return True
    home = (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "").strip()
    if home and (Path(home) / "bin" / "nvcc").is_file():
        return True
    return Path("/usr/local/cuda/bin/nvcc").is_file()


def _prebuilt_wheel(prefix: str) -> Path | None:
    root = longlive_wheels_dir()
    if not root.is_dir():
        return None
    found = sorted(root.glob(f"{prefix}*.whl"))
    return found[-1] if found else None


def _install_wheel(wheel: Path, env: dict[str, str]) -> None:
    subprocess.run(
        [
            python_bin(),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            str(wheel),
        ],
        check=True,
        timeout=300,
        env=env,
    )


def _fouroversix_ready() -> bool:
    try:
        import fouroversix  # noqa: F401
        from fouroversix import _C  # noqa: F401
    except ImportError:
        return False
    return True


def _flash_attn_ready() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def _kv_dequant_ready() -> bool:
    root = longlive_root()
    if not (root / "utils" / "kernel" / "kv_dequant.py").is_file():
        return False
    env = _longlive_inference_env()
    try:
        subprocess.run(
            [
                python_bin(),
                "-c",
                "from utils.kernel.kv_dequant import dequantize_kv_cache_fp4",
            ],
            check=True,
            cwd=str(root),
            env=env,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return True


def _install_fouroversix(progress: ProgressCallback | None = None) -> None:
    if _fouroversix_ready():
        return
    if progress:
        progress("weights:longlive:fouroversix")
    env = _longlive_inference_env()
    wheel = _prebuilt_wheel("fouroversix")
    if wheel is not None:
        try:
            _install_wheel(wheel, env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"{FAIL_CLOSED}: fouroversix wheel install failed (exit {exc.returncode}). {SETUP_HINT}"
            ) from exc
        if _fouroversix_ready():
            return
    if production_stack_locked():
        raise RuntimeError(
            f"{FAIL_CLOSED}: fouroversix wheel missing in production image "
            f"(expected under {longlive_wheels_dir()}). Live compile is forbidden. {SETUP_HINT}"
        )
    if not _nvcc_available():
        raise RuntimeError(
            f"{FAIL_CLOSED}: fouroversix CUDA extension missing and nvcc is not on this image. "
            "Do not pip install -e (needs CUDA 12.8+ toolkit / SM 12.0). "
            f"Expected a prebuilt wheel under {longlive_wheels_dir()}. {SETUP_HINT}"
        )
    env["CUDA_ARCHS"] = _nvfp4_cuda_archs()
    from gpu_worker.stack import ensure_c_compiler

    ensure_c_compiler()
    subprocess.run(
        [python_bin(), "-m", "pip", "install", "--upgrade", *LONGLIVE_NVFP4_BUILD_PACKAGES],
        check=True,
        timeout=600,
        env=env,
    )
    bundled = longlive_root() / "fouroversix"
    if not bundled.is_dir():
        raise RuntimeError(f"{FAIL_CLOSED}: nvcc present but {bundled} is missing. {SETUP_HINT}")
    try:
        subprocess.run(
            [
                python_bin(),
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "-e",
                str(bundled),
            ],
            check=True,
            timeout=1800,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{FAIL_CLOSED}: fouroversix compile failed (exit {exc.returncode}). {SETUP_HINT}"
        ) from exc
    if not _fouroversix_ready():
        raise RuntimeError(f"{FAIL_CLOSED}: fouroversix import failed after install. {SETUP_HINT}")


def _install_flash_attn(progress: ProgressCallback | None = None) -> None:
    if _flash_attn_ready():
        return
    if progress:
        progress("weights:longlive:flash_attn")
    env = _longlive_inference_env()
    wheel = _prebuilt_wheel("flash")
    if wheel is not None:
        try:
            _install_wheel(wheel, env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"{FAIL_CLOSED}: flash-attn wheel install failed (exit {exc.returncode}). {SETUP_HINT}"
            ) from exc
        if _flash_attn_ready():
            return
    if production_stack_locked():
        raise RuntimeError(
            f"{FAIL_CLOSED}: flash-attn wheel missing in production image "
            f"(expected under {longlive_wheels_dir()}). Live compile is forbidden. {SETUP_HINT}"
        )
    if not _nvcc_available():
        raise RuntimeError(
            f"{FAIL_CLOSED}: flash-attn missing and nvcc is not on this image. {SETUP_HINT}"
        )
    subprocess.run(
        [
            python_bin(),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "flash-attn",
        ],
        check=True,
        timeout=1800,
        env=env,
    )
    if not _flash_attn_ready():
        raise RuntimeError(f"{FAIL_CLOSED}: flash-attn import failed after install. {SETUP_HINT}")


def _build_kv_dequant_kernel(progress: ProgressCallback | None = None) -> None:
    if _kv_dequant_ready():
        return
    if progress:
        progress("weights:longlive:kv_dequant")
    kernel_dir = longlive_root() / "utils" / "kernel"
    if not (kernel_dir / "setup.py").is_file():
        raise RuntimeError(f"{FAIL_CLOSED}: LongLive KV dequant setup missing at {kernel_dir}. {SETUP_HINT}")
    if production_stack_locked():
        raise RuntimeError(
            f"{FAIL_CLOSED}: KV dequant extension missing in production image. "
            f"Live compile is forbidden. {SETUP_HINT}"
        )
    if not _nvcc_available():
        raise RuntimeError(
            f"{FAIL_CLOSED}: utils/kernel KV dequant missing and nvcc is not on this image. {SETUP_HINT}"
        )
    from gpu_worker.stack import ensure_c_compiler

    ensure_c_compiler()
    env = _longlive_inference_env()
    subprocess.run(
        [python_bin(), "setup.py", "build_ext", "--inplace"],
        check=True,
        cwd=str(kernel_dir),
        timeout=1800,
        env=env,
    )
    if not _kv_dequant_ready():
        raise RuntimeError(f"{FAIL_CLOSED}: utils/kernel KV dequant extension failed to build. {SETUP_HINT}")


def _install_nvfp4_stack(progress: ProgressCallback | None = None) -> None:
    _install_fouroversix(progress)
    _install_flash_attn(progress)
    _build_kv_dequant_kernel(progress)


def missing_nvfp4_requirements() -> list[str]:
    missing: list[str] = []
    if not _fouroversix_ready():
        missing.append("fouroversix (NVFP4 matmul/quantize)")
    if not _flash_attn_ready():
        missing.append("flash-attn (Blackwell attention kernels)")
    if not _kv_dequant_ready():
        missing.append("utils/kernel longlive_kv_dequant_cuda (FP4 KV cache)")
    return missing


def _ensure_wan_layout() -> Path:
    """Official inference.py hardcodes cwd-relative wan_models/Wan2.2-TI2V-5B/."""
    dest = wan_dir()
    dest.mkdir(parents=True, exist_ok=True)
    official = longlive_root() / "wan_models" / WAN_ARCH_NAME
    if official.resolve() == dest.resolve():
        return dest
    official.parent.mkdir(parents=True, exist_ok=True)
    if official.is_symlink():
        if official.resolve() == dest.resolve():
            return dest
        official.unlink()
    elif official.exists():
        return dest
    official.symlink_to(dest, target_is_directory=True)
    return dest


def _wan_sidecar_files() -> tuple[str, ...]:
    files = list(WAN_INFERENCE_FILES)
    if _wrapper_loads_original_transformer():
        files.extend(WAN_TRANSFORMER_SHARDS)
    return tuple(files)


def longlive_spatial_size() -> tuple[int, int]:
    try:
        width = int(os.environ.get("LONGLIVE_WIDTH") or LONGLIVE_NATIVE_WIDTH)
        height = int(os.environ.get("LONGLIVE_HEIGHT") or LONGLIVE_NATIVE_HEIGHT)
    except ValueError:
        return LONGLIVE_NATIVE_WIDTH, LONGLIVE_NATIVE_HEIGHT
    return max(64, width), max(64, height)


def longlive_video_seconds(latent_frames: int = LONGLIVE_DEFAULT_LATENT_FRAMES) -> float:
    """Video seconds for a latent-frame count (Wan 4× temporal, 24fps)."""
    video_frames = max(1, (int(latent_frames) - 1) * TEMPORAL_COMPRESSION + 1)
    return video_frames / 24.0


def longlive_num_output_frames(seconds: float) -> int:
    """Latent frames for Wan 4× temporal compression, snapped to the AR block size."""
    video_frames = max(5, int(round(float(seconds) * 24)))
    latents = (video_frames - 1) // TEMPORAL_COMPRESSION + 1
    while latents % NUM_FRAME_PER_BLOCK != 0:
        latents += 1
    return max(NUM_FRAME_PER_BLOCK, latents)


def select_longlive_mode(segment: dict) -> str:
    """I2V when a real first frame exists (Toonflow continue / establishing still)."""
    first = segment.get("first_frame_path")
    if first and not is_directory_like(first) and Path(str(first)).is_file():
        return "i2v"
    return "t2v"


def i2v_first_chunk_video_frames(
    *,
    num_frame_per_block: int = NUM_FRAME_PER_BLOCK,
    temporal_compression_ratio: int = TEMPORAL_COMPRESSION,
) -> int:
    """RGB frames MultiVideoConcatDataset demands for the first chunk (Wan 5B: 29).

    Official I2V does not need that source video: inference.py uses ImagePromptDataset
    when data_path has no ``video/`` subdirectory, conditioning on a single still.
    """
    first_chunk_latent_frames = max(1, int(num_frame_per_block))
    return 1 + (first_chunk_latent_frames - 1) * int(temporal_compression_ratio)


def _i2v_images(data_path: Path) -> list[Path]:
    root = Path(data_path)
    image_dir = root / "images" if (root / "images").is_dir() else root
    found: list[Path] = []
    for ext in I2V_IMAGE_EXTENSIONS:
        found.extend(image_dir.glob(f"*{ext}"))
        found.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(
        {p.resolve(): p for p in found if p.is_file() and p.stat().st_size >= MIN_I2V_STILL_BYTES}.values(),
        key=lambda p: p.name,
    )


def assert_i2v_image_layout(data_path: Path) -> Path:
    """Refuse the MultiVideoConcatDataset path that raises 'no video can provide the first chunk'.

    NVlabs inference.py: ``i2v`` + ``data_path/video/`` → sample first_chunk_frames from mp4.
    A 1-frame conditioner (or empty video/) cannot fill that chunk. Official 2.0 I2V is a
    first-frame still under ``images/`` plus ``prompts/<stem>.txt``.
    """
    root = Path(data_path)
    video_dir = root / "video"
    if video_dir.exists():
        raise RuntimeError(
            "longlive i2v data_path has video/ — official inference.py would use "
            "MultiVideoConcatDataset and demand "
            f"{i2v_first_chunk_video_frames()} source frames for the first chunk. "
            "Use images/ + prompts/ (ImagePromptDataset), not a conditioner mp4."
        )
    images = _i2v_images(root)
    if not images:
        raise RuntimeError(
            f"longlive i2v first chunk empty: no first-frame image under {root}/images/. "
            "Stage the locked turnaround / plate / keyframe png."
        )
    image = images[0]
    prompt_dir = root / "prompts" if (root / "prompts").is_dir() else image.parent
    prompt = prompt_dir / f"{image.stem}.txt"
    if not prompt.is_file() or not prompt.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"longlive i2v missing prompt for {image.name}: expected {prompt}")
    return image


def missing_longlive_requirements() -> list[str]:
    missing: list[str] = []
    root = longlive_root()
    if not (root / "inference.py").is_file():
        missing.append(f"LONGLIVE_ROOT inference.py ({root / 'inference.py'})")
    ckpt = generator_ckpt()
    if not ckpt.is_file():
        missing.append(f"LONGLIVE_GENERATOR_CKPT ({ckpt})")
    wan = wan_dir()
    for rel in _wan_sidecar_files():
        path = wan / rel
        if not path.is_file() or path.stat().st_size < 32:
            missing.append(f"LONGLIVE_WAN_DIR {rel} ({path})")
    missing.extend(missing_nvfp4_requirements())
    return missing


def _skip_ensure() -> bool:
    return os.environ.get("AF_SKIP_WEIGHTS", "").strip().lower() in {"1", "true", "yes", "on"}


def _hf_token() -> str | None:
    raw = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return raw or None


def _hf_kwargs() -> dict:
    token = _hf_token()
    return {"token": token} if token else {}


def _clone_longlive_repo(progress: ProgressCallback | None = None) -> Path:
    root = longlive_root()
    if (root / "inference.py").is_file():
        return root
    if progress:
        progress("weights:longlive:clone")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--single-branch",
            "--branch",
            "main",
            "--depth",
            "1",
            LONGLIVE_REPO_URL,
            str(root),
        ],
        check=True,
        timeout=180,
    )
    if not (root / "inference.py").is_file():
        raise RuntimeError(f"git clone {LONGLIVE_REPO_URL} left no inference.py at {root}")
    _patch_wrapper_skip_wan_init()
    _patch_wrapper_t5_lowmem()
    _patch_inference_mmap_load()
    return root


def _download_generator(progress: ProgressCallback | None = None) -> Path:
    ckpt = generator_ckpt()
    if ckpt.is_file() and ckpt.stat().st_size > 32:
        return ckpt
    if progress:
        progress("weights:longlive:generator")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=LONGLIVE_CKPT_REPO,
        filename=LONGLIVE_CKPT_FILE,
        local_dir=str(ckpt.parent),
        **_hf_kwargs(),
    )
    src = Path(path)
    if src.resolve() != ckpt.resolve():
        shutil.copy2(src, ckpt)
    if not ckpt.is_file() or ckpt.stat().st_size < 32:
        raise RuntimeError(f"LongLive generator missing after Hub pull: {ckpt}. {SETUP_HINT}")
    return ckpt


def _download_wan(progress: ProgressCallback | None = None) -> Path:
    """Pull only the sidecar files inference.py opens. Never snapshot the full Wan repo."""
    dest = _ensure_wan_layout()
    needed = _wan_sidecar_files()
    pending = [rel for rel in needed if not (dest / rel).is_file() or (dest / rel).stat().st_size < 32]
    if not pending:
        return dest
    if progress:
        progress("weights:longlive:wan")
    from huggingface_hub import hf_hub_download

    for rel in pending:
        hf_hub_download(
            repo_id=WAN_REPO,
            filename=rel,
            local_dir=str(dest),
            **_hf_kwargs(),
        )
    missing = [rel for rel in needed if not (dest / rel).is_file() or (dest / rel).stat().st_size < 32]
    if missing:
        raise RuntimeError("longlive Wan sidecars missing: " + ", ".join(missing) + ". " + SETUP_HINT)
    return dest


def _install_longlive_python(progress: ProgressCallback | None = None) -> None:
    if progress:
        progress("weights:longlive:pip")
    subprocess.run(
        [python_bin(), "-m", "pip", "install", "--upgrade", *LONGLIVE_PIP_PACKAGES],
        check=True,
        timeout=900,
    )


def ensure_longlive(progress: ProgressCallback | None = None) -> dict[str, Any]:
    """Clone NVlabs/LongLive and pull Hub weights onto this card. Never R2."""
    if _skip_ensure():
        return {"ok": True, "skipped": True, "missing": missing_longlive_requirements()}
    _clone_longlive_repo(progress)
    _patch_wrapper_skip_wan_init()
    _patch_wrapper_t5_lowmem()
    _patch_inference_mmap_load()
    _install_torch_mmap_sitecustomize()
    _assert_nvfp4_generator()
    _download_generator(progress)
    _download_wan(progress)
    _install_longlive_python(progress)
    _install_nvfp4_stack(progress)
    missing = missing_longlive_requirements()
    if missing:
        raise RuntimeError(f"{FAIL_CLOSED}: longlive not ready: " + "; ".join(missing) + ". " + SETUP_HINT)
    return {
        "ok": True,
        "skipped": False,
        "root": str(longlive_root()),
        "generator_ckpt": str(generator_ckpt()),
        "wan_dir": str(wan_dir()),
        "nvfp4_sampling_steps": nvfp4_sampling_steps(),
        "hf_token": bool(_hf_token()),
    }


def longlive_status() -> dict[str, Any]:
    missing = missing_longlive_requirements()
    return {
        "backend": "longlive",
        "root": str(longlive_root()),
        "generator_ckpt": str(generator_ckpt()),
        "wan_dir": str(wan_dir()),
        "ready": not missing,
        "missing": missing,
        "hint": SETUP_HINT,
    }


def _latent_hw(width: int, height: int) -> tuple[int, int]:
    """Wan 2.2 TI2V-5B spatial compression is 16×."""
    return max(1, height // 16), max(1, width // 16)


def write_inference_yaml(
    dest: Path,
    *,
    data_path: Path,
    output_folder: Path,
    seconds: float,
    seed: int,
    i2v: bool,
) -> Path:
    width, height = longlive_spatial_size()
    latent_h, latent_w = _latent_hw(width, height)
    frames = longlive_num_output_frames(seconds)
    ckpt = _assert_nvfp4_generator()
    lora = lora_ckpt()
    steps = nvfp4_sampling_steps()
    lines = [
        "model_kwargs:",
        f"  model_name: {WAN_ARCH_NAME}",
        "  timestep_shift: 5.0",
        f"  num_frame_per_block: {NUM_FRAME_PER_BLOCK}",
        "  local_attn_size: 32",
        "  sink_size: 8",
        "use_ema: false",
        f"output_folder: {output_folder}",
        "num_samples: 1",
        "save_latents_only: false",
        "save_with_index: true",
        f"num_output_frames: {frames}",
        "merge_lora: false",
        f"data_path: {data_path}",
        "data:",
        f"  data_path: {data_path}",
        "  image_or_video_shape:",
        "  - 1",
        f"  - {frames}",
        "  - 48",
        f"  - {latent_h}",
        f"  - {latent_w}",
        "inference:",
        f"  sampling_steps: {steps}",
        *(
            ["  independent_first_frame: true"]
            if i2v
            else []
        ),
        "  sink_size: 8",
        "  guidance_scale: 1.0",
        "  multi_shot_sink: true",
        "  multi_shot_rope_offset: 8",
        "  kv_quant: true",
        "  kv_quant_scale_rule: mse",
        "  kv_quant_backend: cuda",
        "  streaming_vae: false",
        "  async_vae: false",
        "  vae_type: wan",
        "checkpoints:",
        f"  generator_ckpt: {ckpt}",
        "model_quant: true",
        "model_quant_use_transformer_engine: false",
        "model_quant_te_inference_only: true",
        "model_quant_te_low_precision_weights: true",
        "model_quant_te_fallback_to_fouroversix: false",
        "model_quant_scale_rule: mse",
        "model_quant_activation_scale_rule: mse",
        "model_quant_weight_scale_rule: mse",
        "model_quant_gradient_scale_rule: mse",
        "torch_compile: false",
        "i2v: " + ("true" if i2v else "false"),
        "algorithm:",
        "  i2v: " + ("true" if i2v else "false"),
        "  independent_first_frame: " + ("true" if i2v else "false"),
        "logging:",
        f"  seed: {int(seed)}",
    ]
    if lora is not None and lora.is_file():
        ckpt_idx = lines.index("checkpoints:")
        lines.insert(ckpt_idx + 2, f"  lora_ckpt: {lora}")
        lines.extend(
            [
                "adapter:",
                "  type: lora",
                "  rank: 128",
                "  alpha: 128",
                "  dropout: 0.0",
                "  dtype: bfloat16",
            ]
        )
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def write_prompt_job(
    workdir: Path,
    segment: dict,
    *,
    i2v: bool,
) -> Path:
    """Official inference data_path: prompts.txt (T2V) or images/+prompts/ (I2V).

    Do not write ``video/``. That selects MultiVideoConcatDataset, which samples
    ``first_chunk_frames`` (29 RGB frames on Wan 5B) from source mp4s and raises
    ``ValueError: no video can provide the first chunk`` for a still / 1-frame clip.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    leftover_video = workdir / "video"
    if leftover_video.exists():
        shutil.rmtree(leftover_video, ignore_errors=True)
    caption = segment_prompt(segment).strip()
    if not i2v:
        prompts = workdir / "prompts.txt"
        prompts.write_text(caption + "\n", encoding="utf-8")
        return prompts
    first = Path(str(segment.get("first_frame_path") or ""))
    if not first.is_file() or first.stat().st_size < MIN_I2V_STILL_BYTES:
        raise RuntimeError(f"longlive i2v {segment.get('id') or '?'} missing first_frame_path")
    images = workdir / "images"
    prompts_dir = workdir / "prompts"
    images.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    dest = images / "0.png"
    shutil.copy2(first, dest)
    if dest.stat().st_size < MIN_I2V_STILL_BYTES:
        raise RuntimeError(f"longlive i2v {segment.get('id') or '?'} staged empty first frame")
    text = caption or str(segment.get("id") or "shot")
    (prompts_dir / "0.txt").write_text(text + "\n", encoding="utf-8")
    assert_i2v_image_layout(workdir)
    return workdir


def _find_output_mp4(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    videos = sorted(folder.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return videos[0] if videos else None


def submit_longlive(
    segment: dict,
    dest: Path,
    *,
    root: Path | None = None,
    progress: ProgressCallback | None = None,
    run: bool = True,
) -> dict[str, Any]:
    """Run official inference.py. Raises if weights/repo are missing — never fakes dest."""
    _ = root  # callers stage first_frame onto the segment before submit
    missing = missing_longlive_requirements()
    if missing:
        raise RuntimeError(f"{FAIL_CLOSED}: longlive not ready: " + "; ".join(missing) + ". " + SETUP_HINT)
    _assert_nvfp4_generator()
    _ensure_wan_layout()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    workdir = dest.parent / f".longlive-{segment.get('id') or 'shot'}"
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    mode = select_longlive_mode(segment)
    data_path = write_prompt_job(workdir / "data", segment, i2v=mode == "i2v")
    out_dir = workdir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = write_inference_yaml(
        workdir / "inference.yaml",
        data_path=data_path,
        output_folder=out_dir,
        seconds=float(segment.get("duration") or 8.0),
        seed=int(segment.get("seed") or 11),
        i2v=mode == "i2v",
    )
    if not run:
        return {"shot": segment.get("id"), "mode": mode, "yaml": str(yaml_path), "skipped": "dry_run"}
    if progress:
        progress(f"anim:{segment.get('id')}")
    timeout = float(os.environ.get("LONGLIVE_SHOT_TIMEOUT_S") or 2400)
    cmd = [python_bin(), str(longlive_root() / "inference.py"), "--config_path", str(yaml_path)]
    from gpu_worker.stack import stop_comfy_for_longlive

    stop_comfy_for_longlive()
    _install_torch_mmap_sitecustomize()
    print(
        {
            "longlive_cmd": cmd,
            "cwd": str(longlive_root()),
            "mode": mode,
            "seconds": float(segment.get("duration") or 8.0),
            "frames": longlive_num_output_frames(float(segment.get("duration") or 8.0)),
        },
        flush=True,
    )
    try:
        # Stream inference.py so the card logs prove this is LongLive, not H3.
        subprocess.run(
            cmd,
            check=True,
            cwd=str(longlive_root()),
            timeout=timeout,
            env=_longlive_inference_env(),
        )
    except subprocess.CalledProcessError as exc:
        if is_longlive_fatal_error(exc) or exc.returncode in (-9, 9, 137):
            raise RuntimeError(
                f"{FAIL_CLOSED}: inference SIGKILL/exit {exc.returncode} for {segment.get('id')} "
                f"(T5/VAE/generator exploded the cgroup, or fouroversix missing). {SETUP_HINT}"
            ) from exc
        raise RuntimeError(
            f"{FAIL_CLOSED}: longlive inference failed {segment.get('id')}: exit {exc.returncode}. {SETUP_HINT}"
        ) from exc
    produced = _find_output_mp4(out_dir)
    if produced is None or produced.stat().st_size < 32:
        raise RuntimeError(f"longlive produced no video for {segment.get('id')}. {SETUP_HINT}")
    shutil.copy2(produced, dest)
    last_rel = dest.with_name("last.png")
    try:
        extract_last_frame(dest, last_rel)
        segment["last_frame_path"] = str(last_rel)
    except Exception as exc:  # noqa: BLE001 — chain can retry extract
        segment["last_frame_error"] = str(exc)[:200]
    return {
        "shot": segment.get("id"),
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "mode": mode,
        "backend": "longlive",
        "last_frame": segment.get("last_frame_path"),
    }

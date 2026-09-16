"""Pull H3 / anime SDXL stills / Comfy models from HuggingFace onto instance-local disk.

Weights are **never** stored on R2. After the Vast box is destroyed, the files
are gone and the next machine pulls HuggingFace again. That is intentional.

Runtime files live under
ComfyUI/models/{diffusion_models,text_encoders,vae,checkpoints,ipadapter,clip_vision,visual_qc}.
`extra_model_paths.yaml` points Comfy at that local tree only.

Stills (Animagine XL 4.0 + IP-Adapter plus SDXL + ViT-H CLIP vision + OpenCLIP
ViT-B-32/openai for visual QC, ~9.6GB) and H3 (NVFP4 DiT + TE + VAE) are separate
downloads. Boot starts H3 in the background and only blocks Comfy / design/keyframe
on still weights. Anim must join H3 and confirm files first.

Visual QC CLIP (``models/visual_qc/``) is **not** the IP-Adapter ViT-H encoder under
``models/clip_vision/``. ``visual_qc.py`` loads offline from ``AF_VISUAL_QC_CLIP_*``
env vars; it must never silently download weights at runtime.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from pathlib import Path
from typing import Callable

from anime_factory.models import IMAGE_CKPT, IMAGE_MODEL

# HuggingFace repos / files Comfy 生图 + H3 need. First boot may download.
# Comfy-Org/MiniMax-H3 ships pruned DiT as int8_convrot / fp8_scaled / bf16 only
# (~21GB INT8). Blackwell NVFP4 (~12.5GB) is community: lilcheaty/MiniMax-H3-NVFP4.
# TE + video/audio VAE stay on Comfy-Org.
# Stills: cagliostrolab/animagine-xl-4.0 single-file SDXL (~6.9GB, ungated) plus the
# IP-Adapter plus SDXL weights and the ViT-H image encoder it needs for identity.
H3_ORG_REPO = "Comfy-Org/MiniMax-H3"
H3_DIT_REPO = "lilcheaty/MiniMax-H3-NVFP4"
STILL_CKPT_REPO = IMAGE_MODEL
STILL_CKPT_FILE = IMAGE_CKPT
STILL_CKPT_DEST = f"models/checkpoints/{STILL_CKPT_FILE}"
IPADAPTER_REPO = "h94/IP-Adapter"
IPADAPTER_FILE = "ip-adapter-plus_sdxl_vit-h.safetensors"
IPADAPTER_DEST = f"models/ipadapter/{IPADAPTER_FILE}"
# ip-adapter-plus_sdxl_vit-h wants the ViT-H encoder (models/), not the bigG one
# under sdxl_models/. Comfy loads it by filename from models/clip_vision/.
CLIP_VISION_FILE = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
CLIP_VISION_DEST = f"models/clip_vision/{CLIP_VISION_FILE}"

# Visual QC: OpenCLIP ViT-B-32/openai (~605MB). Separate from IP-Adapter ViT-H above.
OPEN_CLIP_TORCH_VERSION = "2.32.0"
VISUAL_QC_CLIP_REPO = "timm/vit_base_patch32_clip_224.openai"
VISUAL_QC_CLIP_WEIGHT_HF = "open_clip_model.safetensors"
VISUAL_QC_CLIP_CONFIG_HF = "open_clip_config.json"
VISUAL_QC_CLIP_DIR = "models/visual_qc"
VISUAL_QC_CLIP_WEIGHT_DEST = f"{VISUAL_QC_CLIP_DIR}/{VISUAL_QC_CLIP_WEIGHT_HF}"
VISUAL_QC_CLIP_CONFIG_DEST = f"{VISUAL_QC_CLIP_DIR}/{VISUAL_QC_CLIP_CONFIG_HF}"
VISUAL_QC_CLIP_MODEL_NAME = "ViT-B-32"
VISUAL_QC_CLIP_PRETRAINED_TAG = "openai"
# HF LFS size 605_143_284; reject truncated Hub pulls without downloading in tests.
VISUAL_QC_CLIP_MIN_BYTES = 600_000_000
VISUAL_QC_CLIP_CONFIG_MIN_BYTES = 32

H3_CORE_FILES: list[dict[str, str]] = [
    {
        "repo": H3_ORG_REPO,
        "hf": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "dest": "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    },
    {
        "repo": H3_ORG_REPO,
        "hf": "vae/minimax_h3_video_vae_fp16.safetensors",
        "dest": "models/vae/minimax_h3_video_vae_fp16.safetensors",
    },
]
H3_FL2VA_FILES: list[dict[str, str]] = [
    {
        "repo": H3_DIT_REPO,
        "hf": "minimax_h3_fl2va_pruned_nvfp4.safetensors",
        "dest": "models/diffusion_models/minimax_h3_fl2va_pruned_nvfp4.safetensors",
    },
]
H3_REF2VA_FILES: list[dict[str, str]] = [
    {
        "repo": H3_DIT_REPO,
        "hf": "minimax_h3_ref2va_pruned_nvfp4.safetensors",
        "dest": "models/diffusion_models/minimax_h3_ref2va_pruned_nvfp4.safetensors",
    },
]
# Audio VAE is never decoded (CosyVoice2 is the mix). Do not download it.
H3_FILES: list[dict[str, str]] = [*H3_CORE_FILES, *H3_FL2VA_FILES, *H3_REF2VA_FILES]
STILL_FILES: list[dict[str, str]] = [
    {
        "repo": STILL_CKPT_REPO,
        "hf": STILL_CKPT_FILE,
        "dest": STILL_CKPT_DEST,
    },
    {
        "repo": IPADAPTER_REPO,
        "hf": f"sdxl_models/{IPADAPTER_FILE}",
        "dest": IPADAPTER_DEST,
    },
    {
        "repo": IPADAPTER_REPO,
        "hf": "models/image_encoder/model.safetensors",
        "dest": CLIP_VISION_DEST,
    },
    {
        "repo": VISUAL_QC_CLIP_REPO,
        "hf": VISUAL_QC_CLIP_WEIGHT_HF,
        "dest": VISUAL_QC_CLIP_WEIGHT_DEST,
        "min_bytes": str(VISUAL_QC_CLIP_MIN_BYTES),
    },
    {
        "repo": VISUAL_QC_CLIP_REPO,
        "hf": VISUAL_QC_CLIP_CONFIG_HF,
        "dest": VISUAL_QC_CLIP_CONFIG_DEST,
        "min_bytes": str(VISUAL_QC_CLIP_CONFIG_MIN_BYTES),
    },
]
RUNTIME_WEIGHT_FILES: list[dict[str, str]] = [*STILL_FILES, *H3_FILES]
# Historical alias used by tests; stills are anime SDXL, not a Kolors snapshot.
H3_AND_KOLORS_FILES = RUNTIME_WEIGHT_FILES

_h3_lock = threading.Lock()
_h3_thread: threading.Thread | None = None
_h3_error: BaseException | None = None
_h3_done = threading.Event()


def reset_h3_weight_job() -> None:
    """Tests only: drop background-job bookkeeping."""
    global _h3_thread, _h3_error
    with _h3_lock:
        _h3_thread = None
        _h3_error = None
        _h3_done.set()
        _h3_done.clear()


def hf_token() -> str | None:
    """Hub token from the instance env only. Never used to store weights on R2."""
    raw = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return raw or None


def hf_local_dir_for(dest: Path) -> str:
    """local_dir for hf_hub_download so repo paths land on dest, not dest.parent/<subdir>."""
    models_root = dest
    while models_root.name != "models" and models_root.parent != models_root:
        models_root = models_root.parent
    return str(models_root if models_root.name == "models" else dest.parent)


def hf_download_local_dir(filename: str, dest: Path) -> str:
    """Hub local_dir for a repo-relative name so the blob lands on dest without a second copy."""
    rel = filename.replace("\\", "/").lstrip("/")
    if "/" in rel:
        return hf_local_dir_for(dest)
    return str(dest.parent)


def extra_model_paths_yaml(comfy_dir: Path) -> str:
    """Comfy extra_model_paths pointing at this instance's disk only."""
    root = comfy_dir.resolve().as_posix().rstrip("/")
    return (
        "# Instance-local Comfy models. Do not point this at R2.\n"
        "anime-factory:\n"
        f"    base_path: {root}/\n"
        "    checkpoints: models/checkpoints/\n"
        "    clip: models/text_encoders/\n"
        "    clip_vision: models/clip_vision/\n"
        "    diffusion_models: models/diffusion_models/\n"
        "    ipadapter: models/ipadapter/\n"
        "    text_encoders: models/text_encoders/\n"
        "    unet: models/diffusion_models/\n"
        "    vae: models/vae/\n"
    )


def write_extra_model_paths(comfy_dir: Path) -> Path:
    dest = comfy_dir / "extra_model_paths.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(extra_model_paths_yaml(comfy_dir), encoding="utf-8")
    return dest


def _hub_kwargs() -> dict:
    token = hf_token()
    return {"token": token} if token else {}


def _hf_file(repo: str, filename: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    # Repo-relative names (diffusion_models/foo.safetensors) must land on dest.
    # local_dir=dest.parent nested a second copy and doubled disk on the 160GB card.
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=hf_download_local_dir(filename, dest),
        **_hub_kwargs(),
    )
    src = Path(path)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
        try:
            src.unlink()
        except OSError:
            pass


def _file_meets_contract(dest: Path, item: dict[str, str] | None = None) -> bool:
    if not dest.is_file():
        return False
    size = dest.stat().st_size
    if size <= 0:
        return False
    if item:
        min_bytes = item.get("min_bytes")
        if min_bytes and size < int(min_bytes):
            return False
        digest = (item.get("sha256") or "").strip().lower()
        if digest:
            got = hashlib.sha256(dest.read_bytes()).hexdigest()
            if got != digest:
                return False
    return True


def _item_present(dest: Path, item: dict[str, str] | None = None) -> bool:
    if dest.is_file():
        return _file_meets_contract(dest, item)
    return dest.is_dir() and any(dest.rglob("*"))


def visual_qc_clip_paths(comfy_dir: Path | str | None = None) -> dict[str, Path | str]:
    """Local OpenCLIP ViT-B-32/openai paths for ``visual_qc.py`` offline loading."""
    root = Path(comfy_dir or os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI")
    weight = Path(
        os.environ.get("AF_VISUAL_QC_CLIP_WEIGHTS") or root / VISUAL_QC_CLIP_WEIGHT_DEST
    )
    config = Path(
        os.environ.get("AF_VISUAL_QC_CLIP_CONFIG") or root / VISUAL_QC_CLIP_CONFIG_DEST
    )
    clip_dir = Path(os.environ.get("AF_VISUAL_QC_CLIP_DIR") or weight.parent)
    return {
        "dir": clip_dir,
        "weights": weight,
        "config": config,
        "model": VISUAL_QC_CLIP_MODEL_NAME,
        "pretrained": VISUAL_QC_CLIP_PRETRAINED_TAG,
    }


def visual_qc_clip_env(comfy_dir: Path | str | None = None) -> dict[str, str]:
    """Env map for subprocesses / ``visual_qc.py`` — no Hub URLs, local files only."""
    paths = visual_qc_clip_paths(comfy_dir)
    return {
        "AF_VISUAL_QC_CLIP_DIR": str(paths["dir"]),
        "AF_VISUAL_QC_CLIP_WEIGHTS": str(paths["weights"]),
        "AF_VISUAL_QC_CLIP_CONFIG": str(paths["config"]),
        "AF_OPEN_CLIP_MODEL": str(paths["model"]),
        "AF_OPEN_CLIP_PRETRAINED": str(paths["pretrained"]),
        "OPEN_CLIP_TORCH_VERSION": OPEN_CLIP_TORCH_VERSION,
    }


def validate_visual_qc_clip_weights(comfy_dir: Path | str | None = None) -> None:
    """Fail closed when visual QC CLIP files are missing or truncated."""
    paths = visual_qc_clip_paths(comfy_dir)
    weight = Path(paths["weights"])
    config = Path(paths["config"])
    weight_item = next(
        (item for item in STILL_FILES if item["dest"] == VISUAL_QC_CLIP_WEIGHT_DEST),
        None,
    )
    config_item = next(
        (item for item in STILL_FILES if item["dest"] == VISUAL_QC_CLIP_CONFIG_DEST),
        None,
    )
    if not _file_meets_contract(weight, weight_item):
        raise RuntimeError(
            f"visual QC CLIP weights missing or truncated: {weight} "
            f"(need >={VISUAL_QC_CLIP_MIN_BYTES} bytes, not IP-Adapter ViT-H)"
        )
    if not _file_meets_contract(config, config_item):
        raise RuntimeError(f"visual QC CLIP config missing: {config}")


def _skip_weights() -> bool:
    return os.environ.get("AF_SKIP_WEIGHTS", "").strip().lower() in {"1", "true", "yes", "on"}


def _empty_materialize(root: Path, yaml_path: Path) -> dict:
    return {
        "skipped": True,
        "hits": [],
        "misses": [],
        "downloaded": [],
        "source": "huggingface",
        "root": str(root),
        "extra_model_paths": str(yaml_path),
    }


def ensure_weights(
    names: list[str],
    dest_dir: Path | str,
    hf_download: Callable[[str], bytes],
) -> dict:
    """Place named blobs on local disk via HuggingFace. Never R2."""
    root = Path(dest_dir)
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    hits: list[str] = []
    misses: list[str] = []
    for name in names:
        dest = root / name
        if dest.is_file() and dest.stat().st_size > 0:
            hits.append(name)
            files[name] = {"bytes": dest.read_bytes(), "source": "local", "path": str(dest)}
            continue
        misses.append(name)
        blob = hf_download(name)
        dest.write_bytes(blob)
        files[name] = {"bytes": blob, "source": "huggingface", "path": str(dest)}
    return {"files": files, "hits": hits, "misses": misses, "source": "huggingface"}


def runtime_weight_bytes(comfy_dir: Path | str | None = None) -> int:
    """Count completed and in-progress local model bytes for the selected backend."""
    backend = ""
    try:
        from anime_factory.video_backend import locked_video_backend, select_video_backend

        backend = locked_video_backend() or select_video_backend()
    except Exception:  # noqa: BLE001 — accounting must not crash boot
        backend = (os.environ.get("AF_VIDEO_BACKEND") or "h3").strip().lower()
    total = 0
    if backend != "longlive":
        root = Path(comfy_dir or os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI") / "models"
        total += _dir_bytes(root)
        return total
    still_root = Path(comfy_dir or os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI") / "models"
    for rel in (
        "checkpoints",
        "ipadapter",
        "clip_vision",
        "visual_qc",
    ):
        total += _dir_bytes(still_root / rel)
    try:
        from gpu_worker.longlive import generator_ckpt, longlive_root, wan_dir

        for path in (generator_ckpt(), wan_dir(), longlive_root() / "checkpoints"):
            total += _dir_bytes(path if path.is_dir() else path.parent)
    except Exception:  # noqa: BLE001
        pass
    return total


def _dir_bytes(root: Path) -> int:
    total = 0
    if root.is_file():
        try:
            return root.stat().st_size
        except OSError:
            return 0
    if not root.is_dir():
        return 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                if "minimax_h3" in path.name and "longlive" in str(root).lower():
                    continue
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _materialize_items(
    items: list[dict[str, str]],
    root: Path,
    progress: Callable[[str], None] | None = None,
) -> dict:
    hits: list[str] = []
    misses: list[str] = []
    downloaded: list[str] = []
    for item in items:
        dest = root / item["dest"]
        label = item["dest"]
        if progress:
            progress(f"startup_stage:weights:{label}")
            progress(f"startup_bytes:{runtime_weight_bytes(root)}")
        if _item_present(dest, item):
            hits.append(label)
            continue
        misses.append(label)
        _hf_file(item["repo"], item["hf"], dest)
        downloaded.append(label)
        if progress:
            progress(f"startup_bytes:{runtime_weight_bytes(root)}")
    return {
        "hits": hits,
        "misses": misses,
        "downloaded": downloaded,
        "source": "huggingface",
        "root": str(root),
        "hf_token": bool(hf_token()),
    }


def ensure_still_weights(
    comfy_dir: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Anime SDXL checkpoint + IP-Adapter only. Comfy can start and draw stills after this."""
    root = Path(comfy_dir or os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI")
    yaml_path = write_extra_model_paths(root)
    if _skip_weights():
        return _empty_materialize(root, yaml_path)
    out = _materialize_items(STILL_FILES, root, progress=progress)
    out["extra_model_paths"] = str(yaml_path)
    out["kind"] = "stills"
    return out


def _refuse_h3_on_longlive() -> None:
    try:
        from anime_factory.video_backend import locked_video_backend, select_video_backend

        backend = locked_video_backend() or select_video_backend()
    except Exception:  # noqa: BLE001
        backend = (os.environ.get("AF_VIDEO_BACKEND") or "").strip().lower()
    cap = (os.environ.get("AF_IMAGE_CAPABILITY") or "").strip().lower()
    if backend == "longlive" or cap == "longlive":
        raise RuntimeError(
            "capability_mismatch:video_backend: refuse H3 weight download on LongLive image/path"
        )


def h3_files_for_modes(modes: set[str] | None = None) -> list[dict[str, str]]:
    """Core TE+video VAE plus only the DiTs the board actually needs.

    ``None`` keeps the sequential full pull (tests / one-shot). An empty set or
    ``{"core"}`` is boot: TE + video VAE only. Audio VAE is never included.
    """
    files = list(H3_CORE_FILES)
    if modes is None:
        files.extend(H3_FL2VA_FILES)
        files.extend(H3_REF2VA_FILES)
        return files
    wanted = {str(m or "").strip().lower() for m in modes if str(m or "").strip()}
    if not wanted or wanted == {"core"}:
        return files
    if any(m.startswith("fl2va") for m in wanted):
        files.extend(H3_FL2VA_FILES)
    if "ref2va" in wanted:
        files.extend(H3_REF2VA_FILES)
    return files


def ensure_h3_weights(
    comfy_dir: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
    modes: set[str] | None = None,
) -> dict:
    """H3 NVFP4 DiT + TE + video VAE. Must finish before anim sampling. Never audio VAE."""
    _refuse_h3_on_longlive()
    root = Path(comfy_dir or os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI")
    yaml_path = write_extra_model_paths(root)
    if _skip_weights():
        return _empty_materialize(root, yaml_path)
    items = h3_files_for_modes(modes)
    out = _materialize_items(items, root, progress=progress)
    out["extra_model_paths"] = str(yaml_path)
    out["kind"] = "h3"
    out["modes"] = sorted(modes) if modes else ["core", "fl2va", "ref2va"]
    return out


def missing_h3_weight_labels(
    comfy_dir: Path | str | None = None,
    modes: set[str] | None = None,
) -> list[str]:
    root = Path(comfy_dir or os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI")
    return [item["dest"] for item in h3_files_for_modes(modes) if not _item_present(root / item["dest"], item)]


def start_h3_weights_background(
    comfy_dir: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
    modes: set[str] | None = None,
) -> threading.Thread | None:
    """Pull H3 core (and optional DiTs) while stills run. Anim must join this first."""
    global _h3_thread, _h3_error
    _refuse_h3_on_longlive()
    root = Path(comfy_dir or os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI")
    with _h3_lock:
        if _h3_thread is not None and _h3_thread.is_alive():
            return _h3_thread
        if _h3_done.is_set() and _h3_error is None and not missing_h3_weight_labels(root, modes={"core"} if modes is None else modes):
            return _h3_thread
        _h3_error = None
        _h3_done.clear()

        def run() -> None:
            global _h3_error
            try:
                # Boot downloads TE + video VAE only; DiTs wait for the board.
                ensure_h3_weights(root, progress=progress, modes=modes if modes is not None else set())
            except BaseException as exc:  # noqa: BLE001 — join_h3_weights re-raises
                _h3_error = exc
            finally:
                _h3_done.set()

        _h3_thread = threading.Thread(target=run, name="h3-weights", daemon=False)
        _h3_thread.start()
        return _h3_thread


def join_h3_weights(
    comfy_dir: Path | str | None = None,
    timeout_s: float | None = None,
    modes: set[str] | None = None,
) -> dict:
    """Block until H3 core files are on disk. Call this before H3 sampling, not before stills."""
    _refuse_h3_on_longlive()
    root = Path(comfy_dir or os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI")
    thread: threading.Thread | None
    with _h3_lock:
        thread = _h3_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            raise TimeoutError("h3 weights download still running")
    if _h3_error is not None:
        raise RuntimeError(f"h3 weights download failed: {_h3_error}") from _h3_error
    if _skip_weights():
        return {"ok": True, "skipped": True, "missing": []}
    needed = modes if modes is not None else set()
    missing = missing_h3_weight_labels(root, modes=needed)
    if missing:
        # No background job (tests / AF_SKIP) — pull now so anim still has files.
        ensure_h3_weights(root, modes=needed)
        missing = missing_h3_weight_labels(root, modes=needed)
    if missing:
        raise RuntimeError(f"h3 weights missing after join: {missing}")
    return {"ok": True, "skipped": False, "missing": []}


def ensure_h3_dits_for_shots(
    shots: list[dict],
    comfy_dir: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Download only FL2VA / Ref2VA DiTs the board will actually sample."""
    from gpu_worker.h3 import select_mode

    modes = {select_mode(shot) for shot in shots or []}
    return ensure_h3_weights(comfy_dir, progress=progress, modes=modes)


def materialize_runtime_weights(
    comfy_dir: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Place still weights then H3 under ComfyUI from HuggingFace onto local disk.

    Prefer ensure_still_weights + start_h3_weights_background at boot so stills
    do not wait for H3. This helper stays sequential for tests and one-shot pulls.
    Never restore from object storage and never upload Hub files back to it.
    """
    root = Path(comfy_dir or os.environ.get("COMFYUI_DIR") or "/opt/ComfyUI")
    yaml_path = write_extra_model_paths(root)
    still = ensure_still_weights(root, progress=progress)
    h3 = ensure_h3_weights(root, progress=progress)
    return {
        "hits": list(still.get("hits") or []) + list(h3.get("hits") or []),
        "misses": list(still.get("misses") or []) + list(h3.get("misses") or []),
        "downloaded": list(still.get("downloaded") or []) + list(h3.get("downloaded") or []),
        "source": "huggingface",
        "root": str(root),
        "extra_model_paths": str(yaml_path),
        "hf_token": bool(hf_token()),
        "skipped": bool(still.get("skipped") or h3.get("skipped")),
        "stills": still,
        "h3": h3,
    }

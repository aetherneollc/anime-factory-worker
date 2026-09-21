"""H3 workflow helpers: Hailuo ref2va pack + last-frame continue. Picture only — no H3 speech."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from anime_factory.models import (
    H3_GEN_HEIGHT,
    H3_GEN_WIDTH,
    H3_MAX_REFS,
    H3_OOM_FALLBACK_HEIGHT,
    H3_OOM_FALLBACK_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / "workflows"

DEFAULT_H3_PROMPT = (
    "original cinematic animation for this story, named character and scene refs already bound, "
    "camera and acting motion must change this shot, 24fps, no text, no watermark, no spoken dialogue."
)
CONTINUE_PROMPT = (
    "延续上一镜头, continue from the previous last frame, "
    "same identity, same costume, same set"
)
MOTION_KEEP_ALIVE = (
    "camera and subject motion are the changing terms this shot; "
    "do not freeze a still photograph; keep face and costume of the named character refs; "
    "no face drift, no costume swap, 禁止五官漂移/换装; "
    "do not generate spoken dialogue"
)
FL2VA_UNET = "minimax_h3_fl2va_pruned_nvfp4.safetensors"
REF2VA_UNET = "minimax_h3_ref2va_pruned_nvfp4.safetensors"
H3_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
# Downloaded on the box but never decoded into the mix. Locked CosyVoice2 is episode dialogue.
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
FORBIDDEN_H3_SPEECH_NODES = ("VAEDecodeAudio",)
H3_NATIVE_WIDTH = H3_GEN_WIDTH
H3_NATIVE_HEIGHT = H3_GEN_HEIGHT
H3_LOW_VRAM_WIDTH = H3_OOM_FALLBACK_WIDTH
H3_LOW_VRAM_HEIGHT = H3_OOM_FALLBACK_HEIGHT
H3_NATIVE_VRAM_MB = 32_000
H3_LOW_VRAM_HEAD_CHUNKS = 16
H3_LOW_VRAM_FF_CHUNKS = 8
H3_LOW_VRAM_SEQ_THRESHOLD = 4096

_H3_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "cuda out of memory",
    "cudamalloc",
    "cuda error",
    "allocator",
    "allocat",
    "oom",
    "sigkill",
    "killed",
    "ran out of memory",
    "insufficient memory",
    "nvml error",
    "execution_error",
    "torch.cuda.outofmemoryerror",
)


def load_workflow(name: str) -> dict:
    path = WORKFLOWS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def is_directory_like(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    if text == "" or text.endswith("/") or text in {".", "./", "input", "input/", "input\\"}:
        return True
    return False


def is_h3_oom(error: BaseException | str | None) -> bool:
    """Structured OOM classification for Comfy execution, CUDA allocator, and host kills."""
    if error is None:
        return False
    text = str(error).lower().replace("-", " ").replace("_", " ")
    if any(marker.replace("_", " ") in text for marker in _H3_OOM_MARKERS):
        return True
    if re.search(r"\boom\b", text):
        return True
    return False


def prune_empty_last_frame(workflow: dict) -> dict:
    """Drop LoadImage nodes whose image is empty / a directory (IsADirectoryError)."""
    graph = copy.deepcopy(workflow)
    prompt = graph.get("prompt", graph)
    drop_ids = []
    for nid, node in list(prompt.items()):
        if not isinstance(node, dict):
            continue
        ctype = node.get("class_type") or ""
        inputs = node.get("inputs") or {}
        role = (node.get("_meta") or {}).get("role") or inputs.get("role")
        image_val = inputs.get("image")
        if ctype in {"LoadImage", "LoadImageOutput"} and (
            role in {"last_frame", "last-frame"} or "last_frame" in nid.lower()
        ):
            if is_directory_like(image_val):
                drop_ids.append(nid)
                continue
        if ctype in {"LoadImage", "LoadImageOutput"} and is_directory_like(image_val):
            drop_ids.append(nid)
    for nid in drop_ids:
        prompt.pop(nid, None)
    # unlink references
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for k, v in list(inputs.items()):
            if isinstance(v, list) and v and v[0] in drop_ids:
                inputs.pop(k, None)
            if k in {"last_frame", "last_image"} and is_directory_like(v):
                inputs.pop(k, None)
    return graph if "prompt" in graph else prompt


def _has_ref_pack(segment: dict) -> bool:
    """Character sheet pack only. A location plate alone is an empty fl2va establishing still."""
    return _has_character_identity(segment)


def _has_character_identity(segment: dict) -> bool:
    refs = segment.get("refs") or []
    if any("char" in str(r) or "sheet" in str(r) or "costume" in str(r) for r in refs):
        return True
    cid = str(segment.get("character_id") or "").strip()
    if cid and segment.get("on_camera") is not False:
        return True
    return False


def select_mode(segment: dict) -> str:
    """Hailuo ref pack wins. Chain tails stay ref2va with last_frame as a guide, not I2V-only."""
    last = segment.get("last_frame_path")
    mode = str(segment.get("h3_mode") or "").strip()
    chain_index = int(segment.get("chain_index") or 0)
    if _has_ref_pack(segment):
        return "ref2va"
    if chain_index > 0 or segment.get("match_cut"):
        if last and not is_directory_like(last):
            return "fl2va_first_last"
        return "fl2va_first_last" if mode == "fl2va_first_last" else "fl2va_first"
    if mode in {"ref2va", "fl2va_first", "fl2va_first_last"}:
        return mode
    if last and not is_directory_like(last):
        return "fl2va_first_last"
    return "fl2va_first"


def image_filename(value: Any) -> str | None:
    """Basename of a real image path. Never invent f1.png for a missing first frame."""
    if value is None or is_directory_like(value):
        return None
    name = Path(str(value)).name.strip()
    return name or None


def ref_image_filenames(segment: dict) -> list[str]:
    """Basenames for LoadImage nodes. Sheets/plates as *.png, never extensionless tokens."""
    chain_tail = int(segment.get("chain_index") or 0) > 0 or bool(segment.get("chain_source_last_frame"))
    chain_first = image_filename(segment.get("first_frame_path")) if chain_tail else None
    names: list[str] = []
    try:
        from anime_factory.h3_storyboard import ref_bind_names

        ordered = list(ref_bind_names(segment))
    except Exception:  # noqa: BLE001
        ordered = list(segment.get("refs") or [])
    for raw in ordered[:H3_MAX_REFS]:
        name = image_filename(raw) or str(raw or "").strip()
        if not name:
            continue
        if "." not in name:
            name = f"{name}.png"
        if chain_tail and chain_first and name == chain_first:
            continue
        if name not in names:
            names.append(name)
    return names[:H3_MAX_REFS]


def ref_autogrow_input_key(index: int) -> str:
    """Comfy 0.34 MiniMaxH3ReferenceToVideo Autogrow slot.

    execute() is ``execute(..., ref_images=None, ...)`` and iterates
    ``(ref_images or {}).values()``. API prompts must use dotted keys
    ``ref_images.ref_image_0`` (0-based). A top-level ``ref_image_1`` is an
    unexpected kwarg. A bare ``ref_images`` IMAGE link is a tensor, so
    ``.values()`` / ``or {}`` crashes.
    """
    return f"ref_images.ref_image_{int(index)}"


def extract_last_frame(video_path: str | Path, dest: str | Path) -> Path:
    """ffmpeg last frame of a rendered segment — used as the next chain first_frame."""
    video = Path(video_path)
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmds = (
        [
            "ffmpeg",
            "-y",
            "-sseof",
            "-0.05",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out),
        ],
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-update",
            "1",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out),
        ],
    )
    last_exc: BaseException | None = None
    for cmd in cmds:
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            last_exc = None
            break
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_exc = exc
    if last_exc is not None:
        raise RuntimeError(f"last-frame extract failed: {video}") from last_exc
    if not out.is_file() or out.stat().st_size < 32:
        raise RuntimeError(f"last-frame extract produced no image: {video}")
    return out


def apply_chain_first_frame(segment: dict, last_frame_path: str | Path) -> dict:
    """Index >0 first_frame is the previous last_frame, mixed with the same character/scene refs."""
    nxt = dict(segment)
    path = str(last_frame_path)
    nxt["first_frame_path"] = path
    nxt["chain_source_last_frame"] = path
    nxt["status"] = "prepared"
    if _has_ref_pack(nxt):
        nxt["h3_mode"] = "ref2va"
    elif nxt.get("last_frame_path") and not is_directory_like(nxt.get("last_frame_path")):
        nxt["h3_mode"] = "fl2va_first_last"
    else:
        nxt["h3_mode"] = "fl2va_first"
    return nxt


def h3_length_frames(seconds: float) -> int:
    """Snap duration to MiniMax H3's 17k+5 frame grid at 24 fps (same as Comfy)."""
    frames = max(5, int(round(float(seconds) * 24)))
    while frames % 17 != 5:
        frames += 1
    return frames


def _camera_motion_clause(cam: str) -> str:
    text = str(cam or "").strip() or "Static Shot"
    if text.lower() in {"static shot", "static", "locked-off", "locked off"}:
        return "subtle body and face motion, speaking gesture, do not freeze as a still"
    return f"camera move: {text}"


def segment_prompt(segment: dict) -> str:
    """Compile picture prompt from cuts/manifest. Never pollute with dialogue `line`."""
    try:
        from anime_factory.h3_storyboard import compile_picture_prompt

        return compile_picture_prompt(segment)
    except Exception:  # noqa: BLE001 — fall back for minimal unit stubs
        prompt = ""
        for key in ("h3_prompt", "first_frame_prompt", "action", "prompt"):
            val = segment.get(key)
            if isinstance(val, dict):
                val = val.get("zh") or val.get("en") or val.get("ja")
            if val and str(val).strip():
                prompt = str(val).strip()
                break
        if not prompt:
            prompt = DEFAULT_H3_PROMPT
        if int(segment.get("chain_index") or 0) > 0 or segment.get("chain_source_last_frame"):
            if CONTINUE_PROMPT not in prompt:
                prompt = f"{CONTINUE_PROMPT}. {prompt}"
        cam = "Static Shot"
        cuts = segment.get("cuts") or []
        if cuts and isinstance(cuts[0], dict) and cuts[0].get("camera"):
            cam = str(cuts[0]["camera"])
        elif segment.get("camera"):
            cam = str(segment["camera"])
        return f"{prompt}, camera: {cam}, {_camera_motion_clause(cam)}, {MOTION_KEEP_ALIVE}"


def gpu_vram_mb() -> int | None:
    """Return actual visible VRAM in MiB; explicit GPU inventory env is a probe fallback."""
    raw_mb = (os.environ.get("GPU_VRAM_MB") or "").strip()
    raw_gb = (os.environ.get("GPU_VRAM_GB") or "").strip()
    try:
        if raw_mb:
            return max(0, int(float(raw_mb)))
        if raw_gb:
            return max(0, int(float(raw_gb) * 1024))
    except ValueError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            total = int(torch.cuda.get_device_properties(0).total_memory)
            if total > 0:
                return int(total / (1024 * 1024))
    except Exception:  # noqa: BLE001 — hardware probe only
        pass
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "replace")
        first = raw.strip().splitlines()[0] if raw.strip() else ""
        return int(float(first)) if first else None
    except Exception:  # noqa: BLE001 — hardware probe only
        return None


def _align_dim(value: int, max_value: int) -> int:
    aligned = max(32, min(max_value, int(value) // 32 * 32))
    return max(32, aligned)


def h3_resolution_profile(
    segment: dict,
    detected_vram_mb: int | None = None,
    *,
    oom_fallback: bool | None = None,
) -> dict[str, Any]:
    """Primary 1024×576 delivery-upscaled to 1280×720; OOM latch uses 864×480."""
    use_fallback = bool(
        oom_fallback
        if oom_fallback is not None
        else segment.get("h3_oom_fallback") or segment.get("h3_downgraded")
    )
    if use_fallback:
        gen_w = H3_OOM_FALLBACK_WIDTH
        gen_h = H3_OOM_FALLBACK_HEIGHT
        tier = "oom_fallback_480p"
        downgraded = True
        reason = str(segment.get("h3_downgrade_reason") or "oom_fallback")
    else:
        gen_w = H3_GEN_WIDTH
        gen_h = H3_GEN_HEIGHT
        tier = "gen_576p"
        downgraded = False
        reason = "primary_1024x576"
    vram_mb = gpu_vram_mb() if detected_vram_mb is None else detected_vram_mb
    width = _align_dim(int(segment.get("width") or gen_w), gen_w)
    height = _align_dim(int(segment.get("height") or gen_h), gen_h)
    if width > gen_w or height > gen_h:
        width, height = gen_w, gen_h
    return {
        "gen_width": width,
        "gen_height": height,
        "width": width,
        "height": height,
        "delivery_width": VIDEO_WIDTH,
        "delivery_height": VIDEO_HEIGHT,
        "head_chunks": H3_LOW_VRAM_HEAD_CHUNKS,
        "chunks": H3_LOW_VRAM_FF_CHUNKS,
        "seq_threshold": H3_LOW_VRAM_SEQ_THRESHOLD,
        "vram_mb": vram_mb,
        "tier": tier,
        "downgraded": downgraded,
        "reason": reason,
    }


def h3_spatial_size(segment: dict, *, oom_fallback: bool | None = None) -> tuple[int, int]:
    """MiniMax H3 dimensions aligned to its 32px latent tile grid."""
    profile = h3_resolution_profile(segment, oom_fallback=oom_fallback)
    return int(profile["width"]), int(profile["height"])


def _low_vram_model_chain(unet_node: str = "1") -> dict[str, Any]:
    return {
        "lv_attn": {
            "class_type": "MiniMaxLowVRAMAttention",
            "inputs": {"model": [unet_node, 0], "head_chunks": H3_LOW_VRAM_HEAD_CHUNKS},
        },
        "lv_ff": {
            "class_type": "MiniMaxChunkFeedForward",
            "inputs": {
                "model": ["lv_attn", 0],
                "chunks": H3_LOW_VRAM_FF_CHUNKS,
                "seq_threshold": H3_LOW_VRAM_SEQ_THRESHOLD,
            },
        },
    }


def native_h3_graph(segment: dict, mode: str) -> dict:
    """Comfy 0.34+ API graph. Stub class MiniMaxH3 400s; use MiniMaxH3ImageToVideo."""
    seconds = float(segment.get("duration") or 8.0)
    length = h3_length_frames(seconds)
    prompt = segment_prompt(segment)
    seed = int(segment.get("seed") or 11)
    steps = int(os.environ.get("H3_SAMPLER_STEPS") or 8)
    first = image_filename(segment.get("first_frame_path"))
    last = segment.get("last_frame_path")
    chain_tail = int(segment.get("chain_index") or 0) > 0 or bool(segment.get("chain_source_last_frame"))
    if chain_tail and not first:
        raise RuntimeError(
            f"chain tail {segment.get('id') or '?'} refuses submit without last.png"
        )
    unet = REF2VA_UNET if mode == "ref2va" else FL2VA_UNET
    cond_type = "MiniMaxH3ReferenceToVideo" if mode == "ref2va" else "MiniMaxH3ImageToVideo"
    width, height = h3_spatial_size(segment)
    cond_inputs: dict[str, Any] = {
        "clip": ["2", 0],
        "vae": ["3", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": length,
    }
    if mode == "ref2va":
        # Picture only. Never wire H3/Hailuo audio_vae — CosyVoice2 is the episode mix.
        cond_inputs["ref_image_size"] = "match"
        tags = []
        for i, filename in enumerate(ref_image_filenames(segment)):
            nid = f"r{i + 1}"
            cond_inputs[ref_autogrow_input_key(i)] = [nid, 0]
            tags.append(f"<<<image_{i + 1}>>>")
        if tags:
            cond_inputs["prompt"] = (
                f"{', '.join(tags)} are the locked character-sheet and scene-plate "
                f"reference pack for this story. Keep face and costume of these refs; "
                f"no face drift; no costume swap; 禁止五官漂移/换装. "
                f"Do not mint a new first frame. {prompt}"
            )
    else:
        if not first:
            raise RuntimeError(
                f"fl2va segment {segment.get('id') or '?'} missing first_frame_path"
            )
        cond_inputs["first_frame"] = ["5", 0]
        if last and not is_directory_like(last):
            cond_inputs["last_frame"] = ["last", 0]
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
        "16": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": ["lv_ff", 0], "shift_video": 12.0, "shift_audio": 3.0},
        },
        "6": {"class_type": cond_type, "inputs": cond_inputs},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["16", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0},
        },
        "10": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["16", 0], "conditioning": ["6", 0]},
        },
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["7", 0],
                "guider": ["10", 0],
                "sampler": ["8", 0],
                "sigmas": ["9", 0],
                "latent_image": ["6", 1],
            },
        },
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "14": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["12", 0], "fps": 24},
        },
        "15": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["14", 0],
                "filename_prefix": f"h3/{segment.get('id') or 'shot'}",
                "format": "auto",
                "codec": "auto",
            },
        },
    }
    graph.update(_low_vram_model_chain())
    if mode == "ref2va":
        for i, filename in enumerate(ref_image_filenames(segment)):
            graph[f"r{i + 1}"] = {
                "class_type": "LoadImage",
                "_meta": {"role": ref_autogrow_input_key(i)},
                "inputs": {"image": filename},
            }
        # Optional composition guide only when a real first frame exists. Never LoadImage f1.png.
        if first:
            graph["5"] = {
                "class_type": "LoadImage",
                "_meta": {"role": "first_frame"},
                "inputs": {"image": first},
            }
            graph["guide"] = {
                "class_type": "MiniMaxH3AddGuide",
                "inputs": {
                    "positive": ["6", 0],
                    "vae": ["3", 0],
                    "latent": ["6", 1],
                    "image": ["5", 0],
                    "frame_idx": 0,
                },
            }
            graph["10"]["inputs"]["conditioning"] = ["guide", 0]
        elif chain_tail:
            raise RuntimeError(
                f"chain tail {segment.get('id') or '?'} refuses submit without last.png"
            )
    else:
        graph["5"] = {
            "class_type": "LoadImage",
            "_meta": {"role": "first_frame"},
            "inputs": {"image": first},
        }
    # Never decode or mux H3/Hailuo speech. Locked CosyVoice2 is mixed at compose.
    for nid, node in list(graph.items()):
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in FORBIDDEN_H3_SPEECH_NODES:
            graph.pop(nid, None)
            continue
        if (node.get("inputs") or {}).get("vae_name") == H3_AUDIO_VAE:
            graph.pop(nid, None)
    if "14" in graph:
        graph["14"]["inputs"].pop("audio", None)
    if last and not is_directory_like(last) and mode != "ref2va":
        graph["last"] = {
            "class_type": "LoadImage",
            "_meta": {"role": "last_frame"},
            "inputs": {"image": last},
        }
    return graph


def prepare_workflow(segment: dict, template: dict | None = None) -> dict:
    mode = select_mode(segment)
    if template is None:
        graph = {"prompt": native_h3_graph(segment, mode)}
    else:
        graph = copy.deepcopy(template)
    if mode != "fl2va_first_last" or is_directory_like(segment.get("last_frame_path")):
        graph = prune_empty_last_frame(graph)
    return graph

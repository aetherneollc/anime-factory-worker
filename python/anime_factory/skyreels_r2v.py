"""SkyReels V3 reference-to-video wrapper.

Official CLI (SkyworkAI/SkyReels-V3 ``generate_video.py``, commit
``28c771e8456341be6a213e3d1133ed1fd19bf75d``):

    python3 generate_video.py \\
        --task_type reference_to_video \\
        --ref_imgs img1.png,img2.png \\
        --prompt "..." \\
        --duration 5 \\
        --resolution 720P \\
        --seed 42 \\
        --model_id Skywork/SkyReels-V3-R2V-14B \\
        --low_vram \\
        --offload

There is no ``--fps`` flag and no ``--aspect`` flag. ``generate_video.py``
writes non-avatar clips at 24 fps (``fps = 24``). Aspect is chosen inside
``ReferenceToVideoPipeline.generate_video`` by snapping the first reference
image to the closest 720P bucket (``16:9`` → 1280×720, ``9:16`` → 720×1280).
This wrapper accepts only those two aspects and records the choice; the
caller must supply a first reference whose orientation matches.

``--low_vram`` is on by default (5090 32GB). ``SKYREELS_OFFLOAD`` toggles
``--offload`` (default on; set ``0`` to omit it). ``--use_usp`` is never
passed (it cannot be combined with ``--low_vram``).

Klein VRAM is released before the subprocess so the two models stay serial.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SKYREELS_MODEL_ID = "Skywork/SkyReels-V3-R2V-14B"
MODEL_ID = SKYREELS_MODEL_ID
# generate_video.py auto-selects this alias when --model_id is omitted.
# We pass the HF repo id from the model card explicitly.
SKYREELS_CODE_DEFAULT_MODEL_ID = "Skywork/SkyReels-V3-Reference2Video"
TASK_TYPE = "reference_to_video"
DEFAULT_RESOLUTION = "720P"
DEFAULT_DURATION_S = 5
DEFAULT_FPS = 24
ALLOWED_ASPECTS = ("16:9", "9:16")
ALLOWED_RESOLUTIONS = ("480P", "540P", "720P")
max_references = 4
MAX_REFS = max_references
# 720P buckets from skyreels_v3/config.py ASPECT_RATIO_CONFIG (height, width).
ASPECT_BUCKETS = {
    "16:9": (720, 1280),
    "9:16": (1280, 720),
}


class SkyReelsR2VError(RuntimeError):
    """R2V CLI prerequisites failed."""


@dataclass
class SkyReelsGenerateResult:
    path: str
    argv: list[str]
    dry_run: bool
    aspect: str
    fps: int = DEFAULT_FPS
    resolution: str = DEFAULT_RESOLUTION
    duration_s: int = DEFAULT_DURATION_S
    low_vram: bool = True
    offload: bool = True
    model_id: str = SKYREELS_MODEL_ID
    meta: dict = field(default_factory=dict)


def skyreels_model_id() -> str:
    raw = (os.environ.get("SKYREELS_MODEL_ID") or SKYREELS_MODEL_ID).strip()
    return raw or SKYREELS_MODEL_ID


def skyreels_root() -> Path:
    return Path(os.environ.get("SKYREELS_ROOT") or "/opt/SkyReels-V3")


def skyreels_dry_run(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return str(os.environ.get("SKYREELS_DRY_RUN") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def skyreels_offload_enabled(explicit: bool | None = None) -> bool:
    """``SKYREELS_OFFLOAD`` toggles ``--offload``. Unset means on."""
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("SKYREELS_OFFLOAD")
    if raw is None or str(raw).strip() == "":
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def low_vram_enabled(explicit: bool | None = None) -> bool:
    """``--low_vram`` defaults on. ``SKYREELS_OFFLOAD=0`` turns it off too.

    The official pipeline treats ``--low_vram`` as offload, so leaving the
    flag on would ignore an explicit offload disable.
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("SKYREELS_LOW_VRAM")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    off = os.environ.get("SKYREELS_OFFLOAD")
    if off is not None and str(off).strip().lower() in {"0", "false", "no", "off"}:
        return False
    return True


def normalize_aspect(aspect: str | None) -> str:
    text = str(aspect or "16:9").strip().replace(" ", "")
    if text not in ALLOWED_ASPECTS:
        raise SkyReelsR2VError(
            f"R2V aspect must be one of {ALLOWED_ASPECTS}, got {aspect!r}"
        )
    return text


def _reference_path_and_role(item: Any) -> tuple[str, str]:
    if isinstance(item, str):
        return item.strip(), "other"
    if isinstance(item, Mapping):
        return str(item.get("path") or "").strip(), str(item.get("role") or "other").strip().lower()
    path = str(getattr(item, "path", "") or "").strip()
    role = str(getattr(item, "role", "") or "other").strip().lower()
    return path, role or "other"


def truncate_references(refs: Sequence[Any]) -> tuple[list[str], dict[str, Any]]:
    """Keep at most ``max_references`` images. Character paths come first.

    Empty input is a refusal. This adapter does not substitute a preview frame.
    Dropped paths are recorded in the returned meta.
    """
    characters: list[str] = []
    others: list[str] = []
    seen: set[str] = set()
    for item in refs or []:
        path, role = _reference_path_and_role(item)
        if not path or path in seen:
            continue
        seen.add(path)
        if role == "character":
            characters.append(path)
        else:
            others.append(path)
    ordered = characters + others
    kept = ordered[:max_references]
    dropped = ordered[max_references:]
    meta: dict[str, Any] = {
        "max_references": max_references,
        "input_count": len(ordered),
        "kept_count": len(kept),
        "truncated": bool(dropped),
        "dropped_references": dropped,
        "priority": "character",
    }
    if not kept:
        raise SkyReelsR2VError(
            "SkyReels R2V requires references; refusing to substitute a preview frame"
        )
    return kept, meta


def build_r2v_argv(
    ref_imgs: Sequence[str],
    prompt: str,
    *,
    duration_s: int = DEFAULT_DURATION_S,
    resolution: str = DEFAULT_RESOLUTION,
    seed: int = 42,
    low_vram: bool = True,
    offload: bool = True,
    model_id: str | None = None,
    python: str | None = None,
) -> list[str]:
    """Exact flags from ``generate_video.py`` argparse. No invented flags."""
    refs = [str(p).strip() for p in ref_imgs if str(p).strip()]
    if not refs or len(refs) > MAX_REFS:
        raise SkyReelsR2VError(f"reference_to_video requires 1–{MAX_REFS} --ref_imgs, got {len(refs)}")
    if resolution not in ALLOWED_RESOLUTIONS:
        raise SkyReelsR2VError(
            f"--resolution must be one of {ALLOWED_RESOLUTIONS}, got {resolution!r}"
        )
    duration = int(duration_s)
    if duration < 1:
        raise SkyReelsR2VError(f"--duration must be >= 1, got {duration}")
    exe = python or sys.executable
    argv = [
        exe,
        "generate_video.py",
        "--task_type",
        TASK_TYPE,
        "--ref_imgs",
        ",".join(refs),
        "--prompt",
        prompt or "",
        "--duration",
        str(duration),
        "--resolution",
        resolution,
        "--seed",
        str(int(seed)),
        "--model_id",
        model_id or skyreels_model_id(),
    ]
    if low_vram:
        argv.append("--low_vram")
    if offload:
        argv.append("--offload")
    return argv


def _newest_mp4(root: Path) -> Path | None:
    folder = root / "result" / TASK_TYPE
    if not folder.is_dir():
        return None
    files = sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def generate_reference_video(
    *,
    ref_imgs: Sequence[str],
    prompt: str,
    duration_s: float = DEFAULT_DURATION_S,
    resolution: str = DEFAULT_RESOLUTION,
    aspect: str = "16:9",
    fps: int = DEFAULT_FPS,
    seed: int | None = None,
    low_vram: bool | None = None,
    offload: bool | None = None,
    dry_run: bool | None = None,
    out_path: str | None = None,
) -> SkyReelsGenerateResult:
    """Run official ``generate_video.py --task_type reference_to_video``.

    Truncates to ``max_references`` before any GPU work. Empty references
    refuse. Dry-run (``SKYREELS_DRY_RUN=1``) returns the argv and does not
    spawn the process. Klein VRAM is released only after references validate.
    """
    kept, trunc_meta = truncate_references(ref_imgs)
    from anime_factory.flux2_klein import release_klein_vram

    release_klein_vram()
    chosen_aspect = normalize_aspect(aspect)
    if int(fps) != DEFAULT_FPS:
        raise SkyReelsR2VError(
            f"generate_video.py hardcodes {DEFAULT_FPS} fps for reference_to_video, got {fps}"
        )
    use_low = low_vram_enabled(low_vram)
    use_offload = skyreels_offload_enabled(offload)
    seed_i = 42 if seed is None else int(seed)
    argv = build_r2v_argv(
        kept,
        prompt,
        duration_s=int(duration_s),
        resolution=resolution,
        seed=seed_i,
        low_vram=use_low,
        offload=use_offload,
    )
    dry = skyreels_dry_run(dry_run)
    height, width = ASPECT_BUCKETS[chosen_aspect]
    meta = {
        "aspect": chosen_aspect,
        "fps": DEFAULT_FPS,
        "bucket_hw": [height, width],
        "note": "aspect is not a CLI flag; first ref image selects the 720P bucket",
        "argv": argv,
        **trunc_meta,
    }
    if dry:
        return SkyReelsGenerateResult(
            path=out_path or f"result/{TASK_TYPE}/dry-run.mp4",
            argv=argv,
            dry_run=True,
            aspect=chosen_aspect,
            fps=DEFAULT_FPS,
            resolution=resolution,
            duration_s=int(duration_s),
            low_vram=use_low,
            offload=use_offload,
            model_id=skyreels_model_id(),
            meta=meta,
        )

    root = skyreels_root()
    script = root / "generate_video.py"
    if not script.is_file():
        raise SkyReelsR2VError(f"missing {script}; set SKYREELS_ROOT or SKYREELS_DRY_RUN=1")
    for ref in kept:
        if not Path(ref).is_file():
            raise SkyReelsR2VError(f"reference image missing: {ref}")
    try:
        subprocess.run(argv, cwd=str(root), check=True)
    except subprocess.CalledProcessError as exc:
        raise SkyReelsR2VError(f"generate_video.py failed: {exc}") from exc
    produced = _newest_mp4(root)
    if produced is None:
        raise SkyReelsR2VError(f"no mp4 under {root / 'result' / TASK_TYPE}")
    path = str(produced)
    if out_path:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(produced.read_bytes())
        path = str(dest)
    return SkyReelsGenerateResult(
        path=path,
        argv=argv,
        dry_run=False,
        aspect=chosen_aspect,
        fps=DEFAULT_FPS,
        resolution=resolution,
        duration_s=int(duration_s),
        low_vram=use_low,
        offload=use_offload,
        model_id=skyreels_model_id(),
        meta=meta,
    )

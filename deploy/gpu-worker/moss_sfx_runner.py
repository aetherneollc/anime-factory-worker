#!/usr/bin/env python3
"""Isolated MOSS-SoundEffect v2 runner. Runs in its OWN Python 3.12 /
torch-2.9-cu128 environment — never imported by the torch-2.8 worker process.

Invoked by `anime_factory.sfx_moss.MossSoundEffectClient` with the exact argv
contract of `moss_command(...)`:

    <python3.12> moss_sfx_runner.py \
        --pipeline OpenMOSS/MOSS-TTS/moss_soundeffect_v2 \
        --model-repo OpenMOSS-Team/MOSS-SoundEffect-v2.0 \
        --commit 934d6826b084c46a0d033402174d5f8ac4ed2519 \
        --prompt "<text>" --duration <s> --seed <int> \
        --device cuda:0 --weights-dir <dir> --out <path.wav>

Guarantees enforced here, not merely documented:
  - startup validation: Python 3.12, torch/torchaudio 2.9 built for CUDA 12.8
    (cu128); a mismatched environment exits before any download or import of
    model code;
  - pinned identities: the argv --model-repo/--commit/--pipeline must equal
    the constants below (which mirror `anime_factory.sfx_common`); the model
    snapshot is downloaded at the pinned HF revision only;
  - lazy weights: nothing is downloaded at import/parse time — the snapshot is
    pulled into --weights-dir only when a generation is actually invoked, and
    a completed pinned snapshot is reused;
  - deterministic output: seed is applied via torch.manual_seed and forwarded
    to the pipeline; output is written as mono PCM16 WAV via the stdlib (never
    a float WAV a downstream QC could misread);
  - secrets: every error path is scrubbed with `redact_secrets` so HF_TOKEN or
    any *_KEY/_TOKEN/_SECRET env value can never reach stdout/stderr.

Official upstream usage this wraps (MOSS-TTS/moss_soundeffect_v2 README):

    from moss_soundeffect_v2 import MossSoundEffectPipeline
    pipe = MossSoundEffectPipeline.from_pretrained(model_dir,
        torch_dtype=torch.bfloat16, device="cuda")
    audio = pipe(prompt=..., seconds=..., num_inference_steps=...,
        cfg_scale=..., sigma_shift=..., seed=...)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import wave
from pathlib import Path

PINNED_MODEL_REPO = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
# MOSS-TTS source tree commit the pipeline package is installed from.
PINNED_SOURCE_COMMIT = "934d6826b084c46a0d033402174d5f8ac4ed2519"
PINNED_PIPELINE_MODULE = "OpenMOSS/MOSS-TTS/moss_soundeffect_v2"
# HF model repo revision the weights snapshot is pinned to.
PINNED_MODEL_REVISION = "e35df4d82fbe87fcd5d14e5d100e349c0c3c076d"

REQUIRED_PYTHON = (3, 12)
REQUIRED_TORCH = "2.9.0+cu128"
REQUIRED_TORCHAUDIO = "2.9.0+cu128"
REQUIRED_TORCHVISION = "0.24.0+cu128"
REQUIRED_CUDA = "12.8"  # cu128

DEFAULT_STEPS = 100
DEFAULT_CFG_SCALE = 4.0
DEFAULT_SIGMA_SHIFT = 5.0
MAX_SECONDS = 30.0
SNAPSHOT_MARKER = ".moss_sfx_snapshot.json"

_SECRET_ENV_NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.I)
_MIN_SECRET_LEN = 6


class RunnerError(RuntimeError):
    """Any validation/generation failure. Message must already be redacted."""


def redact_secrets(text: object) -> str:
    """Strip secret-looking env values (HF_TOKEN, *_API_KEY, …) out of text."""
    out = str(text or "")
    values = [
        v
        for k, v in os.environ.items()
        if v and len(v) >= _MIN_SECRET_LEN and _SECRET_ENV_NAME_RE.search(k)
    ]
    for value in sorted(set(values), key=len, reverse=True):
        out = out.replace(value, "<redacted>")
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated MOSS-SoundEffect v2 runner")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    parser.add_argument("--sigma-shift", type=float, default=DEFAULT_SIGMA_SHIFT)
    return parser.parse_args(argv)


def validate_pins(args: argparse.Namespace) -> None:
    """The caller's pins and this runner's pins must agree exactly."""
    if args.model_repo != PINNED_MODEL_REPO:
        raise RunnerError(f"model repo {args.model_repo!r} != pinned {PINNED_MODEL_REPO!r}")
    if args.commit != PINNED_SOURCE_COMMIT:
        raise RunnerError(f"source commit {args.commit!r} != pinned {PINNED_SOURCE_COMMIT!r}")
    if args.pipeline != PINNED_PIPELINE_MODULE:
        raise RunnerError(f"pipeline {args.pipeline!r} != pinned {PINNED_PIPELINE_MODULE!r}")
    if not (0.0 < float(args.duration) <= MAX_SECONDS):
        raise RunnerError(f"duration {args.duration} outside (0, {MAX_SECONDS}]s")


def validate_python(version_info: tuple = tuple(sys.version_info)) -> None:
    if tuple(version_info[:2]) != REQUIRED_PYTHON:
        found = ".".join(str(v) for v in version_info[:3])
        raise RunnerError(
            f"MOSS-SoundEffect v2 requires Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}, "
            f"running {found}"
        )


def validate_torch(torch_mod: object) -> None:
    version = getattr(torch_mod, "__version__", "")
    if version != REQUIRED_TORCH:
        raise RunnerError(f"torch {version!r} is not the required {REQUIRED_TORCH!r}")
    cuda = getattr(getattr(torch_mod, "version", None), "cuda", None)
    if str(cuda or "") != REQUIRED_CUDA:
        raise RunnerError(
            f"torch CUDA build {cuda!r} is not the required cu128 ({REQUIRED_CUDA})"
        )


def validate_torchaudio(torchaudio_mod: object) -> None:
    version = getattr(torchaudio_mod, "__version__", "")
    if version != REQUIRED_TORCHAUDIO:
        raise RunnerError(
            f"torchaudio {version!r} is not the required {REQUIRED_TORCHAUDIO!r}"
        )


def validate_torchvision(torchvision_mod: object) -> None:
    version = getattr(torchvision_mod, "__version__", "")
    if version != REQUIRED_TORCHVISION:
        raise RunnerError(
            f"torchvision {version!r} is not the required {REQUIRED_TORCHVISION!r}"
        )


def ensure_weights(weights_dir: str, *, snapshot_download=None) -> str:
    """Lazily materialize the pinned model snapshot. Reuse a completed one.

    Called only from a real generation — never at import/parse time — so
    weights are pulled the first time a cue actually falls back to MOSS.
    """
    root = Path(weights_dir)
    marker = root / SNAPSHOT_MARKER
    if marker.is_file():
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recorded = {}
        if recorded.get("revision") == PINNED_MODEL_REVISION:
            return str(root)
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as snapshot_download  # noqa: PLC0415
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=PINNED_MODEL_REPO,
        revision=PINNED_MODEL_REVISION,
        local_dir=str(root),
    )
    marker.write_text(
        json.dumps({"repo": PINNED_MODEL_REPO, "revision": PINNED_MODEL_REVISION}),
        encoding="utf-8",
    )
    return str(root)


def to_mono_float_samples(audio: object) -> list[float]:
    """(B, C, T) tensor / nested lists -> one mono float list in [-1, 1]."""
    if hasattr(audio, "detach"):  # torch tensor without importing torch here
        audio = audio.detach().float().cpu().tolist()
    data = audio
    # Unwrap batch then average channels.
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
        batch = data[0]
        if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], (list, tuple)):
            channels = [list(ch) for ch in batch]
        else:
            channels = [list(batch)]
    else:
        channels = [list(data or [])]
    if not channels or not channels[0]:
        raise RunnerError("pipeline returned no audio samples")
    n = min(len(ch) for ch in channels)
    inv = 1.0 / len(channels)
    return [sum(ch[i] for ch in channels) * inv for i in range(n)]


def write_pcm16_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    """Stdlib PCM16 mono WAV writer — output is always integer PCM, never a
    float WAV that a downstream PCM16 decoder would read as garbage."""
    if not samples:
        raise RunnerError("no samples to write")
    if int(sample_rate) <= 0:
        raise RunnerError(f"invalid sample rate {sample_rate!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for value in samples:
        v = max(-1.0, min(1.0, float(value)))
        frames += struct.pack("<h", int(round(v * 32767)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(bytes(frames))


def _default_pipeline_factory(model_dir: str, device: str):
    import torch  # noqa: PLC0415 — isolated env only
    from moss_soundeffect_v2 import MossSoundEffectPipeline  # noqa: PLC0415

    return MossSoundEffectPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device=device,
    )


def generate(
    args: argparse.Namespace,
    *,
    pipeline_factory=None,
    snapshot_download=None,
    torch_mod=None,
) -> Path:
    model_dir = ensure_weights(args.weights_dir, snapshot_download=snapshot_download)
    if torch_mod is not None and hasattr(torch_mod, "manual_seed"):
        torch_mod.manual_seed(int(args.seed))
    factory = pipeline_factory or _default_pipeline_factory
    pipe = factory(model_dir, args.device)
    audio = pipe(
        prompt=args.prompt,
        seconds=round(float(args.duration), 1),
        num_inference_steps=int(args.steps),
        cfg_scale=float(args.cfg_scale),
        sigma_shift=float(args.sigma_shift),
        seed=int(args.seed),
    )
    sample_rate = int(getattr(pipe, "sample_rate", 48000) or 48000)
    out = Path(args.out)
    write_pcm16_wav(out, to_mono_float_samples(audio), sample_rate)
    return out


def main(
    argv: list[str] | None = None,
    *,
    torch_mod=None,
    torchaudio_mod=None,
    torchvision_mod=None,
    pipeline_factory=None,
    snapshot_download=None,
) -> int:
    try:
        args = parse_args(list(sys.argv[1:] if argv is None else argv))
        validate_python()
        validate_pins(args)
        if torch_mod is None:
            import torch as torch_mod  # noqa: PLC0415 — lazy: only a real invocation imports torch
        if torchaudio_mod is None:
            import torchaudio as torchaudio_mod  # noqa: PLC0415
        if torchvision_mod is None:
            import torchvision as torchvision_mod  # noqa: PLC0415
        validate_torch(torch_mod)
        validate_torchaudio(torchaudio_mod)
        validate_torchvision(torchvision_mod)
        out = generate(
            args,
            pipeline_factory=pipeline_factory,
            snapshot_download=snapshot_download,
            torch_mod=torch_mod,
        )
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — one redacted line, nonzero exit
        sys.stderr.write(f"moss_sfx_runner: {type(exc).__name__}: {redact_secrets(exc)}\n")
        return 1
    sys.stdout.write(f"moss_sfx_runner: wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

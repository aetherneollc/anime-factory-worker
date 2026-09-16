"""MOSS-SoundEffect v2 fallback: an isolated-subprocess wrapper, never an import.

Upstream contract (pinned, must not drift silently):
  - model: OpenMOSS-Team/MOSS-SoundEffect-v2.0
  - source commit: 934d6826b084c46a0d033402174d5f8ac4ed2519
  - pipeline module: OpenMOSS/MOSS-TTS/moss_soundeffect_v2
  - official runtime: isolated Python 3.12, torch/torchaudio/torchvision 2.9 cu128

This worker process runs torch 2.8. MOSS-SoundEffect v2 needs torch 2.9 cu128,
so it cannot share this interpreter — it is invoked as a subprocess in a
separate, pre-built Python 3.12 environment configured entirely by env vars
(`MOSS_SFX_*`). No `torch`, `torchaudio`, or MOSS-TTS import happens in this
module. Docker wiring for that isolated environment is out of scope here
(owned by another branch/Dockerfile); this module only defines and enforces
the calling contract, so it works once that environment exists.

Weights are pulled by the isolated subprocess itself, into
`MOSS_SFX_WEIGHTS_DIR`, and only when `generate()` is actually called — i.e.
only when a cue has already fallen back off Freesound. This worker never
fetches MOSS weights itself and never writes them to R2.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from anime_factory.sfx_common import (
    SfxCue,
    SfxProvenance,
    SfxResult,
    assert_audio_qc,
    cache_dir_for,
    cache_lookup,
    cache_record,
    cache_write,
    content_hash,
    deterministic_seed,
)
from anime_factory.tts import decode_pcm16_mono

log = logging.getLogger("anime_factory.sfx.moss")

MOSS_MODEL_REPO = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
MOSS_SOURCE_COMMIT = "934d6826b084c46a0d033402174d5f8ac4ed2519"
MOSS_PIPELINE_MODULE = "OpenMOSS/MOSS-TTS/moss_soundeffect_v2"
MOSS_REQUIRED_PYTHON = "3.12"
MOSS_REQUIRED_TORCH = "2.9"
MOSS_REQUIRED_CUDA = "cu128"

DEFAULT_TIMEOUT_S = 300.0
RunnerFn = Callable[[list[str], dict, float], "subprocess.CompletedProcess"]


class MossConfigError(RuntimeError):
    """MOSS_SFX_PYTHON / MOSS_SFX_SCRIPT / MOSS_SFX_WEIGHTS_DIR missing from env."""


class MossGenerationError(RuntimeError):
    """The isolated subprocess exited non-zero, timed out, or wrote no audio."""


@dataclass(frozen=True)
class MossRunnerConfig:
    """Everything needed to shell into the isolated torch-2.9 environment."""

    python_executable: str
    script_path: str
    weights_dir: str
    device: str = "cuda:0"
    timeout_s: float = DEFAULT_TIMEOUT_S
    extra_args: tuple[str, ...] = ()


def moss_runner_config_from_env() -> MossRunnerConfig | None:
    """None means "not configured" — the orchestrator must treat that as a hard failure,
    not silently skip the fallback."""
    python_exe = (os.environ.get("MOSS_SFX_PYTHON") or "").strip()
    script = (os.environ.get("MOSS_SFX_SCRIPT") or "").strip()
    weights_dir = (os.environ.get("MOSS_SFX_WEIGHTS_DIR") or "").strip()
    if not (python_exe and script and weights_dir):
        return None
    timeout_raw = (os.environ.get("MOSS_SFX_TIMEOUT_S") or "").strip()
    return MossRunnerConfig(
        python_executable=python_exe,
        script_path=script,
        weights_dir=weights_dir,
        device=(os.environ.get("MOSS_SFX_DEVICE") or "cuda:0").strip(),
        timeout_s=float(timeout_raw) if timeout_raw else DEFAULT_TIMEOUT_S,
    )


def _default_runner(args: list[str], env: dict, timeout_s: float) -> "subprocess.CompletedProcess":
    return subprocess.run(args, env=env, capture_output=True, timeout=timeout_s)  # noqa: S603


def moss_command(config: MossRunnerConfig, cue: SfxCue, seed: int, duration: float, out_path: Path) -> list[str]:
    """The exact subprocess argv contract an isolated runner script must accept."""
    return [
        config.python_executable,
        config.script_path,
        "--pipeline",
        MOSS_PIPELINE_MODULE,
        "--model-repo",
        MOSS_MODEL_REPO,
        "--commit",
        MOSS_SOURCE_COMMIT,
        "--prompt",
        cue.query,
        "--duration",
        f"{duration:.3f}",
        "--seed",
        str(seed),
        "--device",
        config.device,
        "--weights-dir",
        config.weights_dir,
        "--out",
        str(out_path),
        *config.extra_args,
    ]


class MossSoundEffectClient:
    """Owns exactly one contract: subprocess in, WAV bytes out. No model code here."""

    def __init__(self, config: MossRunnerConfig | None = None, runner: RunnerFn | None = None):
        self._config = config
        self._config_loaded = config is not None
        self.runner = runner or _default_runner

    @property
    def config(self) -> MossRunnerConfig | None:
        if not self._config_loaded:
            self._config = moss_runner_config_from_env()
            self._config_loaded = True
        return self._config

    def generate(self, cue: SfxCue, out_path: Path) -> bytes:
        config = self.config
        if config is None:
            raise MossConfigError(
                "MOSS_SFX_PYTHON/MOSS_SFX_SCRIPT/MOSS_SFX_WEIGHTS_DIR not set; cannot run the "
                f"{MOSS_MODEL_REPO}@{MOSS_SOURCE_COMMIT[:12]} fallback"
            )
        seed = cue.seed if cue.seed is not None else deterministic_seed(cue.cue_key, salt=MOSS_SOURCE_COMMIT)
        duration = float(cue.duration_target or 2.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        args = moss_command(config, cue, seed, duration, out_path)
        env = dict(os.environ)
        env["MOSS_SFX_WEIGHTS_DIR"] = config.weights_dir
        try:
            result = self.runner(args, env, config.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise MossGenerationError(f"{cue.cue_key}: MOSS subprocess timed out after {config.timeout_s}s") from exc
        returncode = getattr(result, "returncode", 1)
        if returncode != 0:
            stderr = getattr(result, "stderr", b"") or b""
            tail = stderr[-400:].decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)[-400:]
            raise MossGenerationError(f"{cue.cue_key}: MOSS subprocess exit {returncode}: {tail}")
        if not out_path.is_file() or out_path.stat().st_size <= 44:
            raise MossGenerationError(f"{cue.cue_key}: MOSS subprocess wrote no audio at {out_path}")
        return out_path.read_bytes()


def resolve_via_moss(client: MossSoundEffectClient, cue: SfxCue, root: Path) -> SfxResult:
    """Generate + validate with the same deterministic QC as the Freesound path.

    A cache hit skips the subprocess entirely — MOSS weights are only pulled
    (by the isolated subprocess) the first time a given cue actually falls back.
    """
    cache_root = cache_dir_for(root, cue)
    cached = cache_lookup(cache_root, cue.cue_key)
    if cached is not None:
        return cached

    seed = cue.seed if cue.seed is not None else deterministic_seed(cue.cue_key, salt=MOSS_SOURCE_COMMIT)
    duration = float(cue.duration_target or 2.0)
    tmp_out = cache_root / f"_gen_{cue.cue_key}.wav"
    wav = client.generate(cue, tmp_out)
    assert_audio_qc(wav, label=f"{cue.cue_key}:moss", target_duration=cue.duration_target, duration_tolerance=cue.duration_tolerance)
    digest = content_hash(wav)
    local_path = cache_write(cache_root, digest, wav)
    try:
        tmp_out.unlink(missing_ok=True)
    except OSError:
        pass
    rate, samples = decode_pcm16_mono(wav)
    provenance = SfxProvenance(
        source="moss",
        content_hash=digest,
        sample_rate=rate,
        duration=len(samples) / float(rate or 1),
        query=cue.query,
        tags=cue.tags,
        seed=seed,
        model=MOSS_MODEL_REPO,
        model_commit=MOSS_SOURCE_COMMIT,
        prompt=cue.query,
        cached=False,
    )
    cache_record(cache_root, cue.cue_key, local_path, provenance)
    return SfxResult(cue_key=cue.cue_key, status="resolved", local_path=str(local_path), provenance=provenance)

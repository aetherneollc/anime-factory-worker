"""MOSS-SoundEffect v2 fallback: isolated-subprocess contract, determinism, QC."""

from __future__ import annotations

import subprocess

import pytest

from anime_factory.sfx_common import SfxCue, deterministic_seed
from anime_factory.sfx_moss import (
    MOSS_MODEL_REPO,
    MOSS_SOURCE_COMMIT,
    MossConfigError,
    MossGenerationError,
    MossRunnerConfig,
    MossSoundEffectClient,
    moss_command,
    resolve_via_moss,
)
from anime_factory.tts import voiced_dummy_wav


def _config(tmp_path) -> MossRunnerConfig:
    return MossRunnerConfig(
        python_executable="/opt/moss-sfx/bin/python3.12",
        script_path="/opt/moss-sfx/run_soundeffect.py",
        weights_dir=str(tmp_path / "weights"),
        device="cuda:0",
        timeout_s=30.0,
    )


def test_no_config_raises_moss_config_error(tmp_path, monkeypatch):
    for key in ("MOSS_SFX_PYTHON", "MOSS_SFX_SCRIPT", "MOSS_SFX_WEIGHTS_DIR"):
        monkeypatch.delenv(key, raising=False)
    client = MossSoundEffectClient()
    cue = SfxCue(cue_key="thunder", query="distant thunder rumble", duration_target=3.0)
    with pytest.raises(MossConfigError):
        client.generate(cue, tmp_path / "out.wav")


def test_command_is_deterministic_and_pins_model_commit(tmp_path):
    config = _config(tmp_path)
    cue = SfxCue(cue_key="thunder", query="distant thunder rumble", duration_target=3.0)
    seed = deterministic_seed(cue.cue_key, salt=MOSS_SOURCE_COMMIT)
    out = tmp_path / "thunder.wav"
    args = moss_command(config, cue, seed, 3.0, out)
    assert MOSS_MODEL_REPO in args
    assert MOSS_SOURCE_COMMIT in args
    assert "distant thunder rumble" in args
    assert "3.000" in args
    assert str(seed) in args
    # Recomputing the command for the same cue must be byte-identical.
    args2 = moss_command(config, cue, seed, 3.0, out)
    assert args == args2
    # A different cue_key changes the seed deterministically, not randomly.
    other_seed = deterministic_seed("rain", salt=MOSS_SOURCE_COMMIT)
    assert other_seed != seed
    assert deterministic_seed("thunder", salt=MOSS_SOURCE_COMMIT) == seed


def _fake_runner_writing(wav_bytes: bytes):
    def runner(args, env, timeout_s):
        out_path = args[args.index("--out") + 1]
        with open(out_path, "wb") as fh:
            fh.write(wav_bytes)
        return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=b"")

    return runner


def test_generate_writes_and_returns_wav_bytes(tmp_path):
    config = _config(tmp_path)
    wav = voiced_dummy_wav(2.0, freq=180.0)
    client = MossSoundEffectClient(config=config, runner=_fake_runner_writing(wav))
    cue = SfxCue(cue_key="thunder", query="distant thunder rumble", duration_target=2.0)
    out = tmp_path / "gen.wav"
    result = client.generate(cue, out)
    assert result == wav
    assert out.is_file()


def test_nonzero_exit_raises_generation_error(tmp_path):
    config = _config(tmp_path)

    def runner(args, env, timeout_s):
        return subprocess.CompletedProcess(args, returncode=1, stdout=b"", stderr=b"cuda oom")

    client = MossSoundEffectClient(config=config, runner=runner)
    cue = SfxCue(cue_key="thunder", query="thunder", duration_target=2.0)
    with pytest.raises(MossGenerationError):
        client.generate(cue, tmp_path / "out.wav")


def test_timeout_raises_generation_error(tmp_path):
    config = _config(tmp_path)

    def runner(args, env, timeout_s):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout_s)

    client = MossSoundEffectClient(config=config, runner=runner)
    cue = SfxCue(cue_key="thunder", query="thunder", duration_target=2.0)
    with pytest.raises(MossGenerationError):
        client.generate(cue, tmp_path / "out.wav")


def test_resolve_via_moss_validates_with_shared_qc(tmp_path):
    config = _config(tmp_path)
    silent = voiced_dummy_wav(0.01, freq=100.0)  # far too short / near-silent
    client = MossSoundEffectClient(config=config, runner=_fake_runner_writing(silent))
    cue = SfxCue(cue_key="whisper_fail", query="a whisper", duration_target=2.0)
    from anime_factory.sfx_common import SfxError

    with pytest.raises(SfxError):
        resolve_via_moss(client, cue, tmp_path)


def test_resolve_via_moss_success_records_provenance(tmp_path):
    config = _config(tmp_path)
    wav = voiced_dummy_wav(2.0, freq=180.0)
    client = MossSoundEffectClient(config=config, runner=_fake_runner_writing(wav))
    cue = SfxCue(cue_key="rain_loop", query="light rain loop", duration_target=2.0, seed=42)
    result = resolve_via_moss(client, cue, tmp_path)
    assert result.status == "resolved"
    assert result.provenance.source == "moss"
    assert result.provenance.model == MOSS_MODEL_REPO
    assert result.provenance.model_commit == MOSS_SOURCE_COMMIT
    assert result.provenance.seed == 42


def test_resolve_via_moss_cache_hit_skips_subprocess(tmp_path):
    config = _config(tmp_path)
    wav = voiced_dummy_wav(2.0, freq=180.0)
    client = MossSoundEffectClient(config=config, runner=_fake_runner_writing(wav))
    cue = SfxCue(cue_key="rain_loop2", query="light rain loop", duration_target=2.0, seed=7)
    first = resolve_via_moss(client, cue, tmp_path)
    assert first.status == "resolved"

    def exploding_runner(args, env, timeout_s):
        raise AssertionError("subprocess must not run on a cache hit")

    client2 = MossSoundEffectClient(config=config, runner=exploding_runner)
    second = resolve_via_moss(client2, cue, tmp_path)
    assert second.provenance.cached is True

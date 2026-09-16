"""deploy/gpu-worker/moss_sfx_runner.py: pins, runtime validation, lazy weights,
PCM WAV output, secret redaction — all with mocked torch/pipeline (this test
env runs neither Python-version games nor torch 2.9)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "deploy" / "gpu-worker" / "moss_sfx_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("moss_sfx_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _fake_torch(version="2.9.1+cu128", cuda="12.8"):
    seeds: list[int] = []
    mod = SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda),
        bfloat16=object(),
        manual_seed=seeds.append,
    )
    return mod, seeds


def _fake_torchaudio(version="2.9.0"):
    return SimpleNamespace(__version__=version)


def _argv(tmp_path, **overrides) -> list[str]:
    values = {
        "--pipeline": runner.PINNED_PIPELINE_MODULE,
        "--model-repo": runner.PINNED_MODEL_REPO,
        "--commit": runner.PINNED_SOURCE_COMMIT,
        "--prompt": "a heavy wooden door creaking open",
        "--duration": "2.000",
        "--seed": "1234",
        "--device": "cuda:0",
        "--weights-dir": str(tmp_path / "weights"),
        "--out": str(tmp_path / "out.wav"),
    }
    values.update(overrides)
    argv: list[str] = []
    for key, value in values.items():
        argv.extend([key, str(value)])
    return argv


class FakePipe:
    sample_rate = 48000

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        # (B, C, T) nested lists — a quiet ramp, deterministic.
        t = int(0.2 * self.sample_rate)
        return [[[0.25 * (i % 100) / 100.0 for i in range(t)]]]


def _prepin_weights(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / runner.SNAPSHOT_MARKER).write_text(
        json.dumps({"repo": runner.PINNED_MODEL_REPO, "revision": runner.PINNED_MODEL_REVISION}),
        encoding="utf-8",
    )
    return weights


def test_argv_contract_matches_sfx_moss_command(tmp_path):
    """The runner must accept exactly what MossSoundEffectClient sends."""
    from anime_factory.sfx_common import SfxCue
    from anime_factory.sfx_moss import MossRunnerConfig, moss_command

    config = MossRunnerConfig("python3.12", str(RUNNER_PATH), str(tmp_path / "w"))
    cue = SfxCue(cue_key="door", query="door creak", duration_target=2.0)
    argv = moss_command(config, cue, 7, 2.0, tmp_path / "o.wav")[2:]  # drop python + script
    args = runner.parse_args(argv)
    assert args.model_repo == runner.PINNED_MODEL_REPO
    assert args.commit == runner.PINNED_SOURCE_COMMIT
    runner.validate_pins(args)  # pins agree between worker contract and runner


def test_validate_python_rejects_non_312():
    runner.validate_python((3, 12, 4))
    with pytest.raises(runner.RunnerError):
        runner.validate_python((3, 11, 9))
    with pytest.raises(runner.RunnerError):
        runner.validate_python((3, 13, 0))


def test_validate_torch_requires_29_cu128():
    good, _ = _fake_torch()
    runner.validate_torch(good)
    bad_version, _ = _fake_torch(version="2.8.0+cu128")
    with pytest.raises(runner.RunnerError):
        runner.validate_torch(bad_version)
    bad_cuda, _ = _fake_torch(cuda="12.4")
    with pytest.raises(runner.RunnerError):
        runner.validate_torch(bad_cuda)
    runner.validate_torchaudio(_fake_torchaudio())
    with pytest.raises(runner.RunnerError):
        runner.validate_torchaudio(_fake_torchaudio("2.8.0"))


def test_validate_pins_rejects_drift(tmp_path):
    args = runner.parse_args(_argv(tmp_path, **{"--commit": "deadbeef" * 5}))
    with pytest.raises(runner.RunnerError):
        runner.validate_pins(args)
    args2 = runner.parse_args(_argv(tmp_path, **{"--model-repo": "SomeoneElse/other-model"}))
    with pytest.raises(runner.RunnerError):
        runner.validate_pins(args2)
    args3 = runner.parse_args(_argv(tmp_path, **{"--duration": "31.0"}))
    with pytest.raises(runner.RunnerError):
        runner.validate_pins(args3)


def test_ensure_weights_is_lazy_and_pinned(tmp_path):
    calls: list[dict] = []

    def fake_snapshot(**kwargs):
        calls.append(kwargs)
        Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)

    weights = tmp_path / "weights"
    out = runner.ensure_weights(str(weights), snapshot_download=fake_snapshot)
    assert out == str(weights)
    assert len(calls) == 1
    assert calls[0]["repo_id"] == runner.PINNED_MODEL_REPO
    assert calls[0]["revision"] == runner.PINNED_MODEL_REVISION  # pinned, never 'main'
    # A completed pinned snapshot is reused without a second download.
    runner.ensure_weights(str(weights), snapshot_download=fake_snapshot)
    assert len(calls) == 1


def test_main_end_to_end_writes_pcm16_wav_and_seeds(tmp_path):
    _prepin_weights(tmp_path)
    torch_mod, seeds = _fake_torch()
    pipe = FakePipe()

    code = runner.main(
        _argv(tmp_path),
        torch_mod=torch_mod,
        torchaudio_mod=_fake_torchaudio(),
        pipeline_factory=lambda model_dir, device: pipe,
        snapshot_download=lambda **kw: (_ for _ in ()).throw(AssertionError("weights already pinned")),
    )
    assert code == 0
    assert seeds == [1234]  # torch.manual_seed applied
    assert pipe.calls and pipe.calls[0]["seed"] == 1234
    assert pipe.calls[0]["prompt"] == "a heavy wooden door creaking open"
    out = tmp_path / "out.wav"
    assert out.is_file()
    from anime_factory.tts import decode_pcm16_mono

    rate, samples = decode_pcm16_mono(out.read_bytes())
    assert rate == 48000
    assert samples, "output must decode as integer PCM16 mono"


def test_main_never_leaks_tokens_on_failure(tmp_path, monkeypatch, capsys):
    secret = "hf_secret_token_abcdef0123"
    monkeypatch.setenv("HF_TOKEN", secret)
    _prepin_weights(tmp_path)
    torch_mod, _ = _fake_torch()

    def exploding_factory(model_dir, device):
        raise RuntimeError(f"401 for token {secret}")

    code = runner.main(
        _argv(tmp_path),
        torch_mod=torch_mod,
        torchaudio_mod=_fake_torchaudio(),
        pipeline_factory=exploding_factory,
    )
    assert code == 1
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "<redacted>" in captured.err


def test_main_fails_before_weights_on_bad_torch(tmp_path):
    """Runtime validation runs before any snapshot download."""
    torch_mod, _ = _fake_torch(version="2.8.0+cu124", cuda="12.4")

    def must_not_download(**kwargs):
        raise AssertionError("weights must not download when the runtime is invalid")

    code = runner.main(
        _argv(tmp_path),
        torch_mod=torch_mod,
        torchaudio_mod=_fake_torchaudio(),
        pipeline_factory=lambda m, d: FakePipe(),
        snapshot_download=must_not_download,
    )
    assert code == 1
    assert not (tmp_path / "weights").exists()


def test_write_pcm16_wav_clamps_out_of_range(tmp_path):
    out = tmp_path / "clamp.wav"
    runner.write_pcm16_wav(out, [2.0, -2.0, 0.5], 48000)
    from anime_factory.tts import decode_pcm16_mono

    _, samples = decode_pcm16_mono(out.read_bytes())
    assert samples[0] == 32767
    assert samples[1] == -32767

"""The NVFP4 production path: one real pipeline build, many samples on that object.

Only CUDA and the NVlabs tree are faked (tests/longlive_fakes.py). Everything under
``gpu_worker.longlive*`` executes for real, so these assertions are about the shipped
code path rather than a rehearsal of it.
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest

from gpu_worker import longlive, longlive_batch, longlive_pipeline
from gpu_worker.longlive import FAIL_CLOSED
from gpu_worker.longlive_batch import LongLiveBatchRunner, submit_longlive_batch
from tests.longlive_fakes import install_official_stubs

TAKE_SECONDS = 8.0
# longlive_num_output_frames(8.0): 192 video frames -> 48 latents, already a
# multiple of NUM_FRAME_PER_BLOCK, so 6 autoregressive blocks.
EXPECTED_FRAMES = 48
EXPECTED_BLOCKS = 6


@pytest.fixture
def longlive_env(tmp_path, monkeypatch):
    """Point the worker at a scratch LONGLIVE_ROOT and stub the upstream imports."""
    root = tmp_path / "LongLive"
    (root / "checkpoints").mkdir(parents=True)
    monkeypatch.setenv("LONGLIVE_ROOT", str(root))
    monkeypatch.setenv("LONGLIVE_WAN_DIR", str(root / "wan_models" / "Wan2.2-TI2V-5B"))
    monkeypatch.setenv("LONGLIVE_GENERATOR_CKPT", str(root / "checkpoints" / "model_4o6.pt"))
    monkeypatch.delenv("LONGLIVE_LORA_CKPT", raising=False)
    monkeypatch.delenv("AF_SKIP_WEIGHTS", raising=False)
    # Weight/extension presence is probed elsewhere; these tests are about sampling.
    monkeypatch.setattr(longlive_batch, "missing_longlive_requirements", lambda: [])
    log: dict = {}
    install_official_stubs(monkeypatch, log)
    return log


def _imported_modules(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _code_strings(tree: ast.Module) -> list[str]:
    """String literals excluding docstrings, so prose about upstream is allowed."""
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        head = body[0]
        if isinstance(head, ast.Expr) and isinstance(head.value, ast.Constant):
            if isinstance(head.value.value, str):
                docstrings.add(id(head.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _keyframe(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 120)
    return path


def _takes(tmp_path: Path, *, i2v: bool = True) -> list[dict]:
    rows = []
    for idx, (take_id, scene) in enumerate((("take-01", "cafe"), ("take-02", "cafe"))):
        row = {
            "id": take_id,
            "take_id": take_id,
            "duration": TAKE_SECONDS,
            "scene_id": scene,
            "seed": 11 + idx,
            "h3_prompt": f"prompt for {take_id}",
            "keyframe_source": "qc_keyframe" if idx == 0 else "prior_last_frame",
        }
        if i2v:
            row["first_frame_path"] = str(_keyframe(tmp_path, f"kf-{take_id}"))
        rows.append(row)
    return rows


def _dests(tmp_path: Path, takes: list[dict]) -> dict[str, Path]:
    return {t["take_id"]: tmp_path / "out" / f"{t['take_id']}.mp4" for t in takes}


def test_pipeline_is_constructed_and_set_up_exactly_once(longlive_env, tmp_path):
    takes = _takes(tmp_path)
    dests = _dests(tmp_path, takes)

    out = submit_longlive_batch(takes, dests)

    log = longlive_env
    assert len(log["constructed"]) == 1, "pipeline must be constructed once per batch"
    assert len(log["setup"]) == 1, "setup_nvfp4_pipeline must run once"
    assert log["setup"][0]["pipeline_id"] == log["constructed"][0]
    assert out["model_load_count"] == 1
    assert out["inference_calls"] == 2


def test_every_take_samples_with_the_same_resident_pipeline(longlive_env, tmp_path):
    takes = _takes(tmp_path)
    dests = _dests(tmp_path, takes)

    submit_longlive_batch(takes, dests)

    calls = longlive_env["inference"]
    assert len(calls) == 2, "each take must be a real pipeline.inference call"
    assert {c["pipeline_id"] for c in calls} == {longlive_env["constructed"][0]}
    assert calls[0]["prompt"].startswith("prompt for take-01")
    assert calls[1]["prompt"].startswith("prompt for take-02")
    # inference.py clears the VAE cache between samples.
    assert longlive_env["clear_cache"] == 2
    assert longlive_env["seeds"] == [11, 12]


def test_outputs_map_to_the_requested_dest_not_the_newest_file(longlive_env, tmp_path):
    takes = _takes(tmp_path)
    dests = _dests(tmp_path, takes)

    out = submit_longlive_batch(takes, dests)

    saved = [item["path"] for item in longlive_env["saved"]]
    assert saved == [str(dests["take-01"]), str(dests["take-02"])]
    assert out["outputs"]["take-01"] == str(dests["take-01"])
    assert out["outputs"]["take-02"] == str(dests["take-02"])
    for dest in dests.values():
        assert dest.is_file() and dest.stat().st_size >= 32
    marker = json.loads((dests["take-01"].parent / "longlive_batch.json").read_text(encoding="utf-8"))
    assert marker["model_load_count"] == 1
    assert marker["inference_calls"] == 2
    assert marker["take_ids"] == ["take-01", "take-02"]


def test_batch_never_spawns_inference_py(longlive_env, tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError(f"production batch must not spawn a subprocess: {args!r}")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(subprocess, "Popen", explode)
    monkeypatch.setattr(subprocess, "check_call", explode)
    monkeypatch.setattr(subprocess, "check_output", explode)

    takes = _takes(tmp_path)
    submit_longlive_batch(takes, _dests(tmp_path, takes))

    for module in (longlive_batch, longlive_pipeline):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        assert "subprocess" not in _imported_modules(tree), module.__name__
        literals = _code_strings(tree)
        assert not any("inference.py" in text for text in literals), module.__name__


def test_i2v_uses_official_image_prompt_dataset_and_clean_first_latent(longlive_env, tmp_path):
    takes = _takes(tmp_path)
    submit_longlive_batch(takes, _dests(tmp_path, takes))

    datasets = longlive_env["datasets"]
    assert len(datasets) == 2
    for entry in datasets:
        # 44 x 80 latents at spatial ratio 16 -> 704 x 1280, per docs/getting_started.md.
        assert entry["image_size"] == (704, 1280)
        assert entry["num_blocks"] == EXPECTED_BLOCKS
        assert entry["images"] == ["0.png"]
        staged = Path(entry["data_path"])
        assert (staged / "images" / "0.png").is_file()
        assert (staged / "prompts" / "0.txt").read_text(encoding="utf-8").strip()
        assert not (staged / "video").exists(), "video/ would select MultiVideoConcatDataset"

    # inference.py: [B,C,H,W] -> unsqueeze(2) -> encode_to_latent.
    assert longlive_env["encode_to_latent"] == [(1, 3, 1, 704, 1280)] * 2
    for call in longlive_env["inference"]:
        assert call["initial_latent"] == (1, 1, 48, 44, 80)
        assert call["noise"] == (1, EXPECTED_FRAMES, 48, 44, 80)
        assert call["blocks"] == EXPECTED_BLOCKS


def test_t2v_batch_uses_official_single_prompt_inputs(longlive_env, tmp_path):
    takes = _takes(tmp_path, i2v=False)
    for take in takes:
        take["keyframe_source"] = ""
    submit_longlive_batch(takes, _dests(tmp_path, takes))

    assert "datasets" not in longlive_env, "t2v must not build an image dataset"
    for call in longlive_env["inference"]:
        assert call["initial_latent"] is None
        assert call["noise"] == (1, EXPECTED_FRAMES, 48, 44, 80)


def test_mixed_mode_batch_fails_closed(longlive_env, tmp_path):
    takes = _takes(tmp_path)
    takes[1].pop("first_frame_path")
    takes[1]["keyframe_source"] = ""

    with pytest.raises(RuntimeError, match="mixes i2v and t2v"):
        submit_longlive_batch(takes, _dests(tmp_path, takes))
    assert "constructed" not in longlive_env


def test_missing_continuation_frame_fails_closed_instead_of_downgrading(longlive_env, tmp_path):
    takes = _takes(tmp_path)
    # take-02 is planned i2v (prior_last_frame) but its still never materialised.
    takes[1].pop("first_frame_path")

    with pytest.raises(RuntimeError, match="refusing to downgrade"):
        submit_longlive_batch(takes, _dests(tmp_path, takes))
    assert len(longlive_env["constructed"]) == 1
    assert len(longlive_env["inference"]) == 1


def test_missing_official_helper_fails_closed(longlive_env, tmp_path, monkeypatch):
    import utils.inference_utils as stub  # type: ignore[import-not-found]

    monkeypatch.delattr(stub, "setup_nvfp4_pipeline")
    takes = _takes(tmp_path)

    with pytest.raises(RuntimeError, match="official LongLive inference API unavailable"):
        submit_longlive_batch(takes, _dests(tmp_path, takes))


def test_class_import_is_not_counted_as_a_load(longlive_env, tmp_path, monkeypatch):
    """The prior bug returned the pipeline CLASS and still reported model_load_count==1."""
    klass = type("CausalDiffusionInferencePipeline", (), {"inference": lambda self: None})
    monkeypatch.setattr(
        longlive_pipeline, "build_pipeline", lambda *_a, **_k: (klass, "device(cuda)")
    )
    runner = LongLiveBatchRunner()

    with pytest.raises(RuntimeError, match="expected an initialized"):
        runner.load_model(takes=_takes(tmp_path), workdir=tmp_path / "batch")
    assert runner.model_load_count == 0
    assert runner.pipeline is None


def test_skip_weights_never_yields_placeholder_media(longlive_env, tmp_path, monkeypatch):
    monkeypatch.setenv("AF_SKIP_WEIGHTS", "1")
    takes = _takes(tmp_path)
    dests = _dests(tmp_path, takes)

    with pytest.raises(RuntimeError, match=FAIL_CLOSED):
        submit_longlive_batch(takes, dests)
    assert not any(dest.exists() for dest in dests.values())


def test_config_load_rejects_wrong_sampling_steps(longlive_env, tmp_path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_NVFP4_SAMPLING_STEPS", "4")
    yaml_path = longlive.write_inference_yaml(
        tmp_path / "inference.yaml",
        data_path=tmp_path / "data",
        output_folder=tmp_path / "out",
        seconds=TAKE_SECONDS,
        seed=11,
        i2v=True,
    )
    api = longlive_pipeline.import_official_api()
    monkeypatch.setenv("LONGLIVE_NVFP4_SAMPLING_STEPS", "2")
    with pytest.raises(RuntimeError, match="sampling_steps=2"):
        longlive_pipeline.load_official_config(api, yaml_path)


def test_written_yaml_round_trips_through_normalize_config(longlive_env, tmp_path):
    yaml_path = longlive.write_inference_yaml(
        tmp_path / "inference.yaml",
        data_path=tmp_path / "data",
        output_folder=tmp_path / "out",
        seconds=TAKE_SECONDS,
        seed=7,
        i2v=True,
    )
    api = longlive_pipeline.import_official_api()
    config = longlive_pipeline.load_official_config(api, yaml_path)

    assert config.sampling_steps == 2
    assert config.independent_first_frame is True
    assert config.i2v is True
    assert config.seed == 7
    assert config.model_quant is True
    assert config.model_quant_use_transformer_engine is False
    assert config.num_frame_per_block == 8
    assert list(config.image_or_video_shape) == [1, EXPECTED_FRAMES, 48, 44, 80]
    assert config.model_kwargs.model_name == "Wan2.2-TI2V-5B"

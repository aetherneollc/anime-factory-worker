"""LongLive 2.0 image/backend contract, packing, one-load batch, and teardown."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.dockerfile_parse import parse_dockerfile
from gpu_worker.images import resolve_capability_profile
from gpu_worker.stack_contracts import LONGLIVE_STACK, parse_pins_env
from anime_factory.compose import SCALE_PAD_FILTER
from anime_factory.models import VIDEO_HEIGHT, VIDEO_WIDTH
from anime_factory.longlive_workflow import (
    LongLiveWorkflowError,
    load_longlive_short_workflow,
    pack_short_takes,
    validate_longlive_short_workflow,
    validate_packed_takes,
)
from anime_factory.video_backend import (
    CAPABILITY_MISMATCH,
    VideoBackendLockError,
    lock_video_backend,
)
from gpu_worker.boot import assert_lease_capabilities
from gpu_worker.images import image_capabilities, image_can_lease
from gpu_worker.longlive import FAIL_CLOSED, LONGLIVE_NATIVE_HEIGHT, LONGLIVE_NATIVE_WIDTH, LONGLIVE_REF
from gpu_worker.longlive_batch import LongLiveBatchRunner, submit_longlive_batch
from gpu_worker.session import BatchBudget, CpuPostQueue, LeaseRuntime
from gpu_worker.weights import H3_FILES, ensure_h3_weights, h3_files_for_modes


def test_image_backend_capability_mismatch_fails_closed(monkeypatch):
    monkeypatch.setenv("AF_IMAGE_CAPABILITY", "h3")
    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    with pytest.raises(VideoBackendLockError, match=CAPABILITY_MISMATCH):
        lock_video_backend()


def test_longlive_image_refuses_h3_backend(monkeypatch):
    monkeypatch.setenv("AF_IMAGE_CAPABILITY", "longlive")
    monkeypatch.setenv("AF_VIDEO_BACKEND", "h3")
    with pytest.raises(VideoBackendLockError, match=CAPABILITY_MISMATCH):
        lock_video_backend()


def test_longlive_image_can_lease_with_h3_false():
    caps = image_capabilities("docker.io/aetherneo/anime-factory-gpu-longlive")
    assert caps["longlive"] is True
    assert caps["h3"] is False
    assert caps["comfy"] is True
    assert image_can_lease(caps) is True
    assert_lease_capabilities(caps)


def test_zero_h3_files_on_longlive_path(tmp_path, monkeypatch):
    monkeypatch.setenv("AF_IMAGE_CAPABILITY", "longlive")
    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    lock_video_backend()
    pulled = []
    monkeypatch.setattr("gpu_worker.weights._hf_file", lambda *a, **k: pulled.append(a))
    with pytest.raises(RuntimeError, match="refuse H3"):
        ensure_h3_weights(tmp_path)
    assert pulled == []
    assert not any("minimax_h3" in p.name for p in tmp_path.rglob("*"))
    assert not any("audio_vae" in item["dest"] for item in H3_FILES)


def test_h3_dits_are_lazy_by_board_mode():
    core = h3_files_for_modes(set())
    assert any("video_vae" in item["dest"] for item in core)
    assert not any("fl2va" in item["dest"] for item in core)
    assert not any("ref2va" in item["dest"] for item in core)
    fl = h3_files_for_modes({"fl2va_first"})
    assert any("fl2va" in item["dest"] for item in fl)
    assert not any("ref2va" in item["dest"] for item in fl)
    ref = h3_files_for_modes({"ref2va"})
    assert any("ref2va" in item["dest"] for item in ref)


def test_longlive_short_workflow_schema_and_silent_native():
    data = load_longlive_short_workflow()
    validate_longlive_short_workflow(data)
    assert data["silent"] is True
    assert data["audio"] is False
    # Continuation is i2v from the prior last frame; no KV/latent carry-over claimed.
    assert data["continuation_mechanism"] == "i2v_first_frame"
    with pytest.raises(LongLiveWorkflowError, match="continuation_mechanism"):
        validate_longlive_short_workflow({**data, "continuation_mechanism": "kv_cache"})
    assert data["native_width"] == LONGLIVE_NATIVE_WIDTH == 1280
    assert data["native_height"] == LONGLIVE_NATIVE_HEIGHT == 704
    assert data["compose"]["scale_pad_only"] is True
    assert data["compose"]["delivery_width"] == VIDEO_WIDTH
    assert data["compose"]["delivery_height"] == VIDEO_HEIGHT
    assert "force_original_aspect_ratio=decrease" in SCALE_PAD_FILTER
    assert f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}" in SCALE_PAD_FILTER


def test_longlive_pack_scene_boundaries_two_to_three_takes():
    shots = [
        {"id": "s1", "duration": 40, "scene_id": "cafe", "character_id": "hero", "h3_prompt": "enter"},
        {"id": "s2", "duration": 40, "scene_id": "cafe", "character_id": "hero", "h3_prompt": "talk"},
        {"id": "s3", "duration": 40, "scene_id": "street", "character_id": "hero", "h3_prompt": "leave"},
    ]
    takes = pack_short_takes(shots)
    assert 2 <= len(takes) <= 3
    assert all(float(t["duration"]) <= 60 for t in takes)
    assert takes[0]["keyframe_source"] == "qc_keyframe"
    cafe = [t for t in takes if t.get("scene_id") == "cafe"]
    if len(cafe) > 1:
        assert cafe[1]["keyframe_source"] == "prior_last_frame"
        assert cafe[1]["continue_from"] == cafe[0]["take_id"]
    street = [t for t in takes if t.get("scene_id") == "street"]
    assert street
    assert street[0]["keyframe_source"] == "new_keyframe_hard_cut"
    for take in takes:
        assert take["take_id"]
        assert take["end_frame"] > take["start_frame"]
        assert "cast" in take
        assert "prompt" in take


def test_scene_change_must_hard_cut():
    with pytest.raises(LongLiveWorkflowError, match="new_keyframe_hard_cut"):
        validate_packed_takes(
            [
                {
                    "take_id": "take-01",
                    "duration": 40,
                    "scene_id": "a",
                    "keyframe_source": "qc_keyframe",
                },
                {
                    "take_id": "take-02",
                    "duration": 40,
                    "scene_id": "b",
                    "keyframe_source": "prior_last_frame",
                },
            ]
        )


def _stub_loader(monkeypatch, mode: str = "i2v") -> dict[str, int]:
    """Replace only the model build; staging, mapping and fail-closed stay real."""
    loads = {"n": 0}

    def load(self, *, takes, workdir):
        if self.pipeline is None:
            loads["n"] += 1
            self.pipeline = object()
            self.mode = mode
            self.batch_dir = Path(workdir)
            self.model_load_count += 1
        return self.pipeline

    monkeypatch.setattr(LongLiveBatchRunner, "load_model", load)
    return loads


def test_one_model_load_maps_outputs_by_take_id(tmp_path, monkeypatch):
    loads = _stub_loader(monkeypatch)

    def infer(*, take, dest, **_):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 64 + b"ftyp")
        return dest

    (tmp_path / "a.png").write_bytes(b"\x89PNG" + b"x" * 64)
    (tmp_path / "b.png").write_bytes(b"\x89PNG" + b"x" * 64)
    takes = [
        {"take_id": "take-01", "duration": 8, "h3_prompt": "one", "first_frame_path": str(tmp_path / "a.png")},
        {"take_id": "take-02", "duration": 8, "h3_prompt": "two", "first_frame_path": str(tmp_path / "b.png")},
    ]
    dests = {
        "take-01": tmp_path / "take-01.mp4",
        "take-02": tmp_path / "take-02.mp4",
    }
    runner = LongLiveBatchRunner()
    out = submit_longlive_batch(takes, dests, infer=infer, runner=runner)
    assert out["model_load_count"] == 1
    assert loads["n"] == 1
    assert out["h3_downloads"] == 0
    assert out["native_audio"] is False
    assert runner.pipeline is None  # release VRAM before an isolated MOSS-SFX process
    assert out["outputs"]["take-01"].endswith("take-01.mp4")
    assert out["outputs"]["take-02"].endswith("take-02.mp4")
    marker = json.loads((tmp_path / "longlive_batch.json").read_text(encoding="utf-8"))
    assert marker["model_load_count"] == 1
    assert marker["h3_downloads"] == 0


def test_batch_rejects_ambiguous_dest_mapping(tmp_path, monkeypatch):
    _stub_loader(monkeypatch)
    shared = tmp_path / "same.mp4"
    with pytest.raises(RuntimeError, match="ambiguous"):
        submit_longlive_batch(
            [
                {"take_id": "take-01", "duration": 8, "h3_prompt": "a"},
                {"take_id": "take-02", "duration": 8, "h3_prompt": "b"},
            ],
            {"take-01": shared, "take-02": shared},
            infer=lambda **_: shared,
        )


def test_batch_fail_closed_without_placeholder(tmp_path, monkeypatch):
    _stub_loader(monkeypatch, mode="t2v")

    def infer(**_):
        return None

    dest = tmp_path / "missing.mp4"
    with pytest.raises(RuntimeError, match=FAIL_CLOSED):
        submit_longlive_batch(
            [{"take_id": "take-01", "duration": 8, "h3_prompt": "x"}],
            {"take-01": dest},
            infer=infer,
        )
    assert not dest.exists()


def test_run_anim_longlive_one_load_and_native_dims(tmp_path, monkeypatch):
    from gpu_worker import session

    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    monkeypatch.setenv("AF_IMAGE_CAPABILITY", "longlive")
    frame = tmp_path / "f1.png"
    frame.write_bytes(b"\x89PNG" + b"x" * 64)
    shots = [
        {
            "id": "take-01",
            "take_id": "take-01",
            "duration": 8,
            "scene_id": "cafe",
            "first_frame_path": str(frame),
            "h3_prompt": "walk",
            "keyframe_source": "qc_keyframe",
        },
        {
            "id": "take-02",
            "take_id": "take-02",
            "duration": 8,
            "scene_id": "street",
            "first_frame_path": str(frame),
            "h3_prompt": "cut",
            "keyframe_source": "new_keyframe_hard_cut",
        },
    ]
    monkeypatch.setattr(session, "_board_shots", lambda _root: shots)

    def load(self, *, takes, workdir):
        if self.pipeline is None:
            self.pipeline = object()
            self.mode = "i2v"
            self.batch_dir = Path(workdir)
            self.model_load_count += 1
        return self.pipeline

    monkeypatch.setattr("gpu_worker.longlive_batch.LongLiveBatchRunner.load_model", load)
    monkeypatch.setattr(session, "stop_comfy_for_longlive", lambda: {"ok": True})

    loads = {"n": 0}

    def infer(**kwargs):
        loads["n"] += 1
        dest = Path(kwargs["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 64 + b"ftyp")
        return dest

    monkeypatch.setattr(session, "_longlive_infer_hook", infer)
    monkeypatch.setattr(session, "_keep_native_strip_audio", lambda *_a, **_k: None)
    monkeypatch.setattr(
        session,
        "_probe_video",
        lambda _path: {
            "width": 1280,
            "height": 704,
            "duration": 8,
            "frames": 192,
            "size_bytes": 80,
            "audio_streams": 0,
        },
    )
    monkeypatch.setattr(session, "incremental_qc_segment", lambda *_a, **_k: "pass")
    monkeypatch.setattr(session, "mark_completed_passing", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "record_generation_result", lambda *_a, **_k: None)
    monkeypatch.setattr(session, "checkpoint_and_upload", lambda _conn, dest: Path(dest))
    monkeypatch.setattr(session, "put_file", lambda *_a, **_k: {"ok": True})

    def fake_extract(_video, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x89PNG" + b"x" * 64)
        return dest

    monkeypatch.setattr(session, "extract_last_frame", fake_extract)
    scaled = []
    monkeypatch.setattr(
        session,
        "_normalize_h3_for_qc",
        lambda *a, **k: scaled.append(a) or (_ for _ in ()).throw(AssertionError("must not scale native")),
    )

    out = session.run_anim("story-ll", tmp_path, object(), router=None)
    assert out["failed"] == []
    assert set(out["generated"]) == {"take-01", "take-02"}
    assert out["model_load_count"] == 1
    assert out["native_audio"] is False
    assert out["resolution"]["width"] == 1280
    assert out["resolution"]["height"] == 704
    assert out["resolution"]["scale_pad_only"] is True
    assert loads["n"] == 2
    assert scaled == []
    assert out["mapped"]["take-01"]
    assert out["mapped"]["take-02"]


def test_post_checkpoint_idle_uses_seconds_and_queue(monkeypatch):
    monkeypatch.delenv("VAST_IDLE_SECONDS", raising=False)
    now = [0.0]
    runtime = LeaseRuntime(
        instance_id="1",
        started_at=0.0,
        idle_since=0.0,
        clock=lambda: now[0],
    )
    runtime.budget = BatchBudget(idle_minutes=15, post_checkpoint_idle_seconds=30)
    runtime.phase = "production"
    runtime.mark_idle()
    runtime.record_checkpoint()
    now[0] = 29.0
    assert not runtime.idle_expired(comfy_inflight=0, pending_jobs=0)
    now[0] = 30.0
    assert runtime.idle_expired(comfy_inflight=0, pending_jobs=0)
    assert not runtime.idle_expired(comfy_inflight=0, pending_jobs=1)
    assert not runtime.idle_expired(comfy_inflight=1, pending_jobs=0)


def test_cpu_post_queue_drains_after_gpu():
    order = []
    q = CpuPostQueue()
    q.submit(lambda: order.append("cpu") or "ok")
    order.append("gpu")
    assert q.drain() == ["ok"]
    assert order == ["cpu", "gpu"]
    q.close()


def test_clone_pins_nvlabs_commit():
    assert LONGLIVE_REF == "6b36d20ec6f7958d29d11a704dfa64611a9f2572"
    src = (Path(__file__).resolve().parents[1] / "python" / "gpu_worker" / "longlive.py").read_text(
        encoding="utf-8"
    )
    clone = src.split("def _clone_longlive_repo")[1].split("\ndef ")[0]
    assert "LONGLIVE_REF" in clone
    assert "--branch" not in clone
    assert "fetch" in clone
    assert "--depth" in clone


def test_dockerfile_longlive_contract():
    root = Path(__file__).resolve().parents[1]
    deploy = root / "deploy" / "gpu-worker"
    pins = parse_pins_env(deploy / "pins.env")
    text = (deploy / "Dockerfile.longlive").read_text(encoding="utf-8")
    assert pins["LONGLIVE_PYTORCH_IMAGE"] in text
    assert "devel-ubuntu" not in text
    assert "12.8.1-devel-ubuntu24.04" not in text
    assert text.count("FROM nvidia/cuda") == 0
    assert "ARG MOSS_IMAGE=" in text
    assert "ARG KV_DEQUANT_IMAGE=" in text
    assert "FROM ${MOSS_IMAGE} AS moss-sfx" in text
    assert "FROM ${KV_DEQUANT_IMAGE} AS kv-dequant" in text
    assert "6b36d20ec6f7958d29d11a704dfa64611a9f2572" in text
    assert "AF_IMAGE_CAPABILITY=longlive" in text
    assert 'org.aetherneo.anime-factory.h3="false"' in text
    assert 'org.aetherneo.anime-factory.longlive="true"' in text
    assert "fouroversix" in text
    assert "flash_attn" in text
    assert "setup.py build_ext --inplace" not in text
    assert "setup.py bdist_wheel" not in text
    assert "minimax_h3" not in text.lower()
    assert "huggingface-cli download" not in text
    assert "model_4o6.pt" in text  # path env, not a baked COPY
    assert "COPY --from=nvfp4" not in text
    assert "COPY --from=kv-dequant /opt/longlive-wheels /opt/longlive-wheels" in text
    assert "COPY --from=moss-sfx /opt/moss-sfx /opt/moss-sfx" in text
    assert "MOSS_SFX_PYTHON=/opt/moss-sfx/bin/python3.12" in text
    assert "pip install --no-cache-dir --index-url https://download.pytorch.org/whl" not in text
    targets = json.loads((root / "deploy" / "docker-targets.json").read_text(encoding="utf-8"))
    ll = next(t for t in targets["targets"] if t["id"] == "longlive")
    assert ll["enabled"] is False
    assert "h3" not in ll["capabilities"]
    assert "longlive" in ll["capabilities"]
    h3 = next(t for t in targets["targets"] if t["id"] == "h3")
    assert "longlive" not in h3["capabilities"]


def test_longlive_profile_matches_pins_and_dockerfile():
    root = Path(__file__).resolve().parents[1]
    deploy = root / "deploy" / "gpu-worker"
    pins = parse_pins_env(deploy / "pins.env")
    profile = resolve_capability_profile("longlive-nvfp4-sm120")
    assert profile.expected_torch == pins["LONGLIVE_TORCH"]
    assert profile.expected_torchvision == pins["LONGLIVE_TORCHVISION"]
    assert profile.expected_torchaudio == pins["LONGLIVE_TORCHAUDIO"]
    assert profile.expected_flash_attn == pins["FLASH_ATTN_VERSION"]
    assert profile.expected_torch == LONGLIVE_STACK.torch
    text = (deploy / "Dockerfile.longlive").read_text(encoding="utf-8")
    assert f"ARG LONGLIVE_TORCH={pins['LONGLIVE_TORCH'].split('+')[0]}" in text
    assert f"ARG LONGLIVE_TORCHVISION={pins['LONGLIVE_TORCHVISION'].split('+')[0]}" in text
    assert f"ARG LONGLIVE_TORCHAUDIO={pins['LONGLIVE_TORCHAUDIO'].split('+')[0]}" in text


def test_dockerfile_longlive_kv_dequant_sm120a_patch_contract():
    root = Path(__file__).resolve().parents[1]
    deploy = root / "deploy" / "gpu-worker"
    pins = parse_pins_env(deploy / "pins.env")
    patch = (deploy / "patches" / "kv_dequant_sm120a.patch").read_text(encoding="utf-8")
    assert "compute_100a,code=sm_100a" in patch
    assert "compute_120a,code=sm_120a" in patch
    text = (deploy / "Dockerfile.kv-dequant").read_text(encoding="utf-8")
    assert pins["LONGLIVE_PYTORCH_DEVEL_IMAGE"] in text
    assert "kv_dequant_sm120a.patch" in text
    assert "patch failed closed" in text
    assert "compute_120a,code=sm_120a" in text
    assert "kv_dequant gencode contract violated" in text
    assert "setup.py bdist_wheel" in text
    assert "FROM scratch" in text
    longlive = (deploy / "Dockerfile.longlive").read_text(encoding="utf-8")
    assert "kv_dequant_sm120a.patch" not in longlive
    assert "setup.py bdist_wheel" not in longlive
    assert "setup.py build_ext" not in longlive


def test_dockerfile_longlive_uses_pinned_release_wheels():
    root = Path(__file__).resolve().parents[1]
    deploy = root / "deploy" / "gpu-worker"
    text = (deploy / "Dockerfile.longlive").read_text(encoding="utf-8")
    pins = dict(
        line.split("=", 1)
        for line in (deploy / "pins.env").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    fetch = (deploy / "fetch_pinned_wheel.sh").read_text(encoding="utf-8")
    assert "sha256sum --check --strict" in fetch
    assert "urllib.parse.unquote" in fetch
    assert "not a valid wheel filename" in fetch
    assert "FORCE_BUILD=1" not in text
    assert "FLASH_ATTENTION_FORCE_BUILD" not in text
    assert "Dao-AILab/flash-attention.git" not in text
    assert "NVIDIA/cutlass.git" not in text
    assert pins["FOUROVERSIX_WHEEL_SHA256"] in text
    assert pins["FLASH_ATTN_WHEEL_SHA256"] in text
    assert "fetch_pinned_wheel.sh" in text
    assert "/opt/longlive-wheels/fouroversix.whl" not in text
    assert "/opt/longlive-wheels/flash_attn.whl" not in text
    assert "pip install --no-cache-dir --no-deps /opt/longlive-wheels/*.whl" in text
    assert "importlib.metadata" in text


def test_dockerfile_longlive_runtime_skips_cuda_extension_import():
    text = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "gpu-worker"
        / "Dockerfile.longlive"
    ).read_text(encoding="utf-8")
    runtime = text.rsplit("FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime", 1)[1]
    fail_closed = runtime.split("longlive_fail_closed", 1)[0]
    assert "import flash_attn" not in fail_closed
    assert "from fouroversix import _C" not in fail_closed
    assert "importlib.metadata" in fail_closed
    assert "dist_has_so" in fail_closed


def test_dockerfile_longlive_matches_official_nvfp4_stack():
    """docs/getting_started.md "NVFP4 Environment" + the NVFP4-S2 model card."""
    root = Path(__file__).resolve().parents[1]
    deploy = root / "deploy" / "gpu-worker"
    text = (deploy / "Dockerfile.longlive").read_text(encoding="utf-8")
    pins = dict(
        line.split("=", 1)
        for line in (deploy / "pins.env").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    assert pins["LONGLIVE_TORCH"] == "2.10.0+cu128"
    assert pins["LONGLIVE_TORCHVISION"] == "0.25.0+cu128"
    assert pins["LONGLIVE_TORCHAO"] == "0.16.0"
    assert pins["FLASH_ATTN_VERSION"] == "2.8.3"
    for arg in (
        "ARG LONGLIVE_TORCH=2.10.0",
        "ARG LONGLIVE_TORCHVISION=0.25.0",
        "ARG LONGLIVE_TORCHAO=0.16.0",
        f"ARG FLASH_ATTN_WHEEL_SHA256={pins['FLASH_ATTN_WHEEL_SHA256']}",
        f"ARG FOUROVERSIX_WHEEL_SHA256={pins['FOUROVERSIX_WHEEL_SHA256']}",
    ):
        assert arg in text, arg
    assert pins["FLASH_ATTN_WHEEL_URL"] in text
    assert pins["FOUROVERSIX_WHEEL_URL"] in text
    assert "my-pytorch-builds" not in text
    assert "FORCE_BUILD=1" not in text
    # Runtime stage must stay build-tool free and version-locked.
    assert 'shutil.which("nvcc")' in text
    assert "longlive_fail_closed" in text
    assert "LONGLIVE_NVFP4_SAMPLING_STEPS=2" in text
    assert pins["H3_PYTORCH_IMAGE"].split("@", 1)[0] not in text


def test_dockerfiles_parse_as_buildkit_instructions():
    """Heredoc bodies must not leave a bare `&&` continuation as a new instruction."""
    root = Path(__file__).resolve().parents[1]
    for name in ("Dockerfile", "Dockerfile.longlive", "Dockerfile.moss", "Dockerfile.kv-dequant"):
        path = root / "deploy" / "gpu-worker" / name
        instructions, errors = parse_dockerfile(path)
        assert not errors, f"{name}: {errors}"
        assert instructions, name
        assert instructions[0] in {"FROM", "ARG"}, name


def test_dockerfile_longlive_has_no_trailing_whitespace():
    root = Path(__file__).resolve().parents[1]
    for name in ("Dockerfile", "Dockerfile.longlive", "Dockerfile.moss", "Dockerfile.kv-dequant"):
        text = (root / "deploy" / "gpu-worker" / name).read_text(encoding="utf-8")
        offenders = [i + 1 for i, line in enumerate(text.splitlines()) if line != line.rstrip()]
        assert not offenders, f"{name} trailing whitespace on lines {offenders}"
        assert text.endswith("\n")


def test_inference_config_matches_official_nvfp4_s2_keys(tmp_path, monkeypatch):
    from gpu_worker import longlive

    monkeypatch.setenv("LONGLIVE_GENERATOR_CKPT", str(tmp_path / "model_4o6.pt"))
    monkeypatch.delenv("LONGLIVE_LORA_CKPT", raising=False)
    config = longlive.build_inference_config(
        data_path=tmp_path / "data",
        output_folder=tmp_path / "out",
        seconds=8.0,
        seed=11,
        i2v=True,
    )
    # Every top-level key must be one normalize_config actually keeps.
    assert set(config) <= longlive.OFFICIAL_TOP_LEVEL_KEYS
    assert config["inference"]["sampling_steps"] == 2  # NVFP4-S2 model card
    assert config["inference"]["independent_first_frame"] is True
    assert config["i2v"] is True
    assert config["model_quant"] is True
    assert config["model_quant_use_transformer_engine"] is False
    assert config["merge_lora"] is False
    assert config["torch_compile"] is False
    assert config["data"]["image_or_video_shape"] == [1, 48, 48, 44, 80]
    assert config["num_output_frames"] == 48
    assert "adapter" not in config
    # T2V must not claim the i2v conditioning flags.
    t2v = longlive.build_inference_config(
        data_path=tmp_path / "data",
        output_folder=tmp_path / "out",
        seconds=8.0,
        seed=11,
        i2v=False,
    )
    assert "i2v" not in t2v
    assert "independent_first_frame" not in t2v["inference"]


def test_inference_config_rejects_unknown_keys(tmp_path, monkeypatch):
    from gpu_worker import longlive

    monkeypatch.setenv("LONGLIVE_GENERATOR_CKPT", str(tmp_path / "model_4o6.pt"))
    config = longlive.build_inference_config(
        data_path=tmp_path / "data",
        output_folder=tmp_path / "out",
        seconds=8.0,
        seed=11,
        i2v=True,
    )
    with pytest.raises(RuntimeError, match="official runtime ignores"):
        longlive.validate_inference_config({**config, "algorithm_i2v": True})
    with pytest.raises(RuntimeError, match="unknown inference keys"):
        bad = {**config, "inference": {**config["inference"], "denoise_steps": 2}}
        longlive.validate_inference_config(bad)


def test_lora_on_materialized_nvfp4_fails_closed(tmp_path, monkeypatch):
    from gpu_worker import longlive

    monkeypatch.setenv("LONGLIVE_GENERATOR_CKPT", str(tmp_path / "model_4o6.pt"))
    monkeypatch.setenv("LONGLIVE_LORA_CKPT", str(tmp_path / "lora.pt"))
    with pytest.raises(RuntimeError, match="cannot be applied on top of"):
        longlive.build_inference_config(
            data_path=tmp_path / "data",
            output_folder=tmp_path / "out",
            seconds=8.0,
            seed=11,
            i2v=True,
        )


def test_i2v_staging_uses_official_split_layout(tmp_path):
    from gpu_worker import longlive

    frame = tmp_path / "kf.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 96)
    data = longlive.write_prompt_job(
        tmp_path / "data",
        {"id": "take-01", "h3_prompt": "hero walks", "first_frame_path": str(frame)},
        i2v=True,
    )
    assert (data / "images" / "0.png").is_file()
    assert (data / "prompts" / "0.txt").read_text(encoding="utf-8").strip()
    assert not (data / "video").exists()
    assert longlive.assert_i2v_image_layout(data).name == "0.png"
    # A video/ subdirectory would hand inference.py to MultiVideoConcatDataset,
    # which needs 29 RGB frames for the first chunk on Wan 5B.
    (data / "video").mkdir()
    with pytest.raises(RuntimeError, match="MultiVideoConcatDataset"):
        longlive.assert_i2v_image_layout(data)
    assert longlive.i2v_first_chunk_video_frames() == 29


def test_planned_mode_covers_continuation_before_its_frame_exists(tmp_path):
    from gpu_worker import longlive

    first = {"take_id": "take-01", "keyframe_source": "qc_keyframe"}
    cont = {"take_id": "take-02", "keyframe_source": "prior_last_frame"}
    assert longlive.select_longlive_mode(cont) == "t2v"
    assert longlive.planned_longlive_mode(cont) == "i2v"
    frame = tmp_path / "kf.png"
    frame.write_bytes(b"\x89PNG" + b"x" * 64)
    first["first_frame_path"] = str(frame)
    assert longlive.batch_longlive_mode([first, cont]) == "i2v"
    with pytest.raises(RuntimeError, match="mixes i2v and t2v"):
        longlive.batch_longlive_mode([first, {"take_id": "take-03"}])


def test_docker_workflow_reuses_moss_and_pinned_wheels():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "docker.yml").read_text(encoding="utf-8")
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    assert "deploy/gpu-worker/Dockerfile.moss" in workflow
    assert "deploy/gpu-worker/Dockerfile.kv-dequant" in workflow
    assert "MOSS_IMAGE=" in workflow
    assert "KV_DEQUANT_IMAGE=" in workflow
    assert "timeout-minutes: 45" in workflow
    assert "timeout-minutes: 90" not in workflow
    assert "timeout-minutes: 360" not in workflow
    assert "needs: [matrix, moss, kv-dequant]" in workflow
    assert "Dockerfile.moss" in makefile
    assert "Dockerfile.kv-dequant" in makefile
    assert "MOSS_CHECK_IMAGE" in makefile
    assert "KV_DEQUANT_CHECK_IMAGE" in makefile
    assert "--build-arg MOSS_IMAGE=" in makefile
    assert "--build-arg KV_DEQUANT_IMAGE=" in makefile
    assert "nvidia/cuda:12.8.1-runtime-ubuntu24.04" in makefile


def test_incremental_r2_skips_existing(tmp_path, monkeypatch):
    from anime_factory import r2_client

    dest = tmp_path / "board.json"
    dest.write_bytes(b"abc")
    downloaded = []

    monkeypatch.setattr(
        r2_client,
        "list_prefix",
        lambda prefix, max_keys=2000: [
            {"key": "stories/s/board.json", "size": 3},
            {"key": "stories/s/episodes/EP002/board.json", "size": 4},
        ],
    )
    monkeypatch.setattr(
        r2_client,
        "download_file",
        lambda key, dest_path: downloaded.append(key) or dest_path.write_bytes(b"new") or True,
    )
    written = r2_client.download_prefix(
        "stories/s/",
        tmp_path,
        strip_prefix="stories/s/",
        skip_existing=True,
        key_filter=lambda key: "EP002" not in key,
    )
    assert "board.json" in written
    assert downloaded == []

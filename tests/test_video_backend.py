"""Video line selection: MiniMax H3 (default) vs LongLive 2.0."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from types import ModuleType
import sys

import pytest

from anime_factory.directors.common import expand_shots_to_segments
from anime_factory.models import H3_MAX_SECONDS
from anime_factory.video_backend import (
    DEFAULT_VIDEO_BACKEND,
    LONGLIVE_MAX_SECONDS,
    max_seconds_for_backend,
    normalize_video_backend,
    select_video_backend,
    write_factory_backend,
)
from gpu_worker.longlive import (
    FAIL_CLOSED,
    SETUP_HINT,
    GENERATOR_DEFAULT,
    LONGLIVE_CKPT_FILE,
    LONGLIVE_CKPT_REPO,
    LONGLIVE_DEFAULT_LATENT_FRAMES,
    LONGLIVE_NVFP4_SAMPLING_STEPS,
    WAN_INFERENCE_FILES,
    WAN_TRANSFORMER_SHARDS,
    _assert_nvfp4_generator,
    _install_fouroversix,
    _patch_wrapper_t5_lowmem,
    assert_i2v_image_layout,
    ensure_longlive,
    i2v_first_chunk_video_frames,
    is_longlive_fatal_error,
    longlive_num_output_frames,
    longlive_video_seconds,
    missing_longlive_requirements,
    nvfp4_sampling_steps,
    select_longlive_mode,
    submit_longlive,
    write_inference_yaml,
    write_prompt_job,
)
from gpu_worker.session import _board_shots, _prepare_longlive_shot, _submit_video


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch):
    monkeypatch.delenv("AF_VIDEO_BACKEND", raising=False)
    monkeypatch.delenv("AF_VIDEO_BACKEND_LOCKED", raising=False)
    monkeypatch.delenv("AF_IMAGE_CAPABILITY", raising=False)
    monkeypatch.delenv("LONGLIVE_MAX_SECONDS", raising=False)
    monkeypatch.delenv("LONGLIVE_ROOT", raising=False)
    monkeypatch.delenv("LONGLIVE_GENERATOR_CKPT", raising=False)
    monkeypatch.delenv("LONGLIVE_WAN_DIR", raising=False)
    monkeypatch.delenv("LONGLIVE_WHEELS", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)


def test_normalize_defaults_to_h3():
    assert normalize_video_backend(None) == "h3"
    assert normalize_video_backend("") == "h3"
    assert normalize_video_backend("H3") == "h3"
    assert normalize_video_backend("wan") == "h3"
    assert DEFAULT_VIDEO_BACKEND == "h3"


def test_normalize_longlive_aliases():
    assert normalize_video_backend("longlive") == "longlive"
    assert normalize_video_backend("LongLive 2.0") == "longlive"
    assert normalize_video_backend("longlive-2.0-5b") == "longlive"


def test_select_priority_segment_over_file_over_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    write_factory_backend(tmp_path, "h3")
    assert select_video_backend(root=tmp_path) == "h3"
    assert select_video_backend({"video_backend": "longlive"}, root=tmp_path) == "longlive"
    assert select_video_backend(root=tmp_path / "missing") == "longlive"
    monkeypatch.delenv("AF_VIDEO_BACKEND")
    assert select_video_backend() == "h3"


def test_missing_factory_keeps_in_flight_h3(tmp_path: Path):
    """Live 舔狗 shoot has no factory.json — must stay H3."""
    assert not (tmp_path / "factory.json").exists()
    assert select_video_backend(root=tmp_path) == "h3"


def test_max_seconds_longlive_does_not_split_12s(tmp_path: Path):
    write_factory_backend(tmp_path, "longlive")
    assert max_seconds_for_backend("longlive") == LONGLIVE_MAX_SECONDS
    assert max_seconds_for_backend("h3") == H3_MAX_SECONDS
    shots = [{"id": "S01", "duration": 12.0, "scene_id": "sc01"}]
    segs = expand_shots_to_segments(shots, max_s=max_seconds_for_backend("longlive"))
    assert len(segs) == 1
    assert segs[0]["duration"] == 12.0
    h3_segs = expand_shots_to_segments(shots, max_s=H3_MAX_SECONDS)
    assert len(h3_segs) >= 2


def test_max_seconds_longlive_keeps_scene_length_take():
    shots = [{"id": "S01", "duration": 40.0, "scene_id": "sc01", "chain_id": "sc01"}]
    segs = expand_shots_to_segments(shots, max_s=max_seconds_for_backend("longlive"))
    assert len(segs) == 1
    assert segs[0]["duration"] == 40.0
    h3_segs = expand_shots_to_segments(shots, max_s=H3_MAX_SECONDS)
    assert len(h3_segs) >= 5


def test_longlive_i2v_when_first_frame_exists(tmp_path: Path):
    frame = tmp_path / "last.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
    assert select_longlive_mode({"first_frame_path": str(frame)}) == "i2v"
    assert select_longlive_mode({"first_frame_path": "input/"}) == "t2v"
    assert select_longlive_mode({}) == "t2v"


def test_longlive_missing_weights_does_not_fake_mp4(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_ROOT", str(tmp_path / "no-repo"))
    monkeypatch.setenv("LONGLIVE_GENERATOR_CKPT", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("LONGLIVE_WAN_DIR", str(tmp_path / "no-wan"))
    dest = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError, match="longlive not ready") as caught:
        submit_longlive({"id": "s001", "duration": 8}, dest, root=tmp_path)
    assert "LONGLIVE_GENERATOR_CKPT" in str(caught.value)
    assert SETUP_HINT.split()[0] in str(caught.value)
    assert not dest.exists()
    assert missing_longlive_requirements()


def _stub_wan_sidecars(wan: Path) -> None:
    from gpu_worker.longlive import WAN_INFERENCE_FILES

    for rel in WAN_INFERENCE_FILES:
        path = wan / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sidecar" * 8)


def test_longlive_prepare_yaml_and_prompts(tmp_path: Path, monkeypatch):
    repo = tmp_path / "LongLive"
    repo.mkdir()
    (repo / "inference.py").write_text("# stub\n", encoding="utf-8")
    ckpt = tmp_path / "gen.pt"
    ckpt.write_bytes(b"ckpt")
    wan = tmp_path / "wan"
    _stub_wan_sidecars(wan)
    monkeypatch.setenv("LONGLIVE_ROOT", str(repo))
    monkeypatch.setenv("LONGLIVE_GENERATOR_CKPT", str(ckpt))
    monkeypatch.setenv("LONGLIVE_WAN_DIR", str(wan))
    monkeypatch.setattr("gpu_worker.longlive.missing_nvfp4_requirements", lambda: [])
    yaml_path = write_inference_yaml(
        tmp_path / "inf.yaml",
        data_path=tmp_path / "prompts.txt",
        output_folder=tmp_path / "out",
        seconds=8.0,
        seed=7,
        i2v=False,
    )
    text = yaml_path.read_text(encoding="utf-8")
    assert "Wan2.2-TI2V-5B" in text
    assert "wan_model_dir" not in text
    assert "adapter:" not in text
    assert "lora_ckpt" not in text
    assert "sampling_steps: 2" in text
    assert "model_quant: true" in text
    assert "model_quant_use_transformer_engine: false" in text
    assert "kv_quant: true" in text
    assert str(ckpt) in text
    assert longlive_num_output_frames(8.0) % 8 == 0
    assert longlive_num_output_frames(64.0) == 384
    prompts = write_prompt_job(tmp_path / "data", {"h3_prompt": "clerk turns", "duration": 8}, i2v=False)
    assert "clerk turns" in prompts.read_text(encoding="utf-8")
    dry = submit_longlive({"id": "s001", "h3_prompt": "walk", "duration": 8}, tmp_path / "clip.mp4", run=False)
    assert dry["skipped"] == "dry_run"
    assert not (tmp_path / "clip.mp4").exists()


def test_submit_video_dispatches_longlive(tmp_path: Path, monkeypatch):
    seen = {}

    def fake_submit(shot, dest, root=None, progress=None, run=True):
        seen["shot"] = shot["id"]
        seen["dest"] = str(dest)
        dest.write_bytes(b"mp4")
        return {"backend": "longlive", "shot": shot["id"]}

    monkeypatch.setattr("gpu_worker.session.submit_longlive", fake_submit)
    dest = tmp_path / "s001.mp4"
    out = _submit_video(None, {"id": "s001", "duration": 8}, tmp_path, dest, None, "longlive")
    assert out["backend"] == "longlive"
    assert seen["shot"] == "s001"
    assert dest.is_file()


def test_submit_video_h3_requires_router():
    with pytest.raises(RuntimeError, match="no_comfy_router"):
        _submit_video(None, {"id": "s001"}, Path("/tmp"), Path("/tmp/x.mp4"), None, "h3")


def test_generator_default_is_nvfp4_s2():
    assert GENERATOR_DEFAULT.endswith("model_4o6.pt")
    assert "nvfp4_s2" in GENERATOR_DEFAULT
    assert LONGLIVE_CKPT_REPO.endswith("LongLive-2.0-5B-NVFP4-S2")
    assert LONGLIVE_CKPT_FILE == "model_4o6.pt"
    assert nvfp4_sampling_steps() == LONGLIVE_NVFP4_SAMPLING_STEPS == 2


def test_setup_hint_does_not_require_hf_token():
    assert "HF_TOKEN is required" not in SETUP_HINT
    assert "model_4o6.pt" in SETUP_HINT
    assert "NVFP4-S2" in SETUP_HINT
    assert "model_bf16.pt" not in SETUP_HINT
    assert "training repo" in SETUP_HINT


def test_patch_wrapper_skips_original_wan_transformer(tmp_path: Path, monkeypatch):
    from gpu_worker.longlive import _patch_wrapper_skip_wan_init, _wrapper_loads_original_transformer

    repo = tmp_path / "LongLive"
    wrapper = repo / "utils" / "wan_5b_wrapper.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        "if is_causal:\n"
        "        self.model = CausalWanModel.from_pretrained(\n"
        '            f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size,\n'
        "            num_frame_per_block=num_frame_per_block)\n"
        "    else:\n"
        '        self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("LONGLIVE_ROOT", str(repo))
    assert _wrapper_loads_original_transformer()
    assert _patch_wrapper_skip_wan_init()
    text = wrapper.read_text(encoding="utf-8")
    assert "CausalWanModel.from_pretrained" not in text
    assert "CausalWanModel.from_config" in text
    assert "CausalWanModel.load_config" in text
    assert not _wrapper_loads_original_transformer()


def test_patch_inference_mmap_avoids_full_cpu_copy(tmp_path: Path, monkeypatch):
    from gpu_worker.longlive import _patch_inference_mmap_load

    repo = tmp_path / "LongLive"
    repo.mkdir()
    inf = repo / "inference.py"
    inf.write_text(
        'generator_checkpoint = torch.load(generator_ckpt_path, map_location="cpu")\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("LONGLIVE_ROOT", str(repo))
    assert _patch_inference_mmap_load()
    text = inf.read_text(encoding="utf-8")
    assert "mmap=True" in text
    assert _patch_inference_mmap_load()


def test_longlive_sitecustomize_mmaps_cpu_torch_load(tmp_path: Path, monkeypatch):
    from gpu_worker.longlive import _install_torch_mmap_sitecustomize, _longlive_inference_env

    repo = tmp_path / "LongLive"
    repo.mkdir()
    monkeypatch.setenv("LONGLIVE_ROOT", str(repo))
    path = _install_torch_mmap_sitecustomize()
    assert path.is_file()
    assert "mmap" in path.read_text(encoding="utf-8")
    env = _longlive_inference_env()
    assert str(repo) in env["PYTHONPATH"]
    assert env["MALLOC_ARENA_MAX"] == "2"


def test_ensure_longlive_skip_weights(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AF_SKIP_WEIGHTS", "1")
    monkeypatch.setenv("LONGLIVE_ROOT", str(tmp_path / "no-repo"))
    monkeypatch.setenv("LONGLIVE_GENERATOR_CKPT", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("LONGLIVE_WAN_DIR", str(tmp_path / "no-wan"))
    out = ensure_longlive()
    assert out["skipped"] is True
    assert out["missing"]


def test_ensure_longlive_downloads(monkeypatch, tmp_path: Path):
    root = tmp_path / "LongLive"
    ckpt = tmp_path / "model_4o6.pt"
    wan = tmp_path / "wan"
    monkeypatch.setenv("LONGLIVE_ROOT", str(root))
    monkeypatch.setenv("LONGLIVE_GENERATOR_CKPT", str(ckpt))
    monkeypatch.setenv("LONGLIVE_WAN_DIR", str(wan))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["git", "init"]:
            Path(cmd[2]).mkdir(parents=True, exist_ok=True)
            return None
        if cmd and cmd[0] == "git" and "-C" in cmd:
            dest = Path(cmd[cmd.index("-C") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            if "fetch" in cmd or "checkout" in cmd:
                (dest / "inference.py").write_text("# stub\n", encoding="utf-8")
            return None
        if cmd[:2] == ["git", "clone"]:
            dest = Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "inference.py").write_text("# stub\n", encoding="utf-8")
        return None

    def fake_hub_download(**kwargs):
        dest = Path(kwargs["local_dir"]) / kwargs["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ckpt" * 16)
        return str(dest)

    def fake_snapshot(**kwargs):
        raise AssertionError("must not snapshot_download Wan-AI/Wan2.2-TI2V-5B")

    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = fake_hub_download
    hub.snapshot_download = fake_snapshot
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    def fake_nvfp4(progress=None):
        return None

    monkeypatch.setattr("gpu_worker.longlive.subprocess.run", fake_run)
    monkeypatch.setattr("gpu_worker.longlive._install_nvfp4_stack", fake_nvfp4)
    monkeypatch.setattr("gpu_worker.longlive.missing_nvfp4_requirements", lambda: [])
    out = ensure_longlive()
    assert out["ok"] is True
    assert out["nvfp4_sampling_steps"] == 2
    assert ckpt.is_file()
    assert (root / "inference.py").is_file()
    assert (wan / "config.json").is_file()
    assert (wan / "Wan2.2_VAE.pth").is_file()
    assert (wan / "models_t5_umt5-xxl-enc-bf16.pth").is_file()
    assert not (wan / "diffusion_pytorch_model-00001-of-00003.safetensors").exists()
    assert any("fetch" in cmd and "6b36d20ec6f7958d29d11a704dfa64611a9f2572" in cmd for cmd in calls)


def _sheet_png(tag: str = "sheet") -> bytes:
    """A locked sheet has to survive the fail-closed size check."""
    from anime_factory.design import synthetic_still_png

    return synthetic_still_png(1536, 512, tag=tag, placeholder=True)


def _tiny_png() -> bytes:
    # 1×1 RGB PNG so ImagePromptDataset / PIL can open it if a test loads the file.
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x00\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def test_prepare_longlive_shot_uses_locked_sheet(tmp_path: Path):
    sheet = tmp_path / "assets" / "characters" / "hero" / "sheet_front.png"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_bytes(_sheet_png("hero"))
    (tmp_path / "assets" / "index.json").write_text(
        '{"characters":{"hero":{"selected":"sheet_front.png","history":["sheet_front.png"],'
        '"qc_verdict":"pass","qc_version":"visual_qc_v1"}},"scenes":{}}',
        encoding="utf-8",
    )
    prepared = _prepare_longlive_shot({"id": "s001", "character_id": "hero"}, tmp_path)
    staged = tmp_path / "episodes" / "EP001" / "keyframes" / "s001" / "f1.png"
    assert Path(prepared["first_frame_path"]) == staged
    assert staged.read_bytes() == sheet.read_bytes()


def test_prepare_longlive_shot_falls_back_to_plate(tmp_path: Path):
    plate = tmp_path / "assets" / "scenes" / "store" / "plate_base.png"
    plate.parent.mkdir(parents=True, exist_ok=True)
    plate.write_bytes(_sheet_png("store-plate"))
    (tmp_path / "assets" / "index.json").write_text(
        '{"characters":{},"scenes":{"store":{"selected":"plate_base.png","history":["plate_base.png"],'
        '"qc_verdict":"pass","qc_version":"visual_qc_v1"}}}',
        encoding="utf-8",
    )
    prepared = _prepare_longlive_shot({"id": "s001", "plate_id": "plate_store"}, tmp_path)
    staged = tmp_path / "episodes" / "EP001" / "keyframes" / "s001" / "f1.png"
    assert Path(prepared["first_frame_path"]) == staged
    assert staged.read_bytes() == plate.read_bytes()


def test_official_latent_frames_are_about_64s():
    assert LONGLIVE_DEFAULT_LATENT_FRAMES == 384
    assert longlive_num_output_frames(64.0) == 384
    assert abs(longlive_video_seconds(384) - 63.875) < 0.01
    assert set(WAN_TRANSFORMER_SHARDS).isdisjoint(WAN_INFERENCE_FILES)


def test_board_shots_longlive_keeps_40s_take(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    write_factory_backend(tmp_path, "longlive")
    ep = tmp_path / "episodes" / "EP001"
    ep.mkdir(parents=True)
    (ep / "board.json").write_text(
        json.dumps(
            {
                "shots": [
                    {
                        "id": "s001",
                        "duration": 40.0,
                        "scene_id": "sc01",
                        "chain_id": "sc01",
                        "h3_prompt": "clerk at the counter",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    shots = _board_shots(tmp_path)
    assert len(shots) == 1
    assert shots[0]["duration"] == 40.0


def test_board_shots_h3_still_splits_12s(tmp_path: Path):
    write_factory_backend(tmp_path, "h3")
    ep = tmp_path / "episodes" / "EP001"
    ep.mkdir(parents=True)
    (ep / "board.json").write_text(
        json.dumps(
            {
                "shots": [
                    {
                        "id": "s001",
                        "duration": 12.0,
                        "scene_id": "sc01",
                        "chain_id": "sc01",
                        "h3_prompt": "clerk at the counter",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    shots = _board_shots(tmp_path)
    assert len(shots) >= 2
    assert max(float(s["duration"]) for s in shots) <= H3_MAX_SECONDS + 1e-9


def test_i2v_first_chunk_is_29_video_frames_on_wan5b():
    """MultiVideoConcatDataset first chunk — a 1-frame mp4 cannot satisfy this."""
    assert i2v_first_chunk_video_frames() == 29
    assert longlive_num_output_frames(28.19) == 176


def test_write_prompt_job_i2v_uses_image_not_video(tmp_path: Path):
    frame = tmp_path / "f1.png"
    frame.write_bytes(_tiny_png())
    data = write_prompt_job(
        tmp_path / "data",
        {"id": "s001", "h3_prompt": "clerk turns at the counter", "first_frame_path": str(frame)},
        i2v=True,
    )
    assert data == tmp_path / "data"
    assert not (data / "video").exists()
    staged = data / "images" / "0.png"
    prompt = data / "prompts" / "0.txt"
    assert staged.is_file() and staged.stat().st_size >= 32
    assert staged.read_bytes() == frame.read_bytes()
    assert "clerk turns at the counter" in prompt.read_text(encoding="utf-8")
    assert assert_i2v_image_layout(data) == staged


def test_assert_i2v_image_layout_rejects_video_dir(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "0.png").write_bytes(_tiny_png())
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "0.txt").write_text("ok\n", encoding="utf-8")
    (tmp_path / "video" / "shot").mkdir(parents=True)
    (tmp_path / "video" / "shot" / "0.mp4").write_bytes(b"not-enough-frames")
    with pytest.raises(RuntimeError, match="first chunk"):
        assert_i2v_image_layout(tmp_path)


def test_assert_i2v_image_layout_rejects_empty_still(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "0.txt").write_text("ok\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="first chunk empty"):
        assert_i2v_image_layout(tmp_path)


def test_write_prompt_job_i2v_strips_leftover_video_dir(tmp_path: Path):
    leftover = tmp_path / "data" / "video" / "shot"
    leftover.mkdir(parents=True)
    (leftover / "0.mp4").write_bytes(b"stale")
    frame = tmp_path / "f1.png"
    frame.write_bytes(_tiny_png())
    data = write_prompt_job(
        tmp_path / "data",
        {"id": "s001", "h3_prompt": "walk", "first_frame_path": str(frame)},
        i2v=True,
    )
    assert not (data / "video").exists()
    assert_i2v_image_layout(data)


def test_write_inference_yaml_i2v_flags(tmp_path: Path):
    yaml_path = write_inference_yaml(
        tmp_path / "inf.yaml",
        data_path=tmp_path / "data",
        output_folder=tmp_path / "out",
        seconds=28.19,
        seed=11,
        i2v=True,
    )
    text = yaml_path.read_text(encoding="utf-8")
    assert "i2v: true" in text
    assert "independent_first_frame: true" in text
    assert "num_output_frames: 176" in text
    assert "model_quant: true" in text
    assert "sampling_steps: 2" in text


def test_h3_backend_unchanged_by_longlive_defaults():
    assert normalize_video_backend("h3") == "h3"
    assert max_seconds_for_backend("h3") == H3_MAX_SECONDS


def test_dockerfile_h3_only_no_compile():
    root = Path(__file__).resolve().parents[1]
    text = (root / "deploy" / "gpu-worker" / "Dockerfile").read_text(encoding="utf-8")
    pins = dict(
        line.split("=", 1)
        for line in (root / "deploy" / "gpu-worker" / "pins.env").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    assert text.count("FROM nvidia/cuda") == 0
    assert pins["H3_PYTORCH_IMAGE"] in text
    assert "ARG MOSS_IMAGE=" in text
    assert "FROM ${MOSS_IMAGE} AS moss-sfx" in text
    assert "13.0.0-runtime-ubuntu24.04" not in text
    assert "devel-ubuntu" not in text
    assert '"2.13.0+cu130"' in text
    assert '"0.28.0+cu130"' in text
    assert '"2.11.0+cu130"' in text
    assert "KJ_NODES_REF=d3cfe21625e5170126ce06fbfcfe1d88108688c3" in text
    assert "org.aetherneo.anime-factory.h3=\"true\"" in text
    assert "org.aetherneo.anime-factory.comfy=\"true\"" in text
    assert "org.aetherneo.anime-factory.image-gen=\"true\"" in text
    assert "PIP_ONLY_BINARY=:all:" in text
    assert "PYTHONPATH=/app/python" in text
    assert 'Pillow>=10,<12' in text
    assert "huggingface_hub" in text
    assert "boto3" in text
    assert "open_clip_torch==2.32.0" in text
    assert "python -m venv --system-site-packages /opt/venv" in text
    assert "pip install --no-cache-dir --only-binary=:all: --index-url https://download.pytorch.org/whl" not in text
    forbidden = (
        "nvfp4",
        "LongLive",
        "longlive-wheels",
        "fouroversix",
        "flash_attn",
        "flash-attn",
        "pip wheel",
        "setup.py build_ext",
        "python3-dev",
        "ninja-build",
        " AS nvfp4",
        "org.aetherneo.anime-factory.longlive",
        "LONGLIVE_",
        "CUDA_ARCHS",
        "build_ext",
        "no-build-isolation",
    )
    lowered = text.lower()
    for token in forbidden:
        assert token.lower() not in lowered, token
    pip_install_lines = [line for line in text.splitlines() if "pip install" in line]
    assert pip_install_lines, "expected pip install steps in Dockerfile"
    for line in pip_install_lines:
        assert "--only-binary=:all:" in line, line
    assert '"--only-binary=:all:"' in text
    assert "COPY --from=moss-sfx /opt/moss-sfx /opt/moss-sfx" in text
    assert 'shutil.which("nvcc")' in text
    assert "MOSS_SFX_PYTHON=/opt/moss-sfx/bin/python3.12" in text
    assert not re.search(r"^RUN .*pip install -e /opt/LongLive/fouroversix", text, re.M)
    assert not re.search(r"^PY\s*\n\s*&&", text, re.M)
    start = (Path(__file__).resolve().parents[1] / "deploy" / "gpu-worker" / "start.sh").read_text(
        encoding="utf-8"
    )
    assert "/opt/venv/bin" in start
    assert "longlive" not in start.lower()


def test_refuse_bf16_generator(tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "model_bf16.pt"
    ckpt.write_bytes(b"nope")
    monkeypatch.setenv("LONGLIVE_GENERATOR_CKPT", str(ckpt))
    with pytest.raises(RuntimeError, match=FAIL_CLOSED):
        _assert_nvfp4_generator()


def test_patch_wrapper_t5_uses_bf16_cpu(tmp_path: Path, monkeypatch):
    repo = tmp_path / "LongLive"
    wrapper = repo / "utils" / "wan_5b_wrapper.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        "self.text_encoder = umt5_xxl(\n"
        "            encoder_only=True,\n"
        "            return_tokenizer=False,\n"
        "            dtype=torch.float32,\n"
        "            device=torch.device('cpu')\n"
        "        ).eval().requires_grad_(False)\n"
        "        self.text_encoder.load_state_dict(\n"
        '            torch.load("wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",\n'
        "                       map_location='cpu', weights_only=False)\n"
        "        )\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LONGLIVE_ROOT", str(repo))
    assert _patch_wrapper_t5_lowmem()
    text = wrapper.read_text(encoding="utf-8")
    assert "dtype=torch.bfloat16" in text
    assert "device=torch.device('cpu')" in text
    assert "torch.cuda.is_available()" not in text
    assert "mmap=True" in text
    assert "dtype=torch.float32" not in text


def test_patch_wrapper_t5_reverts_cuda_to_cpu(tmp_path: Path, monkeypatch):
    repo = tmp_path / "LongLive"
    wrapper = repo / "utils" / "wan_5b_wrapper.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        "self.text_encoder = umt5_xxl(\n"
        "            encoder_only=True,\n"
        "            return_tokenizer=False,\n"
        "            dtype=torch.bfloat16,\n"
        "            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "        ).eval().requires_grad_(False)\n"
        "        self.text_encoder.load_state_dict(\n"
        '            torch.load("wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",\n'
        '                       map_location="cpu", weights_only=False, mmap=True)\n'
        "        )\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LONGLIVE_ROOT", str(repo))
    assert _patch_wrapper_t5_lowmem()
    text = wrapper.read_text(encoding="utf-8")
    assert "device=torch.device('cpu')" in text
    assert "torch.cuda.is_available()" not in text


def test_fouroversix_skips_when_ready(monkeypatch):
    calls = []
    monkeypatch.setattr("gpu_worker.longlive._fouroversix_ready", lambda: True)
    monkeypatch.setattr(
        "gpu_worker.longlive.subprocess.run",
        lambda *a, **k: calls.append(a[0]) or None,
    )
    _install_fouroversix()
    assert calls == []


def test_fouroversix_fail_closed_without_nvcc(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LONGLIVE_WHEELS", str(tmp_path / "missing-wheels"))
    monkeypatch.setattr("gpu_worker.longlive._fouroversix_ready", lambda: False)
    monkeypatch.setattr("gpu_worker.longlive._nvcc_available", lambda: False)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise AssertionError(f"must not compile fouroversix: {cmd}")

    monkeypatch.setattr("gpu_worker.longlive.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match=FAIL_CLOSED) as caught:
        _install_fouroversix()
    assert "nvcc" in str(caught.value).lower()
    assert calls == []
    assert not any("-e" in str(c) for c in calls)


def test_fouroversix_installs_prebuilt_wheel(tmp_path: Path, monkeypatch):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = wheels / "fouroversix-1.0.5-cp312-cp312-linux_x86_64.whl"
    wheel.write_bytes(b"wheel")
    monkeypatch.setenv("LONGLIVE_WHEELS", str(wheels))
    ready = [False]
    calls = []

    def fake_ready():
        return ready[0]

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        ready[0] = True
        return None

    monkeypatch.setattr("gpu_worker.longlive._fouroversix_ready", fake_ready)
    monkeypatch.setattr("gpu_worker.longlive._nvcc_available", lambda: False)
    monkeypatch.setattr("gpu_worker.longlive.subprocess.run", fake_run)
    _install_fouroversix()
    assert any("fouroversix-1.0.5" in str(part) for cmd in calls for part in cmd)
    assert not any(part == "-e" for cmd in calls for part in cmd)


def test_longlive_fatal_error_matches_sigkill_not_exit1():
    err = subprocess.CalledProcessError(-9, ["python", "inference.py"])
    assert is_longlive_fatal_error(err)
    assert is_longlive_fatal_error("longlive inference failed s001: exit -9")
    assert is_longlive_fatal_error(f"{FAIL_CLOSED}: fouroversix")
    assert is_longlive_fatal_error(
        "OutOfMemoryError:CUDA out of memory. Tried to allocate 1.30 GiB. "
        "GPU 0 has a total capacity of 31.36 GiB of which 1.30 GiB is free."
    )
    assert is_longlive_fatal_error({"id": "take-01", "error": "CUDA out of memory"})
    assert not is_longlive_fatal_error("longlive inference failed s001: exit 1")
    assert not is_longlive_fatal_error(None)

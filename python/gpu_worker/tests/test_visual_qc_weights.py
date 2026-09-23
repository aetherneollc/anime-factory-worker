"""Offline contracts for visual QC OpenCLIP weights and GPU image pinning."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gpu_worker import weights

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO_ROOT / "deploy" / "gpu-worker" / "Dockerfile"
START_SH = REPO_ROOT / "deploy" / "gpu-worker" / "start.sh"


def test_still_files_include_openclip_vit_b32_not_vit_h():
    by_dest = {item["dest"]: item for item in weights.still_weight_files("kolors")}
    clip_qc = by_dest[weights.VISUAL_QC_CLIP_WEIGHT_DEST]
    config_qc = by_dest[weights.VISUAL_QC_CLIP_CONFIG_DEST]
    assert weights.CLIP_VISION_DEST not in by_dest
    clip_ip = {item["dest"]: item for item in weights.still_weight_files("animagine")}[weights.CLIP_VISION_DEST]

    assert clip_qc["repo"] == "timm/vit_base_patch32_clip_224.openai"
    assert clip_qc["hf"] == "open_clip_model.safetensors"
    assert clip_qc["dest"] == "models/visual_qc/open_clip_model.safetensors"
    assert int(clip_qc["min_bytes"]) == weights.VISUAL_QC_CLIP_MIN_BYTES

    assert config_qc["repo"] == "timm/vit_base_patch32_clip_224.openai"
    assert config_qc["hf"] == "open_clip_config.json"
    assert config_qc["dest"] == "models/visual_qc/open_clip_config.json"

    assert clip_ip["dest"] == "models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    assert "visual_qc" not in clip_ip["dest"]
    assert clip_qc["dest"] != clip_ip["dest"]


def test_visual_qc_clip_paths_and_env_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("AF_VISUAL_QC_CLIP_WEIGHTS", raising=False)
    monkeypatch.delenv("AF_VISUAL_QC_CLIP_CONFIG", raising=False)
    monkeypatch.delenv("AF_VISUAL_QC_CLIP_DIR", raising=False)

    paths = weights.visual_qc_clip_paths(tmp_path)
    assert paths["model"] == "ViT-B-32"
    assert paths["pretrained"] == "openai"
    assert paths["weights"] == tmp_path / weights.VISUAL_QC_CLIP_WEIGHT_DEST
    assert paths["config"] == tmp_path / weights.VISUAL_QC_CLIP_CONFIG_DEST

    env = weights.visual_qc_clip_env(tmp_path)
    assert env["AF_OPEN_CLIP_MODEL"] == "ViT-B-32"
    assert env["AF_OPEN_CLIP_PRETRAINED"] == "openai"
    assert env["OPEN_CLIP_TORCH_VERSION"] == weights.OPEN_CLIP_TORCH_VERSION
    assert env["AF_VISUAL_QC_CLIP_WEIGHTS"].endswith("open_clip_model.safetensors")
    assert "clip_vision" not in env["AF_VISUAL_QC_CLIP_WEIGHTS"]


def test_visual_qc_clip_env_respects_overrides(monkeypatch, tmp_path):
    custom = tmp_path / "custom" / "qc.safetensors"
    custom.parent.mkdir(parents=True)
    custom.write_bytes(b"x" * weights.VISUAL_QC_CLIP_MIN_BYTES)
    cfg = tmp_path / "custom" / "open_clip_config.json"
    cfg.write_bytes(b"{" + b"x" * weights.VISUAL_QC_CLIP_CONFIG_MIN_BYTES + b"}")
    monkeypatch.setenv("AF_VISUAL_QC_CLIP_WEIGHTS", str(custom))
    monkeypatch.setenv("AF_VISUAL_QC_CLIP_CONFIG", str(cfg))
    monkeypatch.setenv("AF_VISUAL_QC_CLIP_DIR", str(custom.parent))

    paths = weights.visual_qc_clip_paths()
    assert paths["weights"] == custom
    assert paths["config"] == cfg
    weights.validate_visual_qc_clip_weights()


def test_validate_visual_qc_clip_rejects_truncated_weight(tmp_path):
    dest = tmp_path / weights.VISUAL_QC_CLIP_WEIGHT_DEST
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"truncated")
    cfg = tmp_path / weights.VISUAL_QC_CLIP_CONFIG_DEST
    cfg.write_bytes(b'{"model_cfg":{}}')

    with pytest.raises(RuntimeError, match="truncated"):
        weights.validate_visual_qc_clip_weights(tmp_path)


def test_ensure_still_weights_pulls_openclip_files(tmp_path, monkeypatch):
    pulled: list[str] = []

    def fake_hf(repo, filename, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if filename == weights.VISUAL_QC_CLIP_WEIGHT_HF:
            dest.write_bytes(b"W" * weights.VISUAL_QC_CLIP_MIN_BYTES)
        elif filename == weights.VISUAL_QC_CLIP_CONFIG_HF:
            dest.write_bytes(b"{" + b"x" * weights.VISUAL_QC_CLIP_CONFIG_MIN_BYTES + b"}")
        else:
            dest.write_bytes(b"HF:" + filename.encode())
        pulled.append(f"{repo}:{filename}")

    monkeypatch.setattr(weights, "_hf_file", fake_hf)
    out = weights.ensure_still_weights(tmp_path)
    assert out["kind"] == "stills"
    assert f"{weights.VISUAL_QC_CLIP_REPO}:{weights.VISUAL_QC_CLIP_WEIGHT_HF}" in pulled
    assert f"{weights.VISUAL_QC_CLIP_REPO}:{weights.VISUAL_QC_CLIP_CONFIG_HF}" in pulled
    weights.validate_visual_qc_clip_weights(tmp_path)


def test_runtime_weight_manifest_excludes_r2_and_keeps_h3_contract():
    assert all("r2" not in item.get("repo", "").lower() for item in weights.RUNTIME_WEIGHT_FILES)
    assert all(item.get("snapshot") != "1" for item in weights.RUNTIME_WEIGHT_FILES)
    h3_dest = {item["dest"] for item in weights.H3_FILES}
    assert "models/diffusion_models/minimax_h3_fl2va_pruned_nvfp4.safetensors" in h3_dest


def test_dockerfile_pins_open_clip_torch_at_build_time():
    text = DOCKERFILE.read_text(encoding="utf-8")
    install_block = text.split("RUN chmod +x /usr/local/bin/af-start", 1)[1].split(
        "&& python3 - <<'PY'", 1
    )[0]
    assert "--only-binary=:all:" in install_block
    assert '"open_clip_torch==2.32.0"' in install_block
    assert text.count("open_clip_torch==") == 1
    assert "OPEN_CLIP_TORCH_VERSION=2.32.0" in text
    assert "AF_VISUAL_QC_CLIP_WEIGHTS=/opt/ComfyUI/models/visual_qc/open_clip_model.safetensors" in text
    assert "AF_OPEN_CLIP_MODEL=ViT-B-32" in text
    assert "AF_OPEN_CLIP_PRETRAINED=openai" in text
    assert "models/visual_qc" in text
    skip_block = text.split("skip =", 1)[1].split("]", 1)[0]
    assert "open_clip_torch" in skip_block
    assert "open_clip_torch==2.32.0" in install_block


def test_start_sh_does_not_pip_install_open_clip():
    text = START_SH.read_text(encoding="utf-8")
    assert "pip install" not in text
    assert "open_clip" not in text.lower()


def test_visual_qc_clip_env_is_consumed_by_anime_factory_visual_qc(tmp_path, monkeypatch):
    from anime_factory import visual_qc

    for name in (
        "AF_VISUAL_QC_CLIP_WEIGHTS",
        "AF_VISUAL_QC_CLIP_CONFIG",
        "AF_VISUAL_QC_CLIP_DIR",
        "AF_OPEN_CLIP_MODEL",
        "AF_OPEN_CLIP_PRETRAINED",
    ):
        monkeypatch.delenv(name, raising=False)

    env = weights.visual_qc_clip_env(tmp_path)
    weight = Path(env["AF_VISUAL_QC_CLIP_WEIGHTS"])
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"W" * weights.VISUAL_QC_CLIP_MIN_BYTES)
    Path(env["AF_VISUAL_QC_CLIP_CONFIG"]).write_bytes(b"{" + b"x" * weights.VISUAL_QC_CLIP_CONFIG_MIN_BYTES + b"}")

    for key, value in env.items():
        if key == "OPEN_CLIP_TORCH_VERSION":
            continue
        monkeypatch.setenv(key, value)

    assert visual_qc._clip_weight_path() == weight
    assert visual_qc._clip_cache_dir() == Path(env["AF_VISUAL_QC_CLIP_DIR"])
    assert visual_qc._clip_config_path() == Path(env["AF_VISUAL_QC_CLIP_CONFIG"])
    assert visual_qc._clip_model_env() == (weights.VISUAL_QC_CLIP_MODEL_NAME, weights.VISUAL_QC_CLIP_PRETRAINED_TAG)

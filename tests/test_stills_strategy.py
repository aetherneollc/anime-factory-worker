"""Contracts, backend selection, QC gate, character_refs=master.png only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anime_factory.backends import resolve_backend_env
from anime_factory.backends.control import (
    ControlBackendError,
    ControlRequest,
    get_control_backend,
    select_control_backend,
)
from anime_factory.backends.image import (
    ImageBackendError,
    select_image_backend,
)
from anime_factory.backends.video_ext import (
    Hunyuan15VideoGenerateBackend,
    VideoGenerateRequest,
    gated_video_generate,
    get_video_generate_backend,
    normalize_pluggable_video_backend,
    select_pluggable_video_backend,
)
from anime_factory.contracts import (
    MASTER_FILENAME,
    CharacterRefError,
    KeyframeContract,
    KeyframeQcGateError,
    QcReport,
    ShotContract,
    assert_character_refs,
    assert_keyframe_allows_video,
    b_tier_control_payload,
    character_master_relpath,
    is_master_ref,
)
from anime_factory.hunyuan21 import inference_params, master_prompt
from anime_factory.stills_phase import phase_a_loop, run_keyframe_phase, run_video_phase
from anime_factory.video_backend import normalize_video_backend, select_video_backend


# --- character_refs / contracts -------------------------------------------------


def test_is_master_ref_only_master_png():
    assert is_master_ref("characters/alice/master.png")
    assert is_master_ref("/abs/assets/characters/bob/master.png")
    assert not is_master_ref("characters/alice/sheet_front.png")
    assert not is_master_ref("characters/alice/")
    assert not is_master_ref("characters/alice")
    assert not is_master_ref("master.jpg")


def test_assert_character_refs_rejects_sheets_and_dirs():
    assert assert_character_refs(["characters/alice/master.png"]) == [
        "characters/alice/master.png"
    ]
    with pytest.raises(CharacterRefError):
        assert_character_refs(["characters/alice/sheet_front.png"])
    with pytest.raises(CharacterRefError):
        assert_character_refs(["characters/alice/"])
    with pytest.raises(CharacterRefError):
        assert_character_refs(["characters/alice"])


def test_shot_contract_b_requires_master_and_identity():
    shot = ShotContract.from_dict(
        {
            "shot_id": "s001",
            "tier": "B",
            "prompt": "alice at the dock",
            "character_refs": ["characters/alice/master.png"],
            "sheet_refs": ["characters/alice/sheet_turnaround.png"],
        }
    )
    assert shot.control_mode == "identity"
    payload = b_tier_control_payload(shot)
    assert payload["character_refs"] == ["characters/alice/master.png"]
    assert payload["sheet_refs"] == ["characters/alice/sheet_turnaround.png"]
    assert payload["control_mode"] == "identity"

    with pytest.raises(CharacterRefError):
        ShotContract.from_dict(
            {
                "shot_id": "s002",
                "tier": "B",
                "prompt": "x",
                "character_refs": ["characters/alice/sheet_front.png"],
            }
        )
    with pytest.raises(CharacterRefError):
        ShotContract.from_dict({"shot_id": "s003", "tier": "B", "prompt": "x"})


def test_character_master_relpath():
    assert character_master_relpath("alice") == f"characters/alice/{MASTER_FILENAME}"


# --- backend selection / env adapters -------------------------------------------


def test_image_backend_env_and_still_adapter(monkeypatch):
    monkeypatch.delenv("IMAGE_BACKEND", raising=False)
    monkeypatch.delenv("STILL_BACKEND", raising=False)
    assert select_image_backend() == "hunyuan21"

    monkeypatch.setenv("STILL_BACKEND", "kolors")
    assert select_image_backend() == "kolors"

    monkeypatch.setenv("IMAGE_BACKEND", "hunyuan21")
    assert select_image_backend() == "hunyuan21"

    monkeypatch.setenv("IMAGE_BACKEND", "nope")
    with pytest.raises(ImageBackendError):
        select_image_backend()


def test_control_backend_default_and_selection(monkeypatch):
    monkeypatch.delenv("CONTROL_BACKEND", raising=False)
    assert select_control_backend() == "kolors_ipadapter"
    monkeypatch.setenv("CONTROL_BACKEND", "composite")
    assert select_control_backend() == "composite"
    monkeypatch.setenv("CONTROL_BACKEND", "bad")
    with pytest.raises(ControlBackendError):
        select_control_backend()


def test_video_backend_hunyuan15_and_legacy_adapter(monkeypatch):
    monkeypatch.delenv("VIDEO_BACKEND", raising=False)
    monkeypatch.delenv("AF_VIDEO_BACKEND", raising=False)
    monkeypatch.delenv("AF_VIDEO_BACKEND_LOCKED", raising=False)
    assert normalize_video_backend("hunyuan15") == "hunyuan15"
    assert normalize_pluggable_video_backend("hy15") == "hunyuan15"
    assert normalize_video_backend("wan") == "h3"
    assert normalize_video_backend("longlive") == "longlive"

    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    assert select_pluggable_video_backend() == "longlive"
    monkeypatch.setenv("VIDEO_BACKEND", "hunyuan15")
    assert select_video_backend() == "hunyuan15"
    assert select_pluggable_video_backend() == "hunyuan15"


def test_resolve_backend_env_snapshot(monkeypatch):
    monkeypatch.setenv("IMAGE_BACKEND", "hunyuan21")
    monkeypatch.setenv("CONTROL_BACKEND", "kolors_ipadapter")
    monkeypatch.setenv("VIDEO_BACKEND", "h3")
    snap = resolve_backend_env()
    assert snap == {
        "IMAGE_BACKEND": "hunyuan21",
        "CONTROL_BACKEND": "kolors_ipadapter",
        "VIDEO_BACKEND": "h3",
    }


# --- Phase B control stub -------------------------------------------------------


def test_control_single_ref_identity_stub():
    backend = get_control_backend("kolors_ipadapter")
    result = backend.apply(
        ControlRequest(
            prompt="alice standing",
            character_refs=["characters/alice/master.png"],
            control_mode="identity",
            reference_pngs=[b"\x89PNG\r\n\x1a\n" + b"ref"],
        )
    )
    assert result.backend == "kolors_ipadapter"
    assert result.used_refs == ["characters/alice/master.png"]
    assert result.png.startswith(b"\x89PNG")

    with pytest.raises(ControlBackendError):
        backend.apply(
            ControlRequest(
                prompt="x",
                character_refs=[
                    "characters/alice/master.png",
                    "characters/bob/master.png",
                ],
                control_mode="identity",
            )
        )
    with pytest.raises(CharacterRefError):
        ControlRequest(
            prompt="x",
            character_refs=["characters/alice/sheet_front.png"],
            control_mode="identity",
        )


# --- QC gate --------------------------------------------------------------------


def test_keyframe_qc_gate_blocks_video():
    fail = KeyframeContract(
        shot_id="s001",
        path="episodes/EP001/keyframes/s001/keyframe_final.png",
        final_path="episodes/EP001/keyframes/s001/keyframe_final.png",
        status="qc_fail",
        qc=QcReport(verdict="fail", reasons=["identity_miss"]),
    )
    with pytest.raises(KeyframeQcGateError):
        assert_keyframe_allows_video(fail)

    ok = KeyframeContract(
        shot_id="s001",
        final_path="episodes/EP001/keyframes/s001/keyframe_final.png",
        status="qc_pass",
        qc=QcReport(verdict="pass"),
    )
    assert assert_keyframe_allows_video(ok).qc_passed


def test_gated_video_generate_respects_qc(monkeypatch):
    monkeypatch.setenv("VIDEO_BACKEND", "h3")
    fail = KeyframeContract(
        shot_id="s001",
        final_path="episodes/EP001/keyframes/s001/keyframe_final.png",
        status="qc_fail",
        qc=QcReport(verdict="fail", reasons=["vlm_count"]),
    )
    with pytest.raises(KeyframeQcGateError):
        gated_video_generate(
            fail,
            VideoGenerateRequest(
                shot_id="s001",
                keyframe_final=fail.final_path,
            ),
        )


def test_hunyuan15_consumes_only_keyframe_final():
    backend = Hunyuan15VideoGenerateBackend()
    with pytest.raises(Exception):
        backend.generate(
            VideoGenerateRequest(shot_id="s1", keyframe_final="episodes/x/f1.png")
        )
    result = backend.generate(
        VideoGenerateRequest(
            shot_id="s1",
            keyframe_final="episodes/EP001/keyframes/s1/keyframe_final.png",
            rife=True,
        )
    )
    assert result.backend == "hunyuan15"
    assert result.meta["keyframe_final"].endswith("keyframe_final.png")
    assert result.meta["rife"]["enabled"] is True


# --- Phase A loop ---------------------------------------------------------------


def test_phase_a_loop_qc_pass_then_h3(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HUNYUAN21_DRY_RUN", "1")
    monkeypatch.setenv("IMAGE_BACKEND", "hunyuan21")
    monkeypatch.setenv("VIDEO_BACKEND", "h3")
    shot = ShotContract(
        shot_id="s001",
        tier="A",
        prompt="rainy alley first frame, alice in leather jacket",
        seed=42,
    )
    out = phase_a_loop(shot, story_root=tmp_path, episode_code="EP001", dry_run=True)
    assert out["blocked"] is False
    assert out["image_backend"] == "hunyuan21"
    assert out["video_backend"] == "h3"
    final = Path(out["keyframe"]["final_path"])
    assert final.name == "keyframe_final.png"
    assert final.is_file()
    assert out["keyframe"]["qc"]["verdict"] == "pass"
    assert out["video"]["backend"] == "h3"


def test_phase_a_blocks_video_on_qc_fail(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HUNYUAN21_DRY_RUN", "1")
    monkeypatch.setenv("IMAGE_BACKEND", "hunyuan21")
    monkeypatch.setenv("VIDEO_BACKEND", "h3")

    def _fail_qc(*_a, **_k):
        return QcReport(verdict="fail", reasons=["clip_identity"], scores={})

    import anime_factory.stills_phase as sp

    monkeypatch.setattr(sp, "score_keyframe_qc", _fail_qc)
    shot = {"shot_id": "s002", "tier": "A", "prompt": "dock night"}
    out = phase_a_loop(shot, story_root=tmp_path, episode_code="EP001", dry_run=True)
    assert out["blocked"] is True
    assert out["video"] is None
    with pytest.raises(KeyframeQcGateError):
        run_video_phase(KeyframeContract.from_dict(out["keyframe"]))


def test_tier_b_uses_control_not_directory_scan(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HUNYUAN21_DRY_RUN", "1")
    monkeypatch.setenv("IMAGE_BACKEND", "hunyuan21")
    monkeypatch.setenv("CONTROL_BACKEND", "kolors_ipadapter")
    monkeypatch.setenv("HUNYUAN21_DRY_RUN", "1")
    shot = ShotContract(
        shot_id="s010",
        tier="B",
        prompt="alice close-up",
        character_refs=["characters/alice/master.png"],
        sheet_refs=["characters/alice/sheet_turnaround.png"],
    )
    # Place a decoy sheet that must never be read as a control ref.
    decoy = tmp_path / "assets" / "characters" / "alice"
    decoy.mkdir(parents=True)
    (decoy / "sheet_front.png").write_bytes(b"\x89PNG\r\n\x1a\nsheet")
    (decoy / "master.png").write_bytes(b"\x89PNG\r\n\x1a\nmaster")
    kf = run_keyframe_phase(
        shot,
        story_root=tmp_path,
        episode_code="EP001",
        reference_pngs=[(decoy / "master.png").read_bytes()],
    )
    assert kf.backend == "kolors_ipadapter"
    assert kf.meta["control"]["character_refs"] == ["characters/alice/master.png"]
    assert "sheet_front.png" not in json.dumps(kf.meta)


def test_hunyuan21_inference_params_and_master_prompt():
    distilled = inference_params("hunyuanimage-v2.1-distilled")
    assert distilled["num_inference_steps"] == 8
    base = inference_params("hunyuanimage-v2.1")
    assert base["num_inference_steps"] == 28
    assert "leather" in master_prompt("1boy, leather jacket")


def test_dockerfile_hy_skeleton_parses():
    from tests.dockerfile_parse import parse_dockerfile

    root = Path(__file__).resolve().parents[1]
    path = root / "deploy" / "gpu-worker" / "Dockerfile.hy"
    instructions, errors = parse_dockerfile(path)
    assert not errors
    assert "FROM" in instructions
    text = path.read_text(encoding="utf-8")
    assert "IMAGE_BACKEND=hunyuan21" in text
    assert "anime-factory-worker-hy" in text
    assert "baked" in text.lower()
    assert "cu128" in text.lower() or "cuda12.8" in text.lower()
    targets = json.loads((root / "deploy" / "docker-targets.json").read_text(encoding="utf-8"))
    hy = next(t for t in targets["targets"] if t["id"] == "hy")
    assert hy["enabled"] is True
    assert hy["image_name"] == "anime-factory-worker-hy"


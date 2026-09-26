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
    SkyReelsV3R2VBackend,
    VideoGenerateRequest,
    gated_video_generate,
    normalize_pluggable_video_backend,
    select_pluggable_video_backend,
)
from anime_factory.legacy_backends import H3VideoGenerateBackend
from anime_factory.contracts import (
    MASTER_FILENAME,
    CharacterRefError,
    KeyframeContract,
    KeyframeQcGateError,
    QcReport,
    RefPackContract,
    RefPackImage,
    ShotContract,
    assert_character_refs,
    assert_keyframe_allows_video,
    assert_ref_pack_allows_video,
    b_tier_control_payload,
    character_master_relpath,
    is_master_ref,
)
from anime_factory.flux2_klein import KLEIN_4B_MODEL_ID, edit_klein_refs, generate_klein_t2i
from anime_factory.skyreels_r2v import (
    SKYREELS_MODEL_ID,
    SkyReelsR2VError,
    build_r2v_argv,
    generate_reference_video,
    truncate_references,
)
from anime_factory.stills_phase import (
    build_reference_pack,
    generate_tier_c_keyframe,
    phase_a_loop,
    register_character_master,
    run_keyframe_phase,
    run_video_phase,
)
from anime_factory.models import IMAGE_BACKENDS, VIDEO_BACKENDS
from anime_factory.video_backend import ProductionBackendError, normalize_video_backend, select_video_backend


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
    assert IMAGE_BACKENDS == ("flux2_klein4b",)
    assert select_image_backend() == "flux2_klein4b"

    monkeypatch.setenv("STILL_BACKEND", "kolors")
    assert select_image_backend() == "flux2_klein4b"

    monkeypatch.setenv("IMAGE_BACKEND", "flux2_klein4b")
    assert select_image_backend() == "flux2_klein4b"

    monkeypatch.setenv("IMAGE_BACKEND", "kolors")
    with pytest.raises(ImageBackendError):
        select_image_backend()

    monkeypatch.setenv("IMAGE_BACKEND", "nope")
    with pytest.raises(ImageBackendError):
        select_image_backend()


def test_control_backend_default_and_selection(monkeypatch):
    monkeypatch.delenv("CONTROL_BACKEND", raising=False)
    assert select_control_backend() == "flux2_klein_ref"
    monkeypatch.setenv("CONTROL_BACKEND", "composite")
    assert select_control_backend() == "composite"
    monkeypatch.setenv("CONTROL_BACKEND", "bad")
    with pytest.raises(ControlBackendError):
        select_control_backend()


def test_production_enums_reject_legacy_backends(monkeypatch):
    monkeypatch.delenv("VIDEO_BACKEND", raising=False)
    monkeypatch.delenv("AF_VIDEO_BACKEND", raising=False)
    monkeypatch.delenv("AF_VIDEO_BACKEND_LOCKED", raising=False)
    monkeypatch.delenv("AF_IMAGE_CAPABILITY", raising=False)
    assert VIDEO_BACKENDS == ("skyreels_v3_r2v",)
    assert normalize_video_backend(None) == "skyreels_v3_r2v"
    assert normalize_pluggable_video_backend("r2v") == "skyreels_v3_r2v"
    assert select_video_backend() == "skyreels_v3_r2v"
    for legacy in ("h3", "wan", "longlive", "kolors"):
        with pytest.raises(ProductionBackendError):
            normalize_video_backend(legacy)
    monkeypatch.setenv("AF_VIDEO_BACKEND", "longlive")
    with pytest.raises(ProductionBackendError):
        select_pluggable_video_backend()
    monkeypatch.setenv("VIDEO_BACKEND", "h3")
    with pytest.raises(ProductionBackendError):
        select_video_backend()
    with pytest.raises(ProductionBackendError):
        select_pluggable_video_backend()


def test_resolve_backend_env_snapshot(monkeypatch):
    monkeypatch.setenv("IMAGE_BACKEND", "flux2_klein4b")
    monkeypatch.setenv("CONTROL_BACKEND", "flux2_klein_ref")
    monkeypatch.setenv("VIDEO_BACKEND", "skyreels_v3_r2v")
    snap = resolve_backend_env()
    assert snap == {
        "IMAGE_BACKEND": "flux2_klein4b",
        "VIDEO_BACKEND": "skyreels_v3_r2v",
    }
    assert "CONTROL_BACKEND" not in snap


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
    monkeypatch.delenv("VIDEO_BACKEND", raising=False)
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


def test_h3_rollback_still_generates():
    backend = H3VideoGenerateBackend()
    result = backend.generate(
        VideoGenerateRequest(
            shot_id="s1",
            keyframe_final="episodes/EP001/keyframes/s1/keyframe_final.png",
            rife=True,
        )
    )
    assert result.backend == "h3"
    assert result.meta["delegated"] == "gpu_worker.h3"


# --- Phase A loop ---------------------------------------------------------------


def test_phase_a_loop_qc_pass_then_skyreels(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLUX2_KLEIN_DRY_RUN", "1")
    monkeypatch.setenv("SKYREELS_DRY_RUN", "1")
    monkeypatch.delenv("IMAGE_BACKEND", raising=False)
    monkeypatch.delenv("VIDEO_BACKEND", raising=False)
    register_character_master(tmp_path, "alice", "assets/characters/alice/master.png")
    shot = ShotContract(
        shot_id="s001",
        tier="A",
        prompt="rainy alley first frame, alice in leather jacket",
        seed=42,
    )
    out = phase_a_loop(shot, story_root=tmp_path, episode_code="EP001", dry_run=True)
    assert out["blocked"] is False
    assert out["image_backend"] == "flux2_klein4b"
    assert out["video_backend"] == "skyreels_v3_r2v"
    final = Path(out["keyframe"]["final_path"])
    assert final.name == "keyframe_final.png"
    assert final.is_file()
    assert out["keyframe"]["qc"]["verdict"] == "pass"
    assert out["video"]["backend"] == "skyreels_v3_r2v"
    assert "keyframe_final" not in json.dumps(out["video"].get("references"))


def test_phase_a_blocks_video_on_qc_fail(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLUX2_KLEIN_DRY_RUN", "1")
    monkeypatch.setenv("IMAGE_BACKEND", "flux2_klein4b")
    monkeypatch.delenv("VIDEO_BACKEND", raising=False)

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
    monkeypatch.setenv("FLUX2_KLEIN_DRY_RUN", "1")
    monkeypatch.setenv("IMAGE_BACKEND", "flux2_klein4b")
    monkeypatch.setenv("CONTROL_BACKEND", "kolors_ipadapter")
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


def test_tier_c_klein_refs_and_composite_refine(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLUX2_KLEIN_DRY_RUN", "1")
    shot = ShotContract(
        shot_id="s030",
        tier="C",
        prompt="alice and bob on the dock",
        character_refs=[
            "characters/alice/master.png",
            "characters/bob/master.png",
        ],
        scene_refs=["scenes/dock.png"],
    )
    pngs = [b"\x89PNG\r\n\x1a\n" + f"ref{i}".encode() for i in range(3)]
    png, backend, meta = generate_tier_c_keyframe(shot, reference_pngs=pngs)
    assert backend == "flux2_klein_ref"
    assert png.startswith(b"\x89PNG")
    assert "fallback" not in meta

    import anime_factory.backends.control as control

    def _boom(_self, _request):
        raise RuntimeError("klein unavailable")

    monkeypatch.setattr(control.Flux2KleinRefControlBackend, "apply", _boom)
    png2, backend2, meta2 = generate_tier_c_keyframe(
        shot,
        reference_pngs=pngs,
        env={"FLUX2_KLEIN_DRY_RUN": "1"},
    )
    assert backend2 == "flux2_klein_ref"
    assert meta2["fallback"] == "composite"
    assert meta2["refine"] is True
    assert png2.startswith(b"\x89PNG")
    kf = run_keyframe_phase(
        shot,
        story_root=tmp_path,
        episode_code="EP001",
        reference_pngs=pngs,
        env={"FLUX2_KLEIN_DRY_RUN": "1", "IMAGE_BACKEND": "flux2_klein4b"},
    )
    pack = json.loads((Path(kf.final_path).parent / "ref_pack.json").read_text(encoding="utf-8"))
    packed = pack.get("references") or pack.get("images") or []
    assert all("keyframe_final" not in item["path"] for item in packed)


def test_klein_dry_run_t2i_and_refs(monkeypatch):
    monkeypatch.setenv("FLUX2_KLEIN_DRY_RUN", "1")
    png = generate_klein_t2i("full body master", 768, 1344, seed=1, dry_run=True)
    assert png.startswith(b"\x89PNG")
    edited = edit_klein_refs(
        "same character",
        [png, png],
        1280,
        720,
        seed=2,
        dry_run=True,
    )
    assert edited.startswith(b"\x89PNG")
    assert KLEIN_4B_MODEL_ID == "black-forest-labs/FLUX.2-klein-4B"
    with pytest.raises(Exception):
        edit_klein_refs("too many", [png] * 5, dry_run=True)


def test_skyreels_dry_run_argv():
    result = generate_reference_video(
        ref_imgs=["characters/alice/master.png", "characters/bob/master.png"],
        prompt="they walk",
        duration_s=5,
        aspect="16:9",
        dry_run=True,
    )
    argv = result.argv
    assert result.dry_run is True
    assert result.model_id == SKYREELS_MODEL_ID == "Skywork/SkyReels-V3-R2V-14B"
    assert "--task_type" in argv and argv[argv.index("--task_type") + 1] == "reference_to_video"
    assert "--resolution" in argv and argv[argv.index("--resolution") + 1] == "720P"
    assert "--duration" in argv and argv[argv.index("--duration") + 1] == "5"
    assert "--low_vram" in argv
    assert "--offload" in argv
    assert "--aspect" not in argv
    assert "--fps" not in argv
    built = build_r2v_argv(["a.png"], "p", low_vram=False, offload=False)
    assert "--low_vram" not in built
    assert "--offload" not in built


def test_ref_pack_gate_and_master_png():
    pack = RefPackContract(
        shot_id="s1",
        images=[
            RefPackImage(path="characters/alice/master.png", role="character", qc_status="pass"),
            RefPackImage(path="scenes/dock.png", role="scene", qc_status="pass"),
        ],
        qc_status="pass",
    )
    assert_ref_pack_allows_video(pack)
    with pytest.raises(CharacterRefError):
        RefPackImage(path="characters/alice/sheet_front.png", role="character", qc_status="pass")
    many = RefPackContract(
        shot_id="s2",
        references=[
            RefPackImage(path=f"characters/c{i}/master.png", role="character", qc_status="pass")
            for i in range(5)
        ],
        qc_status="pass",
    )
    assert len(many.references) == 5
    assert_ref_pack_allows_video(many)
    blocked = RefPackContract(
        shot_id="s3",
        images=[RefPackImage(path="preview.png", role="preview", qc_status="fail")],
        qc_status="fail",
    )
    with pytest.raises(KeyframeQcGateError):
        assert_ref_pack_allows_video(blocked)


def test_consecutive_shots_share_stable_reference_pack(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLUX2_KLEIN_DRY_RUN", "1")
    monkeypatch.setenv("SKYREELS_DRY_RUN", "1")
    monkeypatch.delenv("IMAGE_BACKEND", raising=False)
    monkeypatch.delenv("VIDEO_BACKEND", raising=False)
    monkeypatch.delenv("AF_VIDEO_BACKEND", raising=False)
    register_character_master(tmp_path, "alice", "assets/characters/alice/master.png")
    register_character_master(tmp_path, "bob", "assets/characters/bob/master.png")
    last = tmp_path / "episodes" / "EP001" / "keyframes" / "s001" / "keyframe_final.png"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"\x89PNG\r\n\x1a\nlast-frame")
    shot_a = ShotContract(shot_id="s001", tier="A", prompt="dock night", seed=1)
    shot_b = ShotContract(
        shot_id="s002",
        tier="B",
        prompt="dock day",
        character_refs=["assets/characters/alice/master.png"],
        seed=2,
    )
    pack_a = build_reference_pack(shot_a, story_root=tmp_path, qc=QcReport(verdict="pass"), status="qc_pass")
    pack_b = build_reference_pack(
        shot_b,
        story_root=tmp_path,
        preview_path=str(last),
        qc=QcReport(verdict="pass"),
        status="qc_pass",
    )
    paths_a = [item.path for item in pack_a.references]
    paths_b = [item.path for item in pack_b.references]
    assert paths_a == paths_b == [
        "assets/characters/alice/master.png",
        "assets/characters/bob/master.png",
    ]
    blob = json.dumps(pack_a.to_dict()) + json.dumps(pack_b.to_dict())
    assert "keyframe_final" not in blob
    assert "last-frame" not in blob
    kf = run_keyframe_phase(shot_a, story_root=tmp_path, episode_code="EP001", dry_run=True)
    video = run_video_phase(kf, prompt="dock night", camera="wide", motion="push in", duration=5, seed=7)
    assert video.backend == "skyreels_v3_r2v"
    assert video.camera == "wide"
    assert video.motion == "push in"
    assert video.references == paths_a
    assert "--task_type" in video.meta.get("argv", [])


def test_skyreels_truncates_characters_first_and_refuses_empty():
    refs = [
        RefPackImage(path=f"characters/c{i}/master.png", role="character", qc_status="pass")
        for i in range(5)
    ]
    refs.append(RefPackImage(path="scenes/dock.png", role="scene", qc_status="pass"))
    kept, meta = truncate_references(refs)
    assert len(kept) == 4
    assert kept == [f"characters/c{i}/master.png" for i in range(4)]
    assert meta["truncated"] is True
    assert meta["max_references"] == 4
    assert "characters/c4/master.png" in meta["dropped_references"]
    assert "scenes/dock.png" in meta["dropped_references"]
    with pytest.raises(SkyReelsR2VError):
        truncate_references([])
    with pytest.raises(SkyReelsR2VError):
        generate_reference_video(ref_imgs=[], prompt="walk", dry_run=True)
    backend = SkyReelsV3R2VBackend()
    with pytest.raises(SkyReelsR2VError):
        backend.generate(
            VideoGenerateRequest(
                shot_id="s",
                keyframe_final="episodes/EP001/keyframes/s/keyframe_final.png",
                prompt="walk",
            )
        )


def test_dockerfile_sr3_skeleton_parses():
    from tests.dockerfile_parse import parse_dockerfile

    root = Path(__file__).resolve().parents[1]
    path = root / "deploy" / "gpu-worker" / "Dockerfile.sr3"
    instructions, errors = parse_dockerfile(path)
    assert not errors
    assert "FROM" in instructions
    text = path.read_text(encoding="utf-8")
    assert "IMAGE_BACKEND=flux2_klein4b" in text
    assert "VIDEO_BACKEND=skyreels_v3_r2v" in text
    assert "anime-factory-worker-sr3" in text
    assert "requirements.txt" not in text
    assert "sha256:" in text
    assert not (root / "deploy" / "gpu-worker" / "Dockerfile.hy").exists()
    targets = json.loads((root / "deploy" / "docker-targets.json").read_text(encoding="utf-8"))
    ids = [t["id"] for t in targets["targets"]]
    assert "hy" not in ids
    sr3 = next(t for t in targets["targets"] if t["id"] == "sr3")
    assert sr3["enabled"] is True
    assert "python/" in sr3["watch_paths"]
    for frozen in ("h3", "longlive"):
        row = next(t for t in targets["targets"] if t["id"] == frozen)
        assert "python/" not in row["watch_paths"]
        assert not any(str(p).startswith("python") for p in row["watch_paths"])


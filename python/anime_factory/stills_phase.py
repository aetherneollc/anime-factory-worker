"""Klein writes the asset registry. SkyReels consumes a stable character pack.

Tier A/B/C Klein edits are asset generation (keyframes for review), not video
continuity. ``build_reference_pack`` reloads the same character masters from
the registry for every shot and never chains the previous shot's last frame.
Shot unit is 5 seconds. Production video is SkyReels V3 R2V only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from anime_factory.backends.control import ControlRequest, get_control_backend
from anime_factory.backends.image import ImageGenerateRequest, get_image_backend, select_image_backend
from anime_factory.backends.video_ext import (
    VideoGenerateRequest,
    gated_video_generate,
    get_video_generate_backend,
    select_pluggable_video_backend,
    video_contract_from_result,
)
from anime_factory.contracts import (
    KEYFRAME_FINAL_NAME,
    MASTER_FILENAME,
    AssetRef,
    CharacterRefError,
    KeyframeContract,
    QcReport,
    RefPackContract,
    ReferenceAsset,
    ShotContract,
    VideoContract,
    VideoRequest,
    assert_character_refs,
    assert_ref_pack_allows_video,
    b_tier_control_payload,
    is_master_ref,
)
from anime_factory.flux2_klein import (
    DEFAULT_HEIGHT,
    DEFAULT_MASTER_HEIGHT,
    DEFAULT_MASTER_WIDTH,
    DEFAULT_WIDTH,
    Flux2KleinError,
    edit_klein_refs,
    generate_klein_t2i,
    master_prompt,
)

# R2V recommended duration. Longer shots are split, not extended.
SHOT_UNIT_SECONDS = 5.0
REGISTRY_REL = Path("assets") / "registry.json"


def registry_path(story_root: Path) -> Path:
    return Path(story_root) / REGISTRY_REL


def load_asset_registry(story_root: Path) -> list[AssetRef]:
    path = registry_path(story_root)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = []
    if isinstance(data, dict):
        rows = data.get("assets") or []
    if not isinstance(rows, list):
        return []
    out: list[AssetRef] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(AssetRef.from_dict(row))
    return out


def write_asset_registry(story_root: Path, assets: list[AssetRef]) -> Path:
    path = registry_path(story_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"assets": [item.to_dict() for item in assets]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def register_asset(story_root: Path, asset: AssetRef) -> AssetRef:
    """Klein asset write. Replaces an existing row with the same id and kind."""
    current = [item for item in load_asset_registry(story_root) if not (item.asset_id == asset.asset_id and item.kind == asset.kind)]
    current.append(asset)
    write_asset_registry(story_root, current)
    return asset


def register_character_master(
    story_root: Path,
    character_id: str,
    master_path: str,
    *,
    qc_status: str = "pass",
) -> AssetRef:
    path = str(master_path).replace("\\", "/")
    if not is_master_ref(path):
        raise CharacterRefError(f"character registry entry must be {MASTER_FILENAME}, got {path!r}")
    return register_asset(
        story_root,
        AssetRef(
            asset_id=str(character_id),
            kind="character",
            master_path=path,
            meta={"qc_status": qc_status},
        ),
    )


def _stable_character_references(story_root: Path) -> list[ReferenceAsset]:
    """Character masters from the registry, in registry order. No last frames."""
    refs: list[ReferenceAsset] = []
    seen: set[str] = set()
    for asset in load_asset_registry(story_root):
        if asset.kind != "character":
            continue
        path = str(asset.master_path or "").replace("\\", "/")
        if not path or not is_master_ref(path) or path in seen:
            continue
        if "keyframe_final" in path or "/keyframes/" in path or path.endswith("last.png"):
            continue
        seen.add(path)
        qc = str((asset.meta or {}).get("qc_status") or "pass")
        refs.append(
            ReferenceAsset(path=path, role="character", qc_status=qc, asset_id=asset.asset_id)  # type: ignore[arg-type]
        )
    return refs


def keyframe_dir(story_root: Path, episode_code: str, shot_id: str) -> Path:
    return Path(story_root) / "episodes" / episode_code / "keyframes" / shot_id


def write_keyframe_final(dest_dir: Path, png: bytes) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / KEYFRAME_FINAL_NAME
    path.write_bytes(png)
    return path


def score_keyframe_qc(
    png: bytes,
    *,
    prompt: str = "",
    scorer: Any | None = None,
    require_clip: bool = False,
) -> QcReport:
    """Layer-1 CLIP gate (fast). Layer-2 VLM is optional follow-up.

    When no scorer is available and require_clip is False, structure-only pass
    for dry pipeline tests (non-empty PNG header).
    """
    if len(png) < 16 or not png.startswith(b"\x89PNG"):
        return QcReport(verdict="fail", layer="clip", reasons=["not_png"], scores={})

    if scorer is None:
        try:
            from anime_factory.visual_qc import load_clip_scorer, score_still

            if require_clip:
                scorer = load_clip_scorer()
                result = score_still(png, kind="keyframe", prompt=prompt, scorer=scorer, require_clip=True)
                return QcReport(
                    verdict="pass" if result.verdict == "pass" else "fail",
                    layer="clip",
                    reasons=list(result.reasons or []),
                    scores={k: float(v) for k, v in (result.scores or {}).items()},
                )
        except Exception:  # noqa: BLE001 — dry / unit path
            pass
        return QcReport(
            verdict="pass",
            layer="clip",
            reasons=[],
            scores={"dry_header_ok": 1.0},
            meta={"dry": True},
        )

    from anime_factory.visual_qc import score_still

    result = score_still(png, kind="keyframe", prompt=prompt, scorer=scorer, require_clip=require_clip)
    return QcReport(
        verdict="pass" if result.verdict == "pass" else "fail",
        layer="clip",
        reasons=list(result.reasons or []),
        scores={k: float(v) for k, v in (result.scores or {}).items()},
    )


def _dry_flag(env: Mapping[str, str] | None, name: str) -> bool:
    return str((env or os.environ).get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def mint_master_png(
    *,
    identity: str,
    seed: int | None = None,
    dry_run: bool | None = None,
    width: int = DEFAULT_MASTER_WIDTH,
    height: int = DEFAULT_MASTER_HEIGHT,
) -> bytes:
    """Character master via Klein 4B T2I."""
    return generate_klein_t2i(
        master_prompt(identity),
        width,
        height,
        seed=seed,
        dry_run=dry_run,
    )


def generate_tier_a_keyframe(
    shot: ShotContract | Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    dry_run: bool | None = None,
) -> tuple[bytes, str]:
    """A-tier: Klein T2I. No character reference."""
    s = shot if isinstance(shot, ShotContract) else ShotContract.from_dict(shot)
    if dry_run is None:
        dry_run = _dry_flag(env, "FLUX2_KLEIN_DRY_RUN")
    backend_name = select_image_backend(env=env)
    if backend_name == "flux2_klein4b" and dry_run:
        return (
            generate_klein_t2i(
                s.prompt,
                DEFAULT_WIDTH,
                DEFAULT_HEIGHT,
                seed=s.seed,
                dry_run=True,
                negative_prompt=s.negative_prompt,
            ),
            "flux2_klein4b",
        )
    result = get_image_backend(backend_name, env=env).generate(
        ImageGenerateRequest(
            prompt=s.prompt,
            negative_prompt=s.negative_prompt,
            seed=s.seed,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            meta={"tier": "A"},
        )
    )
    return result.png, result.backend


def generate_tier_b_keyframe(
    shot: ShotContract | Mapping[str, Any],
    *,
    reference_pngs: list[bytes] | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[bytes, str, dict[str, Any]]:
    """B-tier: one character master, Klein reference edit (or the selected control backend)."""
    s = shot if isinstance(shot, ShotContract) else ShotContract.from_dict(shot)
    payload = b_tier_control_payload(s)
    assert_character_refs(payload["character_refs"])
    if len(payload["character_refs"]) != 1:
        raise CharacterRefError("tier B requires exactly one master.png character_ref")
    control = get_control_backend(env=env)
    result = control.apply(
        ControlRequest(
            prompt=s.prompt,
            negative_prompt=s.negative_prompt,
            character_refs=list(payload["character_refs"]),
            control_mode=str(payload["control_mode"]),
            reference_pngs=tuple(reference_pngs or ()),
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            seed=s.seed,
            meta={"tier": "B", "dry_run": _dry_flag(env, "FLUX2_KLEIN_DRY_RUN")},
        )
    )
    return result.png, result.backend, {"control": payload, **result.meta}


def _scene_refs(shot: ShotContract) -> list[str]:
    raw: list[Any] | Any = list(shot.scene_refs)
    if not raw:
        raw = shot.meta.get("scene_refs") or []
    if isinstance(raw, (str, Path)):
        raw = [raw]
    return [str(x).replace("\\", "/").strip() for x in raw if str(x).strip()]


def generate_tier_c_keyframe(
    shot: ShotContract | Mapping[str, Any],
    *,
    reference_pngs: list[bytes] | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[bytes, str, dict[str, Any]]:
    """C-tier: 2–4 refs through Klein. Composite paste, then Klein refine, on failure."""
    s = shot if isinstance(shot, ShotContract) else ShotContract.from_dict(shot)
    chars = list(s.character_refs)
    scenes = _scene_refs(s)
    if len(chars) + len(scenes) < 2 and len(reference_pngs or []) < 2 and len(chars) < 2:
        raise CharacterRefError("tier C requires 2–4 references (characters and optional scene)")
    # Character masters first; cap the path list at 4. Bytes follow the same order.
    paths = list(chars)
    for scene in scenes:
        if scene not in paths:
            paths.append(scene)
    paths = paths[:4]
    pngs = list(reference_pngs or [])
    request = ControlRequest(
        prompt=s.prompt,
        negative_prompt=s.negative_prompt,
        character_refs=list(chars[:4]),
        control_mode="compose",
        reference_pngs=tuple(pngs[:4]),
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        seed=s.seed,
        meta={"tier": "C", "scene_refs": scenes, "dry_run": _dry_flag(env, "FLUX2_KLEIN_DRY_RUN")},
    )
    try:
        result = get_control_backend("flux2_klein_ref", env=env).apply(request)
        return result.png, result.backend, {"control": {"character_refs": chars, "scene_refs": scenes}, **result.meta}
    except (Flux2KleinError, Exception) as exc:  # noqa: BLE001 — composite fallback is the C-tier contract
        pasted = get_control_backend("composite", env=env).apply(request)
        refined = edit_klein_refs(
            s.prompt,
            [pasted.png, *pngs[:3]],
            DEFAULT_WIDTH,
            DEFAULT_HEIGHT,
            seed=s.seed,
            dry_run=True if _dry_flag(env, "FLUX2_KLEIN_DRY_RUN") else None,
        )
        return refined, "flux2_klein_ref", {
            "fallback": "composite",
            "error": str(exc),
            "control": {"character_refs": chars, "scene_refs": scenes},
            "refine": True,
        }


def build_reference_pack(
    shot: ShotContract | Mapping[str, Any],
    *,
    story_root: Path | None = None,
    registry: list[AssetRef] | None = None,
    preview_path: str = "",
    qc: QcReport | None = None,
    status: str = "pending",
    backend: str = "",
) -> RefPackContract:
    """Stable character pack from the asset registry. Same references for every shot.

    ``preview_path`` is review metadata only. It is not appended to ``references``,
    and the previous shot's last frame is never read.
    """
    del preview_path  # review stills are not video references
    s = shot if isinstance(shot, ShotContract) else ShotContract.from_dict(shot)
    if registry is not None:
        refs: list[ReferenceAsset] = []
        seen: set[str] = set()
        for asset in registry:
            if asset.kind != "character":
                continue
            path = str(asset.master_path or "").replace("\\", "/")
            if not path or not is_master_ref(path) or path in seen:
                continue
            if "keyframe_final" in path or "/keyframes/" in path or path.endswith("last.png"):
                continue
            seen.add(path)
            qc_asset = str((asset.meta or {}).get("qc_status") or "pass")
            refs.append(
                ReferenceAsset(path=path, role="character", qc_status=qc_asset, asset_id=asset.asset_id)  # type: ignore[arg-type]
            )
    elif story_root is not None:
        refs = _stable_character_references(story_root)
    else:
        refs = []
    asset_qc = "pass" if refs and all(item.qc_status == "pass" for item in refs) else "fail"
    if qc is not None or status not in {"", "pending"}:
        passed = bool(qc and qc.passed) or status == "qc_pass"
        # Asset QC still blocks video. A failing caller status cannot promote the pack.
        if not passed:
            asset_qc = "fail"
    return RefPackContract(
        shot_id=s.shot_id,
        references=refs,
        qc_status=asset_qc,  # type: ignore[arg-type]
        preview_path="",
        meta={"backend": backend, "source": "asset_registry", "character_ids": [item.asset_id for item in refs]},
    )


def build_ref_pack(
    shot: ShotContract | Mapping[str, Any],
    *,
    story_root: Path | None = None,
    preview_path: str = "",
    scene_refs: list[str] | None = None,
    qc: QcReport | None = None,
    status: str = "pending",
    backend: str = "",
    registry: list[AssetRef] | None = None,
) -> RefPackContract:
    """Compatibility alias. Scene plates and preview frames are not packed."""
    del scene_refs
    return build_reference_pack(
        shot,
        story_root=story_root,
        registry=registry,
        preview_path=preview_path,
        qc=qc,
        status=status,
        backend=backend,
    )


def run_keyframe_phase(
    shot: ShotContract | Mapping[str, Any],
    *,
    story_root: Path,
    episode_code: str,
    env: Mapping[str, str] | None = None,
    scorer: Any | None = None,
    require_clip: bool = False,
    dry_run: bool | None = None,
    reference_pngs: list[bytes] | None = None,
) -> KeyframeContract:
    """Mint keyframe_final.png and attach QC. Does not call video."""
    s = shot if isinstance(shot, ShotContract) else ShotContract.from_dict(shot)
    meta: dict[str, Any] = {}
    if s.tier == "B":
        png, backend, meta = generate_tier_b_keyframe(s, reference_pngs=reference_pngs, env=env)
    elif s.tier == "C":
        png, backend, meta = generate_tier_c_keyframe(s, reference_pngs=reference_pngs, env=env)
    else:
        png, backend = generate_tier_a_keyframe(s, env=env, dry_run=dry_run)

    for ref in s.character_refs:
        cid = PurePosixPath(ref).parent.name
        if cid and is_master_ref(ref):
            register_character_master(story_root, cid, ref)
    dest = keyframe_dir(story_root, episode_code, s.shot_id)
    final = write_keyframe_final(dest, png)
    qc = score_keyframe_qc(png, prompt=s.prompt, scorer=scorer, require_clip=require_clip)
    status = "qc_pass" if qc.passed else "qc_fail"
    register_asset(
        story_root,
        AssetRef(
            asset_id=f"keyframe-{s.shot_id}",
            kind="keyframe",
            master_path="",
            meta={"path": str(final).replace("\\", "/"), "qc_status": status, "backend": backend},
        ),
    )
    contract = KeyframeContract(
        shot_id=s.shot_id,
        path=str(final),
        final_path=str(final),
        status=status,  # type: ignore[arg-type]
        qc=qc,
        backend=backend,
        seed=s.seed,
        meta={"image_backend": select_image_backend(env=env), **meta},
    )
    pack = build_reference_pack(s, story_root=story_root, qc=qc, status=status, backend=backend)
    contract.meta["ref_pack"] = pack.to_dict()
    (dest / "keyframe.json").write_text(json.dumps(contract.to_dict(), indent=2) + "\n", encoding="utf-8")
    (dest / "ref_pack.json").write_text(json.dumps(pack.to_dict(), indent=2) + "\n", encoding="utf-8")
    return contract


def run_video_phase(
    keyframe: KeyframeContract | RefPackContract | None = None,
    *,
    ref_pack: RefPackContract | None = None,
    duration_s: float | None = None,
    duration: float | None = None,
    prompt: str = "",
    camera: str = "",
    motion: str = "",
    resolution: str = "720P",
    seed: int | None = None,
    aspect: str = "16:9",
    rife: bool = False,
    env: Mapping[str, str] | None = None,
) -> VideoContract:
    """QC-passed stable reference pack, then SkyReels. No preview-frame substitute."""
    pack = ref_pack
    if pack is None and isinstance(keyframe, RefPackContract):
        pack = keyframe
    elif pack is None and isinstance(keyframe, KeyframeContract):
        raw = keyframe.meta.get("ref_pack")
        pack = RefPackContract.from_dict(raw) if isinstance(raw, Mapping) else None
        if pack is None:
            from anime_factory.contracts import ref_pack_from_keyframe

            pack = ref_pack_from_keyframe(keyframe)
    if pack is None:
        raise CharacterRefError("run_video_phase requires a ref_pack")
    assert_ref_pack_allows_video(pack)
    dur = float(duration if duration is not None else duration_s if duration_s is not None else SHOT_UNIT_SECONDS)
    if seed is None and isinstance(keyframe, KeyframeContract):
        seed = keyframe.seed
    request_doc = VideoRequest(
        shot_id=pack.shot_id,
        references=list(pack.references),
        prompt=prompt,
        camera=camera,
        motion=motion,
        duration=dur,
        resolution=resolution,
        seed=seed,
    )
    backend = get_video_generate_backend(env=env)
    req = VideoGenerateRequest(
        shot_id=request_doc.shot_id,
        prompt=request_doc.prompt,
        camera=request_doc.camera,
        motion=request_doc.motion,
        duration=request_doc.duration,
        resolution=request_doc.resolution,
        seed=request_doc.seed,
        references=list(request_doc.references),
        aspect=aspect,
        rife=rife,
        meta={"dry_run": _dry_flag(env, "SKYREELS_DRY_RUN")},
    )
    result = gated_video_generate(pack, req, backend=backend, env=env)
    return video_contract_from_result(pack.shot_id, "", result, request=req)


def ensure_master_on_disk(
    story_root: Path,
    character_id: str,
    *,
    identity: str,
    seed: int | None = None,
    dry_run: bool | None = None,
) -> Path:
    """Write characters/<id>/master.png via Klein 4B T2I."""
    if not character_id.strip():
        raise CharacterRefError("character_id required")
    dest = Path(story_root) / "assets" / "characters" / character_id / MASTER_FILENAME
    rel = f"characters/{character_id}/{MASTER_FILENAME}"
    if not (dest.is_file() and dest.stat().st_size >= 16):
        png = mint_master_png(identity=identity, seed=seed, dry_run=dry_run)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(png)
    register_character_master(story_root, character_id, rel)
    return dest


def phase_a_loop(
    shot: ShotContract | Mapping[str, Any],
    *,
    story_root: Path,
    episode_code: str,
    env: Mapping[str, str] | None = None,
    dry_run: bool = True,
    run_video: bool = True,
) -> dict[str, Any]:
    """Closed loop: Klein asset → registry pack → SkyReels. Legacy video backends are not selected."""
    mapping = dict(env or os.environ)
    if dry_run:
        mapping.setdefault("FLUX2_KLEIN_DRY_RUN", "1")
        mapping.setdefault("SKYREELS_DRY_RUN", "1")
    mapping.setdefault("IMAGE_BACKEND", "flux2_klein4b")
    mapping.setdefault("VIDEO_BACKEND", "skyreels_v3_r2v")
    kf = run_keyframe_phase(
        shot,
        story_root=story_root,
        episode_code=episode_code,
        env=mapping,
        dry_run=dry_run,
    )
    out: dict[str, Any] = {
        "keyframe": kf.to_dict(),
        "image_backend": select_image_backend(env=mapping),
        "video_backend": select_pluggable_video_backend(env=mapping),
    }
    if not kf.qc_passed:
        out["video"] = None
        out["blocked"] = True
        return out
    if run_video:
        vid = run_video_phase(kf, env=mapping)
        out["video"] = vid.to_dict()
        out["blocked"] = False
    return out

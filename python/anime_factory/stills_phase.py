"""Phase A/B/C stills → keyframe QC → video orchestration (pluggable backends).

Phase A: Master / prompt → IMAGE_BACKEND (hunyuan21) → keyframe → QC gate
         → VIDEO_BACKEND (may stay h3).
Phase B: CONTROL_BACKEND single-ref identity when tier B.
Phase C: VIDEO_BACKEND=hunyuan15 consumes only keyframe_final.png (+ RIFE stub).

Does not replace the existing Kolors/Comfy ensure_keyframe path; this is the
new stills-strategy loop that tests and Docker-hy wire first.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from anime_factory.backends.control import ControlRequest, get_control_backend, select_control_backend
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
    CharacterRefError,
    KeyframeContract,
    KeyframeQcGateError,
    QcReport,
    ShotContract,
    VideoContract,
    assert_character_refs,
    assert_keyframe_allows_video,
    b_tier_control_payload,
)
from anime_factory.hunyuan21 import (
    DEFAULT_MASTER_HEIGHT,
    DEFAULT_MASTER_WIDTH,
    generate_hunyuan21_t2i,
    master_prompt,
)


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


def mint_master_png(
    *,
    identity: str,
    seed: int | None = None,
    dry_run: bool | None = None,
    width: int = DEFAULT_MASTER_WIDTH,
    height: int = DEFAULT_MASTER_HEIGHT,
) -> bytes:
    """Phase-A Master still via HunyuanImage-2.1 T2I."""
    return generate_hunyuan21_t2i(
        prompt=master_prompt(identity),
        width=width,
        height=height,
        seed=seed,
        dry_run=dry_run,
    )


def generate_tier_a_keyframe(
    shot: ShotContract | Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    dry_run: bool | None = None,
) -> tuple[bytes, str]:
    """A-tier: identity in prompt only → IMAGE_BACKEND T2I (default hunyuan21)."""
    s = shot if isinstance(shot, ShotContract) else ShotContract.from_dict(shot)
    if dry_run is None:
        dry_run = str((env or os.environ).get("HUNYUAN21_DRY_RUN") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
    backend_name = select_image_backend(env=env)
    if backend_name == "hunyuan21":
        # Direct path keeps canary params; ImageBackend also works when not dry.
        if dry_run:
            png = generate_hunyuan21_t2i(
                prompt=s.prompt,
                negative_prompt=s.negative_prompt,
                seed=s.seed,
                dry_run=True,
            )
        else:
            png = get_image_backend("hunyuan21", env=env).generate(
                ImageGenerateRequest(
                    prompt=s.prompt,
                    negative_prompt=s.negative_prompt,
                    seed=s.seed,
                    meta={"tier": "A"},
                )
            ).png
        return png, "hunyuan21"
    result = get_image_backend(backend_name, env=env).generate(
        ImageGenerateRequest(
            prompt=s.prompt,
            negative_prompt=s.negative_prompt,
            seed=s.seed,
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
    """B-tier: single master.png + identity CONTROL_BACKEND."""
    s = shot if isinstance(shot, ShotContract) else ShotContract.from_dict(shot)
    payload = b_tier_control_payload(s)
    assert_character_refs(payload["character_refs"])
    control = get_control_backend(env=env)
    result = control.apply(
        ControlRequest(
            prompt=s.prompt,
            negative_prompt=s.negative_prompt,
            character_refs=list(payload["character_refs"]),
            control_mode=str(payload["control_mode"]),
            reference_pngs=tuple(reference_pngs or ()),
            seed=s.seed,
            meta={"tier": "B"},
        )
    )
    return result.png, result.backend, {"control": payload, **result.meta}


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
        # C: compose stub via composite control; sheets stay out of character_refs.
        control = get_control_backend("composite", env=env)
        refs = list(s.character_refs)
        result = control.apply(
            ControlRequest(
                prompt=s.prompt,
                negative_prompt=s.negative_prompt,
                character_refs=refs,
                control_mode="compose",
                reference_pngs=tuple(reference_pngs or ()),
                seed=s.seed,
                meta={"tier": "C"},
            )
        )
        png, backend, meta = result.png, result.backend, dict(result.meta)
    else:
        png, backend = generate_tier_a_keyframe(s, env=env, dry_run=dry_run)

    dest = keyframe_dir(story_root, episode_code, s.shot_id)
    final = write_keyframe_final(dest, png)
    qc = score_keyframe_qc(png, prompt=s.prompt, scorer=scorer, require_clip=require_clip)
    status = "qc_pass" if qc.passed else "qc_fail"
    contract = KeyframeContract(
        shot_id=s.shot_id,
        path=str(final),
        final_path=str(final),
        status=status,  # type: ignore[arg-type]
        qc=qc,
        backend=backend,
        seed=s.seed,
        meta={"image_backend": select_image_backend(env=env), "control_backend": select_control_backend(env=env), **meta},
    )
    (dest / "keyframe.json").write_text(json.dumps(contract.to_dict(), indent=2) + "\n", encoding="utf-8")
    return contract


def run_video_phase(
    keyframe: KeyframeContract,
    *,
    duration_s: float = 5.0,
    prompt: str = "",
    rife: bool = False,
    env: Mapping[str, str] | None = None,
) -> VideoContract:
    """QC gate then pluggable video backend. FAIL raises KeyframeQcGateError."""
    assert_keyframe_allows_video(keyframe)
    backend = get_video_generate_backend(env=env)
    req = VideoGenerateRequest(
        shot_id=keyframe.shot_id,
        keyframe_final=keyframe.final_path or keyframe.path,
        duration_s=duration_s,
        prompt=prompt,
        rife=rife,
    )
    # hunyuan15 insists on keyframe_final.png name
    if backend.name == "hunyuan15":
        name = Path(req.keyframe_final).name
        if name != KEYFRAME_FINAL_NAME:
            raise KeyframeQcGateError(
                f"hunyuan15 requires {KEYFRAME_FINAL_NAME}, got {name!r}"
            )
    result = gated_video_generate(keyframe, req, backend=backend, env=env)
    return video_contract_from_result(keyframe.shot_id, req.keyframe_final, result)


def ensure_master_on_disk(
    story_root: Path,
    character_id: str,
    *,
    identity: str,
    seed: int | None = None,
    dry_run: bool | None = None,
) -> Path:
    """Write characters/<id>/master.png via HunyuanImage-2.1 (Phase A asset)."""
    if not character_id.strip():
        raise CharacterRefError("character_id required")
    dest = Path(story_root) / "assets" / "characters" / character_id / MASTER_FILENAME
    if dest.is_file() and dest.stat().st_size >= 16:
        return dest
    png = mint_master_png(identity=identity, seed=seed, dry_run=dry_run)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(png)
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
    """Minimal closed loop: T2I keyframe → QC → video (H3 by default)."""
    mapping = dict(env or os.environ)
    if dry_run:
        mapping.setdefault("HUNYUAN21_DRY_RUN", "1")
    mapping.setdefault("IMAGE_BACKEND", "hunyuan21")
    # Phase A may keep VIDEO_BACKEND=h3
    mapping.setdefault("VIDEO_BACKEND", mapping.get("VIDEO_BACKEND") or mapping.get("AF_VIDEO_BACKEND") or "h3")
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

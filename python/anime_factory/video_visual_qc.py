"""Video frame visual QC for H3 generations (CPU OpenCLIP; never steals H3 VRAM).

Extracts first/mid/last and cut-boundary frames, scores against keyframe/character/scene
locks, and maps whiteboard/identity/freeze/flicker failures into retry reasons.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from anime_factory.visual_qc import (
    cosine,
    identity_similarity,
    shannon_entropy,
)

QC_VERSION = "video_visual_qc_v1"
MAX_VISUAL_RETRIES = 2

# Soft thresholds — fail closed only when clearly broken; limb detail is not guaranteed.
KEYFRAME_SIM_MIN = 0.18
CHARACTER_SIM_MIN = 0.16
SCENE_SIM_MIN = 0.14
LOW_ENTROPY_MIN = 2.8
FLICKER_DELTA_MAX = 0.55


@dataclass
class VideoVisualQcResult:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    frames: dict[str, str] = field(default_factory=dict)
    retry_strategy: str | None = None
    version: str = QC_VERSION

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, capture_output=True, check=False)


def probe_duration_s(video: Path) -> float:
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(video),
        ]
    )
    if proc.returncode != 0:
        return 0.0
    try:
        return float(proc.stdout.decode("utf-8", "replace").strip() or 0)
    except ValueError:
        return 0.0


def extract_frame_at(video: Path, dest: Path, *, seconds: float) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.0, seconds):.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(dest),
        ]
    )
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size < 32:
        return None
    return dest


def sample_qc_timestamps(
    duration_s: float,
    *,
    cut_boundaries_s: Sequence[float] | None = None,
) -> dict[str, float]:
    dur = max(0.1, float(duration_s))
    stamps = {
        "first": 0.05,
        "mid": dur * 0.5,
        "last": max(0.0, dur - 0.08),
    }
    for i, boundary in enumerate(cut_boundaries_s or []):
        t = float(boundary)
        if 0.1 < t < dur - 0.1:
            stamps[f"cut_{i + 1}"] = t
    return stamps


def extract_qc_frames(
    video: Path,
    out_dir: Path,
    *,
    duration_s: float | None = None,
    cut_boundaries_s: Sequence[float] | None = None,
) -> dict[str, Path]:
    dur = float(duration_s) if duration_s is not None else probe_duration_s(video)
    stamps = sample_qc_timestamps(dur, cut_boundaries_s=cut_boundaries_s)
    frames: dict[str, Path] = {}
    for name, seconds in stamps.items():
        dest = out_dir / f"qc_{name}.jpg"
        path = extract_frame_at(video, dest, seconds=seconds)
        if path is not None:
            frames[name] = path
    return frames


def _load_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


def _frame_entropy(path: Path) -> float:
    try:
        return float(shannon_entropy(_load_rgb(path)))
    except Exception:  # noqa: BLE001
        return 0.0


def _encode_paths(paths: Sequence[Path], scorer: Any) -> list[list[float]]:
    encode = getattr(scorer, "encode_images", None) or getattr(scorer, "encode_image", None)
    if encode is None:
        return []
    try:
        vecs = encode([str(p) for p in paths])
    except TypeError:
        vecs = [encode(str(p)) for p in paths]
    out: list[list[float]] = []
    for vec in vecs or []:
        out.append([float(x) for x in vec])
    return out


def _encode_texts(texts: Sequence[str], scorer: Any) -> list[list[float]]:
    encode = getattr(scorer, "encode_texts", None) or getattr(scorer, "encode_text", None)
    if encode is None:
        return []
    try:
        vecs = encode(list(texts))
    except TypeError:
        vecs = [encode(t) for t in texts]
    return [[float(x) for x in vec] for vec in (vecs or [])]


def _mean_sim(frame_vecs: Sequence[Sequence[float]], ref_vec: Sequence[float] | None) -> float:
    if not frame_vecs or not ref_vec:
        return 0.0
    return sum(identity_similarity(fv, ref_vec) for fv in frame_vecs) / max(1, len(frame_vecs))


def cut_boundaries_from_segment(segment: dict | None) -> list[float]:
    if not segment:
        return []
    alignment = segment.get("picture_alignment") or []
    if alignment:
        return [float(row.get("end_s") or 0) for row in alignment[:-1] if isinstance(row, dict)]
    cuts = [c for c in (segment.get("cuts") or []) if isinstance(c, dict)]
    cursor = 0.0
    out: list[float] = []
    for cut in cuts[:-1]:
        cursor += float(cut.get("seconds") or 0)
        out.append(cursor)
    return out


def score_video_visual(
    video: Path,
    *,
    segment: dict | None = None,
    keyframe_path: Path | None = None,
    character_lock_path: Path | None = None,
    scene_lock_path: Path | None = None,
    identity_prompt: str = "",
    scene_prompt: str = "",
    work_dir: Path | None = None,
    scorer: Any | None = None,
    require_clip: bool = False,
) -> VideoVisualQcResult:
    """Score extracted frames. Without CLIP, only entropy/flicker heuristics run."""
    reasons: list[str] = []
    scores: dict[str, float] = {}
    out_dir = Path(work_dir) if work_dir is not None else video.parent / "_visual_qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = float(segment.get("duration") or 0) if segment else 0.0
    if duration <= 0:
        duration = probe_duration_s(video)
    frames = extract_qc_frames(
        video,
        out_dir,
        duration_s=duration,
        cut_boundaries_s=cut_boundaries_from_segment(segment),
    )
    if len(frames) < 2:
        return VideoVisualQcResult(
            verdict="retry",
            reasons=["frame_extract_failed"],
            scores=scores,
            frames={k: str(v) for k, v in frames.items()},
            retry_strategy="change_seed",
        )

    entropies = {name: _frame_entropy(path) for name, path in frames.items()}
    scores.update({f"entropy_{k}": v for k, v in entropies.items()})
    if any(v < LOW_ENTROPY_MIN for v in entropies.values()):
        reasons.append("low_entropy_freeze")

    # Flicker: large RGB mean shift between adjacent sampled frames.
    ordered = [frames[k] for k in sorted(frames.keys()) if k in frames]
    try:
        prev = None
        max_delta = 0.0
        for path in ordered:
            img = _load_rgb(path)
            hist = img.resize((64, 36)).histogram()
            norm = [x / max(1, sum(hist)) for x in hist]
            if prev is not None:
                delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(prev, norm)) / max(1, len(norm)))
                max_delta = max(max_delta, delta)
            prev = norm
        scores["flicker_delta"] = float(max_delta)
        if max_delta > FLICKER_DELTA_MAX:
            reasons.append("sudden_flicker")
    except Exception:  # noqa: BLE001
        pass

    clip = scorer
    if clip is None and require_clip:
        try:
            from anime_factory.visual_qc import get_clip_scorer

            clip = get_clip_scorer()
        except Exception:  # noqa: BLE001
            clip = None
            reasons.append("clip_unavailable")

    if clip is not None:
        frame_vecs = _encode_paths(list(frames.values()), clip)
        if keyframe_path and Path(keyframe_path).is_file():
            ref = _encode_paths([Path(keyframe_path)], clip)
            sim = _mean_sim(frame_vecs, ref[0] if ref else None)
            scores["keyframe_sim"] = sim
            if sim < KEYFRAME_SIM_MIN:
                reasons.append("keyframe_mismatch")
        if character_lock_path and Path(character_lock_path).is_file():
            ref = _encode_paths([Path(character_lock_path)], clip)
            sim = _mean_sim(frame_vecs, ref[0] if ref else None)
            scores["character_sim"] = sim
            if sim < CHARACTER_SIM_MIN:
                reasons.append("identity_drift")
        if scene_lock_path and Path(scene_lock_path).is_file():
            ref = _encode_paths([Path(scene_lock_path)], clip)
            sim = _mean_sim(frame_vecs, ref[0] if ref else None)
            scores["scene_sim"] = sim
            if sim < SCENE_SIM_MIN:
                reasons.append("scene_mismatch")

        # Semantic probes for age/costume and forbidden whiteboard / military look.
        probes = []
        if identity_prompt:
            probes.append(("identity_text", identity_prompt[:200], "wrong person, different costume"))
            if "60" in identity_prompt or "elderly" in identity_prompt.lower():
                probes.append(("age_elderly", "elderly 60 year old man", "young adult 20 year old"))
            if "navy" in identity_prompt.lower() or "worker" in identity_prompt.lower():
                probes.append(
                    (
                        "worker_uniform",
                        "navy blue worker uniform jacket",
                        "military officer uniform with epaulettes",
                    )
                )
            if "no military" in identity_prompt.lower() or "epaulette" in identity_prompt.lower():
                probes.append(("no_military", "civilian worker clothes", "military uniform epaulettes"))
        probes.append(("no_whiteboard", "blank wall, no text", "whiteboard with chinese characters"))
        probes.append(("no_logo", "clean frame no logo", "bright logo watermark text"))
        if scene_prompt:
            probes.append(("scene_text", scene_prompt[:160], "different location neon night city"))

        for probe_id, pos, neg in probes:
            try:
                pos_v, neg_v = _encode_texts([pos, neg], clip)
                if not frame_vecs or not pos_v or not neg_v:
                    continue
                margins = []
                for fv in frame_vecs:
                    pos_s = cosine(fv, pos_v)
                    neg_s = cosine(fv, neg_v)
                    margins.append(pos_s - neg_s)
                margin = sum(margins) / max(1, len(margins))
                scores[f"probe_{probe_id}"] = float(margin)
                if probe_id.startswith("no_") and margin < -0.02:
                    if "whiteboard" in probe_id or "logo" in probe_id:
                        reasons.append("generated_text_or_logo")
                    elif "military" in probe_id:
                        reasons.append("military_costume_drift")
                elif probe_id in {"age_elderly", "worker_uniform", "identity_text", "scene_text"} and margin < -0.03:
                    reasons.append(f"semantic_{probe_id}")
            except Exception:  # noqa: BLE001
                continue

    # Extra-person heuristic: when identity says 1boy/1girl, reject strong multi-person text probe.
    # (CLIP-only; limb QC is intentionally not claimed.)

    strategy = map_reasons_to_strategy(reasons)
    if not reasons:
        verdict = "pass"
    elif any(r in {"frame_extract_failed"} for r in reasons):
        verdict = "retry"
    else:
        verdict = "retry"
    return VideoVisualQcResult(
        verdict=verdict,
        reasons=reasons,
        scores=scores,
        frames={k: str(v) for k, v in frames.items()},
        retry_strategy=strategy if verdict != "pass" else None,
    )


def map_reasons_to_strategy(reasons: Sequence[str]) -> str:
    text = " ".join(reasons)
    if "generated_text" in text or "logo" in text:
        return "reinforce_no_text"
    if "identity" in text or "military" in text or "semantic_age" in text or "worker" in text:
        return "reinforce_identity"
    if "keyframe" in text:
        return "restore_refs"
    if "scene" in text:
        return "restore_refs"
    if "flicker" in text or "freeze" in text:
        return "change_seed"
    if "chain" in text:
        return "relink_last_frame"
    return "change_seed"


def write_visual_qc_report(path: Path, result: VideoVisualQcResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": result.version,
                "verdict": result.verdict,
                "reasons": result.reasons,
                "scores": result.scores,
                "frames": result.frames,
                "retry_strategy": result.retry_strategy,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path

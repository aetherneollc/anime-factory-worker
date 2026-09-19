"""Deterministic + identity/eyeline/chain QC. Repair writes a new generation, never overwrites.

Pre-H3 shipping gates (voice gender, staging collapse, prompt diversity,
speaker-vs-frame, freeze-pad) fail closed and trigger script/board rewrite.
They must not set waiting_for_human / human_gate.
"""

from __future__ import annotations

import filecmp
import json
import sqlite3
from pathlib import Path

from anime_factory.config import load_settings
from anime_factory.db import utcnow
from anime_factory.models import H3_MAX_RETRIES, VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH

# MiniMax H3 (and the Comfy path) snap clip length to a 17k+5 frame grid at
# 24 fps, so round(duration * 24) is only the real frame count when the
# duration already sits on the grid — 8.0s does, 5.5s does not (141, not 132).
# Mirrors gpu_worker.h3.h3_length_frames; anime_factory must not import gpu_worker.
H3_FRAME_GRID = 17
H3_FRAME_OFFSET = 5

# LongLive native Wan2.2-TI2V-5B output (1280×704 @ 24fps). Mirrors gpu_worker.longlive;
# anime_factory must not import gpu_worker.
LONGLIVE_NATIVE_WIDTH = 1280
LONGLIVE_NATIVE_HEIGHT = 704
LONGLIVE_TEMPORAL_COMPRESSION = 4
LONGLIVE_NUM_FRAME_PER_BLOCK = 8
LONGLIVE_MODES = frozenset({"i2v", "t2v"})


def h3_grid_frames(seconds: float) -> int:
    """Frames H3 actually renders for a requested duration."""
    frames = max(H3_FRAME_OFFSET, int(round(float(seconds) * VIDEO_FPS)))
    while frames % H3_FRAME_GRID != H3_FRAME_OFFSET:
        frames += 1
    return frames


def h3_grid_step_seconds() -> float:
    return H3_FRAME_GRID / float(VIDEO_FPS)


def longlive_latent_frames(seconds: float) -> int:
    """Latent frames for Wan 4× temporal compression, snapped to the AR block size."""
    video_frames = max(5, int(round(float(seconds) * VIDEO_FPS)))
    latents = (video_frames - 1) // LONGLIVE_TEMPORAL_COMPRESSION + 1
    while latents % LONGLIVE_NUM_FRAME_PER_BLOCK != 0:
        latents += 1
    return max(LONGLIVE_NUM_FRAME_PER_BLOCK, latents)


def longlive_video_frames(seconds: float) -> int:
    """Video frames implied by longlive_latent_frames (Wan 4× temporal, 24fps)."""
    latents = longlive_latent_frames(seconds)
    return max(1, (latents - 1) * LONGLIVE_TEMPORAL_COMPRESSION + 1)


def longlive_duration_step_seconds() -> float:
    """One 8-latent block → 32 video frames @ 24fps."""
    return (LONGLIVE_NUM_FRAME_PER_BLOCK * LONGLIVE_TEMPORAL_COMPRESSION) / float(VIDEO_FPS)


def record_qc(
    conn: sqlite3.Connection,
    episode_code: str,
    segment_id: str | None,
    gate: str,
    verdict: str,
    details: dict,
) -> int:
    # Existing CHECK(gate IN deterministic/text/visual) — fold new checks into visual.
    stored_gate = gate if gate in {"deterministic", "text", "visual"} else "visual"
    payload = dict(details)
    if stored_gate == "visual" and gate != "visual":
        payload.setdefault("check", gate)
    cur = conn.execute(
        """
        INSERT INTO qc_reports (episode_code, segment_id, gate, verdict, details_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (episode_code, segment_id, stored_gate, verdict, json.dumps(payload, ensure_ascii=False), utcnow()),
    )
    conn.commit()
    return int(cur.lastrowid)


def deterministic_check(
    meta: dict,
    segment_duration: float,
    used_mode: str | None = None,
) -> tuple[str, dict]:
    issues = []
    mode = str(used_mode or "")
    if mode in LONGLIVE_MODES:
        exp_w, exp_h = LONGLIVE_NATIVE_WIDTH, LONGLIVE_NATIVE_HEIGHT
        expected_frames = longlive_video_frames(segment_duration)
        duration_tol = longlive_duration_step_seconds() + 1e-6
    else:
        exp_w, exp_h = VIDEO_WIDTH, VIDEO_HEIGHT
        expected_frames = h3_grid_frames(segment_duration)
        duration_tol = h3_grid_step_seconds() + 1e-6
    if meta.get("width") != exp_w or meta.get("height") != exp_h:
        issues.append("resolution")
    if meta.get("frames") is not None and abs(meta["frames"] - expected_frames) > 1:
        issues.append("frames")
    if meta.get("duration") is not None and abs(meta["duration"] - segment_duration) > duration_tol:
        issues.append("duration")
    if meta.get("size_bytes", 10**9) < 4096:
        issues.append("filesize")
    if meta.get("black_frame_ratio", 0) > 0.4:
        issues.append("black_frames")
    if meta.get("require_speech"):
        if meta.get("has_audio_stream") is False:
            issues.append("no_audio_track")
        rms = meta.get("audio_rms")
        if rms is not None and float(rms) < 80.0:
            issues.append("silence")
    tol = load_settings().tts_speed_tolerance
    for lang, wav in (meta.get("wav_duration") or {}).items():
        if wav is not None and abs(wav - segment_duration) > segment_duration * tol + 1e-6:
            issues.append(f"av_sync_{lang}")
    verdict = "fail" if issues else "pass"
    return verdict, {"issues": issues, "meta": meta}


def identity_qc(segment: dict, used_mode: str | None = None) -> tuple[str, dict]:
    """Character shots must keep ref2va identity; chain tails mix last_frame with the same refs."""
    issues: list[str] = []
    mode = used_mode or str(segment.get("h3_mode") or "")
    character = str(segment.get("character_id") or "").strip()
    chain_index = int(segment.get("chain_index") or 0)
    refs = list(segment.get("refs") or [])
    has_char_ref = character or any("char" in str(r) or "sheet" in str(r) or "costume" in str(r) for r in refs)
    if character and chain_index == 0:
        if mode in {"fl2va_first", "fl2va_first_last"} and not segment.get("match_cut"):
            issues.append("character_used_i2v")
        if not refs:
            issues.append("missing_character_refs")
    if chain_index > 0:
        first = segment.get("first_frame_path")
        if not first:
            issues.append("chain_break")
        if has_char_ref:
            if mode not in {"ref2va", "i2v"}:
                issues.append("chain_dropped_refs")
            if not refs:
                issues.append("missing_character_refs")
        kf = str(segment.get("keyframe_path") or "")
        if kf.endswith("/f1.png") and first and Path(str(first)).name.endswith("f1.png"):
            if str(first) == kf:
                issues.append("chain_new_still")
    verdict = "retry" if issues else "pass"
    return verdict, {"issues": issues, "mode": mode, "check": "identity"}


def eyeline_qc(prev: dict | None, segment: dict) -> tuple[str, dict]:
    issues: list[str] = []
    if prev and prev.get("purpose") in {"coverage", "reverse"} and segment.get("purpose") in {"coverage", "reverse"}:
        prev_eye = prev.get("eyeline")
        eye = segment.get("eyeline")
        if prev_eye and eye and prev_eye == eye and prev.get("character_id") != segment.get("character_id"):
            issues.append("eyeline_jump")
    verdict = "retry" if issues else "pass"
    return verdict, {"issues": issues, "check": "eyeline"}


def _same_frame(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    if str(left) == str(right):
        return True
    a, b = Path(str(left)), Path(str(right))
    try:
        if a.is_file() and b.is_file():
            return a.resolve() == b.resolve() or filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False
    if a.name == b.name:
        return True
    return a.name in str(b) or b.name in str(a)


def chain_qc(prev: dict | None, segment: dict) -> tuple[str, dict]:
    issues: list[str] = []
    if int(segment.get("chain_index") or 0) > 0:
        if not prev or prev.get("chain_id") != segment.get("chain_id"):
            issues.append("chain_break")
        elif not _same_frame(prev.get("last_frame_path"), segment.get("first_frame_path")):
            issues.append("chain_break")
    verdict = "retry" if issues else "pass"
    return verdict, {"issues": issues, "check": "chain"}


def continuity_qc(prev: dict | None, segment: dict, used_mode: str | None = None) -> tuple[str, dict]:
    id_v, id_d = identity_qc(segment, used_mode)
    eye_v, eye_d = eyeline_qc(prev, segment)
    ch_v, ch_d = chain_qc(prev, segment)
    issues = list(id_d["issues"]) + list(eye_d["issues"]) + list(ch_d["issues"])
    if issues:
        return "retry", {"issues": issues, "identity": id_d, "eyeline": eye_d, "chain": ch_d}
    return "pass", {"issues": [], "identity": id_d, "eyeline": eye_d, "chain": ch_d}


def incremental_qc_segment(
    conn: sqlite3.Connection,
    episode_code: str,
    segment_id: str,
    meta: dict,
    duration: float,
    visual_verdict: str = "pass",
    visual_details: dict | None = None,
    segment: dict | None = None,
    prev_segment: dict | None = None,
    used_mode: str | None = None,
    video_path: Path | str | None = None,
    story_root: Path | str | None = None,
) -> str:
    d_verdict, d_details = deterministic_check(meta, duration, used_mode)
    record_qc(conn, episode_code, segment_id, "deterministic", d_verdict, d_details)
    record_qc(conn, episode_code, segment_id, "text", "pass", {"dialogue_unchanged": True})
    merged = dict(visual_details or {"frames": ["first", "mid", "last"]})
    v_verdict = visual_verdict
    if segment is not None:
        c_verdict, c_details = continuity_qc(prev_segment, segment, used_mode)
        merged["continuity"] = c_details
        if c_verdict != "pass" and v_verdict == "pass":
            v_verdict = c_verdict
    if video_path and v_verdict == "pass":
        try:
            from anime_factory.video_visual_qc import score_video_visual, write_visual_qc_report

            root = Path(story_root) if story_root else Path(video_path).parents[2]
            sid = str(segment_id)
            keyframe = root / "episodes" / episode_code / "keyframes" / sid / "f1.png"
            char_lock = None
            scene_lock = None
            identity = ""
            if segment:
                cid = str(segment.get("character_id") or "").strip()
                if cid:
                    cand = root / "assets" / "characters" / cid / "sheet_front.png"
                    if cand.is_file():
                        char_lock = cand
                plate = str(segment.get("plate_id") or segment.get("location_id") or "").strip()
                if plate.startswith("plate_"):
                    plate = plate[len("plate_") :]
                if plate:
                    cand = root / "assets" / "scenes" / plate / "plate_base.png"
                    if cand.is_file():
                        scene_lock = cand
                identity = str(segment.get("identity_prompt") or segment.get("first_frame_prompt") or "")
            result = score_video_visual(
                Path(video_path),
                segment=segment,
                keyframe_path=keyframe if keyframe.is_file() else None,
                character_lock_path=char_lock,
                scene_lock_path=scene_lock,
                identity_prompt=identity,
                work_dir=Path(video_path).parent / "_visual_qc",
                require_clip=False,
            )
            merged["video_visual"] = {
                "verdict": result.verdict,
                "reasons": result.reasons,
                "scores": result.scores,
                "retry_strategy": result.retry_strategy,
            }
            write_visual_qc_report(Path(video_path).parent / "visual_qc.json", result)
            if result.verdict != "pass" and v_verdict == "pass":
                v_verdict = result.verdict
        except Exception as exc:  # noqa: BLE001 — visual QC is additive; never wipe deterministic pass silently
            merged["video_visual_error"] = str(exc)[:400]
    record_qc(conn, episode_code, segment_id, "visual", v_verdict, merged)
    if d_verdict == "fail":
        return "fail"
    if v_verdict == "retry":
        return "retry"
    return v_verdict


def shot_version_path(segment_id: str, version: int) -> str:
    return f"shots/{segment_id}/generation-{version:03d}.mp4"


def legacy_shot_version_path(segment_id: str, version: int) -> str:
    return f"shots/{segment_id}/v{version:03d}.mp4"


def existing_generation_file(root: Path, segment_id: str) -> Path | None:
    """Legacy helper: first on-disk mp4. Prefer select_passing_generation for resume."""
    for version in range(1, 32):
        for rel in (shot_version_path(segment_id, version), legacy_shot_version_path(segment_id, version)):
            path = root / rel
            if path.is_file() and path.stat().st_size > 4096:
                return path
    return None


def _qc_score_total(details: Any) -> float:
    if isinstance(details, dict):
        scores = details.get("qc_scores") or details.get("scores") or details
        if isinstance(scores, dict):
            total = 0.0
            for key in ("keyframe_sim", "character_sim", "scene_sim"):
                try:
                    total += float(scores.get(key) or 0)
                except (TypeError, ValueError):
                    pass
            return total
    return 0.0


def select_passing_generation(
    conn: sqlite3.Connection | None,
    root: Path,
    segment_id: str,
) -> tuple[Path, int, str] | None:
    """Resume only a QC-passing generation. Tie-break: QC score, then highest version.

    Never auto-green the first mp4 found on disk.
    """
    candidates: list[tuple[float, int, Path, str]] = []
    if conn is not None:
        try:
            rows = conn.execute(
                """
                SELECT version, path, qc_verdict, status
                FROM generation_results
                WHERE segment_id = ? AND qc_verdict = 'pass'
                ORDER BY version DESC
                """,
                (segment_id,),
            ).fetchall()
        except Exception:  # noqa: BLE001 — older DBs may lack the table
            rows = []
        for row in rows:
            rel = str(row["path"] or "")
            # path may be stories/<id>/shots/... or shots/...
            name = Path(rel).name
            parent = Path(rel).parent.name
            local_candidates = [
                root / "shots" / segment_id / name,
                root / rel,
            ]
            if parent == segment_id:
                local_candidates.append(root / "shots" / segment_id / name)
            # Also try relative after stories/<id>/
            parts = Path(rel).parts
            if "shots" in parts:
                idx = parts.index("shots")
                local_candidates.append(root.joinpath(*parts[idx:]))
            path = next((p for p in local_candidates if p.is_file() and p.stat().st_size > 4096), None)
            if path is None:
                continue
            version = int(row["version"] or 0)
            score = 0.0
            try:
                qc_rows = conn.execute(
                    """
                    SELECT details_json FROM qc_reports
                    WHERE segment_id = ? AND check_name = 'visual'
                    ORDER BY created_at DESC LIMIT 4
                    """,
                    (segment_id,),
                ).fetchall()
                for qc_row in qc_rows:
                    try:
                        details = json.loads(qc_row["details_json"] or "{}")
                    except (TypeError, json.JSONDecodeError):
                        details = {}
                    score = max(score, _qc_score_total(details))
            except Exception:  # noqa: BLE001
                score = 0.0
            candidates.append((score, version, path, "pass"))
        # Prefer approved_generation_id when present.
        try:
            approved = conn.execute(
                "SELECT approved_generation_id, video_path, status FROM segments WHERE id = ?",
                (segment_id,),
            ).fetchone()
        except Exception:  # noqa: BLE001
            approved = None
        if approved and approved["status"] == "completed" and approved["video_path"]:
            rel = str(approved["video_path"])
            parts = Path(rel).parts
            path = None
            if "shots" in parts:
                idx = parts.index("shots")
                cand = root.joinpath(*parts[idx:])
                if cand.is_file() and cand.stat().st_size > 4096:
                    path = cand
            if path is not None:
                # Ensure it is among passing candidates or promote if DB says completed+approved.
                if not any(c[2] == path for c in candidates):
                    # Only trust approved when generation_results also says pass when available.
                    gid = str(approved["approved_generation_id"] or "")
                    ok = True
                    if gid:
                        try:
                            g_row = conn.execute(
                                "SELECT qc_verdict FROM generation_results WHERE id = ?",
                                (gid,),
                            ).fetchone()
                            if g_row is not None and str(g_row["qc_verdict"] or "") != "pass":
                                ok = False
                        except Exception:  # noqa: BLE001
                            pass
                    if ok:
                        candidates.append((1e9, 999, path, "approved"))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    best = candidates[0]
    return best[2], best[1], best[3]


def next_generation_path(root: Path | None, segment_id: str) -> tuple[int, str]:
    version = 1
    if root is not None:
        while True:
            rel = shot_version_path(segment_id, version)
            legacy = root / legacy_shot_version_path(segment_id, version)
            current = root / rel
            if current.is_file() or legacy.is_file():
                version += 1
                continue
            return version, rel
    return version, shot_version_path(segment_id, version)


def record_generation_result(
    conn: sqlite3.Connection,
    segment_id: str,
    version: int,
    path: str,
    seed: int | None,
    h3_mode: str | None,
    status: str,
    qc_verdict: str | None = None,
) -> str:
    gid = f"{segment_id}-g{version:03d}"
    conn.execute(
        """
        INSERT INTO generation_results (id, segment_id, version, path, seed, h3_mode, status, qc_verdict, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET status = excluded.status, qc_verdict = excluded.qc_verdict
        """,
        (gid, segment_id, version, path, seed, h3_mode, status, qc_verdict, utcnow()),
    )
    conn.commit()
    return gid


def plan_repair(segment: dict, issues: list[str], attempt: int) -> dict:
    seed = int(segment.get("seed") or 11) + attempt * 17
    strategy = "change_seed"
    prompt = str(segment.get("h3_prompt") or "")
    new_prompt = prompt
    joined = " ".join(issues)
    if "character_used_i2v" in issues or "missing_character_refs" in issues or "chain_dropped_refs" in issues:
        strategy = "restore_ref2va"
    elif "chain_break" in issues or "relink_last_frame" in joined:
        strategy = "relink_last_frame"
    elif "eyeline_jump" in issues:
        strategy = "flip_eyeline_prompt"
        new_prompt = f"{prompt}, hold established eyeline, do not cross the 180 line"
    elif "generated_text" in joined or "logo" in joined or "reinforce_no_text" in joined:
        strategy = "reinforce_no_text"
        new_prompt = (
            f"{prompt}, no readable text, no whiteboard, no blackboard writing, "
            "no logos, no chinese characters on surfaces"
        )
    elif (
        "identity" in joined
        or "military" in joined
        or "reinforce_identity" in joined
        or "semantic_age" in joined
        or "worker_uniform" in joined
    ):
        strategy = "reinforce_identity"
        new_prompt = (
            f"{prompt}, keep exact locked face age and navy worker uniform, "
            "no military uniform, no epaulettes, no costume swap"
        )
    elif "keyframe" in joined or "restore_refs" in joined or "scene_mismatch" in joined:
        strategy = "restore_refs"
    elif attempt >= 1:
        strategy = "change_seed"
    return {
        "strategy": strategy,
        "seed": seed,
        "h3_prompt": new_prompt,
        "h3_mode": "ref2va" if strategy == "restore_ref2va" else segment.get("h3_mode"),
        "attempt": attempt,
    }


def open_repair_task(
    conn: sqlite3.Connection,
    segment_id: str,
    issues: list[str],
    attempt: int,
    old_prompt: str | None,
    repair: dict,
) -> str:
    tid = f"{segment_id}-repair-{attempt}"
    conn.execute(
        """
        INSERT INTO repair_tasks (id, target_type, target_id, failure_type, diagnosis, repair_strategy,
            old_prompt, new_prompt, attempt, max_attempt, status, created_at)
        VALUES (?, 'segment', ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
        ON CONFLICT(id) DO UPDATE SET status = 'open', diagnosis = excluded.diagnosis
        """,
        (
            tid,
            segment_id,
            ",".join(issues) or "qc_fail",
            json.dumps(issues, ensure_ascii=False),
            repair.get("strategy"),
            old_prompt,
            repair.get("h3_prompt"),
            attempt,
            H3_MAX_RETRIES,
            utcnow(),
        ),
    )
    conn.commit()
    return tid


def apply_retry(conn: sqlite3.Connection, segment_id: str) -> tuple[int, str]:
    row = conn.execute("SELECT retry_count FROM segments WHERE id = ?", (segment_id,)).fetchone()
    n = int(row["retry_count"] if row else 0) + 1
    if n > H3_MAX_RETRIES:
        conn.execute("UPDATE segments SET status = 'failed', retry_count = ? WHERE id = ?", (n, segment_id))
        conn.commit()
        return n, shot_version_path(segment_id, n)
    # Original is generation-001; first retry is generation-002. Never overwrite.
    path = shot_version_path(segment_id, n + 1)
    conn.execute(
        "UPDATE segments SET retry_count = ?, video_path = ?, status = 'running' WHERE id = ?",
        (n, path, segment_id),
    )
    conn.commit()
    return n, path


def passing_index(conn: sqlite3.Connection, episode_code: str) -> dict[str, str]:
    """Keep only the passing version in the index."""
    index: dict[str, str] = {}
    for row in conn.execute(
        "SELECT id, video_path, status FROM segments WHERE episode_code = ?",
        (episode_code,),
    ):
        if row["status"] == "completed" and row["video_path"]:
            index[row["id"]] = row["video_path"]
    return index


def mark_completed_passing(conn: sqlite3.Connection, segment_id: str, video_path: str) -> None:
    gid = None
    try:
        row = conn.execute(
            """
            SELECT id FROM generation_results
            WHERE segment_id = ? AND qc_verdict = 'pass'
            ORDER BY version DESC LIMIT 1
            """,
            (segment_id,),
        ).fetchone()
        if row:
            gid = row["id"]
    except Exception:  # noqa: BLE001
        gid = None
    if gid:
        try:
            conn.execute(
                """
                UPDATE segments
                SET status = 'completed', video_path = ?, approved_generation_id = ?
                WHERE id = ?
                """,
                (video_path, gid, segment_id),
            )
        except Exception:  # noqa: BLE001 — approved_generation_id may be absent
            conn.execute(
                "UPDATE segments SET status = 'completed', video_path = ? WHERE id = ?",
                (video_path, segment_id),
            )
    else:
        conn.execute(
            "UPDATE segments SET status = 'completed', video_path = ? WHERE id = ?",
            (video_path, segment_id),
        )
    conn.commit()

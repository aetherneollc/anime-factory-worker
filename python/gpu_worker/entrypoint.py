"""GPU worker entrypoint: register, heartbeat, pull sqlite, anim∥qc, compose, checkpoint."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from anime_factory.compose import NON_GPU_STAGES, MissingShotError, collect_shots, non_gpu_stages_passed
from anime_factory.db import checkpoint_and_upload, open_db
from anime_factory.models import VIDEO_HEIGHT, VIDEO_WIDTH
from anime_factory.qc import (
    apply_retry,
    h3_grid_frames,
    incremental_qc_segment,
    mark_completed_passing,
    passing_index,
    shot_version_path,
)
from gpu_worker.h3 import prepare_workflow, prune_empty_last_frame
from gpu_worker.registry import GpuRegistry


class PreflightError(RuntimeError):
    pass


def preflight_non_gpu(flags: dict[str, str]) -> None:
    if not non_gpu_stages_passed(flags):
        missing = [s for s in NON_GPU_STAGES if flags.get(s) != "passed"]
        raise PreflightError(f"GPU session refused; non-GPU stages not passed: {missing}")


def _default_qc_meta(sid: str, durations: dict[str, float]) -> dict:
    duration = durations.get(sid, 5.5)
    return {
        "width": VIDEO_WIDTH,
        "height": VIDEO_HEIGHT,
        "duration": duration,
        "frames": h3_grid_frames(duration),
        "size_bytes": 50_000,
    }


def run_anim_qc_incremental(
    conn: sqlite3.Connection,
    episode_code: str,
    segment_ids: list[str],
    qc_meta: dict[str, dict],
    durations: dict[str, float],
    retry_verdicts: dict[str, str] | None = None,
) -> dict[str, str]:
    """Simulate anim∥qc: one qc_reports triple per segment; retry writes v002."""
    retry_verdicts = retry_verdicts or {}
    for sid in segment_ids:
        path = shot_version_path(sid, 1)
        conn.execute(
            "UPDATE segments SET video_path = ?, status = 'running', retry_count = 0 WHERE id = ?",
            (path, sid),
        )
        conn.commit()
        verdict = incremental_qc_segment(
            conn,
            episode_code,
            sid,
            qc_meta.get(sid, _default_qc_meta(sid, durations)),
            durations.get(sid, 5.5),
            visual_verdict=retry_verdicts.get(sid, "pass"),
        )
        if verdict == "retry":
            n, path = apply_retry(conn, sid)
            incremental_qc_segment(
                conn,
                episode_code,
                sid,
                qc_meta.get(sid, _default_qc_meta(sid, durations)),
                durations.get(sid, 5.5),
                visual_verdict="pass",
            )
            mark_completed_passing(conn, sid, path)
        elif verdict == "pass":
            mark_completed_passing(conn, sid, path)
        else:
            conn.execute("UPDATE segments SET status = 'failed' WHERE id = ?", (sid,))
            conn.commit()
    return passing_index(conn, episode_code)


def checkpoint(conn: sqlite3.Connection, dest: Path) -> Path:
    return checkpoint_and_upload(conn, dest)


def main_dry_run(registry: GpuRegistry, instance_id: str) -> dict:
    registry.register(instance_id)
    registry.heartbeat(instance_id, "idle")
    return {"registered": instance_id, "workers": len(registry.workers)}

"""Poll the Cloudflare Worker for waiting_gpu jobs. No SSH from the Worker.

The box (af-start) pulls work after boot: GET /gpu/work, POST /gpu/claim,
then run_gpu_episode. CONTROL_PLANE_URL + optional AF_STORY_ID.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import quote
from typing import Any

AGENT_KEY_HEADER = "X-Studio-Agent-Key"


def control_headers(*, json_body: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "AnimeFactoryGPU/1.0",
        AGENT_KEY_HEADER: os.environ.get("STUDIO_AGENT_KEY") or "",
    }
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _get_json(url: str, timeout: float = 20) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=control_headers(), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def _post_json(url: str, payload: dict[str, Any], timeout: float = 20) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers=control_headers(json_body=True),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def parse_work_payload(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    jobs = payload.get("jobs") or payload.get("items") or []
    waiting = [j for j in jobs if str(j.get("status") or "") == "waiting_gpu"]
    if waiting:
        return waiting
    stories = payload.get("stories") or []
    out = []
    for s in stories:
        sid = str(s.get("story_id") or "").strip()
        if not sid:
            continue
        st = str(s.get("status") or s.get("story_status") or "").strip().lower()
        until = str(s.get("until_stage") or "").strip().lower()
        if (
            s.get("archived") is True
            or s.get("gpu_done") is True
            or st in {"archived", "finished"}
            or until in {"archive", "finished"}
        ):
            continue
        out.append(
            {
                "story_id": sid,
                "episode_code": s.get("episode_code") or "EP001",
                "stage": s.get("stage") or "anim",
                "status": "waiting_gpu",
                "story_status": s.get("status") or s.get("story_status"),
                "until_stage": s.get("until_stage"),
            }
        )
    return out


def parse_batch(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload or payload.get("batch") is None:
        return None
    batch = payload.get("batch")
    if not isinstance(batch, dict):
        return None
    episodes = batch.get("episodes")
    if not isinstance(episodes, list):
        return None
    return batch


def _gpu_item_closed(item: dict[str, Any]) -> bool:
    st = str(item.get("status") or item.get("story_status") or "").strip().lower()
    ep = str(item.get("episode_status") or "").strip().lower()
    until = str(item.get("until_stage") or "").strip().lower()
    return (
        item.get("archived") is True
        or item.get("gpu_done") is True
        or st in {"archived", "finished"}
        or ep in {"archived", "finished"}
        or until in {"archive", "finished"}
    )


def parse_hitchhikers(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    items = payload.get("hitchhikers") or []
    if not isinstance(items, list):
        return []
    return [
        h
        for h in items
        if isinstance(h, dict) and str(h.get("story_id") or "").strip() and not _gpu_item_closed(h)
    ]


def parse_pre_gpu(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    items = payload.get("pre_gpu") or []
    if not isinstance(items, list):
        return []
    return [p for p in items if isinstance(p, dict) and str(p.get("story_id") or "").strip()]


def pick_job(jobs: list[dict[str, Any]], story_id: str = "") -> dict[str, Any] | None:
    if not jobs:
        return None
    want = (story_id or "").strip()
    if want:
        for job in jobs:
            if str(job.get("story_id") or "") == want:
                return job
    return jobs[0]


def fetch_work(base: str, instance_id: str = "") -> dict[str, Any]:
    url = base.rstrip("/") + "/gpu/work"
    iid = (instance_id or os.environ.get("VAST_INSTANCE_ID") or "").strip()
    if iid:
        url += "?vast_instance_id=" + quote(iid)
    try:
        return _get_json(url)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}", "batch": None, "jobs": []}


def claim_job(base: str, instance_id: str, story_id: str, episode_code: str = "EP001") -> dict[str, Any]:
    url = base.rstrip("/") + "/gpu/claim"
    try:
        return _post_json(
            url,
            {
                "vast_instance_id": instance_id,
                "story_id": story_id,
                "episode_code": episode_code,
            },
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}"}


def report_episode(base: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = base.rstrip("/") + "/gpu/report"
    try:
        return _post_json(url, payload)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}"}


def next_assigned_job(
    base: str,
    instance_id: str,
    story_id: str = "",
) -> dict[str, Any] | None:
    """Prefer AF_STORY_ID from the lease env; otherwise claim the first waiting_gpu job."""
    payload = fetch_work(base, instance_id)
    batch = parse_batch(payload)
    if batch:
        sid = str(batch.get("story_id") or "")
        episodes = batch.get("episodes") or []
        if sid and episodes:
            episode = episodes[0] if isinstance(episodes[0], dict) else {}
            job = {
                **episode,
                "story_id": sid,
                "batch_id": batch.get("batch_id"),
                "status": "waiting_gpu",
            }
            claim_job(
                base,
                instance_id,
                sid,
                str(job.get("episode_code") or os.environ.get("AF_EPISODE") or "EP001"),
            )
            return job
    job = pick_job(parse_work_payload(payload), story_id)
    if not job:
        if story_id:
            return {"story_id": story_id, "episode_code": os.environ.get("AF_EPISODE") or "EP001", "status": "waiting_gpu"}
        return None
    sid = str(job.get("story_id") or "")
    ep = str(job.get("episode_code") or os.environ.get("AF_EPISODE") or "EP001")
    if sid and instance_id:
        claim_job(base, instance_id, sid, ep)
    return job

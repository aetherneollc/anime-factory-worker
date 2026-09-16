"""geo / timeline / continuity / wiki export-import. JSON is a human-readable mirror of sqlite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from anime_factory.db import utcnow

WIKI_DIRS = ("characters", "locations", "events", "terms")


def export_geo(conn: sqlite3.Connection, story_id: str, invented: bool, sources: list[dict] | None = None) -> dict:
    locations = conn.execute("SELECT * FROM locations").fetchall()
    regions, settlements, landmarks = [], [], []
    for loc in locations:
        coord = None
        keys = loc.keys()
        if "coord_json" in keys and loc["coord_json"]:
            coord = json.loads(loc["coord_json"])
        item = {
            "id": loc["id"],
            "name": loc["name"],
            "aka": json.loads(loc["aka_json"] or "[]"),
            "type": loc["type"],
            "coord": coord,
            "parent": loc["parent_id"],
        }
        kind = loc["type"] or "settlement"
        if kind == "region":
            regions.append(item)
        elif kind in {"settlement", "city", "town"}:
            settlements.append(item)
        else:
            landmarks.append(item)
    row = conn.execute("SELECT world_mode FROM story WHERE id = ?", (story_id,)).fetchone()
    world_mode = row["world_mode"] if row else "fiction"
    edges = [
        {
            "from": e["from_id"],
            "to": e["to_id"],
            "mode": e["mode"],
            "distance_km": e["distance_km"],
            "travel_time": e["travel_time"],
        }
        for e in conn.execute("SELECT * FROM geo_edges").fetchall()
    ]
    interiors = [
        {
            "id": it["id"],
            "name": it["name"],
            "parent": it["parent"],
            "scene_id": it["scene_id"],
            "asset_path": interior_asset_path(it),
        }
        for it in conn.execute("SELECT * FROM geo_interiors").fetchall()
    ]
    return {
        "story_id": story_id,
        "world_mode": world_mode,
        "invented": invented,
        "regions": regions,
        "settlements": settlements,
        "landmarks": landmarks,
        "edges": edges,
        "interiors": interiors,
        "sources": sources if sources is not None else [],
    }


def export_timeline(conn: sqlite3.Connection, story_id: str) -> dict:
    events = []
    for row in conn.execute("SELECT * FROM timeline_events ORDER BY date, id"):
        events.append(
            {
                "id": row["id"],
                "date": row["date"],
                "title": row["title"],
                "kind": row["kind"],
                "episode": row["episode"],
                "sources": json.loads(row["sources_json"] or "[]"),
                "body": row["body"],
                "invented": bool(row["invented"]),
            }
        )
    return {"story_id": story_id, "events": events}


def export_continuity(conn: sqlite3.Connection, story_id: str, episode_code: str | None = None) -> dict:
    if episode_code:
        row = conn.execute(
            "SELECT * FROM continuity WHERE episode_code = ?", (episode_code,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM continuity ORDER BY version DESC LIMIT 1"
        ).fetchone()
    if row:
        payload = json.loads(row["payload_json"])
        payload.setdefault("story_id", story_id)
        payload.setdefault("after_episode", row["episode_code"])
        payload.setdefault("version", row["version"])
        return payload
    return empty_continuity(story_id, episode_code or "EP000")


def empty_continuity(story_id: str, after_episode: str) -> dict:
    return {
        "story_id": story_id,
        "after_episode": after_episode,
        "version": 0,
        "characters": [],
        "locations": [],
        "props": [],
        "positions": {},
        "open_threads": [],
    }


def save_continuity(conn: sqlite3.Connection, episode_code: str, version: int, payload: dict) -> None:
    conn.execute(
        """
        INSERT INTO continuity (episode_code, version, payload_json, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(episode_code) DO UPDATE SET
            version = excluded.version,
            payload_json = excluded.payload_json,
            created_at = excluded.created_at
        """,
        (episode_code, version, json.dumps(payload, ensure_ascii=False), utcnow()),
    )
    conn.commit()


def write_canon_dir(root: Path, geo: dict, timeline: dict, continuity: dict) -> None:
    canon = root / "canon"
    canon.mkdir(parents=True, exist_ok=True)
    (canon / "geo.json").write_text(json.dumps(geo, ensure_ascii=False, indent=2), encoding="utf-8")
    (canon / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (canon / "continuity.json").write_text(
        json.dumps(continuity, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def regen_wiki(root: Path, continuity: dict, timeline: dict | None = None) -> list[Path]:
    """Derivative wiki from continuity + timeline. Four dirs always exist; files may be empty."""
    wiki = root / "canon" / "wiki"
    written: list[Path] = []
    for name in WIKI_DIRS:
        d = wiki / name
        d.mkdir(parents=True, exist_ok=True)

    for ch in continuity.get("characters") or []:
        path = wiki / "characters" / f"{ch['id']}.md"
        path.write_text(_wiki_md("character", ch), encoding="utf-8")
        written.append(path)
    for loc in continuity.get("locations") or []:
        path = wiki / "locations" / f"{loc['id']}.md"
        path.write_text(_wiki_md("location", loc), encoding="utf-8")
        written.append(path)
    for event in (timeline or {}).get("events") or []:
        path = wiki / "events" / f"{event['id']}.md"
        path.write_text(_wiki_md("event", event), encoding="utf-8")
        written.append(path)
    for thread in continuity.get("open_threads") or []:
        path = wiki / "terms" / f"{thread['id']}.md"
        path.write_text(_wiki_md("term", thread), encoding="utf-8")
        written.append(path)

    for name in WIKI_DIRS:
        d = wiki / name
        if not any(d.iterdir()):
            placeholder = d / ".keep"
            placeholder.write_text("", encoding="utf-8")
            written.append(placeholder)
    return written


def _wiki_md(kind: str, payload: dict) -> str:
    lines = [f"# {payload.get('name', payload.get('title', payload.get('id')))}", "", f"kind: {kind}"]
    for k, v in payload.items():
        if k in {"name", "title", "id"}:
            continue
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def set_geo_graph(conn: sqlite3.Connection, edges: list[dict], interiors: list[dict]) -> None:
    conn.execute("DELETE FROM geo_edges")
    conn.execute("DELETE FROM geo_interiors")
    for edge in edges:
        conn.execute(
            """
            INSERT INTO geo_edges (from_id, to_id, mode, distance_km, travel_time)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                edge["from"],
                edge["to"],
                edge.get("mode") or "walk",
                edge.get("distance_km"),
                edge.get("travel_time"),
            ),
        )
    for interior in interiors:
        conn.execute(
            """
            INSERT INTO geo_interiors (id, name, parent, scene_id, asset_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                interior["id"],
                interior.get("name"),
                interior.get("parent"),
                interior.get("scene_id"),
                interior_asset_path(interior),
            ),
        )
    conn.commit()


def interior_asset_path(interior: Any) -> str | None:
    """Default plate path so oneshot interiors land on an existing location plate."""
    if isinstance(interior, dict):
        existing = interior.get("asset_path")
        scene_id = interior.get("scene_id") or interior.get("parent") or interior.get("id")
    else:
        existing = interior["asset_path"] if "asset_path" in interior.keys() else None
        scene_id = interior["scene_id"] or (interior["parent"] if "parent" in interior.keys() else None) or interior["id"]
    if existing:
        return str(existing)
    if scene_id:
        return f"assets/scenes/{scene_id}/plate_base.png"
    return None


def hydrate_geo_interiors(conn: sqlite3.Connection, geo: dict | None = None) -> dict:
    """Merge sqlite geo_interiors into geo.json. Gate 3 must not die when the JSON mirror is empty."""
    payload = dict(geo or {})
    by_id: dict[str, dict] = {}
    for item in payload.get("interiors") or []:
        iid = str(item.get("id") or "").strip()
        if iid:
            row = dict(item)
            row["id"] = iid
            if not row.get("asset_path"):
                row["asset_path"] = interior_asset_path(row)
            by_id[iid] = row
    try:
        rows = conn.execute("SELECT id, name, parent, scene_id, asset_path FROM geo_interiors").fetchall()
    except sqlite3.Error:
        rows = []
    for row in rows:
        iid = str(row["id"] or "").strip()
        if not iid:
            continue
        item = {
            "id": iid,
            "name": row["name"],
            "parent": row["parent"],
            "scene_id": row["scene_id"],
            "asset_path": interior_asset_path(row),
        }
        existing = by_id.get(iid)
        if existing is None:
            by_id[iid] = item
            continue
        for field in ("name", "parent", "scene_id", "asset_path"):
            if not existing.get(field) and item.get(field):
                existing[field] = item[field]
    payload["interiors"] = list(by_id.values())
    return payload


def load_geo_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

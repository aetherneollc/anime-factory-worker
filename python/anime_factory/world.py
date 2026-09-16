"""Bible + world_mode + Metaso-backed period research. Three modes share one data shape."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from anime_factory.canon import (
    empty_continuity,
    export_geo,
    export_timeline,
    regen_wiki,
    set_geo_graph,
    write_canon_dir,
)
from anime_factory.db import utcnow
from anime_factory.langs import normalize_langs
from anime_factory.models import DEFAULT_STYLE_PRESET, METASO_DEFAULT_BASE_URL, STYLE_PREFIX, FIXED_NEGATIVE

MODE_WALK_KMH = {"walk": 5.0, "horse": 15.0, "boat": 8.0, "rail": 40.0}

POSITIVE_HEADING = re.compile(r"^#{1,3}\s*正面清单", re.M)
NEGATIVE_HEADING = re.compile(r"^#{1,3}\s*负面清单", re.M)


@dataclass
class PeriodLists:
    positive: list[str]
    negative: list[str]


def parse_period_md(text: str) -> PeriodLists:
    pos, neg = [], []
    section = None
    for line in text.splitlines():
        if POSITIVE_HEADING.match(line.strip()):
            section = "pos"
            continue
        if NEGATIVE_HEADING.match(line.strip()):
            section = "neg"
            continue
        if line.startswith("#"):
            section = None
            continue
        item = line.strip().lstrip("-*").strip()
        if not item or section is None:
            continue
        if section == "pos":
            pos.append(item)
        else:
            neg.append(item)
    return PeriodLists(positive=pos, negative=neg)


def parse_travel_minutes(value: str) -> float:
    if not value:
        return 0.0
    hours = minutes = 0.0
    mh = re.search(r"(\d+(?:\.\d+)?)h", value)
    mm = re.search(r"(\d+(?:\.\d+)?)m", value)
    if mh:
        hours = float(mh.group(1))
    if mm:
        minutes = float(mm.group(1))
    if not mh and not mm:
        minutes = float(value)
    return hours * 60.0 + minutes


def travel_time_contradictions(geo: dict, slack: float = 0.35) -> list[str]:
    """Internal consistency: distance vs travel_time given era transport speeds."""
    issues: list[str] = []
    for edge in geo.get("edges") or []:
        km = float(edge.get("distance_km") or 0)
        mode = edge.get("mode") or "walk"
        speed = MODE_WALK_KMH.get(mode, 5.0)
        minutes = parse_travel_minutes(str(edge.get("travel_time") or "0m"))
        if km <= 0 or minutes <= 0:
            issues.append(f"edge {edge.get('from')}->{edge.get('to')} missing distance or travel_time")
            continue
        implied_kmh = km / (minutes / 60.0)
        if abs(implied_kmh - speed) / speed > slack:
            issues.append(
                f"travel_time contradiction {edge.get('from')}->{edge.get('to')}: "
                f"{km}km in {minutes:.0f}m via {mode} implies {implied_kmh:.1f} km/h "
                f"(expected ~{speed} km/h)"
            )
    return issues


def name_vs_aka_violations(geo: dict, text: str | None = None) -> list[str]:
    """Official `name` must not be a later aka (e.g. 北平 vs 京师)."""
    issues: list[str] = []
    names: dict[str, dict] = {}
    aka_to_official: dict[str, str] = {}
    for bucket in ("regions", "settlements", "landmarks"):
        for item in geo.get(bucket) or []:
            names[item["id"]] = item
            for aka in item.get("aka") or []:
                aka_to_official[aka] = item["name"]
                if aka == item["name"]:
                    issues.append(f"{item['id']}: name {item['name']!r} also listed in aka")
    haystack = text or ""
    for aka, official in aka_to_official.items():
        if aka and aka != official and aka in haystack:
            # allow aka when explicitly marked as alias
            if f"aka:{aka}" in haystack or f"（{aka}）" in haystack:
                continue
            issues.append(f"later name {aka!r} used as current name; official is {official!r}")
    for item in names.values():
        for aka in item.get("aka") or []:
            if item["name"] == aka:
                issues.append(f"{item['id']} uses aka as official name")
    return issues


def write_bible(root: Path, world_md: str, style_md: str | None, cast: dict, period_md: str) -> None:
    bible = root / "bible"
    bible.mkdir(parents=True, exist_ok=True)
    (bible / "world.md").write_text(world_md, encoding="utf-8")
    style = style_md if style_md is not None else f"{STYLE_PREFIX}\n\nnegative: {FIXED_NEGATIVE}\n"
    (bible / "style.md").write_text(style, encoding="utf-8")
    (bible / "cast.json").write_text(json.dumps(cast, ensure_ascii=False, indent=2), encoding="utf-8")
    (bible / "period.md").write_text(period_md, encoding="utf-8")


def init_fiction_world(
    conn: sqlite3.Connection,
    story_id: str,
    title: str,
    locations: list[dict],
    edges: list[dict],
    interiors: list[dict],
    timeline: list[dict],
    root: Path | None = None,
    langs: Sequence[str] | None = None,
) -> dict:
    now = utcnow()
    story_lang_list = list(normalize_langs(langs))
    conn.execute(
        """
        INSERT INTO story (id, title, slug, world_mode, style_preset, langs, primary_lang,
                           version, planned_episodes, schema_version, canon_token_count,
                           created_at, updated_at)
        VALUES (?, ?, ?, 'fiction', ?, ?, ?, '1.0.0', 1, 'v1', 0, ?, ?)
        ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
        """,
        (
            story_id,
            title,
            story_id.rsplit("-", 1)[0],
            DEFAULT_STYLE_PRESET,
            json.dumps(story_lang_list),
            story_lang_list[0],
            now,
            now,
        ),
    )
    for loc in locations:
        conn.execute(
            """
            INSERT INTO locations (id, name, aka_json, type, parent_id, plate_variant, damage_state, season, coord_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET name = excluded.name, coord_json = excluded.coord_json
            """,
            (
                loc["id"],
                loc["name"],
                json.dumps(loc.get("aka") or [], ensure_ascii=False),
                loc.get("type", "settlement"),
                loc.get("parent"),
                loc.get("plate_variant"),
                loc.get("damage_state"),
                loc.get("season"),
                json.dumps(loc["coord"], ensure_ascii=False) if loc.get("coord") is not None else None,
            ),
        )
    for ev in timeline:
        conn.execute(
            """
            INSERT INTO timeline_events (id, date, title, kind, episode, body, sources_json, invented)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (
                ev["id"],
                ev.get("date"),
                ev.get("title"),
                ev.get("kind", "fictional"),
                ev.get("episode"),
                ev.get("body"),
                json.dumps(ev.get("sources") or [], ensure_ascii=False),
                1 if ev.get("invented", True) else 0,
            ),
        )
    set_geo_graph(conn, edges, interiors)
    conn.commit()
    geo = export_geo(conn, story_id, invented=True, sources=[])
    tl = export_timeline(conn, story_id)
    continuity = empty_continuity(story_id, "EP000")
    if root is not None:
        write_canon_dir(root, geo, tl, continuity)
        regen_wiki(root, continuity, tl)
        write_bible(
            root,
            world_md=f"# {title}\n\ninvented: true\n",
            style_md=None,
            cast={},
            period_md="# Period\n\n## 正面清单\n\n## 负面清单\n",
        )
    return {"geo": geo, "timeline": tl, "continuity": continuity}


def init_period_world(
    conn: sqlite3.Connection,
    story_id: str,
    title: str,
    world_mode: str,
    locations: list[dict],
    edges: list[dict],
    interiors: list[dict],
    timeline: list[dict],
    period_md: str,
    sources: list[dict],
    root: Path | None = None,
    langs: Sequence[str] | None = None,
) -> dict:
    now = utcnow()
    invented = False
    story_lang_list = list(normalize_langs(langs))
    conn.execute(
        """
        INSERT INTO story (id, title, slug, world_mode, style_preset, langs, primary_lang,
                           version, planned_episodes, schema_version, canon_token_count,
                           created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, '1.0.0', 1, 'v1', 0, ?, ?)
        ON CONFLICT(id) DO UPDATE SET world_mode = excluded.world_mode, updated_at = excluded.updated_at
        """,
        (
            story_id,
            title,
            story_id.rsplit("-", 1)[0],
            world_mode,
            DEFAULT_STYLE_PRESET,
            json.dumps(story_lang_list),
            story_lang_list[0],
            now,
            now,
        ),
    )
    for loc in locations:
        conn.execute(
            """
            INSERT INTO locations (id, name, aka_json, type, parent_id, plate_variant, damage_state, season, coord_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET name = excluded.name, coord_json = excluded.coord_json
            """,
            (
                loc["id"],
                loc["name"],
                json.dumps(loc.get("aka") or [], ensure_ascii=False),
                loc.get("type", "region"),
                loc.get("parent"),
                None,
                None,
                None,
                json.dumps(loc["coord"], ensure_ascii=False) if loc.get("coord") is not None else None,
            ),
        )
    for ev in timeline:
        conn.execute(
            """
            INSERT INTO timeline_events (id, date, title, kind, episode, body, sources_json, invented)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(id) DO NOTHING
            """,
            (
                ev["id"],
                ev.get("date"),
                ev.get("title"),
                ev.get("kind", "real"),
                ev.get("episode"),
                ev.get("body"),
                json.dumps(ev.get("sources") or sources, ensure_ascii=False),
            ),
        )
    set_geo_graph(conn, edges, interiors)
    conn.commit()
    geo = export_geo(conn, story_id, invented=invented, sources=sources)
    tl = export_timeline(conn, story_id)
    continuity = empty_continuity(story_id, "EP000")
    if root is not None:
        write_canon_dir(root, geo, tl, continuity)
        regen_wiki(root, continuity, tl)
        write_bible(root, world_md=f"# {title}\n", style_md=None, cast={}, period_md=period_md)
    return {"geo": geo, "timeline": tl, "period": parse_period_md(period_md)}


class MetasoClient:
    def __init__(self, api_key: str, base_url: str = METASO_DEFAULT_BASE_URL, opener: Callable | None = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.opener = opener

    def search(self, query: str) -> dict[str, Any]:
        body = json.dumps({"q": query, "scope": "webpage"}).encode("utf-8")
        req = Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        if self.opener:
            return self.opener(req)
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

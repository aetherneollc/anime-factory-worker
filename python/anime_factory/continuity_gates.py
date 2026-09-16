"""Continuity three gates: SQL/JSON deterministic. Never embeddings."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from anime_factory.canon import hydrate_geo_interiors
from anime_factory.config import load_settings
from anime_factory.asset_lock import (
    has_locked_identity,
    has_locked_scene,
    is_qc_locked,
    load_assets_index,
    locked_character_relpath,
)
from anime_factory.directors.common import (
    REMOTE_BEATS,
    STAGING_FOR_BEAT,
    STORE_COPY_MARKERS,
    line_beat_id,
    line_blob,
)
from anime_factory.instrument import Counters
from anime_factory.world import name_vs_aka_violations, parse_travel_minutes

MediaHook = Callable[[str], None]


@dataclass
class GateViolation:
    code: str
    message: str
    group: str


@dataclass
class GateResult:
    ok: bool
    violations: list[GateViolation] = field(default_factory=list)
    injected: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    rewrites: int = 0

    def block(self) -> "GateResult":
        self.ok = False
        self.status = "blocked"
        return self


def _load_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def gate1_inject(conn: sqlite3.Connection, episode_code: str, geo: dict) -> dict[str, Any]:
    """Pre-script injection: SQL/JSON slices, never embeddings."""
    continuity_row = conn.execute(
        "SELECT payload_json FROM continuity ORDER BY version DESC LIMIT 1"
    ).fetchone()
    continuity = _load_json(continuity_row["payload_json"] if continuity_row else None, {})
    prev = conn.execute(
        "SELECT episode_code, recap, synopsis FROM episodes WHERE episode_code < ? ORDER BY episode_code DESC LIMIT 1",
        (episode_code,),
    ).fetchone()
    open_threads = conn.execute(
        "SELECT id, planted_episode, must_payoff_by, text FROM foreshadowing WHERE status = 'open'"
    ).fetchall()
    nearby = conn.execute(
        "SELECT id, date, title, kind, episode FROM timeline_events"
    ).fetchall()
    payload = {
        "source": "sql_json",
        "embeddings": False,
        "continuity_slice": continuity,
        "previous_recap": prev["recap"] if prev else None,
        "previous_synopsis": prev["synopsis"] if prev else None,
        "open_threads": [dict(r) for r in open_threads],
        "geo_places": {
            "regions": geo.get("regions") or [],
            "settlements": geo.get("settlements") or [],
            "landmarks": geo.get("landmarks") or [],
            "edges": geo.get("edges") or [],
            "interiors": geo.get("interiors") or [],
        },
        "nearby_timeline": [dict(r) for r in nearby],
    }
    return payload


def _script_lines(script: dict) -> list[dict]:
    lines = []
    for scene in script.get("scenes") or []:
        for line in scene.get("lines") or []:
            item = dict(line)
            item["scene_id"] = scene.get("id")
            item["location_id"] = scene.get("location_id")
            lines.append(item)
    return lines


def check_dead_dialogue(conn: sqlite3.Connection, script: dict, continuity: dict) -> list[GateViolation]:
    alive = {}
    for row in conn.execute("SELECT id, alive FROM characters"):
        alive[row["id"]] = bool(row["alive"])
    for ch in continuity.get("characters") or []:
        alive[ch["id"]] = bool(ch.get("alive", True))
    violations = []
    for line in _script_lines(script):
        cid = line.get("character_id")
        if cid and not alive.get(cid, True):
            violations.append(
                GateViolation("dead_dialogue", f"{cid} is dead but has dialogue", "character")
            )
    return violations


def check_knows(script: dict, continuity: dict) -> list[GateViolation]:
    knows = {ch["id"]: set(ch.get("knows") or []) for ch in continuity.get("characters") or []}
    violations = []
    for line in _script_lines(script):
        cid = line.get("character_id")
        mentioned = set(line.get("mentions_event_ids") or [])
        extra = mentioned - knows.get(cid, set())
        if extra:
            violations.append(
                GateViolation(
                    "unknown_fact",
                    f"{cid} mentions {sorted(extra)} not in knows[]",
                    "character",
                )
            )
    return violations


def check_props(script: dict, continuity: dict) -> list[GateViolation]:
    props = {p["id"]: p for p in continuity.get("props") or []}
    violations = []
    for scene in script.get("scenes") or []:
        for prop_id in scene.get("props") or []:
            rec = props.get(prop_id)
            if rec is None:
                violations.append(GateViolation("prop_missing", f"prop {prop_id} not in continuity", "prop"))
                continue
            if rec.get("state") == "destroyed" and scene.get("prop_state", {}).get(prop_id) == "intact":
                violations.append(
                    GateViolation("prop_destroyed", f"{prop_id} destroyed but depicted intact", "prop")
                )
            holder = rec.get("holder")
            users = scene.get("prop_users", {}).get(prop_id)
            if users and holder and users != holder and not scene.get("handoff"):
                violations.append(
                    GateViolation("prop_holder", f"{prop_id} used by {users} held by {holder}", "prop")
                )
    return violations


def check_teleport(script: dict, geo: dict, continuity: dict) -> list[GateViolation]:
    edges = geo.get("edges") or []

    def travel(a: str, b: str) -> float | None:
        if a == b:
            return 0.0
        for e in edges:
            if {e.get("from"), e.get("to")} == {a, b}:
                return parse_travel_minutes(str(e.get("travel_time") or "0m"))
        return None

    positions = dict(continuity.get("positions") or {})
    violations = []
    prev_loc = None
    for scene in script.get("scenes") or []:
        loc = scene.get("location_id")
        movers = scene.get("characters") or [ln.get("character_id") for ln in scene.get("lines") or []]
        interval = parse_travel_minutes(str(scene.get("interval_from_prev") or "0m"))
        if prev_loc and loc and prev_loc != loc:
            needed = travel(prev_loc, loc)
            if needed is not None and interval < needed:
                violations.append(
                    GateViolation(
                        "teleport",
                        f"interval {interval:.0f}m < travel_time {needed:.0f}m {prev_loc}->{loc}",
                        "spacetime",
                    )
                )
        for cid in movers:
            if not cid:
                continue
            last = positions.get(cid)
            if last and loc and last != loc:
                needed = travel(last, loc)
                scene_interval = interval
                if needed is not None and scene_interval < needed:
                    violations.append(
                        GateViolation(
                            "teleport",
                            f"{cid} teleport {last}->{loc}: {scene_interval:.0f}m < {needed:.0f}m",
                            "spacetime",
                        )
                    )
            if loc:
                positions[cid] = loc
        prev_loc = loc
    return violations


def check_aka_in_script(script: dict, geo: dict) -> list[GateViolation]:
    blob = json.dumps(script, ensure_ascii=False)
    return [
        GateViolation("aka_as_name", msg, "spacetime")
        for msg in name_vs_aka_violations(geo, blob)
    ]


def check_historical_strict(script: dict, timeline: dict, world_mode: str) -> list[GateViolation]:
    if world_mode != "historical_strict":
        return []
    real = {ev["id"]: ev for ev in timeline.get("events") or [] if ev.get("kind") == "real"}
    violations = []
    for alter in script.get("alters_events") or []:
        eid = alter.get("id")
        if eid in real:
            violations.append(
                GateViolation(
                    "historical_alter",
                    f"historical_strict cannot alter kind:real event {eid}",
                    "history",
                )
            )
    for ev in script.get("events") or []:
        if ev.get("id") in real and ev.get("outcome") and ev.get("outcome") != real[ev["id"]].get("title"):
            violations.append(
                GateViolation(
                    "historical_alter",
                    f"cannot change real event outcome {ev['id']}",
                    "history",
                )
            )
    return violations


def _script_text_blob(script: dict) -> str:
    parts = [str(script.get("synopsis") or ""), str(script.get("recap") or "")]
    for loc in script.get("locations") or []:
        parts.append(str(loc.get("name") or loc.get("id") or ""))
        parts.append(str(loc.get("plate_prompt") or ""))
    for prop in script.get("props") or []:
        parts.append(str(prop.get("name") or prop.get("id") or ""))
    for scene in script.get("scenes") or []:
        parts.append(str(scene.get("synopsis") or ""))
        for line in scene.get("lines") or []:
            parts.append(line_blob(line))
    return " ".join(parts).lower()


def bible_beat_ids(script: dict) -> set[str]:
    from anime_factory.directors.common import BEAT_MARKERS

    blob = _script_text_blob(script)
    found: set[str] = set()
    for beat, markers in BEAT_MARKERS:
        if any(marker.lower() in blob for marker in markers):
            found.add(beat)
    return found


def _line_stagings(script: dict) -> list[str]:
    out: list[str] = []
    for scene in script.get("scenes") or []:
        for line in scene.get("lines") or []:
            staging = " ".join(str(line.get("staging") or "").lower().split())
            if staging:
                out.append(staging)
    return out


def check_staging_collapse(script: dict) -> list[GateViolation]:
    """Bible has distinct beats (phone vs store, rain/door/cat) but staging is one store copy-paste."""
    stagings = _line_stagings(script)
    if len(stagings) < 6:
        return []
    unique = set(stagings)
    beats = bible_beat_ids(script)
    locations = {str(sc.get("location_id") or "") for sc in script.get("scenes") or [] if sc.get("location_id")}
    copy_paste = len(unique) <= 1 and any(any(tok in s for tok in STORE_COPY_MARKERS) for s in unique)
    phone_vs_person = bool(beats & {"phone", "wechat", "asleep"}) and bool(
        beats & {"store", "arrive", "door", "rain", "cat", "reject"}
    )
    diverse_bible = len(beats) >= 3 or phone_vs_person or (len(locations) >= 2 and len(beats) >= 2)
    if diverse_bible and (len(unique) <= 1 or copy_paste):
        return [
            GateViolation(
                "staging_collapse",
                f"bible beats {sorted(beats)} but {len(unique)} unique staging note(s); "
                "rewrite staging per beat, do not copy-paste the same store",
                "visual",
            )
        ]
    if len(stagings) >= 8 and len(unique) <= 1 and len(beats) >= 2:
        return [
            GateViolation(
                "staging_collapse",
                f"{len(stagings)} lines share one staging while bible has beats {sorted(beats)}",
                "visual",
            )
        ]
    return []


def repair_collapsed_staging(script: dict) -> dict:
    """Deterministic staging rewrite so QC can repair without waiting_for_human."""
    fixed = json.loads(json.dumps(script, ensure_ascii=False))
    for scene in fixed.get("scenes") or []:
        loc = str(scene.get("location_id") or scene.get("name") or "the set")
        for line in scene.get("lines") or []:
            beat = line_beat_id(line)
            staging = STAGING_FOR_BEAT.get(beat) or STAGING_FOR_BEAT["dialogue"]
            line["staging"] = f"{line.get('character_id') or 'speaker'} {staging} ({loc})"
            line["on_camera"] = beat not in REMOTE_BEATS
    return fixed


def gate2_check(
    conn: sqlite3.Connection,
    script: dict,
    geo: dict,
    timeline: dict,
    continuity: dict,
    world_mode: str,
) -> list[GateViolation]:
    # Explicitly do not touch embeddings.
    _ = Counters.embedding_api
    violations: list[GateViolation] = []
    violations.extend(check_dead_dialogue(conn, script, continuity))
    violations.extend(check_knows(script, continuity))
    violations.extend(check_props(script, continuity))
    violations.extend(check_teleport(script, geo, continuity))
    violations.extend(check_aka_in_script(script, geo))
    violations.extend(check_historical_strict(script, timeline, world_mode))
    violations.extend(check_staging_collapse(script))
    return violations


def run_script_gates(
    conn: sqlite3.Connection,
    episode_code: str,
    drafts: list[dict],
    geo: dict,
    timeline: dict,
    continuity: dict,
    world_mode: str,
    media_hook: MediaHook | None = None,
    max_rewrites: int | None = None,
) -> GateResult:
    """Try drafts[0], then rewrites. Exceeding CONTINUITY_MAX_REWRITES blocks; no png/mp4."""
    settings = load_settings()
    limit = settings.continuity_max_rewrites if max_rewrites is None else max_rewrites
    injected = gate1_inject(conn, episode_code, geo)
    result = GateResult(ok=False, injected=injected)
    png_mp4_written = False

    def write_media(_path: str) -> None:
        nonlocal png_mp4_written
        png_mp4_written = True
        if media_hook:
            media_hook(_path)

    attempts = 0
    last: list[GateViolation] = []
    for i, draft in enumerate(drafts):
        attempts = i
        last = gate2_check(conn, draft, geo, timeline, continuity, world_mode)
        if not last:
            result.ok = True
            result.status = "ok"
            result.rewrites = i
            result.violations = []
            return result
        if i >= limit:
            break
    result.violations = last
    result.rewrites = min(attempts, limit)
    result.block()
    conn.execute(
        "UPDATE episodes SET status = 'blocked', updated_at = datetime('now') WHERE episode_code = ?",
        (episode_code,),
    )
    conn.commit()
    if png_mp4_written:
        raise RuntimeError("blocked episode wrote png/mp4")
    return result


def _interior_aliases(interior_id: str, location_id: str) -> set[str]:
    ids = {s for s in (interior_id, location_id) if s}
    loc = location_id or (interior_id[:-4] if interior_id.endswith("_int") else "")
    if interior_id.startswith("int_") and len(interior_id) > 4:
        loc = loc or interior_id[4:]
    if loc:
        ids.update({loc, f"int_{loc}", f"{loc}_int"})
    return ids


def resolve_segment_interior(segment: dict, interiors: list[dict]) -> dict | None:
    """Match board interior_id / location_id onto geo.interiors, including oneshot aliases."""
    interiors_by_id = {str(it.get("id") or ""): it for it in interiors if it.get("id")}
    interior_id = str(segment.get("interior_id") or "").strip()
    location_id = str(segment.get("location_id") or "").strip()
    if interior_id and interior_id in interiors_by_id:
        return interiors_by_id[interior_id]
    for aid in _interior_aliases(interior_id, location_id):
        if aid in interiors_by_id:
            return interiors_by_id[aid]
    for it in interiors:
        parent = str(it.get("parent") or "")
        scene = str(it.get("scene_id") or "")
        if location_id and location_id in {parent, scene}:
            return it
        if interior_id and interior_id in {parent, scene}:
            return it
    return None


def pictured_character_ids(segment: dict) -> set[str]:
    """Characters that would appear in a keyframe / H3 roll for this unit."""
    purpose = str(segment.get("purpose") or "").lower()
    ids: set[str] = set()
    cid = str(segment.get("character_id") or "").strip()
    if cid and segment.get("on_camera") is not False and purpose not in {"establish", "establishing"}:
        ids.add(cid)
    for ref in segment.get("refs") or []:
        text = str(ref or "").strip()
        if text.startswith("char_") and text.endswith("_sheet"):
            ids.add(text[len("char_") : -len("_sheet")])
    for cut in segment.get("cuts") or []:
        if not isinstance(cut, dict):
            continue
        for other in cut.get("characters") or []:
            if str(other).strip():
                ids.add(str(other).strip())
    for extra in segment.get("characters") or []:
        if str(extra).strip():
            ids.add(str(extra).strip())
    costumes = segment.get("costume_ids") or {}
    if isinstance(costumes, dict):
        for key in costumes:
            if str(key).strip():
                ids.add(str(key).strip())
    return {cid for cid in ids if cid}


def _index_items(assets_index: dict, conn: sqlite3.Connection) -> dict[str, dict]:
    items = {it["id"]: it for it in assets_index.get("items") or [] if isinstance(it, dict) and it.get("id")}
    for row in conn.execute("SELECT id, kind, path, character_id, scene_id FROM assets"):
        items.setdefault(row["id"], {"id": row["id"], "kind": row["kind"], "path": row["path"], "character_id": row["character_id"], "scene_id": row["scene_id"]})
    return items


def _qc_record(cid: str, assets_index: dict, story_root: Path | None, bucket: str) -> dict:
    rec = ((assets_index or {}).get(bucket) or {}).get(cid) or {}
    if not isinstance(rec, dict):
        rec = {}
    if story_root is not None:
        disk = load_assets_index(story_root)
        disk_rec = (disk.get(bucket) or {}).get(cid) or {}
        if isinstance(disk_rec, dict):
            rec = {**disk_rec, **rec}
    return rec


def _character_locked(
    cid: str,
    assets_index: dict,
    conn: sqlite3.Connection,
    story_root: Path | None,
) -> bool:
    from anime_factory.visual_qc import is_current_pass

    _ = conn
    if story_root is not None and is_qc_locked(story_root, character_id=cid):
        return True
    rec = _qc_record(cid, assets_index, story_root, "characters")
    if is_current_pass(rec):
        if story_root is None:
            return True
        return has_locked_identity(story_root, cid)
    return False


def _scene_locked(
    lid: str,
    assets_index: dict,
    story_root: Path | None,
) -> bool:
    from anime_factory.visual_qc import is_current_pass

    if story_root is not None and is_qc_locked(story_root, scene_id=lid):
        return True
    rec = _qc_record(lid, assets_index, story_root, "scenes")
    if is_current_pass(rec):
        if story_root is None:
            return True
        return has_locked_scene(story_root, lid)
    return False


def _identity_violation(cid: str, rec: dict) -> GateViolation:
    verdict = str(rec.get("qc_verdict") or "")
    if verdict == "stale_visual_v1":
        return GateViolation("stale_visual", f"{cid} design asset is stale_visual_v1; refusing H3", "visual")
    if verdict in {"fail", "needs_human"}:
        return GateViolation("qc_failed_identity", f"{cid} visual QC is {verdict}; refusing H3", "visual")
    return GateViolation(
        "unlocked_identity",
        f"{cid} has no QC-passed front sheet; keyframe/anim fail closed",
        "visual",
    )


def gate3_assets(
    conn: sqlite3.Connection,
    segment: dict,
    geo: dict,
    assets_index: dict,
    story_root: Path | None = None,
) -> tuple[list[GateViolation], list[str]]:
    """Segment must cite existing sheet/plate/costume; pictured characters need a locked turnaround."""
    items = _index_items(assets_index, conn)
    if story_root is not None:
        disk = load_assets_index(story_root)
        merged = dict(assets_index or {})
        merged["characters"] = {**(disk.get("characters") or {}), **(merged.get("characters") or {})}
        merged["scenes"] = {**(disk.get("scenes") or {}), **(merged.get("scenes") or {})}
        assets_index = merged
    geo = hydrate_geo_interiors(conn, geo)
    interior_rows = list(geo.get("interiors") or [])
    interiors = {it.get("scene_id"): it for it in interior_rows}
    missing: list[str] = []
    violations: list[GateViolation] = []

    refs = list(segment.get("refs") or [])
    plate_id = segment.get("plate_id")
    if plate_id:
        refs.append(plate_id)
    costumes = segment.get("costume_ids") or {}
    if isinstance(costumes, dict):
        refs.extend(costumes.values())

    for cid in pictured_character_ids(segment):
        aid = locked_character_relpath(cid)
        if aid not in refs:
            refs.append(aid)

    for rid in refs:
        if rid and rid not in items:
            missing.append(rid)
            violations.append(GateViolation("missing_asset", f"asset {rid} not in index", "visual"))

    for cid in pictured_character_ids(segment):
        aid = locked_character_relpath(cid)
        if aid in missing:
            continue
        rec = _qc_record(cid, assets_index, story_root, "characters")
        if not _character_locked(cid, assets_index, conn, story_root):
            violations.append(_identity_violation(cid, rec))

    plate_id = str(segment.get("plate_id") or "").strip()
    scene_lock_id = ""
    if plate_id:
        scene_lock_id = plate_id[len("plate_") :] if plate_id.startswith("plate_") else plate_id
    if scene_lock_id and story_root is not None:
        rec = _qc_record(scene_lock_id, assets_index, story_root, "scenes")
        if not _scene_locked(scene_lock_id, assets_index, story_root):
            verdict = str(rec.get("qc_verdict") or "")
            if verdict == "stale_visual_v1":
                violations.append(
                    GateViolation("stale_visual", f"{scene_lock_id} scene plate is stale_visual_v1; refusing H3", "visual")
                )
            elif verdict in {"fail", "needs_human"}:
                violations.append(
                    GateViolation("qc_failed_identity", f"{scene_lock_id} scene QC is {verdict}; refusing H3", "visual")
                )
            else:
                violations.append(
                    GateViolation(
                        "unlocked_identity",
                        f"{scene_lock_id} scene plate has no current QC pass; refusing H3",
                        "visual",
                    )
                )

    scene_id = segment.get("scene_id") or segment.get("location_scene_id")
    interior_id = str(segment.get("interior_id") or "").strip()
    matched = resolve_segment_interior(segment, interior_rows)
    if interior_id and matched is None:
        violations.append(
            GateViolation("interior_mismatch", f"interior {interior_id} not in geo.interiors", "visual")
        )
    elif scene_id and interiors and matched is None and scene_id not in interiors:
        # allow if interior.scene_id matches
        if not any(it.get("scene_id") == scene_id for it in interior_rows):
            violations.append(
                GateViolation("interior_mismatch", f"scene {scene_id} has no interiors[] match", "visual")
            )
    return violations, missing

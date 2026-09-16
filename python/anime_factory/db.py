"""story.sqlite open, migrate, wal_checkpoint, atomic replace helpers."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "story_sqlite.sql"

CHAPTER10_TABLES = (
    "story",
    "episodes",
    "scenes",
    "segments",
    "cuts",
    "characters",
    "character_relationships",
    "character_voice",
    "locations",
    "factions",
    "props",
    "timeline_events",
    "foreshadowing",
    "assets",
    "memory",
    "qc_reports",
    "continuity",
    "glossary",
    "productions",
    "sequences",
    "beats",
    "shots",
    "generation_plans",
    "generation_tasks",
    "generation_results",
    "repair_tasks",
)

_ADDITIVE_COLUMNS = (
    ("story", "kind", "TEXT NOT NULL DEFAULT 'series'"),
    ("episodes", "kind", "TEXT NOT NULL DEFAULT 'series'"),
    ("segments", "shot_id", "TEXT"),
    ("segments", "chain_id", "TEXT"),
    ("segments", "chain_index", "INTEGER NOT NULL DEFAULT 0"),
    ("segments", "generation_plan_id", "TEXT"),
    ("segments", "approved_generation_id", "TEXT"),
    ("segments", "creative_version", "INTEGER NOT NULL DEFAULT 1"),
    ("shots", "chain_id", "TEXT"),
    ("assets", "selected_image_id", "TEXT"),
    ("assets", "history_json", "TEXT"),
    ("locations", "plate_prompt", "TEXT"),
)

# Canon graph — first-class tables, not factions.__meta_* rows.
GEO_TABLES = ("geo_edges", "geo_interiors")


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def open_db(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_column(conn: sqlite3.Connection, table: str, name: str, spec: str) -> None:
    if table not in table_names(conn):
        return
    if name not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")


def _backfill_shot_layer(conn: sqlite3.Connection) -> None:
    """Existing segment rows become 1:1 shots / chain heads."""
    names = table_names(conn)
    if "segments" not in names or "shots" not in names:
        return
    seg_cols = _table_columns(conn, "segments")
    if "shot_id" in seg_cols:
        conn.execute("UPDATE segments SET shot_id = id WHERE shot_id IS NULL OR shot_id = ''")
    if "chain_id" in seg_cols:
        conn.execute("UPDATE segments SET chain_id = COALESCE(NULLIF(shot_id, ''), id) WHERE chain_id IS NULL OR chain_id = ''")
    conn.execute(
        """
        INSERT OR IGNORE INTO shots (id, episode_code, scene_id, seq, duration, h3_mode, status)
        SELECT COALESCE(NULLIF(shot_id, ''), id), episode_code, scene_id, MIN(seq), SUM(duration),
               MIN(h3_mode), MIN(status)
        FROM segments
        GROUP BY COALESCE(NULLIF(shot_id, ''), id)
        """
    )


def migrate(conn: sqlite3.Connection, schema_sql: str | None = None) -> None:
    sql = schema_sql if schema_sql is not None else SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)
    for table, name, spec in _ADDITIVE_COLUMNS:
        _ensure_column(conn, table, name, spec)
    _backfill_shot_layer(conn)
    conn.commit()


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {row[0] for row in rows}


def wal_checkpoint(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.commit()


def atomic_replace(src: str | Path, dest: str | Path) -> None:
    """Write-then-rename: dest.tmp -> dest. Mirrors Vast sqlite.tmp then overwrite."""
    src_p = Path(src)
    dest_p = Path(dest)
    tmp = dest_p.with_name(dest_p.name + ".tmp")
    data = src_p.read_bytes()
    tmp.write_bytes(data)
    os.replace(tmp, dest_p)


def checkpoint_and_upload(conn: sqlite3.Connection, dest: str | Path) -> Path:
    """wal_checkpoint then atomic tmp replace onto dest (R2-style local stand-in)."""
    wal_checkpoint(conn)
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_p.with_name(dest_p.name + ".tmp")
    backup = sqlite3.connect(str(tmp))
    try:
        conn.backup(backup)
        backup.commit()
    finally:
        backup.close()
    os.replace(tmp, dest_p)
    return dest_p

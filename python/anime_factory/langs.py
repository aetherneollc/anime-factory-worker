"""Language registry read from config/languages.json. TS side reads the same file.

Adding a language is one registry row plus a voice reference and a subtitle font.
No pipeline stage may hardcode ("zh", "en", "ja") again.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from anime_factory.config import load_settings

REGISTRY_REL = "config/languages.json"


class UnknownLangError(ValueError):
    def __init__(self, code: str, known: Iterable[str]):
        super().__init__(f"lang {code!r} is not registered in {REGISTRY_REL}; registered: {sorted(known)}")
        self.code = code


@dataclass(frozen=True)
class LangSpec:
    code: str
    name: str
    tts_voice_key: str
    subtitle_font: str
    rtl: bool = False
    platform_tag_style: str = "intl"


def registry_path(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root) / REGISTRY_REL
    return Path(__file__).resolve().parents[2] / REGISTRY_REL


_CACHE: dict[tuple[str, int], dict[str, LangSpec]] = {}


def load_registry(path: Path | str | None = None) -> dict[str, LangSpec]:
    p = Path(path) if path is not None else registry_path()
    if not p.is_file():
        raise FileNotFoundError(f"language registry missing: {p}")
    stamp = (str(p), p.stat().st_mtime_ns)
    cached = _CACHE.get(stamp)
    if cached is not None:
        return cached
    raw = json.loads(p.read_text(encoding="utf-8"))
    registry: dict[str, LangSpec] = {}
    for code, row in raw.items():
        registry[code] = LangSpec(
            code=code,
            name=row.get("name") or code,
            tts_voice_key=row.get("tts_voice_key") or code,
            subtitle_font=row.get("subtitle_font") or "NotoSans",
            rtl=bool(row.get("rtl")),
            platform_tag_style=row.get("platform_tag_style") or "intl",
        )
    _CACHE.clear()
    _CACHE[stamp] = registry
    return registry


def spec_for(code: str, registry: dict[str, LangSpec] | None = None) -> LangSpec:
    reg = registry if registry is not None else load_registry()
    try:
        return reg[code]
    except KeyError:
        raise UnknownLangError(code, reg.keys()) from None


def resolve(codes: Sequence[str], registry: dict[str, LangSpec] | None = None) -> list[LangSpec]:
    reg = registry if registry is not None else load_registry()
    return [spec_for(code, reg) for code in codes]


def default_langs() -> tuple[str, ...]:
    """APP_LANGS filtered through the registry so a typo blocks instead of silently dropping a track."""
    codes = tuple(load_settings().app_langs)
    resolve(codes)
    return codes


def normalize_langs(codes: Sequence[str] | None) -> tuple[str, ...]:
    if not codes:
        return default_langs()
    seen: list[str] = []
    for code in codes:
        cleaned = str(code).strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    resolve(seen)
    return tuple(seen)


def story_langs(conn: sqlite3.Connection, story_id: str | None = None) -> tuple[str, ...]:
    """Per-story language list. Existing stories keep whatever they were created with."""
    if story_id:
        row = conn.execute("SELECT langs FROM story WHERE id = ?", (story_id,)).fetchone()
    else:
        row = conn.execute("SELECT langs FROM story ORDER BY created_at LIMIT 1").fetchone()
    raw = row["langs"] if row else None
    if not raw:
        return default_langs()
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        parsed = [part.strip() for part in str(raw).split(",")]
    if isinstance(parsed, str):
        parsed = [part.strip() for part in parsed.split(",")]
    return normalize_langs([str(code) for code in parsed if str(code).strip()])


def subtitle_font(code: str, registry: dict[str, LangSpec] | None = None) -> str:
    return spec_for(code, registry).subtitle_font


def tts_voice_key(code: str, registry: dict[str, LangSpec] | None = None) -> str:
    return spec_for(code, registry).tts_voice_key

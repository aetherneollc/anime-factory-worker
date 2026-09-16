"""Duration-budgeted translation. Glossary lock. Split sentence if still over after refits."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from anime_factory.config import load_settings
from anime_factory.models import H3_MAX_SECONDS


@dataclass
class Line:
    id: str
    character_id: str
    zh: str
    en: str = ""
    ja: str = ""
    dur_zh: float = 0.0
    dur_en: float = 0.0
    dur_ja: float = 0.0
    split: bool = False


@dataclass
class TranslateResult:
    lines: list[Line]
    refits: int = 0
    split_ids: list[str] = field(default_factory=list)


def load_glossary(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM glossary")]


def apply_glossary(text: str, glossary: list[dict], lang: str) -> str:
    out = text
    for row in glossary:
        src = row.get("zh") or ""
        dst = row.get(lang) or src
        if src and dst:
            out = out.replace(src, dst)
    return out


def glossary_lock_violations(text: str, glossary: list[dict], lang: str) -> list[str]:
    issues = []
    for row in glossary:
        zh = row.get("zh") or ""
        locked = row.get(lang) or ""
        if not zh or not locked:
            continue
        if zh in text:
            issues.append(f"{zh} not replaced by glossary {locked} for {lang}")
        # wrong alternate spelling
        for other in row.get("forbidden_" + lang, []) if isinstance(row.get("forbidden_" + lang), list) else []:
            if other in text:
                issues.append(f"forbidden spelling {other} for {zh}")
    return issues


def estimate_speech_seconds(text: str, lang: str) -> float:
    """Heuristic used when a duration callback is not provided (tests inject real durs)."""
    n = max(len(text.strip()), 1)
    cps = {"zh": 4.5, "en": 13.0, "ja": 7.0}.get(lang, 10.0)
    return n / cps


def translate_lines(
    lines: list[Line],
    glossary: list[dict],
    translate_fn=None,
    duration_fn=None,
    max_refits: int | None = None,
) -> TranslateResult:
    settings = load_settings()
    limit = settings.translation_max_refits if max_refits is None else max_refits
    duration_fn = duration_fn or (lambda text, lang: estimate_speech_seconds(text, lang))
    translate_fn = translate_fn or (lambda text, lang, budget: _naive_translate(text, lang, glossary))

    result = TranslateResult(lines=list(lines))
    for line in result.lines:
        if not line.dur_zh:
            line.dur_zh = duration_fn(line.zh, "zh")
        budget = min(line.dur_zh * 1.2, H3_MAX_SECONDS)

        def fit(lang: str, current: str) -> str:
            text = apply_glossary(current or translate_fn(line.zh, lang, budget), glossary, lang)
            durs = duration_fn(text, lang)
            tries = 0
            while durs > H3_MAX_SECONDS and tries < limit:
                tries += 1
                result.refits += 1
                text = apply_glossary(translate_fn(line.zh, lang, budget * (0.85**tries)), glossary, lang)
                durs = duration_fn(text, lang)
            if lang == "en":
                line.en, line.dur_en = text, durs
            else:
                line.ja, line.dur_ja = text, durs
            return text

        fit("en", line.en)
        fit("ja", line.ja)

        peak = max(line.dur_zh, line.dur_en, line.dur_ja)
        if peak > H3_MAX_SECONDS:
            line.split = True
            result.split_ids.append(line.id)

    return result


def _naive_translate(text: str, lang: str, glossary: list[dict]) -> str:
    mapped = apply_glossary(text, glossary, lang)
    if lang == "en":
        return mapped if mapped != text else f"[en] {text}"
    if lang == "ja":
        return mapped if mapped != text else f"[ja] {text}"
    return mapped


def segment_duration(line: Line) -> float:
    return min(max(line.dur_zh, line.dur_en, line.dur_ja), H3_MAX_SECONDS)

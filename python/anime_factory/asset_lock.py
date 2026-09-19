"""Toonflow-style identity lock: one selected turnaround per character / plate per scene.

Frozen on-disk contract (`stories/<id>/assets/index.json`):

    {
      "characters": {
        "<cid>": {"selected": "sheet_turnaround.png", "history": ["sheet_turnaround.png"], "parent_id": null}
      },
      "scenes": {
        "<lid>": {"selected": "plate_base.png", "history": ["plate_base.png"]}
      }
    }

Filenames live under `assets/characters/<cid>/` and `assets/scenes/<lid>/`.
Downstream (directors / H3 / session) must bind the selected file, not every png in the folder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

INDEX_REL = "assets/index.json"
REGEN_REL = "assets/regen.json"
TURNAROUND_FILENAME = "sheet_turnaround.png"
FRONT_ALIAS_FILENAME = "sheet_front.png"
PLATE_FILENAME = "plate_base.png"
MIN_BYTES = 1

_CHAR_FALLBACKS = (FRONT_ALIAS_FILENAME, TURNAROUND_FILENAME, "sheet.png")
_SCENE_FALLBACKS = (PLATE_FILENAME, "plate.png")
QC_RECORD_KEYS = (
    "model_version",
    "prompt_version",
    "workflow_version",
    "qc_version",
    "qc_verdict",
    "qc_reasons",
    "qc_scores",
    "qc_seed",
    "qc_attempts",
    "qc_scorer",
    "candidates",
    "source_hash",
    "identity_source_hash",
    "scene_source_hash",
    "prop_source_hash",
    "lighting_source_hash",
)


def locked_character_relpath(cid: str) -> str:
    """Shot.refs identity token. Same id as the front-sheet row historically used."""
    return f"char_{cid}_sheet"


def character_dir(story_root: Path | str, cid: str) -> Path:
    return Path(story_root) / "assets" / "characters" / str(cid)


def scene_dir(story_root: Path | str, lid: str) -> Path:
    return Path(story_root) / "assets" / "scenes" / str(lid)


def empty_assets_index() -> dict[str, Any]:
    return {"characters": {}, "scenes": {}}


def _filename(value: str | None) -> str:
    text = str(value or "").replace("\\", "/").strip()
    if not text or text.endswith("/"):
        return ""
    return Path(text).name


def _file_ok(path: Path | None) -> bool:
    return bool(path) and path.is_file() and path.stat().st_size >= MIN_BYTES


def _copy_qc_fields(rec: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    for key in QC_RECORD_KEYS:
        if key not in obj:
            continue
        value = obj.get(key)
        if key == "qc_reasons":
            rec[key] = [str(item) for item in (value or [])] if isinstance(value, list) else []
        elif key == "qc_scores":
            rec[key] = dict(value) if isinstance(value, dict) else {}
        elif key == "candidates":
            rec[key] = [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []
        elif key == "qc_attempts":
            try:
                rec[key] = int(value)
            except (TypeError, ValueError):
                rec[key] = 0
        elif key == "qc_seed" and value is not None:
            try:
                rec[key] = int(value)
            except (TypeError, ValueError):
                rec[key] = value
        else:
            rec[key] = value
    return rec


def _char_record(raw: Any) -> dict[str, Any]:
    obj = raw if isinstance(raw, dict) else {}
    selected = _filename(obj.get("selected"))
    history: list[str] = []
    seen: set[str] = set()
    for item in obj.get("history") or []:
        name = _filename(item)
        if name and name not in seen:
            seen.add(name)
            history.append(name)
    if selected and selected not in seen:
        history.insert(0, selected)
    parent = obj.get("parent_id")
    parent_id = None if parent in (None, "", "null") else str(parent)
    rec: dict[str, Any] = {
        "selected": selected or None,
        "history": history,
        "parent_id": parent_id,
        "candidates": [],
    }
    return _copy_qc_fields(rec, obj)


def _scene_record(raw: Any) -> dict[str, Any]:
    obj = raw if isinstance(raw, dict) else {}
    selected = _filename(obj.get("selected"))
    history: list[str] = []
    seen: set[str] = set()
    for item in obj.get("history") or []:
        name = _filename(item)
        if name and name not in seen:
            seen.add(name)
            history.append(name)
    if selected and selected not in seen:
        history.insert(0, selected)
    parent = obj.get("parent_id")
    parent_id = None if parent in (None, "", "null") else str(parent)
    rec: dict[str, Any] = {"selected": selected or None, "history": history, "candidates": []}
    if parent_id:
        rec["parent_id"] = parent_id
    return _copy_qc_fields(rec, obj)


def _migrate_items(index: dict[str, Any], items: Iterable[Any]) -> None:
    characters = index.setdefault("characters", {})
    scenes = index.setdefault("scenes", {})
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").lower()
        filename = _filename(item.get("path") or item.get("r2") or item.get("id"))
        if not filename:
            continue
        cid = str(item.get("character_id") or "").strip()
        lid = str(item.get("scene_id") or "").strip()
        if kind in {"character_sheet", "character", "costume_derive"} or cid:
            if not cid:
                continue
            rec = _char_record(characters.get(cid) or {})
            if filename not in rec["history"]:
                rec["history"].append(filename)
            if not rec["selected"] and rec.get("qc_verdict") not in {"fail", "needs_human", "stale_visual_v1"}:
                rec["selected"] = filename
            if rec.get("parent_id") is None and item.get("parent_id"):
                rec["parent_id"] = str(item.get("parent_id"))
            characters[cid] = rec
        elif kind in {"scene_plate", "location", "plate", "scene_derive"} or lid:
            if not lid:
                continue
            rec = _scene_record(scenes.get(lid) or {})
            if filename not in rec["history"]:
                rec["history"].append(filename)
            if not rec["selected"] and rec.get("qc_verdict") not in {"fail", "needs_human", "stale_visual_v1"}:
                rec["selected"] = filename
            scenes[lid] = rec


def normalize_assets_index(raw: Any) -> dict[str, Any]:
    """Accept the frozen contract, plus the legacy `{items: [...]}` library dump."""
    index = empty_assets_index()
    if not isinstance(raw, dict):
        return index
    chars_in = raw.get("characters")
    if isinstance(chars_in, dict):
        for cid, rec in chars_in.items():
            key = str(cid or "").strip()
            if key:
                index["characters"][key] = _char_record(rec)
    scenes_in = raw.get("scenes")
    if isinstance(scenes_in, dict):
        for lid, rec in scenes_in.items():
            key = str(lid or "").strip()
            if key:
                index["scenes"][key] = _scene_record(rec)
    _migrate_items(index, raw.get("items") or [])
    return index


def load_assets_index(story_root: Path | str | None) -> dict[str, Any]:
    if story_root is None:
        return empty_assets_index()
    path = Path(story_root) / INDEX_REL
    if not path.is_file():
        return empty_assets_index()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_assets_index()
    return normalize_assets_index(raw)


def save_assets_index(
    story_root: Path | str,
    index: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    root = Path(story_root)
    dest = root / INDEX_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_assets_index(index)

    def _dump_record(rec: dict[str, Any], *, with_parent: bool) -> dict[str, Any]:
        row: dict[str, Any] = {
            "selected": rec.get("selected"),
            "history": list(rec.get("history") or []),
        }
        if with_parent or rec.get("parent_id") is not None:
            row["parent_id"] = rec.get("parent_id")
        for key in QC_RECORD_KEYS:
            if key in rec:
                row[key] = rec.get(key)
        return row

    payload: dict[str, Any] = {
        "characters": {
            cid: _dump_record(rec, with_parent=True)
            for cid, rec in (normalized.get("characters") or {}).items()
        },
        "scenes": {
            lid: {
                k: v
                for k, v in _dump_record(rec, with_parent=False).items()
                if k != "parent_id" or rec.get("parent_id")
            }
            for lid, rec in (normalized.get("scenes") or {}).items()
        },
    }
    if extra:
        for key, value in extra.items():
            if key in {"characters", "scenes"}:
                continue
            payload[key] = value
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def _first_existing(folder: Path, names: Iterable[str], selected: str | None = None) -> Path | None:
    ordered: list[str] = []
    if selected:
        ordered.append(selected)
    for name in names:
        if name not in ordered:
            ordered.append(name)
    for name in ordered:
        cand = folder / _filename(name)
        if _file_ok(cand):
            return cand
    return None


def _bucket_record(story_root: Path | str | None, *, character_id: str | None = None, scene_id: str | None = None) -> dict[str, Any]:
    if story_root is None:
        return {}
    index = load_assets_index(story_root)
    if character_id:
        return (index.get("characters") or {}).get(str(character_id)) or {}
    if scene_id:
        return (index.get("scenes") or {}).get(str(scene_id)) or {}
    return {}


def is_qc_locked(
    story_root: Path | str | None,
    *,
    character_id: str | None = None,
    scene_id: str | None = None,
) -> bool:
    """True only with a current visual_qc_v1 pass and the selected file on disk."""
    from anime_factory.visual_qc import is_current_pass

    if story_root is None:
        return False
    rec = _bucket_record(story_root, character_id=character_id, scene_id=scene_id)
    if not is_current_pass(rec):
        return False
    selected = _filename(rec.get("selected"))
    if not selected:
        return False
    folder = character_dir(story_root, character_id) if character_id else scene_dir(story_root, str(scene_id))
    return _file_ok(folder / selected)


def locked_character_file(story_root: Path | str, cid: str) -> Path:
    """Selected QC-passed front sheet. Fallbacks are for display, not H3 binding."""
    folder = character_dir(story_root, cid)
    rec = (load_assets_index(story_root).get("characters") or {}).get(str(cid)) or {}
    selected = _filename(rec.get("selected"))
    from anime_factory.visual_qc import is_current_pass

    if is_current_pass(rec) and selected:
        found = folder / selected
        if _file_ok(found):
            return found
        alias = folder / FRONT_ALIAS_FILENAME
        if _file_ok(alias):
            return alias
        return found
    found = _first_existing(folder, _CHAR_FALLBACKS, selected)
    if found is not None:
        return found
    return folder / (selected or FRONT_ALIAS_FILENAME)


def locked_scene_file(story_root: Path | str, lid: str) -> Path:
    folder = scene_dir(story_root, lid)
    rec = (load_assets_index(story_root).get("scenes") or {}).get(str(lid)) or {}
    selected = _filename(rec.get("selected"))
    from anime_factory.visual_qc import is_current_pass

    if is_current_pass(rec) and selected:
        found = folder / selected
        if _file_ok(found):
            return found
        return found
    found = _first_existing(folder, _SCENE_FALLBACKS, selected)
    if found is not None:
        return found
    return folder / (selected or PLATE_FILENAME)


def has_locked_identity(story_root: Path | str | None, cid: str) -> bool:
    if story_root is None or not cid:
        return False
    return is_qc_locked(story_root, character_id=cid)


def has_locked_scene(story_root: Path | str | None, lid: str) -> bool:
    if story_root is None or not lid:
        return False
    return is_qc_locked(story_root, scene_id=lid)


def write_front_alias(turnaround: Path) -> Path | None:
    """Old session.py / keyframe lookups still open sheet_front.png. Copy, do not re-roll."""
    if not _file_ok(turnaround):
        return None
    alias = turnaround.parent / FRONT_ALIAS_FILENAME
    blob = turnaround.read_bytes()
    if not alias.is_file() or alias.read_bytes() != blob:
        alias.write_bytes(blob)
    return alias


def next_history_filename(history: Iterable[str], stem: str, suffix: str = ".png") -> str:
    taken = {_filename(name) for name in history}
    base = f"{stem}{suffix}"
    if base not in taken:
        return base
    n = 2
    while f"{stem}_{n}{suffix}" in taken:
        n += 1
    return f"{stem}_{n}{suffix}"


def _index_extra(story_root: Path | str | None) -> dict[str, Any]:
    if story_root is None:
        return {}
    path = Path(story_root) / INDEX_REL
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in (raw or {}).items() if k not in {"characters", "scenes"}}


def _target_record(
    payload: dict[str, Any],
    *,
    character_id: str | None,
    scene_id: str | None,
    parent_id: str | None = None,
) -> dict[str, Any]:
    if character_id:
        rec = _char_record((payload.get("characters") or {}).get(character_id) or {})
        if rec.get("parent_id") is None and parent_id:
            rec["parent_id"] = str(parent_id)
        payload.setdefault("characters", {})[str(character_id)] = rec
        return rec
    if scene_id:
        rec = _scene_record((payload.get("scenes") or {}).get(scene_id) or {})
        if parent_id and not rec.get("parent_id"):
            rec["parent_id"] = str(parent_id)
        payload.setdefault("scenes", {})[str(scene_id)] = rec
        return rec
    raise ValueError("character_id or scene_id is required")


def _apply_qc_fields(rec: dict[str, Any], qc: Any, *, attempts: int | None = None) -> None:
    lineage = dict(getattr(qc, "lineage", None) or {})
    rec["model_version"] = lineage.get("model_version")
    rec["prompt_version"] = lineage.get("prompt_version")
    rec["workflow_version"] = lineage.get("workflow_version")
    rec["qc_version"] = lineage.get("qc_version")
    rec["qc_verdict"] = str(getattr(qc, "verdict", None) or rec.get("qc_verdict") or "fail")
    rec["qc_reasons"] = list(getattr(qc, "reasons", None) or [])
    rec["qc_scores"] = dict(getattr(qc, "scores", None) or {})
    rec["qc_seed"] = getattr(qc, "seed", None)
    rec["qc_scorer"] = getattr(qc, "scorer", None)
    if attempts is not None:
        rec["qc_attempts"] = int(attempts)
    else:
        rec["qc_attempts"] = int(rec.get("qc_attempts") or 0)


def record_qc_candidate(
    story_root: Path | str | None,
    *,
    character_id: str | None = None,
    scene_id: str | None = None,
    filename: str,
    qc: Any,
    parent_id: str | None = None,
    index: dict[str, Any] | None = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Append a QC'd candidate to history. Never sets selected or steals a current lock."""
    from anime_factory.visual_qc import is_current_pass

    name = _filename(filename)
    payload = normalize_assets_index(index if index is not None else load_assets_index(story_root))
    rec = _target_record(payload, character_id=character_id, scene_id=scene_id, parent_id=parent_id)
    locked = is_current_pass(rec)
    identity = {key: rec.get(key) for key in ("selected", *QC_RECORD_KEYS) if key != "candidates"} if locked else None
    if name and name not in rec["history"]:
        rec["history"].append(name)
    candidates = [dict(item) for item in rec.get("candidates") or [] if isinstance(item, dict)]
    if name:
        candidates.append(
            {
                "filename": name,
                "seed": getattr(qc, "seed", None),
                "verdict": str(getattr(qc, "verdict", None) or "fail"),
                "reasons": list(getattr(qc, "reasons", None) or []),
                "scores": dict(getattr(qc, "scores", None) or {}),
            }
        )
    rec["candidates"] = candidates
    if locked and identity is not None:
        for key, value in identity.items():
            rec[key] = value
        rec["selected"] = identity.get("selected")
    else:
        attempts = int(rec.get("qc_attempts") or 0) + 1
        _apply_qc_fields(rec, qc, attempts=attempts)
        if rec.get("qc_verdict") == "pass":
            rec["qc_verdict"] = "fail"
            rec["qc_reasons"] = list(rec.get("qc_reasons") or []) + ["recorded_without_lock"]
        if attempts >= max_attempts and rec.get("qc_verdict") != "pass":
            rec["qc_verdict"] = "needs_human"
        rec["selected"] = rec.get("selected")  # leave selected unchanged
    if story_root is not None:
        save_assets_index(story_root, payload, extra=_index_extra(story_root))
    return payload


def lock_after_qc(
    story_root: Path | str | None,
    *,
    character_id: str | None = None,
    scene_id: str | None = None,
    filename: str,
    qc: Any,
    parent_id: str | None = None,
    index: dict[str, Any] | None = None,
    set_selected: bool = True,
) -> dict[str, Any]:
    """Lock only when VisualQcResult.verdict == pass. Front is the identity pointer."""
    name = _filename(filename)
    payload = normalize_assets_index(index if index is not None else load_assets_index(story_root))
    rec = _target_record(payload, character_id=character_id, scene_id=scene_id, parent_id=parent_id)
    if name and name not in rec["history"]:
        rec["history"].append(name)
    verdict = str(getattr(qc, "verdict", None) or "")
    if verdict != "pass":
        raise ValueError(f"lock_after_qc requires verdict='pass', got {verdict!r}")
    attempts = int(rec.get("qc_attempts") or 0) + 1
    _apply_qc_fields(rec, qc, attempts=attempts)
    rec["qc_verdict"] = "pass"
    if set_selected and name:
        rec["selected"] = name
        if story_root is not None and character_id:
            src = character_dir(story_root, character_id) / name
            if _file_ok(src) and name != TURNAROUND_FILENAME:
                write_front_alias(src)
    if story_root is not None:
        save_assets_index(story_root, payload, extra=_index_extra(story_root))
    return payload


def mark_stale_visual_v1(story_root: Path | str | None) -> dict[str, Any]:
    """Old selected without current QC becomes stale. Files and history stay."""
    from anime_factory.visual_qc import QC_VERSION, is_current_pass

    payload = load_assets_index(story_root)
    changed = False
    for rec in list((payload.get("characters") or {}).values()) + list((payload.get("scenes") or {}).values()):
        selected = rec.get("selected")
        if not selected:
            continue
        if is_current_pass(rec):
            continue
        rec["qc_verdict"] = "stale_visual_v1"
        rec["qc_version"] = rec.get("qc_version") or QC_VERSION
        rec["qc_reasons"] = list(rec.get("qc_reasons") or []) + ["stale_visual_v1"]
        rec["selected"] = None
        changed = True
    if changed and story_root is not None:
        save_assets_index(story_root, payload, extra=_index_extra(story_root))
    return payload


def mark_stale_source_mismatch(
    story_root: Path | str | None,
    *,
    character_id: str | None = None,
    scene_id: str | None = None,
    reason: str = "source_hash_mismatch",
) -> dict[str, Any]:
    """Clear selected lock when bible/cast source hash no longer matches."""
    payload = load_assets_index(story_root)
    rec = _target_record(payload, character_id=character_id, scene_id=scene_id)
    if not rec.get("selected") and rec.get("qc_verdict") == "stale_source":
        return payload
    rec["qc_verdict"] = "stale_source"
    rec["qc_reasons"] = list(rec.get("qc_reasons") or []) + [reason]
    rec["selected"] = None
    if story_root is not None:
        save_assets_index(story_root, payload, extra=_index_extra(story_root))
    return payload


def sync_character_source_hash(
    story_root: Path | str | None,
    *,
    character_id: str,
    source_hash: str,
    fail_closed_if_locked: bool = False,
) -> dict[str, Any]:
    """Stamp identity source hash. Mismatch marks stale; locked+mismatch can fail closed."""
    from anime_factory.visual_qc import is_current_pass

    payload = normalize_assets_index(load_assets_index(story_root))
    rec = _target_record(payload, character_id=character_id, scene_id=None)
    prev = str(rec.get("source_hash") or rec.get("identity_source_hash") or "").strip()
    new = str(source_hash or "").strip()
    rec["source_hash"] = new
    rec["identity_source_hash"] = new
    if prev and new and prev != new:
        if fail_closed_if_locked and is_current_pass(rec):
            raise ValueError(
                f"locked character {character_id} source_hash changed ({prev} -> {new}); refuse silent rewrite"
            )
        rec["qc_verdict"] = "stale_source"
        rec["qc_reasons"] = list(rec.get("qc_reasons") or []) + ["source_hash_mismatch"]
        rec["selected"] = None
    if story_root is not None:
        save_assets_index(story_root, payload, extra=_index_extra(story_root))
    return payload


def sync_scene_source_hash(
    story_root: Path | str | None,
    *,
    scene_id: str,
    source_hash: str,
    fail_closed_if_locked: bool = False,
) -> dict[str, Any]:
    from anime_factory.visual_qc import is_current_pass

    payload = normalize_assets_index(load_assets_index(story_root))
    rec = _target_record(payload, character_id=None, scene_id=scene_id)
    prev = str(rec.get("source_hash") or rec.get("scene_source_hash") or "").strip()
    new = str(source_hash or "").strip()
    rec["source_hash"] = new
    rec["scene_source_hash"] = new
    if prev and new and prev != new:
        if fail_closed_if_locked and is_current_pass(rec):
            raise ValueError(
                f"locked scene {scene_id} source_hash changed ({prev} -> {new}); refuse silent rewrite"
            )
        rec["qc_verdict"] = "stale_source"
        rec["qc_reasons"] = list(rec.get("qc_reasons") or []) + ["source_hash_mismatch"]
        rec["selected"] = None
    if story_root is not None:
        save_assets_index(story_root, payload, extra=_index_extra(story_root))
    return payload


def append_history_file(
    story_root: Path | str | None,
    *,
    character_id: str | None = None,
    scene_id: str | None = None,
    filename: str,
    parent_id: str | None = None,
) -> dict[str, Any]:
    """Record a file in history without changing selected or QC verdict."""
    name = _filename(filename)
    payload = load_assets_index(story_root)
    rec = _target_record(payload, character_id=character_id, scene_id=scene_id, parent_id=parent_id)
    if name and name not in rec["history"]:
        rec["history"].append(name)
    if story_root is not None:
        save_assets_index(story_root, payload, extra=_index_extra(story_root))
    return payload


def auto_lock_first_success(
    story_root: Path | str | None,
    *,
    character_id: str | None = None,
    scene_id: str | None = None,
    filename: str,
    parent_id: str | None = None,
    index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Legacy helper for explicit tests only. Live generate/lock paths must not call this.

    Does not stamp current QC fields, so has_locked_identity/is_qc_locked stay false.
    """
    name = _filename(filename)
    payload = normalize_assets_index(index if index is not None else load_assets_index(story_root))
    if not name:
        return payload
    if character_id:
        rec = _char_record((payload.get("characters") or {}).get(character_id) or {})
        if name not in rec["history"]:
            rec["history"].append(name)
        if not rec.get("selected"):
            rec["selected"] = name
        if rec.get("parent_id") is None and parent_id:
            rec["parent_id"] = str(parent_id)
        payload.setdefault("characters", {})[str(character_id)] = rec
        if story_root is not None and rec.get("selected") == name:
            src = character_dir(story_root, character_id) / name
            alias = character_dir(story_root, character_id) / FRONT_ALIAS_FILENAME
            # Two-phase sheets already have a portrait at sheet_front.png; keep it.
            if name == TURNAROUND_FILENAME:
                if not _file_ok(alias):
                    write_front_alias(src)
            else:
                write_front_alias(src)
    elif scene_id:
        rec = _scene_record((payload.get("scenes") or {}).get(scene_id) or {})
        if name not in rec["history"]:
            rec["history"].append(name)
        if not rec.get("selected"):
            rec["selected"] = name
        if parent_id and not rec.get("parent_id"):
            rec["parent_id"] = str(parent_id)
        payload.setdefault("scenes", {})[str(scene_id)] = rec
    if story_root is not None:
        extra = {}
        path = Path(story_root) / INDEX_REL
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            extra = {k: v for k, v in (raw or {}).items() if k not in {"characters", "scenes"}}
        save_assets_index(story_root, payload, extra=extra)
    return payload


def set_selected(
    story_root: Path | str,
    *,
    character_id: str | None = None,
    scene_id: str | None = None,
    filename: str,
) -> dict[str, Any]:
    name = _filename(filename)
    payload = load_assets_index(story_root)
    if character_id:
        rec = _char_record((payload.get("characters") or {}).get(character_id) or {})
        if name and name not in rec["history"]:
            rec["history"].append(name)
        rec["selected"] = name or rec.get("selected")
        payload.setdefault("characters", {})[str(character_id)] = rec
        if name and name != TURNAROUND_FILENAME:
            # Keep the front alias pointing at the locked portrait, not the composite strip.
            locked = character_dir(story_root, character_id) / name
            if _file_ok(locked):
                (character_dir(story_root, character_id) / FRONT_ALIAS_FILENAME).write_bytes(locked.read_bytes())
    elif scene_id:
        rec = _scene_record((payload.get("scenes") or {}).get(scene_id) or {})
        if name and name not in rec["history"]:
            rec["history"].append(name)
        rec["selected"] = name or rec.get("selected")
        payload.setdefault("scenes", {})[str(scene_id)] = rec
    save_assets_index(story_root, payload)
    return payload


def load_regen_requests(story_root: Path | str | None) -> list[dict[str, str]]:
    if story_root is None:
        return []
    path = Path(story_root) / REGEN_REL
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = raw.get("requests") if isinstance(raw, dict) else raw
    out: list[dict[str, str]] = []
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        entity_id = str(item.get("id") or "").strip()
        if kind in {"character", "scene"} and entity_id:
            out.append({"kind": kind, "id": entity_id})
    return out


def save_regen_requests(story_root: Path | str, requests: list[dict[str, str]]) -> Path:
    dest = Path(story_root) / REGEN_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"requests": requests}, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def queue_regen_request(story_root: Path | str, kind: str, entity_id: str) -> list[dict[str, str]]:
    requests = load_regen_requests(story_root)
    row = {"kind": str(kind).strip().lower(), "id": str(entity_id).strip()}
    if row not in requests:
        requests.append(row)
    save_regen_requests(story_root, requests)
    return requests


def pop_regen_requests(story_root: Path | str | None) -> list[dict[str, str]]:
    requests = load_regen_requests(story_root)
    if story_root is not None:
        path = Path(story_root) / REGEN_REL
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                save_regen_requests(story_root, [])
    return requests

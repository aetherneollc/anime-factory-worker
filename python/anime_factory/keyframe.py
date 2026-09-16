"""Chain-head first frames only. Gate 3 at the entrance; missing assets bounce to derive.

chain_index > 0 never draws a still; first_frame is the previous segment last_frame.
ref2va heads use character sheets as composition refs and do not mint f1.png.

Keyframes are drawn from the **still** prompt (English, no camera direction) and
validated as real PNGs at the requested size. The 3-byte `f1.png` files on disk
are what the old `>= 1` byte check accepted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from anime_factory.board import assert_still_prompt_clean, shot_visual_prompt
from anime_factory.continuity_gates import gate3_assets
from anime_factory.design import (
    KolorsClient,
    assert_still_blob,
    derive_missing_assets,
    locked_seed,
    period_lists,
    still_file_ok,
    style_md_prefix,
    style_negative,
    style_prompt,
)
from anime_factory.directors.common import is_chain_head, needs_first_frame_still
from anime_factory.models import IMAGE_MODEL, STILL_HEIGHT, STILL_IMAGE_SIZE, STILL_WIDTH
from anime_factory.r2_paths import join_story
from anime_factory.visual_qc import (
    QC_SEED_SALTS,
    require_clip_for_client,
    score_still,
)


def keyframe_still_path(story_root: Path, episode_code: str, segment_id: str) -> Path:
    return Path(story_root) / "episodes" / episode_code / "keyframes" / segment_id / "f1.png"


def _file_ok(path: Path | None, *, allow_placeholder: bool = True) -> bool:
    """A still on disk must be a PNG of real size, not a 3-byte marker."""
    return still_file_ok(Path(path) if path else None, allow_placeholder=allow_placeholder)


# Shared with the GPU session, which used to accept anything `>= 1` byte.
keyframe_file_ok = _file_ok


def _spec_path(item: dict) -> str:
    return str(item.get("path") or item.get("file") or "").strip()


def sheet_file_on_disk(segment: dict, assets_index: dict, story_root: Path | None) -> Path | None:
    """Resolve the locked character sheet (or other composition ref) that already exists on disk."""
    if story_root is None:
        return None
    root = Path(story_root)
    try:
        from anime_factory.asset_lock import locked_character_file, locked_scene_file
    except ImportError:
        locked_character_file = None  # type: ignore[assignment]
        locked_scene_file = None  # type: ignore[assignment]
    items = {str(it.get("id") or ""): it for it in (assets_index.get("items") or [])}
    lock = assets_index.get("characters") if isinstance(assets_index.get("characters"), dict) else {}
    scene_lock = assets_index.get("scenes") if isinstance(assets_index.get("scenes"), dict) else {}
    candidates: list[Path] = []

    def _lock_char(cid: str) -> None:
        if not cid:
            return
        try:
            from anime_factory.asset_lock import is_qc_locked, locked_character_file
        except ImportError:
            is_qc_locked = None  # type: ignore[assignment]
        if is_qc_locked is not None and not is_qc_locked(root, character_id=cid):
            return
        if locked_character_file is not None:
            try:
                candidates.append(Path(locked_character_file(root, cid)))
            except (TypeError, AttributeError, OSError):
                pass
        selected = str((lock.get(cid) or {}).get("selected") or "").strip()
        if selected:
            candidates.append(root / "assets" / "characters" / cid / selected)
        candidates.append(root / "assets" / "characters" / cid / "sheet_front.png")

    def _lock_scene(lid: str) -> None:
        if not lid:
            return
        try:
            from anime_factory.asset_lock import is_qc_locked, locked_scene_file
        except ImportError:
            is_qc_locked = None  # type: ignore[assignment]
        if is_qc_locked is not None and not is_qc_locked(root, scene_id=lid):
            return
        if not lid:
            return
        if locked_scene_file is not None:
            try:
                candidates.append(Path(locked_scene_file(root, lid)))
            except (TypeError, AttributeError, OSError):
                pass
        selected = str((scene_lock.get(lid) or {}).get("selected") or "").strip()
        if selected:
            candidates.append(root / "assets" / "scenes" / lid / selected)
        candidates.append(root / "assets" / "scenes" / lid / "plate_base.png")
        candidates.append(root / "assets" / "locations" / lid / "plate_base.png")

    for rid in list(segment.get("refs") or []):
        text = str(rid or "").strip()
        if not text:
            continue
        if text.startswith("char_") and text.endswith("_sheet"):
            _lock_char(text[len("char_") : -len("_sheet")])
            continue
        if text.startswith("plate_"):
            _lock_scene(text[len("plate_") :])
            continue
        rel = _spec_path(items.get(text) or {})
        if rel:
            candidates.append(root / rel)
        as_path = Path(text)
        candidates.append(as_path if as_path.is_absolute() else root / as_path)
    _lock_char(str(segment.get("character_id") or "").strip())
    plate_id = str(segment.get("plate_id") or "").strip()
    if plate_id:
        _lock_scene(plate_id[len("plate_") :] if plate_id.startswith("plate_") else plate_id)
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if _file_ok(path):
            return path
    return None


def _has_first_frame_file(story_root: Path, episode_code: str, segment: dict) -> bool:
    still = keyframe_still_path(story_root, episode_code, str(segment.get("id") or ""))
    if _file_ok(still):
        return True
    inherited = Path(str(segment.get("first_frame_path") or segment.get("keyframe_path") or ""))
    return _file_ok(inherited)


def assert_keyframe_files(
    story_root: Path,
    episode_code: str,
    segments: list[dict],
    assets_index: dict | None = None,
) -> None:
    """Refuse a succeeded keyframe stage when required stills are missing on disk."""
    missing_f1: list[str] = []
    missing_refs: list[str] = []
    index = assets_index or {"items": []}
    for segment in segments:
        sid = str(segment.get("id") or "")
        if not sid:
            continue
        if needs_first_frame_still(segment):
            if not _has_first_frame_file(story_root, episode_code, segment):
                missing_f1.append(sid)
            continue
        if is_chain_head(segment) and (
            str(segment.get("h3_mode") or "") == "ref2va" or str(segment.get("character_id") or "").strip()
        ):
            if sheet_file_on_disk(segment, index, story_root) is None:
                missing_refs.append(sid)
    if missing_f1 or missing_refs:
        parts = []
        if missing_f1:
            parts.append(f"missing f1.png for fl2va {missing_f1}")
        if missing_refs:
            parts.append(f"missing character sheets for ref2va {missing_refs}")
        raise RuntimeError("keyframe stage has 0 required files on disk: " + "; ".join(parts))


def ensure_keyframe(
    conn: sqlite3.Connection,
    story_id: str,
    episode_code: str,
    segment: dict,
    geo: dict,
    assets_index: dict,
    specs: dict,
    client: KolorsClient,
    period_md: str | None,
    world_mode: str,
    story_root: Path | None = None,
) -> str:
    if not is_chain_head(segment):
        inherited = (
            segment.get("first_frame_path")
            or segment.get("keyframe_path")
            or segment.get("last_frame_path")
        )
        key = str(inherited or "")
        if key:
            conn.execute(
                "UPDATE segments SET keyframe_path = ?, status = 'pending_chain' WHERE id = ?",
                (key, segment["id"]),
            )
            conn.commit()
        else:
            conn.execute(
                "UPDATE segments SET status = 'pending_chain' WHERE id = ?",
                (segment["id"],),
            )
            conn.commit()
        return key

    violations, missing = gate3_assets(conn, segment, geo, assets_index, story_root=story_root)
    if missing:
        derive_missing_assets(
            conn,
            story_id,
            missing,
            specs,
            client,
            period_md,
            world_mode,
            story_root,
            clip_scorer=None,
        )
        assets_index.setdefault("items", [])
        for mid in missing:
            if not any(it.get("id") == mid for it in assets_index["items"]):
                assets_index["items"].append({"id": mid, **specs.get(mid, {})})
        violations, missing = gate3_assets(conn, segment, geo, assets_index, story_root=story_root)
    if missing:
        raise RuntimeError(f"still missing assets after derive: {missing}")
    visual_block = [
        v
        for v in violations
        if v.code in {"interior_mismatch", "unlocked_identity", "stale_visual", "qc_failed_identity"}
    ]
    if visual_block:
        raise RuntimeError(visual_block[0].message)

    if not needs_first_frame_still(segment):
        sheet = sheet_file_on_disk(segment, assets_index, story_root)
        if sheet is None:
            raise RuntimeError(
                f"ref2va {segment.get('id')} has no character sheet or scene plate on disk; refusing fake f1.png"
            )
        rel = sheet.relative_to(story_root).as_posix() if story_root is not None else sheet.name
        key = join_story(story_id, rel)
        conn.execute(
            "UPDATE segments SET keyframe_path = ?, status = 'prepared' WHERE id = ?",
            (key, segment["id"]),
        )
        conn.commit()
        return key

    rel = f"episodes/{episode_code}/keyframes/{segment['id']}/f1.png"
    key = join_story(story_id, rel)
    if story_root is None:
        raise RuntimeError(f"cannot mint or verify keyframe without story_root: {rel}")
    dest = story_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    live = bool(getattr(client, "live", False))
    require_clip = require_clip_for_client(client)
    if not _file_ok(dest, allow_placeholder=not live):
        visual = shot_visual_prompt(segment)
        if not visual:
            raise RuntimeError(
                f"segment {segment.get('id')} has no first_frame_prompt; refusing to draw a keyframe "
                "from the segment id"
            )
        assert_still_prompt_clean(visual, label=str(segment.get("id") or "keyframe"))
        prefix, bible_negative = style_md_prefix(story_root)
        positives, negatives = period_lists(period_md, world_mode)
        prompt = style_prompt(visual, positives, prefix=prefix, kind="keyframe")
        last_reasons: list[str] = []
        written = False
        for salt in QC_SEED_SALTS:
            seed = locked_seed(f"{story_id}:{segment['id']}", salt)
            png = client.generate(
                {
                    "model": IMAGE_MODEL,
                    "prompt": prompt,
                    "negative_prompt": style_negative(negatives, base=bible_negative),
                    "seed": seed,
                    "image_size": STILL_IMAGE_SIZE,
                    "_styled": True,
                    "_kind": "keyframe",
                    "_label": f"keyframe:{segment['id']}",
                }
            )
            assert_still_blob(
                png,
                width=STILL_WIDTH,
                height=STILL_HEIGHT,
                label=f"keyframe {rel}",
                allow_placeholder=not live,
            )
            qc = score_still(
                png,
                kind="keyframe",
                prompt=prompt,
                require_clip=require_clip,
                allow_placeholder=not live,
                seed=seed,
                label=f"keyframe {rel}",
            )
            if qc.passed:
                dest.write_bytes(png)
                written = True
                break
            last_reasons = list(qc.reasons)
        if not written:
            raise RuntimeError(
                f"keyframe QC failed after {len(QC_SEED_SALTS)} seeds for {segment.get('id')}: {last_reasons}"
            )
    if not _file_ok(dest):
        raise RuntimeError(f"keyframe missing on disk after generate: {rel}")
    conn.execute(
        "UPDATE segments SET keyframe_path = ?, status = 'prepared' WHERE id = ?",
        (key, segment["id"]),
    )
    conn.commit()
    return key

"""Chain-head composition frames and per-cut keyframe packages.

chain_index > 0 never draws a still; first_frame is the previous segment last_frame.
H3 v2: ref2va heads also mint composition f1.png plus per-cut f01/f02/... anchors.
Prompts compose locked character identity + scene anchor + prop state + pose.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from anime_factory.board import assert_still_prompt_clean, scrub_still_prompt, shot_visual_prompt
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


def cut_keyframe_path(story_root: Path, episode_code: str, segment_id: str, seq: int) -> Path:
    from anime_factory.h3_storyboard import cut_keyframe_name

    return Path(story_root) / "episodes" / episode_code / "keyframes" / segment_id / cut_keyframe_name(seq)


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


def _identity_lock_text(segment: dict, story_root: Path | None) -> str:
    cid = str(segment.get("character_id") or "").strip()
    if not cid or story_root is None:
        return ""
    try:
        from anime_factory.db import open_db

        db_path = Path(story_root) / "story.sqlite"
        if not db_path.is_file():
            return ""
        conn = open_db(db_path)
        try:
            row = conn.execute(
                "SELECT identity_prompt FROM characters WHERE id = ?",
                (cid,),
            ).fetchone()
            return str((row["identity_prompt"] if row else "") or "")
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return ""


def _scene_anchor_text(segment: dict) -> str:
    for key in ("scene_anchor", "setting", "location_prompt", "plate_prompt"):
        text = str(segment.get(key) or "").strip()
        if text:
            return text
    return str(segment.get("scene_id") or segment.get("location_id") or "").strip()


def compose_cut_still_prompt(segment: dict, cut: dict, *, story_root: Path | None = None) -> str:
    """Full character lock + scene anchor + prop state + current pose."""
    parts: list[str] = []
    identity = _identity_lock_text(segment, story_root)
    if identity:
        parts.append(identity)
    scene = _scene_anchor_text(segment)
    if scene:
        parts.append(scene)
    props = [str(p).strip() for p in (cut.get("props") or segment.get("props") or []) if str(p).strip()]
    if props:
        parts.append("props: " + ", ".join(props))
    pose = str(cut.get("frame_prompt") or cut.get("staging") or cut.get("pose") or "").strip()
    if pose:
        parts.append(pose)
    size = str(cut.get("size") or "MS").strip()
    parts.append(f"framing {size}")
    visual = ", ".join(p for p in parts if p)
    return scrub_still_prompt(visual) or shot_visual_prompt(segment)


def _scene_id_from_segment(segment: dict) -> str:
    """Plate/location id from segment fields or refs (never a character id)."""
    for key in ("location_id", "scene_id", "plate_id"):
        raw = str(segment.get(key) or "").strip()
        if not raw:
            continue
        if raw.startswith("plate_"):
            return raw[len("plate_") :]
        return raw
    for rid in list(segment.get("refs") or []):
        text = str(rid or "").strip()
        if text.startswith("plate_"):
            return text[len("plate_") :]
    return ""


def locked_plate_path(segment: dict, assets_index: dict, story_root: Path | None) -> Path | None:
    """QC-locked scene plate on disk. Same resolution as sheet_file_on_disk._lock_scene; never a sheet."""
    if story_root is None:
        return None
    root = Path(story_root)
    lid = _scene_id_from_segment(segment)
    if not lid:
        return None
    try:
        from anime_factory.asset_lock import is_qc_locked, locked_scene_file
    except ImportError:
        is_qc_locked = None  # type: ignore[assignment]
        locked_scene_file = None  # type: ignore[assignment]
    if is_qc_locked is not None and not is_qc_locked(root, scene_id=lid):
        return None
    candidates: list[Path] = []
    if locked_scene_file is not None:
        try:
            candidates.append(Path(locked_scene_file(root, lid)))
        except (TypeError, AttributeError, OSError):
            pass
    scene_lock = assets_index.get("scenes") if isinstance(assets_index.get("scenes"), dict) else {}
    selected = str((scene_lock.get(lid) or {}).get("selected") or "").strip()
    if selected:
        candidates.append(root / "assets" / "scenes" / lid / selected)
    candidates.append(root / "assets" / "scenes" / lid / "plate_base.png")
    candidates.append(root / "assets" / "locations" / lid / "plate_base.png")
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if _file_ok(path):
            return path
    return None


def locked_plate_png(segment: dict, assets_index: dict, story_root: Path | None) -> bytes | None:
    """Locked scene plate bytes for keyframe IP-Adapter. Never used on plate mint itself."""
    path = locked_plate_path(segment, assets_index, story_root)
    if path is None:
        return None
    return path.read_bytes()


def _mint_still(
    *,
    dest: Path,
    prompt: str,
    label: str,
    story_id: str,
    seed_key: str,
    client: KolorsClient,
    period_md: str | None,
    world_mode: str,
    story_root: Path | None,
    parent_png: bytes | None = None,
    parent_kind: str | None = None,
) -> None:
    prompt = scrub_still_prompt(prompt)
    assert_still_prompt_clean(prompt, label=label)
    prefix, bible_negative = style_md_prefix(story_root)
    positives, negatives = period_lists(period_md, world_mode)
    styled = style_prompt(prompt, positives, prefix=prefix, kind="keyframe")
    live = bool(getattr(client, "live", False))
    require_clip = require_clip_for_client(client)
    last_reasons: list[str] = []
    written = False
    dest.parent.mkdir(parents=True, exist_ok=True)
    for salt in QC_SEED_SALTS:
        seed = locked_seed(f"{story_id}:{seed_key}", salt)
        payload: dict = {
            "model": IMAGE_MODEL,
            "prompt": styled,
            "negative_prompt": style_negative(negatives, base=bible_negative),
            "seed": seed,
            "image_size": STILL_IMAGE_SIZE,
            "_styled": True,
            "_kind": "keyframe",
            "_label": label,
        }
        if parent_png:
            payload["_parent_png"] = parent_png
            payload["_parent_kind"] = parent_kind or "scene_plate"
        png = client.generate(payload)
        assert_still_blob(
            png,
            width=STILL_WIDTH,
            height=STILL_HEIGHT,
            label=label,
            allow_placeholder=not live,
        )
        qc = score_still(
            png,
            kind="keyframe",
            prompt=styled,
            require_clip=require_clip,
            allow_placeholder=not live,
            seed=seed,
            label=label,
        )
        if qc.passed:
            dest.write_bytes(png)
            written = True
            break
        last_reasons = list(qc.reasons)
    if not written:
        raise RuntimeError(f"keyframe QC failed after {len(QC_SEED_SALTS)} seeds for {label}: {last_reasons}")


def assert_keyframe_files(
    story_root: Path,
    episode_code: str,
    segments: list[dict],
    assets_index: dict | None = None,
) -> None:
    """Refuse a succeeded keyframe stage when required stills are missing on disk."""
    missing_f1: list[str] = []
    missing_cuts: list[str] = []
    missing_refs: list[str] = []
    index = assets_index or {"items": []}
    for segment in segments:
        sid = str(segment.get("id") or "")
        if not sid:
            continue
        if needs_first_frame_still(segment):
            if not _has_first_frame_file(story_root, episode_code, segment):
                missing_f1.append(sid)
            cuts = [c for c in (segment.get("cuts") or []) if isinstance(c, dict)]
            for cut in cuts:
                seq = int(cut.get("seq") or 0)
                if seq <= 0:
                    continue
                path = cut_keyframe_path(story_root, episode_code, sid, seq)
                if not _file_ok(path):
                    missing_cuts.append(f"{sid}/f{seq:02d}")
            continue
        if is_chain_head(segment) and (
            str(segment.get("h3_mode") or "") == "ref2va" or str(segment.get("character_id") or "").strip()
        ):
            if sheet_file_on_disk(segment, index, story_root) is None:
                missing_refs.append(sid)
    if missing_f1 or missing_cuts or missing_refs:
        parts = []
        if missing_f1:
            parts.append(f"missing f1.png for chain heads {missing_f1}")
        if missing_cuts:
            parts.append(f"missing per-cut keyframes {missing_cuts}")
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

    if story_root is None:
        raise RuntimeError(f"cannot mint or verify keyframe without story_root: {segment.get('id')}")

    if not needs_first_frame_still(segment):
        sheet = sheet_file_on_disk(segment, assets_index, story_root)
        if sheet is None:
            raise RuntimeError(
                f"ref2va {segment.get('id')} has no character sheet or scene plate on disk; refusing fake f1.png"
            )
        rel = sheet.relative_to(story_root).as_posix()
        key = join_story(story_id, rel)
        conn.execute(
            "UPDATE segments SET keyframe_path = ?, status = 'prepared' WHERE id = ?",
            (key, segment["id"]),
        )
        conn.commit()
        return key

    # Character-bearing heads still need locked sheets even while minting f1.
    if str(segment.get("character_id") or "").strip() and segment.get("on_camera") is not False:
        sheet = sheet_file_on_disk(segment, assets_index, story_root)
        if sheet is None and str(segment.get("h3_mode") or "") == "ref2va":
            raise RuntimeError(
                f"ref2va {segment.get('id')} has no character sheet on disk; refusing composition-only f1.png"
            )

    rel = f"episodes/{episode_code}/keyframes/{segment['id']}/f1.png"
    key = join_story(story_id, rel)
    dest = story_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    live = bool(getattr(client, "live", False))
    if not _file_ok(dest, allow_placeholder=not live):
        visual = shot_visual_prompt(segment)
        if not visual:
            cuts = [c for c in (segment.get("cuts") or []) if isinstance(c, dict)]
            if cuts:
                visual = compose_cut_still_prompt(segment, cuts[0], story_root=story_root)
        if not visual:
            raise RuntimeError(
                f"segment {segment.get('id')} has no first_frame_prompt; refusing to draw a keyframe "
                "from the segment id"
            )
        plate_png = locked_plate_png(segment, assets_index, story_root)
        _mint_still(
            dest=dest,
            prompt=visual,
            label=f"keyframe {rel}",
            story_id=story_id,
            seed_key=str(segment["id"]),
            client=client,
            period_md=period_md,
            world_mode=world_mode,
            story_root=story_root,
            parent_png=plate_png,
            parent_kind="scene_plate" if plate_png else None,
        )
    if not _file_ok(dest):
        raise RuntimeError(f"keyframe missing on disk after generate: {rel}")

    try:
        from anime_factory.h3_storyboard import ensure_cuts_for_segment

        cuts = ensure_cuts_for_segment(segment)
    except Exception:  # noqa: BLE001
        cuts = [c for c in (segment.get("cuts") or []) if isinstance(c, dict)]
    plate_png = locked_plate_png(segment, assets_index, story_root)
    for cut in cuts:
        seq = int(cut.get("seq") or 0)
        if seq <= 0:
            continue
        cut_dest = cut_keyframe_path(story_root, episode_code, str(segment["id"]), seq)
        if _file_ok(cut_dest, allow_placeholder=not live):
            continue
        cut_prompt = compose_cut_still_prompt(segment, cut, story_root=story_root)
        _mint_still(
            dest=cut_dest,
            prompt=cut_prompt,
            label=f"cut-keyframe {segment['id']}/f{seq:02d}",
            story_id=story_id,
            seed_key=f"{segment['id']}:cut{seq}",
            client=client,
            period_md=period_md,
            world_mode=world_mode,
            story_root=story_root,
            parent_png=plate_png,
            parent_kind="scene_plate" if plate_png else None,
        )
        cut["keyframe_path"] = join_story(
            story_id,
            f"episodes/{episode_code}/keyframes/{segment['id']}/f{seq:02d}.png",
        )

    conn.execute(
        "UPDATE segments SET keyframe_path = ?, status = 'prepared' WHERE id = ?",
        (key, segment["id"]),
    )
    conn.commit()
    return key

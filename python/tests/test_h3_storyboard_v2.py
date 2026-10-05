"""H3 storyboard v2 contract + dual-repo prompt compilation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anime_factory.design import CharacterIdentityError, parse_identity_age, validate_character_identity
from anime_factory.h3_storyboard import (
    SCHEMA_VERSION,
    build_board_v2_payload,
    build_reference_manifest,
    compile_picture_prompt,
    compile_segment_v2,
    cut_frame_ranges,
    ensure_cuts_for_segment,
    validate_board_gates,
)
from anime_factory.qc import plan_repair, select_passing_generation
from gpu_worker.h3 import segment_prompt

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "h3_storyboard_v2"


def _two_cut_segment() -> dict:
    return {
        "id": "EP001-01",
        "duration": 8.0,
        "scene_id": "classroom",
        "character_id": "zhou",
        "on_camera": True,
        "refs": ["char_zhou_sheet", "plate_classroom"],
        "cuts": [
            {
                "seq": 1,
                "seconds": 4.0,
                "size": "MS",
                "camera": "Static Shot",
                "characters": ["zhou"],
                "frame_prompt": "60 y/o janitor in navy blue worker uniform jacket holds keys in classroom",
                "beat_ids": ["b1"],
                "line": {"zh": "钥匙呢"},
            },
            {
                "seq": 2,
                "seconds": 4.0,
                "size": "CU",
                "camera": "Push In",
                "characters": ["zhou"],
                "frame_prompt": "close on weathered hands turning the classroom key",
                "beat_ids": ["b2"],
                "props": ["keys"],
            },
        ],
    }


def test_compile_picture_prompt_never_uses_dialogue_line():
    seg = _two_cut_segment()
    seg["line"] = {"zh": "这段对白绝不能进画面提示词"}
    prompt = compile_picture_prompt(seg)
    assert "对白" not in prompt
    assert "绝不能" not in prompt
    assert "navy blue worker uniform" in prompt or "janitor" in prompt
    assert "no readable text" in prompt
    # h3.py wrapper matches
    assert "对白" not in segment_prompt(seg)


def test_cut_alignment_and_manifest_order():
    seg = compile_segment_v2(_two_cut_segment())
    assert seg["schema_version"] == SCHEMA_VERSION
    cuts = ensure_cuts_for_segment(seg)
    assert abs(sum(c["seconds"] for c in cuts) - 8.0) < 1e-6
    rows = cut_frame_ranges(cuts, total_frames=197)
    assert rows[0]["start_frame"] == 0
    assert rows[-1]["end_frame"] == 197
    manifest = build_reference_manifest(seg)
    assert manifest["order"] == ["temporal", "character", "scene", "prop"]
    assert any("zhou" in r or "sheet" in r for r in manifest["characters"])
    assert manifest["refs"][0].startswith("char_") or "sheet" in manifest["refs"][0]


def test_board_gates_reject_whiteboard_and_cross_scene():
    bad = _two_cut_segment()
    bad["cuts"][0]["frame_prompt"] = "character writes chinese on whiteboard"
    validation = validate_board_gates([bad])
    assert validation["ok"] is False
    codes = {g["code"] for g in validation["failed"]}
    assert "no_generated_text" in codes

    cross = _two_cut_segment()
    cross["cuts"][1]["scene_id"] = "rooftop"
    validation2 = validate_board_gates([cross])
    assert any(g["code"] == "cross_scene_cut" for g in validation2["failed"])


def test_janitor_identity_accepts_composite_uniform_and_age():
    identity = (
        "1boy, 60 y/o, short grey hair, brown eyes, navy blue worker uniform jacket, "
        "lean elderly build, no military uniform, no epaulettes"
    )
    out = validate_character_identity(identity, character_id="zhou", name="周伯")
    assert "navy" in out.lower()
    assert parse_identity_age(out) == 60
    with pytest.raises(CharacterIdentityError):
        validate_character_identity(
            "1boy, young adult, short black hair, brown eyes",
            character_id="hero",
        )


def test_plan_repair_maps_visual_reasons():
    repair = plan_repair({"h3_prompt": "base", "seed": 11, "h3_mode": "ref2va"}, ["generated_text_or_logo"], 1)
    assert repair["strategy"] == "reinforce_no_text"
    assert "whiteboard" in repair["h3_prompt"]
    repair2 = plan_repair({"h3_prompt": "base", "seed": 11}, ["identity_drift", "reinforce_identity"], 1)
    assert repair2["strategy"] == "reinforce_identity"


def test_select_passing_generation_ignores_orphan_mp4(tmp_path):
    root = tmp_path
    sid = "EP001-01"
    shot_dir = root / "shots" / sid
    shot_dir.mkdir(parents=True)
    orphan = shot_dir / "generation-001.mp4"
    orphan.write_bytes(b"x" * 5000)
    assert select_passing_generation(None, root, sid) is None

    import sqlite3

    from anime_factory.db import migrate

    db = tmp_path / "story.sqlite"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    migrate(conn)
    passing = shot_dir / "generation-002.mp4"
    passing.write_bytes(b"y" * 5000)
    conn.execute(
        """
        INSERT INTO generation_results (id, segment_id, version, path, seed, h3_mode, status, qc_verdict, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("g2", sid, 2, f"shots/{sid}/generation-002.mp4", 1, "ref2va", "completed", "pass", "t"),
    )
    conn.execute(
        """
        INSERT INTO generation_results (id, segment_id, version, path, seed, h3_mode, status, qc_verdict, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("g1", sid, 1, f"shots/{sid}/generation-001.mp4", 1, "ref2va", "failed", "fail", "t"),
    )
    conn.commit()
    selected = select_passing_generation(conn, root, sid)
    assert selected is not None
    path, version, _ = selected
    assert path.name == "generation-002.mp4"
    assert version == 2
    conn.close()


def test_golden_fixture_parity_if_present():
    fixture = FIXTURES / "two_cut_classroom.json"
    assert fixture.is_file(), "two_cut_classroom.json fixture missing"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    board = build_board_v2_payload(episode_code="EP001", shots=data["shots"], cast=data.get("cast"))
    assert board["schema_version"] == SCHEMA_VERSION
    assert board["validation"]["ok"] is True
    prompt = compile_picture_prompt(board["segments"][0])
    assert data["expected_prompt_substr"] in prompt


def test_resume_rejects_cross_shot_pass_and_unverified_completion(tmp_path):
    import sqlite3
    from anime_factory.db import migrate
    from anime_factory.qc import mark_completed_passing
    conn = sqlite3.connect(tmp_path / 'story.sqlite')
    conn.row_factory = sqlite3.Row
    migrate(conn)
    for sid in ('s1', 's2'):
        folder = tmp_path / 'shots' / sid
        folder.mkdir(parents=True)
        (folder / 'generation-001.mp4').write_bytes(b'x' * 5000)
        (folder / 'generation-004.mp4').write_bytes(b'y' * 5000)
        conn.execute("INSERT INTO segments (id,episode_code,scene_id,seq,status,video_path) VALUES (?,?,?,?,'completed',?)",
                     (sid, 'EP001', 'scene', 1, 'shots/s2/generation-004.mp4'))
    for gid, sid, version, path, verdict in (
        ('s1-g1', 's1', 1, 'shots/s1/generation-001.mp4', 'pass'),
        ('s1-g4', 's1', 4, 'shots/s2/generation-004.mp4', 'pass'),
        ('s2-g1', 's2', 1, 'shots/s2/generation-001.mp4', 'retry'),
    ):
        conn.execute("INSERT INTO generation_results (id,segment_id,version,path,status,qc_verdict,created_at) VALUES (?,?,?,?,'completed',?,'t')",
                     (gid, sid, version, path, verdict))
    conn.commit()
    assert select_passing_generation(conn, tmp_path, 's1')[0].name == 'generation-001.mp4'
    assert select_passing_generation(conn, tmp_path, 's2') is None
    with pytest.raises(ValueError, match='no QC-passing'):
        mark_completed_passing(conn, 's2', 'shots/s2/generation-004.mp4')
    with pytest.raises(ValueError, match='another segment'):
        mark_completed_passing(conn, 's1', 'shots/s2/generation-004.mp4')
    conn.close()


def test_chain_qc_uses_original_frame_after_unique_comfy_staging(tmp_path):
    from anime_factory.qc import chain_qc
    previous = tmp_path / 'prev-last.png'
    copied = tmp_path / 'chain-first.png'
    previous.write_bytes(b'same-frame')
    copied.write_bytes(previous.read_bytes())
    prev = {'chain_id': 'c1', 'last_frame_path': str(previous)}
    shot = {'chain_id': 'c1', 'chain_index': 1,
            'first_frame_path': 's2_first_abcd.png', 'first_frame_source_path': str(copied)}
    assert chain_qc(prev, shot)[0] == 'pass'
    copied.write_bytes(b'wrong-frame')
    assert chain_qc(prev, shot)[0] == 'retry'

"""SIGBUS regression: a short safetensors header must not be treated as mmap-safe.

Vast 51618093 died during design (D1 boot_stage=design, heartbeat 18:09Z) after
the Comfy still prompt succeeded. af-start had no Python traceback. The H3
weight thread was still pulling Hub files in-process; a header that claims
bytes past EOF makes safetensors/xet mmap raise SIGBUS and kill the heartbeat.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from anime_factory.db import open_db
from gpu_worker import weights
from gpu_worker.preflight import is_host_fault

REPO_ROOT = Path(__file__).resolve().parents[1]
START_SH = REPO_ROOT / "deploy" / "gpu-worker" / "start.sh"


def _write_safetensors(path: Path, *, data_len: int, payload: int) -> None:
    header = {"w": {"dtype": "U8", "shape": [data_len], "data_offsets": [0, data_len]}}
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * payload)


def test_truncated_safetensors_fails_contract_before_mmap(tmp_path: Path):
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, data_len=64, payload=8)
    assert path.stat().st_size > 8
    reason = weights.safetensors_mmap_unsafe(path)
    assert reason is not None
    assert "truncated safetensors" in reason
    assert weights._file_meets_contract(path, {"min_bytes": "1"}) is False


def test_complete_safetensors_header_is_mmap_safe(tmp_path: Path):
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, data_len=16, payload=16)
    assert weights.safetensors_mmap_unsafe(path) is None
    assert weights._file_meets_contract(path, None) is True


def test_open_clip_refuses_truncated_checkpoint_without_loading(tmp_path, monkeypatch):
    from anime_factory.visual_qc import ClipScorerUnavailable, OpenClipScorer

    weight = tmp_path / "open_clip_model.safetensors"
    _write_safetensors(weight, data_len=64, payload=8)
    monkeypatch.setenv("AF_VISUAL_QC_CLIP_WEIGHTS", str(weight))
    monkeypatch.setenv("AF_VISUAL_QC_CLIP_DIR", str(tmp_path))
    called = {"load": False}

    class _Boom:
        @staticmethod
        def load_checkpoint(*_a, **_k):
            called["load"] = True
            raise AssertionError("must not mmap a truncated safetensors file")

    monkeypatch.setitem(
        __import__("sys").modules,
        "open_clip",
        type("M", (), {"load_checkpoint": _Boom.load_checkpoint})(),
    )
    with pytest.raises(ClipScorerUnavailable, match="truncated safetensors"):
        OpenClipScorer.from_pretrained()
    assert called["load"] is False


def test_start_disables_xet_before_python():
    text = START_SH.read_text(encoding="utf-8")
    assert "HF_HUB_DISABLE_XET=1" in text
    assert text.index("HF_HUB_DISABLE_XET") < text.index("exec python")


def test_open_db_disables_sqlite_mmap(tmp_path: Path):
    conn = open_db(tmp_path / "story.sqlite")
    try:
        assert conn.execute("PRAGMA mmap_size").fetchone()[0] == 0
    finally:
        conn.close()


def test_mmap_fault_recycles_without_host_rating(monkeypatch):
    seen: dict = {}

    def teardown(instance_id, reason, error=None):
        seen["instance_id"] = instance_id
        seen["reason"] = reason
        seen["error"] = error
        return {"ok": True, "destroyed": False, "dry_run": True}

    monkeypatch.setattr("gpu_worker.session.teardown_for_reason", teardown)
    result = weights.report_mmap_fault("51618093")
    assert result["dry_run"] is True
    assert seen["reason"] == "explicit_abort"
    assert "weight_mmap_failed" in seen["error"]
    assert "SIGBUS" in seen["error"]
    assert is_host_fault(seen["reason"], seen["error"]) is False

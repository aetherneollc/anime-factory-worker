"""docker.yml matrix skips targets whose watch_paths did not change."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "filter_docker_targets",
    ROOT / "deploy" / "filter_docker_targets.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
ZERO_SHA = _mod.ZERO_SHA
select_targets = _mod.select_targets


def _targets() -> list[dict]:
    data = json.loads((ROOT / "deploy" / "docker-targets.json").read_text(encoding="utf-8"))
    return list(data["targets"])


def test_docker_yml_is_valid_and_wires_force_all():
    text = (ROOT / ".github" / "workflows" / "docker.yml").read_text(encoding="utf-8")
    assert "\t" not in text
    assert "deploy/filter_docker_targets.py" in text
    assert "github.event_name == 'workflow_dispatch' && inputs.force_all" in text
    assert "default: false" in text
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        doc = yaml.safe_load(text)
        # YAML 1.1 parses the bare key `on` as boolean True.
        triggers = doc.get("on", doc.get(True))
        assert triggers["workflow_dispatch"]["inputs"]["force_all"]["default"] is False
        assert doc["jobs"]["matrix"]["outputs"]["targets"]


def test_h3_and_longlive_do_not_watch_python():
    by_id = {t["id"]: t for t in _targets()}
    assert "hy" not in by_id
    assert by_id["sr3"]["enabled"] is True
    assert "python/" in by_id["sr3"]["watch_paths"]
    assert by_id["longlive"]["enabled"] is False
    assert by_id["h3"]["enabled"] is True
    for frozen in ("h3", "longlive"):
        paths = by_id[frozen]["watch_paths"]
        assert not any(str(p).startswith("python") for p in paths)


def test_force_all_and_zero_before_select_every_enabled_target():
    targets = _targets()

    def _never(*_a, **_k):
        raise AssertionError("diff must not run")

    forced = select_targets(targets, before="abc", sha="def", force_all=True, diff_quiet=_never)
    assert {t["id"] for t in forced} == {"h3", "sr3"}
    zero = select_targets(targets, before=ZERO_SHA, sha="def", force_all=False, diff_quiet=_never)
    assert {t["id"] for t in zero} == {"h3", "sr3"}
    empty = select_targets(targets, before="", sha="def", force_all=False, diff_quiet=_never)
    assert {t["id"] for t in empty} == {"h3", "sr3"}


def test_quiet_diff_skips_unchanged_and_builds_changed():
    targets = _targets()

    def diff_quiet(_before: str, _sha: str, paths: list[str]) -> bool:
        # Exit 0 (unchanged) unless the sr3 dockerfile is in the path list.
        return "deploy/gpu-worker/Dockerfile.sr3" not in paths

    selected = select_targets(
        targets,
        before="a" * 40,
        sha="b" * 40,
        force_all=False,
        diff_quiet=diff_quiet,
    )
    assert [t["id"] for t in selected] == ["sr3"]


def test_diff_error_counts_as_changed():
    targets = [t for t in _targets() if t["id"] == "h3"]

    def boom(*_a, **_k):
        raise OSError("missing commit")

    selected = select_targets(
        targets,
        before="a" * 40,
        sha="b" * 40,
        force_all=False,
        diff_quiet=boom,
    )
    assert [t["id"] for t in selected] == ["h3"]

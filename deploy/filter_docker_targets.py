"""Select docker-targets whose watch_paths changed.

Used by ``.github/workflows/docker.yml`` matrix job. Same idea as the moss
and kv-dequant reuse checks: ``git diff --quiet BEFORE..SHA -- paths``.
Exit 0 means unchanged (skip). An all-zero ``before`` (first push) counts
as changed. ``FORCE_ALL=true`` (workflow_dispatch input) builds every
enabled target.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

ZERO_SHA = "0" * 40
DiffQuiet = Callable[[str, str, Sequence[str]], bool]


def git_diff_quiet(before: str, sha: str, paths: Sequence[str], *, cwd: Path | None = None) -> bool:
    """True when ``git diff --quiet before..sha -- paths`` finds no changes."""
    cmd = ["git", "diff", "--quiet", f"{before}..{sha}", "--", *list(paths)]
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode == 0:
        return True
    # 1 = differences. Anything else (missing commit) counts as changed.
    return False


def select_targets(
    targets: Sequence[dict],
    *,
    before: str,
    sha: str,
    force_all: bool,
    diff_quiet: DiffQuiet,
) -> list[dict]:
    """Return enabled targets that should build."""
    selected: list[dict] = []
    before_text = (before or "").strip()
    for target in targets:
        if not target.get("enabled", True):
            continue
        if force_all or not before_text or before_text == ZERO_SHA:
            selected.append(target)
            continue
        paths = [str(p) for p in (target.get("watch_paths") or []) if str(p).strip()]
        if not paths:
            selected.append(target)
            continue
        try:
            unchanged = diff_quiet(before_text, sha, paths)
        except Exception:  # noqa: BLE001 — a broken diff must not skip a build
            unchanged = False
        if not unchanged:
            selected.append(target)
    return selected


def _force_all() -> bool:
    return os.environ.get("FORCE_ALL", "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "deploy" / "docker-targets.json").read_text(encoding="utf-8"))
    before = os.environ.get("BEFORE") or ""
    sha = os.environ.get("GITHUB_SHA") or "HEAD"
    selected = select_targets(
        list(data.get("targets") or []),
        before=before,
        sha=sha,
        force_all=_force_all(),
        diff_quiet=lambda b, s, paths: git_diff_quiet(b, s, paths, cwd=root),
    )
    payload = json.dumps(selected)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"targets={payload}\n")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Minimal reimplementation of the BuildKit Dockerfile front-end's line splitting.

Mirrors moby/buildkit frontend/dockerfile/parser: continuation lines are joined into
one logical line first, and only then are heredoc bodies consumed up to their
terminator. Anything after a terminator therefore starts a NEW instruction, so a
``RUN ... <<'PY'`` whose ``&& cleanup`` sits below the ``PY`` line is a parse error
rather than a shell continuation. ``docker buildx build --check`` catches this too,
but it needs a daemon; this keeps the check in the unit suite.
"""

from __future__ import annotations

import re
from pathlib import Path

# parser.go: reHeredoc = `^(\d*)<<(-?)\s*([^<]*)$`
RE_HEREDOC = re.compile(r"^(\d*)<<(-?)\s*([^<]*)$")
INSTRUCTIONS = frozenset(
    {
        "ADD",
        "ARG",
        "CMD",
        "COPY",
        "ENTRYPOINT",
        "ENV",
        "EXPOSE",
        "FROM",
        "HEALTHCHECK",
        "LABEL",
        "MAINTAINER",
        "ONBUILD",
        "RUN",
        "SHELL",
        "STOPSIGNAL",
        "USER",
        "VOLUME",
        "WORKDIR",
    }
)
# parser.go heredocDirectives
HEREDOC_DIRECTIVES = frozenset({"ADD", "COPY", "RUN"})


def parse_dockerfile(path: Path) -> tuple[list[str], list[str]]:
    """Return (instruction names, errors)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    index = 0
    found: list[str] = []
    errors: list[str] = []
    while index < len(lines):
        start = index + 1
        stripped = lines[index].strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        joined = stripped
        while joined.endswith("\\"):
            joined = joined[:-1]
            if index >= len(lines):
                break
            nxt = lines[index]
            index += 1
            if nxt.strip().startswith("#"):
                continue
            joined += nxt.strip()
        name = joined.split(maxsplit=1)[0].upper()
        if name not in INSTRUCTIONS:
            errors.append(f"line {start}: unknown instruction {name!r} in {joined[:60]!r}")
            continue
        found.append(name)
        if name not in HEREDOC_DIRECTIVES or "<<" not in joined:
            continue
        for word in joined.split():
            match = RE_HEREDOC.match(word)
            if not match:
                continue
            terminator = match.group(3).strip().strip("'\"")
            chomp = match.group(2) == "-"
            terminated = False
            while index < len(lines):
                body = lines[index]
                index += 1
                candidate = body.lstrip("\t") if chomp else body
                if candidate == terminator:
                    terminated = True
                    break
            if not terminated:
                errors.append(f"line {start}: unterminated heredoc {terminator!r}")
    return found, errors

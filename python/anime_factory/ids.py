"""story_id = <slug>-<6 hex chars>."""

from __future__ import annotations

import hashlib
import re
import secrets
import unicodedata


_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """ASCII slug; CJK titles fall back to a stable hex stem (pinyin is optional)."""
    normalized = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    slug = _NON_SLUG.sub("-", normalized.lower()).strip("-")
    if slug:
        return slug[:48]
    digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:12]
    return f"story-{digest}"


def new_story_id(title: str, salt: str | None = None) -> str:
    slug = slugify(title)
    token = salt if salt is not None else secrets.token_hex(3)
    if len(token) != 6:
        token = hashlib.sha256(token.encode("utf-8")).hexdigest()[:6]
    return f"{slug}-{token}"

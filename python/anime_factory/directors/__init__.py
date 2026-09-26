"""Shooting grammars. The planner does not select LongLive or H3."""

from __future__ import annotations

from typing import Any, Sequence

from anime_factory.models import H3_MAX_SECONDS, TARGET_EPISODE_SECONDS, normalize_story_kind


def board_from_script(
    script: dict,
    *,
    episode_code: str,
    langs: Sequence[str],
    target_seconds: float = TARGET_EPISODE_SECONDS,
    shot_seconds: float = H3_MAX_SECONDS,
    kind: str | None = "series",
    video_backend: str | None = None,
) -> list[dict[str, Any]]:
    # H3 and LongLive are not selectable here. The argument is ignored.
    del video_backend
    resolved = normalize_story_kind(kind)
    if resolved == "film":
        from anime_factory.directors.film import board_film

        return board_film(
            script,
            episode_code=episode_code,
            langs=langs,
            target_seconds=target_seconds,
            shot_seconds=shot_seconds,
        )
    if resolved == "short":
        from anime_factory.directors.short import board_short

        return board_short(
            script,
            episode_code=episode_code,
            langs=langs,
            target_seconds=target_seconds,
            shot_seconds=shot_seconds,
        )
    from anime_factory.directors.series import board_series

    return board_series(
        script,
        episode_code=episode_code,
        langs=langs,
        target_seconds=target_seconds,
        shot_seconds=shot_seconds,
    )

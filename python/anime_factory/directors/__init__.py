"""Three shooting grammars. H3 only consumes segments after persist_board splits shots."""

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
        from anime_factory.video_backend import normalize_video_backend

        if normalize_video_backend(video_backend) == "longlive":
            from anime_factory.directors.longlive import board_longlive

            return board_longlive(
                script,
                episode_code=episode_code,
                langs=langs,
                target_seconds=target_seconds,
                shot_seconds=shot_seconds,
            )
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

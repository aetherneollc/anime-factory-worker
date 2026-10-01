"""Shooting grammars. The planner selects MiniMax H3 (about 8 seconds).

LongLive and SkyReels are refused by ``normalize_video_backend``. Shot length
is capped at H3's native maximum.
"""

from __future__ import annotations

from typing import Any, Sequence

from anime_factory.models import H3_MAX_SECONDS, TARGET_EPISODE_SECONDS, normalize_story_kind
from anime_factory.video_backend import (
    max_seconds_for_backend,
    normalize_video_backend,
    select_video_backend,
)


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
    chosen = (
        normalize_video_backend(video_backend)
        if video_backend is not None and str(video_backend).strip()
        else select_video_backend()
    )
    cap = float(max_seconds_for_backend(chosen))
    seconds = min(float(shot_seconds), cap)
    resolved = normalize_story_kind(kind)
    if resolved == "film":
        from anime_factory.directors.film import board_film

        shots = board_film(
            script,
            episode_code=episode_code,
            langs=langs,
            target_seconds=target_seconds,
            shot_seconds=seconds,
        )
    elif resolved == "short":
        from anime_factory.directors.short import board_short

        shots = board_short(
            script,
            episode_code=episode_code,
            langs=langs,
            target_seconds=target_seconds,
            shot_seconds=seconds,
        )
    else:
        from anime_factory.directors.series import board_series

        shots = board_series(
            script,
            episode_code=episode_code,
            langs=langs,
            target_seconds=target_seconds,
            shot_seconds=seconds,
        )
    for shot in shots:
        shot["video_backend"] = chosen
    return shots

"""H3 Comfy session helpers: selective warm loads, prep overlap, load accounting."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from gpu_worker.h3 import FL2VA_UNET, REF2VA_UNET, select_mode

ProgressCallback = Callable[[str], None]


def unet_name_for_mode(mode: str) -> str:
    return REF2VA_UNET if str(mode or "").strip() == "ref2va" else FL2VA_UNET


def modes_for_shots(shots: list[dict]) -> set[str]:
    return {select_mode(shot) for shot in shots or []}


def model_load_count_for_modes(modes: set[str] | None) -> int:
    """Distinct H3 DiT loads the board may require (TE/VAE stay cached in Comfy)."""
    if not modes:
        return 0
    return len({unet_name_for_mode(mode) for mode in modes})


class H3SessionTracker:
    """Track which H3 DiT weights were touched this lease (one load per UNet file)."""

    def __init__(self) -> None:
        self._loaded_unets: set[str] = set()

    def note_mode(self, mode: str) -> None:
        self._loaded_unets.add(unet_name_for_mode(mode))

    @property
    def model_load_count(self) -> int:
        return len(self._loaded_unets)

    @property
    def loaded_unets(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaded_unets))


class H3PrepPool:
    """Overlap deterministic Comfy input/ staging with the next GPU wait when safe."""

    def __init__(self, stage_fn: Callable[[dict], dict]) -> None:
        self._stage_fn = stage_fn
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-prep")
        self._future: Future[dict] | None = None

    def prime(self, shot: dict) -> dict:
        if self._future is not None:
            staged = self._future.result()
            self._future = None
            return staged
        return self._stage_fn(shot)

    def schedule(self, shot: dict | None) -> None:
        if self._future is not None:
            self._future.result()
            self._future = None
        if shot is None:
            return
        self._future = self._executor.submit(self._stage_fn, dict(shot))

    def close(self) -> None:
        if self._future is not None:
            self._future.result()
            self._future = None
        self._executor.shutdown(wait=False)


def can_prefetch_staging(shot: dict) -> bool:
    """Chain tails need the previous last frame; only prefetch independent heads."""
    return int(shot.get("chain_index") or 0) <= 0 and not shot.get("chain_source_last_frame")


def instrument_h3_session(shots: list[dict], *, warmed: bool = False) -> dict[str, Any]:
    modes = modes_for_shots(shots)
    payload = {
        "modes": sorted(modes),
        "expected_model_load_count": model_load_count_for_modes(modes),
        "model_load_count": model_load_count_for_modes(modes) if warmed else 0,
        "warmed": bool(warmed),
    }
    return payload

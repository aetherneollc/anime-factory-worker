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


def _continues_chain(prev: dict, shot: dict) -> bool:
    chain_id = str(shot.get("chain_id") or "").strip()
    prev_chain = str(prev.get("chain_id") or "").strip()
    if not chain_id or prev_chain != chain_id:
        return False
    return int(shot.get("chain_index") or 0) == int(prev.get("chain_index") or 0) + 1


def plan_h3_mode_groups(shots: list[dict]) -> list[dict[str, Any]]:
    """Consecutive compose-order groups sharing one DiT; chains stay atomic units.

    Each group is ``{"mode": str, "indices": [int, ...]}``. Reordering never crosses
    a chain boundary or the original board index order.
    """
    groups: list[dict[str, Any]] = []
    for index, shot in enumerate(shots or []):
        mode = select_mode(shot)
        if not groups:
            groups.append({"mode": mode, "indices": [index]})
            continue
        current = groups[-1]
        prev = shots[current["indices"][-1]]
        prev_chain = str(prev.get("chain_id") or "").strip()
        cur_chain = str(shot.get("chain_id") or "").strip()
        same_mode = current["mode"] == mode
        chain_ok = (
            not cur_chain
            or not prev_chain
            or _continues_chain(prev, shot)
            or (cur_chain != prev_chain and int(shot.get("chain_index") or 0) == 0)
        )
        if same_mode and chain_ok:
            current["indices"].append(index)
        else:
            groups.append({"mode": mode, "indices": [index]})
    return groups


def flatten_h3_group_indices(groups: list[dict[str, Any]]) -> list[int]:
    """Return compose-order indices for all planned groups."""
    out: list[int] = []
    for group in groups or []:
        out.extend(int(i) for i in group.get("indices") or [])
    return out


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
    groups = plan_h3_mode_groups(shots)
    payload = {
        "modes": sorted(modes),
        "mode_groups": [
            {"mode": group["mode"], "shot_count": len(group["indices"]), "indices": group["indices"]}
            for group in groups
        ],
        "expected_model_load_count": model_load_count_for_modes(modes),
        "model_load_count": model_load_count_for_modes(modes) if warmed else 0,
        "warmed": bool(warmed),
    }
    return payload

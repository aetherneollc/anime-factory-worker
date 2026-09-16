"""M9 stub: numpy brute-force search. Production gates MUST NOT call this."""

from __future__ import annotations

from anime_factory.instrument import Counters
from anime_factory.models import EMBEDDING_DIMENSION

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional
    np = None  # type: ignore


def brute_force_search(query: bytes | list[float], rows: list[tuple[str, bytes]], top_k: int = 20):
    """Dense scan over memory.embedding BLOBs. Unused by continuity gates."""
    Counters.embedding_search += 1
    Counters.embedding_api += 1
    if np is None:
        return []
    q = np.frombuffer(query, dtype="<f4") if isinstance(query, (bytes, bytearray)) else np.asarray(query, dtype="<f4")
    scored = []
    for mem_id, blob in rows:
        if not blob:
            continue
        vec = np.frombuffer(blob, dtype="<f4")
        if vec.shape[0] != q.shape[0]:
            continue
        score = float(vec @ q)
        scored.append((score, mem_id))
    scored.sort(reverse=True)
    return scored[:top_k]


def empty_embedding() -> None:
    """Nullable BLOB — M0–M8 leave this unset."""
    return None


assert EMBEDDING_DIMENSION == 1024

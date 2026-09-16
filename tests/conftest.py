import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))


@pytest.fixture(autouse=True)
def _clear_video_backend_lock(monkeypatch):
    monkeypatch.delenv("AF_VIDEO_BACKEND_LOCKED", raising=False)
    monkeypatch.delenv("ANIME_FACTORY_GPU_STILLS", raising=False)


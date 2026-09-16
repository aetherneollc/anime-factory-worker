import pytest


@pytest.fixture(autouse=True)
def _clear_video_backend_lock(monkeypatch):
    monkeypatch.delenv("AF_VIDEO_BACKEND_LOCKED", raising=False)

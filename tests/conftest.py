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
    # Host shells sometimes export IMAGE_BACKEND / VIDEO_BACKEND; unit tests must not inherit them.
    monkeypatch.delenv("IMAGE_BACKEND", raising=False)
    monkeypatch.delenv("VIDEO_BACKEND", raising=False)
    monkeypatch.delenv("AF_VIDEO_BACKEND", raising=False)


@pytest.fixture(autouse=True)
def _unit_tests_use_kolors_characters(request, monkeypatch):
    """Production default is CHARACTER_STILL_BACKEND=hunyuan; unit tests inject Kolors/GPU fakes.

    Mark a test with ``@pytest.mark.qwen_character_default`` to assert the real default
    (or other hosted character backends).
    """
    if request.node.get_closest_marker("qwen_character_default"):
        monkeypatch.delenv("CHARACTER_STILL_BACKEND", raising=False)
        return
    monkeypatch.setenv("CHARACTER_STILL_BACKEND", "kolors")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "qwen_character_default: exercise production CHARACTER_STILL_BACKEND (hunyuan/qwen)",
    )


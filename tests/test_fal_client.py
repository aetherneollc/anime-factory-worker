"""fal.ai queue client: polling, upload cache, budget, dry-run."""

from __future__ import annotations

import json

import pytest

from anime_factory.design import png_dimensions
from anime_factory.fal_client import (
    FalApiError,
    FalBudgetExceeded,
    FalClient,
    clear_upload_cache,
    estimate_image_usd,
    reset_episode_budget,
)
from anime_factory.fal_models import (
    DEFAULT_FAL_IMAGE_T2I_MODEL,
    build_r2v_payload,
    generate_fal_image_t2i,
    generate_fal_video,
    resolve_fal_value,
)


@pytest.fixture(autouse=True)
def _reset_fal_state(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_DRY_RUN", raising=False)
    monkeypatch.delenv("FAL_UPLOAD_MODE", raising=False)
    monkeypatch.delenv("FAL_OVERRIDE_CONTROL_PLANE", raising=False)
    monkeypatch.delenv("FAL_HTTP_RETRIES", raising=False)
    monkeypatch.setenv("FAL_RETRY_BASE_S", "0")
    monkeypatch.setenv("FAL_POLL_INTERVAL_S", "0")
    clear_upload_cache()
    reset_episode_budget(20.0)
    yield
    clear_upload_cache()
    reset_episode_budget(20.0)


def test_queue_polls_exact_status_url_then_fetches_response():
    calls: list[tuple[str, str]] = []
    status_hits = {"n": 0}
    status_url = "https://queue.fal.run/owner/app/requests/abc/status"
    response_url = "https://queue.fal.run/owner/app/requests/abc"

    def transport(method, url, body, headers, timeout):
        calls.append((method, url))
        assert headers.get("Authorization") == "Key test-key"
        if method == "POST":
            return 200, json.dumps(
                {
                    "request_id": "abc",
                    "status_url": status_url,
                    "response_url": response_url,
                    "cancel_url": response_url + "/cancel",
                }
            ).encode()
        if url == status_url:
            status_hits["n"] += 1
            status = "IN_QUEUE" if status_hits["n"] == 1 else "COMPLETED"
            return 200, json.dumps({"status": status}).encode()
        if url == response_url:
            return 200, json.dumps({"images": [{"url": "https://cdn.example/out.png"}]}).encode()
        if url == "https://cdn.example/out.png":
            return 200, b"\x89PNG\r\n\x1a\npayload"
        raise AssertionError(f"unexpected {method} {url}")

    client = FalClient("test-key", transport=transport)
    result = client.submit("fal-ai/minimax/hailuo-02/standard/image-to-video", {"prompt": "pan"})
    assert result["images"][0]["url"].endswith("out.png")
    assert [url for _method, url in calls].count(status_url) == 2
    assert response_url in [url for _method, url in calls]
    assert all("/status/status" not in url for _method, url in calls)


def test_status_405_is_not_retried_as_a_rebuilt_url():
    calls: list[str] = []

    def transport(method, url, body, headers, timeout):
        calls.append(url)
        if method == "POST":
            return 200, json.dumps(
                {
                    "request_id": "abc",
                    "status_url": "https://queue.fal.run/owner/app/requests/abc/status",
                    "response_url": "https://queue.fal.run/owner/app/requests/abc",
                }
            ).encode()
        return 405, b"method not allowed"

    client = FalClient("test-key", transport=transport)
    with pytest.raises(FalApiError, match="405"):
        client.submit("owner/app", {"prompt": "x"})
    assert calls == [
        "https://queue.fal.run/owner/app",
        "https://queue.fal.run/owner/app/requests/abc/status",
    ]


def test_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("FAL_HTTP_RETRIES", "3")
    hits = {"n": 0}

    def transport(method, url, body, headers, timeout):
        hits["n"] += 1
        if hits["n"] < 3:
            return 429, b"slow"
        return 200, b"{}"

    client = FalClient("test-key", transport=transport)
    assert client.request("GET", "https://queue.fal.run/ping") == b"{}"
    assert hits["n"] == 3


def test_upload_cache_is_sha256_and_falls_back_to_data_uri(monkeypatch):
    monkeypatch.setenv("FAL_HTTP_RETRIES", "0")
    initiates = {"n": 0}

    def transport(method, url, body, headers, timeout):
        if "storage/upload/initiate" in url:
            initiates["n"] += 1
            if body and b"master.png" in body:
                return 200, json.dumps(
                    {"upload_url": "https://upload.example/put", "file_url": "https://cdn.example/master.png"}
                ).encode()
            return 500, b"down"
        if url == "https://upload.example/put":
            return 200, b""
        raise AssertionError(url)

    client = FalClient("test-key", transport=transport)
    blob = b"master-png-bytes"
    first = client.upload_bytes(blob, filename="master.png")
    second = client.upload_bytes(blob, filename="plate_base.png")
    assert first == second == "https://cdn.example/master.png"
    assert initiates["n"] == 1
    other = client.upload_bytes(b"different-bytes", filename="ref.png")
    assert other.startswith("data:image/png;base64,")
    assert initiates["n"] == 2


def test_budget_circuit_breaker(monkeypatch):
    monkeypatch.setenv("FAL_PRICE_IMAGE_USD", "0.03")
    tracker = reset_episode_budget(0.05)
    tracker.charge(estimate_image_usd())
    with pytest.raises(FalBudgetExceeded):
        tracker.charge(estimate_image_usd())
    assert tracker.spent == pytest.approx(0.03)


def test_dry_run_makes_no_network_requests(tmp_path):
    def transport(*_a, **_k):
        raise AssertionError("network")

    client = FalClient("test-key", transport=transport, dry_run=True)
    assert client.upload_bytes(b"abc").startswith("data:image/png;base64,")
    png = generate_fal_image_t2i("sheet", 320, 180, dry_run=True)
    assert png_dimensions(png) == (320, 180)
    dest = generate_fal_video("i2v", "pan", duration_s=1.0, out_path=tmp_path / "clip.mp4", dry_run=True)
    blob = dest.read_bytes()
    assert b"ftyp" in blob[:32]


def test_kling_r2v_elements_require_an_angle_and_omit_empty_lists():
    payload = build_r2v_payload(
        model="fal-ai/kling-video/o1/reference-to-video",
        prompt="<<<image_1>>> stands",
        char_urls=["https://cdn.example/master.png"],
        scene_urls=[],
        duration_s=6,
    )
    assert payload["elements"] == [
        {
            "frontal_image_url": "https://cdn.example/master.png",
            "reference_image_urls": ["https://cdn.example/master.png"],
        }
    ]
    assert "image_urls" not in payload
    for element in payload["elements"]:
        assert element["reference_image_urls"]
    empty = build_r2v_payload(
        model="fal-ai/kling-video/o1/reference-to-video",
        prompt="plate",
        char_urls=[],
        scene_urls=["https://cdn.example/plate.png"],
    )
    assert "elements" not in empty
    assert empty["image_urls"] == ["https://cdn.example/plate.png"]


def test_fal_config_overrides_defaults_unless_env_flag(monkeypatch):
    monkeypatch.setenv("FAL_IMAGE_T2I_MODEL", "env/model")
    assert resolve_fal_value("image_t2i_model", fal_config={"image_t2i_model": "plane/model"}) == "plane/model"
    monkeypatch.setenv("FAL_OVERRIDE_CONTROL_PLANE", "1")
    assert resolve_fal_value("image_t2i_model", fal_config={"image_t2i_model": "plane/model"}) == "env/model"
    monkeypatch.delenv("FAL_IMAGE_T2I_MODEL", raising=False)
    monkeypatch.delenv("FAL_OVERRIDE_CONTROL_PLANE", raising=False)
    assert resolve_fal_value("image_t2i_model", fal_config={}) == DEFAULT_FAL_IMAGE_T2I_MODEL

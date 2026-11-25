"""Stdlib client for the fal.ai queue API.

No CUDA, torch, or third-party HTTP library. ``FAL_KEY`` is read from the
environment (or passed in by tests). Never embed a real key in source.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

QUEUE_BASE = "https://queue.fal.run"
STORAGE_INITIATE = "https://rest.alpha.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3"
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_FAIL_STATUS = frozenset({"FAILED", "ERROR", "CANCELLED", "CANCELED"})

# In-process content cache. A character master or scene plate is uploaded once per session.
_UPLOAD_CACHE: dict[str, str] = {}

Transport = Callable[[str, str, bytes | None, dict[str, str], float], tuple[int, bytes]]


class FalApiError(RuntimeError):
    """fal.ai HTTP, queue, or result failure."""


class FalBudgetExceeded(FalApiError):
    """Estimated episode spend would pass FAL_MAX_EPISODE_USD."""


class FalBudgetTracker:
    """Per-episode spend cap. Default cap is ``FAL_MAX_EPISODE_USD`` (20)."""

    def __init__(self, cap: float | None = None) -> None:
        if cap is None:
            raw = (os.environ.get("FAL_MAX_EPISODE_USD") or "20").strip()
            try:
                cap = float(raw)
            except ValueError:
                cap = 20.0
        self.cap = float(cap)
        self.spent = 0.0

    def charge(self, amount: float) -> float:
        cost = float(amount)
        if cost < 0:
            raise FalApiError(f"fal budget charge must be >= 0, got {cost}")
        if self.spent + cost > self.cap + 1e-9:
            raise FalBudgetExceeded(
                f"fal episode budget {self.cap:.2f} USD exceeded "
                f"(spent {self.spent:.4f}, next {cost:.4f})"
            )
        self.spent += cost
        return self.spent

    def reset(self, cap: float | None = None) -> None:
        if cap is not None:
            self.cap = float(cap)
        self.spent = 0.0


_BUDGET = FalBudgetTracker()


def episode_budget() -> FalBudgetTracker:
    return _BUDGET


def reset_episode_budget(cap: float | None = None) -> FalBudgetTracker:
    global _BUDGET
    _BUDGET = FalBudgetTracker(cap)
    return _BUDGET


def clear_upload_cache() -> None:
    _UPLOAD_CACHE.clear()


def fal_dry_run(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return str(os.environ.get("FAL_DRY_RUN") or "").strip().lower() in {"1", "true", "yes", "on"}


def estimate_image_usd() -> float:
    raw = (os.environ.get("FAL_PRICE_IMAGE_USD") or "0.03").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.03


def estimate_video_usd(duration_s: float) -> float:
    raw = (os.environ.get("FAL_PRICE_VIDEO_PER_SEC_USD") or "0.05").strip()
    try:
        per = float(raw)
    except ValueError:
        per = 0.05
    return max(0.0, per) * max(float(duration_s or 0.0), 1.0)


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _data_uri(blob: bytes, content_type: str) -> str:
    encoded = base64.b64encode(blob).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def decode_data_uri(url: str) -> bytes:
    _header, encoded = str(url).split(",", 1)
    return base64.b64decode(encoded)


def _needs_auth(url: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return host.endswith("fal.run") or host.endswith("fal.ai")


def _minimal_mp4() -> bytes:
    """ISO BMFF shell with an ftyp box. Used when ffmpeg is not installed."""
    body = b"isom" + b"\x00\x00\x00\x01" + b"isomiso2mp41"
    ftyp = struct.pack(">I", 8 + len(body)) + b"ftyp" + body
    mdat = struct.pack(">I", 8) + b"mdat"
    return ftyp + mdat


def synthetic_video_bytes(
    duration_s: float = 1.0,
    width: int = 1280,
    height: int = 720,
) -> bytes:
    """A playable mp4 when ffmpeg is present, otherwise a minimal ftyp shell."""
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return _minimal_mp4()
    w = max(16, int(width) // 2 * 2)
    h = max(16, int(height) // 2 * 2)
    dur = max(0.2, float(duration_s or 1.0))
    fd, name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    path = Path(name)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={w}x{h}:r=24:d={dur:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "24",
        str(path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        data = path.read_bytes()
        if len(data) >= 32 and b"ftyp" in data[:32]:
            return data
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        path.unlink(missing_ok=True)
    return _minimal_mp4()


def synthetic_image_bytes(width: int, height: int, *, tag: str) -> bytes:
    from anime_factory.design import synthetic_still_png

    return synthetic_still_png(int(width), int(height), tag=tag, placeholder=False)


def media_url_from_result(payload: Any) -> str:
    """First image or video URL in a fal result document."""
    if not isinstance(payload, dict):
        raise FalApiError(f"fal result is not an object: {type(payload).__name__}")
    for key in ("images", "videos"):
        items = payload.get(key)
        if isinstance(items, list) and items:
            first = items[0]
            if isinstance(first, str) and first:
                return first
            if isinstance(first, dict):
                url = first.get("url") or first.get("uri")
                if url:
                    return str(url)
    for key in ("image", "video", "video_url", "image_url"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            url = value.get("url") or value.get("uri")
            if url:
                return str(url)
    raise FalApiError(f"fal result has no media url: keys={list(payload)[:12]}")


class FalClient:
    """Queue submit + poll against the exact ``status_url`` fal returns."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: Transport | None = None,
        dry_run: bool | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("FAL_KEY") or ""
        self.api_key = str(api_key)
        self.transport = transport
        self.dry_run = fal_dry_run(dry_run)

    def _backoff(self, attempt: int) -> None:
        base = _env_float("FAL_RETRY_BASE_S", 0.4)
        if base <= 0:
            return
        time.sleep(min(8.0, base * (2 ** attempt)))

    def _headers(self, url: str, *, json_body: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.api_key and _needs_auth(url):
            headers["Authorization"] = f"Key {self.api_key}"
        return headers

    def _once(self, method: str, url: str, body: bytes | None, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
        if self.transport is not None:
            return self.transport(method, url, body, headers, timeout)
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", None) or resp.getcode() or 200)
                return status, resp.read()
        except urllib.error.HTTPError as exc:
            payload = exc.read() if exc.fp is not None else b""
            return int(exc.code), payload

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        json_body: bool = False,
    ) -> bytes:
        hdrs = self._headers(url, json_body=json_body)
        if headers:
            hdrs.update(headers)
        attempts = max(0, int(_env_float("FAL_HTTP_RETRIES", 4)))
        last_status = 0
        last_body = b""
        for attempt in range(attempts + 1):
            try:
                status, data = self._once(method, url, body, hdrs, timeout)
            except FalApiError:
                raise
            except Exception as exc:  # noqa: BLE001 — transient socket errors retry
                if attempt >= attempts:
                    raise FalApiError(f"fal {method} {url} failed: {exc}") from exc
                self._backoff(attempt)
                continue
            if status in _RETRY_STATUS and attempt < attempts:
                last_status, last_body = status, data
                self._backoff(attempt)
                continue
            if status >= 400:
                snippet = data[:300]
                raise FalApiError(f"HTTP {status} {method} {url}: {snippet!r}")
            return data
        raise FalApiError(f"HTTP {last_status} {method} {url}: {last_body[:300]!r}")

    def _json(self, method: str, url: str, payload: dict | None = None, *, timeout: float = 60.0) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        raw = self.request(method, url, body, timeout=timeout, json_body=payload is not None)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FalApiError(f"fal {method} {url} returned non-JSON") from exc
        if not isinstance(parsed, dict):
            raise FalApiError(f"fal {method} {url} returned {type(parsed).__name__}")
        return parsed

    def reserve(self, amount: float) -> None:
        if self.dry_run:
            return
        episode_budget().charge(amount)

    def upload_bytes(self, blob: bytes, content_type: str = "image/png", filename: str = "ref.png") -> str:
        """Upload once per SHA-256. REST storage, else a data URI models accept."""
        data = bytes(blob or b"")
        digest = hashlib.sha256(data).hexdigest()
        cached = _UPLOAD_CACHE.get(digest)
        if cached:
            return cached
        mode = str(os.environ.get("FAL_UPLOAD_MODE") or "").strip().lower()
        if self.dry_run or mode == "data_uri":
            url = _data_uri(data, content_type)
        else:
            try:
                url = self._storage_put(data, content_type, filename)
            except Exception:  # noqa: BLE001 — data URIs are a supported fal input
                url = _data_uri(data, content_type)
        _UPLOAD_CACHE[digest] = url
        return url

    def _storage_put(self, blob: bytes, content_type: str, filename: str) -> str:
        initiated = self._json(
            "POST",
            STORAGE_INITIATE,
            {"file_name": filename, "content_type": content_type},
            timeout=60.0,
        )
        upload_url = str(initiated.get("upload_url") or "")
        file_url = str(initiated.get("file_url") or "")
        if not upload_url or not file_url:
            raise FalApiError(f"fal storage initiate missing urls: keys={list(initiated)[:8]}")
        self.request(
            "PUT",
            upload_url,
            blob,
            headers={"Content-Type": content_type},
            timeout=120.0,
        )
        return file_url

    def submit(self, endpoint_id: str, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """POST the queue endpoint and poll the returned status_url. Never rebuild it."""
        endpoint = str(endpoint_id or "").strip().strip("/")
        if not endpoint:
            raise FalApiError("fal endpoint id is empty")
        if not self.dry_run and not self.api_key and self.transport is None:
            raise FalApiError("FAL_KEY is not set")
        submitted = self._json("POST", f"{QUEUE_BASE}/{endpoint}", payload, timeout=60.0)
        request_id = str(submitted.get("request_id") or "")
        status_url = str(submitted.get("status_url") or "")
        response_url = str(submitted.get("response_url") or "")
        if not status_url or not response_url:
            raise FalApiError(
                f"fal queue response missing status_url/response_url for {endpoint}: keys={list(submitted)[:8]}"
            )
        deadline = time.time() + float(timeout if timeout is not None else _env_float("FAL_QUEUE_TIMEOUT_S", 600.0))
        interval = _env_float("FAL_POLL_INTERVAL_S", 1.0)
        while True:
            status_doc = self._json("GET", status_url, timeout=60.0)
            status = str(status_doc.get("status") or "").upper()
            if status == "COMPLETED":
                break
            if status in _FAIL_STATUS:
                detail = status_doc.get("error") or status_doc.get("detail") or status
                raise FalApiError(f"fal request {request_id or endpoint} {status}: {detail}")
            if status and status not in {"IN_QUEUE", "IN_PROGRESS"}:
                raise FalApiError(f"fal request {request_id or endpoint} unexpected status {status}")
            if time.time() > deadline:
                raise FalApiError(f"fal request {request_id or endpoint} timed out at {status or 'UNKNOWN'}")
            if interval > 0:
                time.sleep(interval)
        # Always fetch the URL fal returned. Rebuilding it with a subpath is HTTP 405.
        return self._json("GET", response_url, timeout=120.0)

    def download(self, url: str) -> bytes:
        if str(url).startswith("data:"):
            return decode_data_uri(url)
        return self.request("GET", url, timeout=120.0)


def get_fal_client(*, dry_run: bool | None = None, transport: Transport | None = None) -> FalClient:
    return FalClient(transport=transport, dry_run=dry_run)

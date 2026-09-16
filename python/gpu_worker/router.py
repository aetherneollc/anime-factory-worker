"""Minimal comfy-router on :8199 → ComfyUI :8188. Bind loopback only.

`COMFYUI_MAX_CONCURRENT` is enforced here with a real semaphore. It used to be a
config field nothing read, which is why POSTMORTEM_CONCURRENT_COMFYUI_CROSS_CONTAMINATION
could repeat: concurrent `/prompt` submissions to one backend returned `success`
for every job while the pictures came back matched to the wrong prompts.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from anime_factory.models import COMFY_ROUTER_PORT
from gpu_worker.shard import PER_BACKEND_CONCURRENCY

COMFY_BACKEND = os.environ.get("COMFY_BACKENDS", "http://127.0.0.1:8188").split(",")[0].rstrip("/")
# Under the 60s client timeout in ComfyRouter.prompt, so a blocked caller gets a
# 503 it can retry instead of an opaque socket timeout.
PROMPT_WAIT_S = float(os.environ.get("COMFY_ROUTER_PROMPT_WAIT_SECONDS") or 45)
_gate_lock = threading.Lock()
_prompt_gate: threading.BoundedSemaphore | None = None


def max_concurrent() -> int:
    """Concurrent /prompt submissions allowed per router. One backend means one."""
    from anime_factory.config import load_settings

    configured = load_settings().comfyui_max_concurrent
    return max(int(configured or PER_BACKEND_CONCURRENCY), 1)


def prompt_gate() -> threading.BoundedSemaphore:
    global _prompt_gate
    with _gate_lock:
        if _prompt_gate is None:
            _prompt_gate = threading.BoundedSemaphore(max_concurrent())
        return _prompt_gate


def reset_prompt_gate() -> None:
    """Tests only: re-read COMFYUI_MAX_CONCURRENT."""
    global _prompt_gate
    with _gate_lock:
        _prompt_gate = None


def _content_type(headers: dict[str, str] | None) -> str:
    if not headers:
        return "application/json"
    for key, value in headers.items():
        if key.lower() == "content-type":
            return value
    return "application/json"


def _forward(method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, bytes, str]:
    url = COMFY_BACKEND + path
    hdrs = {"Content-Type": _content_type(headers)}
    req = Request(url, data=body, headers=hdrs, method=method)
    try:
        with urlopen(req, timeout=300) as resp:
            data = resp.read()
            ctype = resp.headers.get("Content-Type") or "application/json"
            return getattr(resp, "status", 200) or 200, data, ctype
    except HTTPError as exc:
        return exc.code, exc.read() or b"{}", "application/json"
    except (URLError, TimeoutError, OSError) as exc:
        payload = json.dumps({"error": str(exc), "backend": COMFY_BACKEND}).encode()
        return 502, payload, "application/json"


class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/healthz", "/health"}:
            self._send(200, json.dumps({"ok": True, "backends": [COMFY_BACKEND]}).encode())
            return
        if path == "/system_stats":
            status, body, ctype = _forward("GET", "/system_stats")
            if status == 200:
                try:
                    stats = json.loads(body.decode())
                except json.JSONDecodeError:
                    stats = {"raw": True}
                wrapped = {"backends": {COMFY_BACKEND: stats}, "inflight": {COMFY_BACKEND: 0}}
                self._send(200, json.dumps(wrapped).encode())
                return
            self._send(status, body, ctype)
            return
        if path.startswith("/history/") or path in {"/queue", "/object_info"}:
            status, body, ctype = _forward("GET", self.path)
            self._send(status, body, ctype)
            return
        if path == "/view":
            status, body, ctype = _forward("GET", self.path)
            if status == 200 and not ctype.startswith("application/json"):
                self._send(status, body, ctype or "application/octet-stream")
                return
            self._send(status, body, ctype)
            return
        self._send(404, json.dumps({"error": "not found"}).encode())

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_body()
        headers = {k: v for k, v in self.headers.items()}
        if path == "/prompt":
            gate = prompt_gate()
            if not gate.acquire(timeout=PROMPT_WAIT_S):
                self._send(
                    503,
                    json.dumps(
                        {
                            "error": "comfy_router_busy",
                            "max_concurrent": max_concurrent(),
                            "waited_s": PROMPT_WAIT_S,
                        }
                    ).encode(),
                )
                return
            try:
                status, data, ctype = _forward("POST", self.path, body, headers)
            finally:
                gate.release()
            self._send(status, data, ctype)
            return
        if path in {"/upload/image", "/free"}:
            # /upload/image: Comfy expects multipart; pass through as-is.
            status, data, ctype = _forward("POST", self.path, body, headers)
            self._send(status, data, ctype)
            return
        self._send(404, json.dumps({"error": "not found"}).encode())


def serve(host: str = "127.0.0.1", port: int = COMFY_ROUTER_PORT) -> ThreadingHTTPServer:
    """Bind loopback only — Comfy :8199 is not public. Tunnel dials out."""
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    httpd = ThreadingHTTPServer((host, port), RouterHandler)
    return httpd


def main() -> None:
    host = os.environ.get("COMFY_ROUTER_HOST", "127.0.0.1")
    port = int(os.environ.get("COMFY_ROUTER_PORT") or COMFY_ROUTER_PORT)
    httpd = serve(host, port)
    print(
        json.dumps(
            {
                "router": f"http://{host}:{port}",
                "backend": COMFY_BACKEND,
                "max_concurrent": max_concurrent(),
            }
        ),
        flush=True,
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()

"""comfy-router protocol on :8199, not :8188."""

from __future__ import annotations

import json
import time
import uuid
from typing import Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from anime_factory.config import load_settings


class ComfyRouter:
    def __init__(self, base_url: str | None = None, token: str = "", opener: Callable | None = None):
        settings = load_settings()
        self.base_url = (base_url or settings.comfyui_base_url).rstrip("/")
        if self.base_url.endswith(":8188"):
            raise ValueError("COMFYUI_BASE_URL must point at comfy-router :8199, not :8188")
        self.token = token
        self.opener = opener

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def upload_image(self, name: str, data: bytes, subfolder: str = "") -> dict:
        """Comfy /upload/image is multipart, not raw bytes + X-Filename."""
        boundary = f"----AF{uuid.uuid4().hex}"
        chunks = [
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
                f"Content-Type: image/png\r\n\r\n"
            ).encode(),
            data,
            b"\r\n",
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            ).encode(),
        ]
        if subfolder:
            chunks.append(
                (
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"subfolder\"\r\n\r\n"
                    f"{subfolder}\r\n"
                ).encode()
            )
        chunks.append(f"--{boundary}--\r\n".encode())
        body = b"".join(chunks)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        url = f"{self.base_url}/upload/image"
        req = Request(url, data=body, headers=headers, method="POST")
        if self.opener:
            return self.opener(req)
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read()[:400]
            raise RuntimeError(f"comfy /upload/image HTTP {exc.code}: {detail!r}") from exc

    def prompt(self, workflow: dict, client_id: str) -> str:
        payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
        req = Request(f"{self.base_url}/prompt", data=payload, headers=self._headers(), method="POST")
        try:
            if self.opener:
                data = self.opener(req)
            else:
                with urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read()[:800]
            raise RuntimeError(f"comfy /prompt HTTP {exc.code}: {detail!r}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"comfy /prompt bad body: {data!r}"[:400])
        if data.get("node_errors"):
            raise RuntimeError(f"comfy node_errors: {data['node_errors']}"[:800])
        pid = data.get("prompt_id")
        if not pid:
            raise RuntimeError(f"comfy /prompt missing prompt_id: {data}"[:400])
        return pid

    def history(self, prompt_id: str) -> dict:
        req = Request(f"{self.base_url}/history/{prompt_id}", headers=self._headers(), method="GET")
        if self.opener:
            return self.opener(req)
        with urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if resp.status == 404 or (isinstance(body, dict) and body.get("error") == "404"):
            raise RuntimeError("prompt_id lost after router restart; resubmit")
        return body

    def queue(self) -> dict:
        req = Request(f"{self.base_url}/queue", headers=self._headers(), method="GET")
        if self.opener:
            body = self.opener(req)
        else:
            with urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise RuntimeError(f"comfy /queue bad body: {body!r}"[:400])
        return body

    def inflight_count(self) -> int:
        """Count running and queued prompts; malformed/failed probes raise for fail-closed idle."""
        body = self.queue()
        total = 0
        for key in ("queue_running", "queue_pending"):
            value = body.get(key) or []
            if not isinstance(value, list):
                raise RuntimeError(f"comfy /queue {key} is not a list")
            total += len(value)
        return total

    def free(self, unload_models: bool = True, free_memory: bool = True) -> dict:
        """Ask Comfy to drop the loaded still checkpoint before H3 sampling on 32GB."""
        payload = json.dumps({"unload_models": unload_models, "free_memory": free_memory}).encode()
        req = Request(f"{self.base_url}/free", data=payload, headers=self._headers(), method="POST")
        if self.opener:
            data = self.opener(req)
            return data if isinstance(data, dict) else {"ok": True}
        try:
            with urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except HTTPError as exc:
            detail = exc.read()[:400]
            raise RuntimeError(f"comfy /free HTTP {exc.code}: {detail!r}") from exc
        if not raw.strip():
            return {"ok": True}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"ok": True, "raw": raw[:200]}
        return parsed if isinstance(parsed, dict) else {"ok": True}

    def view(self, filename: str, subfolder: str = "", type_: str = "output") -> bytes:
        q = f"filename={filename}&subfolder={subfolder}&type={type_}"
        req = Request(f"{self.base_url}/view?{q}", headers=self._headers(), method="GET")
        if self.opener:
            data = self.opener(req)
            return data if isinstance(data, (bytes, bytearray)) else b""
        with urlopen(req, timeout=120) as resp:
            return resp.read()


def unwrap_history(history: dict | None, prompt_id: str | None = None) -> dict:
    if isinstance(history, dict) and prompt_id and prompt_id in history:
        body = history[prompt_id]
        return body if isinstance(body, dict) else {}
    return history if isinstance(history, dict) else {}


def extract_execution_error(history: dict | None, prompt_id: str | None = None) -> dict[str, str] | None:
    """Pull exception type/message/node from Comfy history. Never dump full status."""
    body = unwrap_history(history, prompt_id)
    status = body.get("status") if isinstance(body.get("status"), dict) else {}
    messages = status.get("messages") or body.get("messages") or []
    for item in messages:
        kind, payload = None, None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            kind, payload = item[0], item[1]
        elif isinstance(item, dict):
            kind, payload = item.get("type"), item
        if str(kind) != "execution_error" or not isinstance(payload, dict):
            continue
        return {
            "exception_type": str(payload.get("exception_type") or "Error"),
            "exception_message": str(payload.get("exception_message") or payload.get("message") or ""),
            "node_id": str(payload.get("node_id") or ""),
            "node_type": str(payload.get("node_type") or ""),
        }
    return None


def format_execution_error(err: dict, shot_id: str = "") -> str:
    return (
        f"h3 {shot_id} execution_error: {err.get('exception_type')}: "
        f"{err.get('exception_message')} node={err.get('node_type')}:{err.get('node_id')}"
    )

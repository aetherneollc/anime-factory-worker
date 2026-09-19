"""Vast.ai v0 client. Dry-run default. Never DELETE from list endpoint. Never live-lease in tests."""

from __future__ import annotations

import json
import re
from typing import Any, Callable
from urllib.request import Request, urlopen

from anime_factory.config import load_settings
from anime_factory.instrument import Counters
from anime_factory.models import VAST_API_BASE

VAST_V1_API_BASE = "https://console.vast.ai/api/v1"
from gpu_worker.images import AGENT_IMAGE
from gpu_worker.offers import search_payload

HttpOpener = Callable[[Request], dict]

DEFAULT_LEASE_IMAGE = f"{AGENT_IMAGE}:main"
DEFAULT_DISK_GB = 200


class VastSafetyError(RuntimeError):
    pass


def quote_vast_env_value(value: str) -> str:
    """Match cf-control quoteVastEnvValue — Vast parses env as a docker -e shell string."""
    if not re.search(r"""[\s'"$`\\]""", value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


class VastClient:
    def __init__(self, api_key: str = "", opener: HttpOpener | None = None, dry_run: bool | None = None):
        self.api_key = api_key
        self.opener = opener
        settings = load_settings()
        # Explicit dry_run=False (one-shot produce / VAST_DRY_RUN=0) wins.
        # ANIME_FACTORY_LIVE_VAST=1 means this process may DELETE even if the
        # image baked VAST_DRY_RUN=1.
        if dry_run is None:
            self.dry_run = False if settings.live_vast else settings.vast_dry_run
        else:
            self.dry_run = dry_run
        self.calls: list[tuple[str, str]] = []

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = path if path.startswith("http") else f"{VAST_API_BASE}{path}"
        self.calls.append((method, url))
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = Request(url, data=body, headers=self._headers(), method=method)
        if self.opener:
            return self.opener(req)
        if self.dry_run:
            return {"dry_run": True, "url": url, "method": method, "payload": payload}
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_instances_v1(self, limit: int = 128) -> dict:
        """GET /api/v1/instances — list-only; safe for canary baseline/confirm."""
        bounded = max(1, min(int(limit), 512))
        url = f"{VAST_V1_API_BASE}/instances?limit={bounded}"
        self.calls.append(("GET", url))
        req = Request(url, headers=self._headers(), method="GET")
        if self.opener:
            return self.opener(req)
        if self.dry_run:
            return {"dry_run": True, "instances": []}
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def search_offers(self, query: str | None = None, filters: dict | None = None) -> dict:
        """POST /bundles/ with dict filters. gpu_ram is MB (gte 32000), not a q= string."""
        payload = filters or search_payload()
        if query:
            # Legacy string callers: still send dict filters; ignore q= which Vast 400s.
            payload = search_payload()
        return self._request("POST", "/bundles/", payload)

    def lease_body(
        self,
        image: str = DEFAULT_LEASE_IMAGE,
        disk_gb: int = DEFAULT_DISK_GB,
        env: dict[str, str] | None = None,
        extra: dict | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "label": "anime-factory-gpu",
            # ssh replaces ENTRYPOINT — onstart must start the agent. jupyter wrapping
            # rebuilds a sidecar and injects CDI gpu=N, which 1-GPU hosts reject as gpu=1.
            "runtype": "args ssh",
            "onstart": "/usr/local/bin/af-start",
            "python_utf8": True,
            "lang": "en",
            "target_state": "running",
        }
        if env:
            # Vast accepts env as "-e K=V" string. Quote values with spaces/$/`/\.
            parts: list[str] = []
            for key, raw in env.items():
                if raw is None:
                    continue
                text = str(raw)
                if not text.strip() or any(ch in text for ch in "\r\n"):
                    continue
                parts.append(f"-e {key}={quote_vast_env_value(text)}")
            if parts:
                body["env"] = " ".join(parts)
        if extra:
            body.update(extra)
        return body

    def lease(self, offer_id: str, payload: dict | None = None) -> dict:
        settings = load_settings()
        if settings.vast_allow_replace != 0:
            raise VastSafetyError("VAST_ALLOW_REPLACE must be 0")
        url_path = f"/asks/{offer_id}/"
        body = payload if payload is not None else self.lease_body()
        if self.dry_run:
            # Do not PUT /asks/ — never spend GPU money from this client in dry-run.
            self.calls.append(("DRY_SKIP", f"{VAST_API_BASE}{url_path}"))
            return {"dry_run": True, "skipped": True, "would_put": f"{VAST_API_BASE}{url_path}", "payload": body}
        Counters.vast_asks_put += 1
        return self._request("PUT", url_path, body)

    def get_instance(self, instance_id: str) -> dict:
        """MUST use v0, never /api/v1/instances/{id}/."""
        if "/api/v1/" in instance_id:
            raise VastSafetyError("v1 instance lookup is forbidden")
        path = f"/instances/{instance_id}/"
        Counters.vast_instance_get += 1
        return self._request("GET", path)

    def destroy(self, instance_id: str, registered_ids: set[str]) -> dict:
        if instance_id not in registered_ids:
            raise VastSafetyError(f"refuse to destroy unregistered id {instance_id}")
        path = f"/instances/{instance_id}/"
        if self.dry_run:
            self.calls.append(("DELETE", f"{VAST_API_BASE}{path}"))
            return {"dry_run": True, "id": instance_id}
        Counters.vast_instance_delete += 1
        return self._request("DELETE", path)

    def instance_lookup_url(self, instance_id: str) -> str:
        return f"{VAST_API_BASE}/instances/{instance_id}/"

"""DeepSeek (official first) + Qwen clients with role lock. Tests mock HTTP."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable
from urllib.request import Request, urlopen

from anime_factory.instrument import Counters
from anime_factory.models import (
    DEEPSEEK_BACKUP_MODEL,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL_FLASH,
    QC_MODEL,
    ROLE_ASSISTANT,
    ROLE_DIRECTOR,
    ROLE_MODELS,
    ROLE_QC,
    SILICONFLOW_BASE_URL,
    SIMPLE_MODEL,
)

log = logging.getLogger("anime_factory.llm")

HttpOpener = Callable[[Request], dict[str, Any]]

# deepseek-v4-flash reasons before it answers; a full trilingual episode script spends
# ~28k reasoning tokens and takes minutes. 120s cut every real script off mid-thought.
DEFAULT_TIMEOUT_S = 900.0


class RoleLockError(ValueError):
    pass


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict,
    opener: HttpOpener | None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={**headers, "Content-Type": "application/json"}, method="POST")
    Counters.llm_requests += 1
    if opener:
        return opener(req)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class LlmClient:
    def __init__(
        self,
        deepseek_key: str | None = None,
        siliconflow_keys: list[str] | None = None,
        opener: HttpOpener | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        self.deepseek_key = deepseek_key or ""
        self.siliconflow_keys = siliconflow_keys or []
        self.opener = opener
        self.timeout = timeout
        self.failovers: list[str] = []
        self._sf_i = 0

    def _sf_key(self) -> str:
        if not self.siliconflow_keys:
            return ""
        key = self.siliconflow_keys[self._sf_i % len(self.siliconflow_keys)]
        self._sf_i += 1
        return key

    def chat(self, role: str, messages: list[dict], **kwargs: Any) -> str:
        expected = ROLE_MODELS.get(role)
        if expected is None:
            raise RoleLockError(f"unknown role {role}")
        model = kwargs.get("model", expected)
        if model != expected:
            raise RoleLockError(f"role {role} locked to {expected}, got {model}")
        if role == ROLE_DIRECTOR:
            return self._director(messages, **kwargs)
        if role == ROLE_ASSISTANT:
            return self._siliconflow(SIMPLE_MODEL, messages, **kwargs)
        if role == ROLE_QC:
            return self._siliconflow(QC_MODEL, messages, **kwargs)
        raise RoleLockError(role)

    def _director(self, messages: list[dict], **kwargs: Any) -> str:
        try:
            payload = self._payload(DEEPSEEK_MODEL_FLASH, messages, **kwargs)
            # Thinking tokens share max_tokens; a 8k/default budget returns HTTP 200 with empty content.
            payload.setdefault("max_tokens", 65536)
            payload["thinking"] = {"type": "disabled"}
            payload["reasoning_effort"] = "none"
            data = _post_json(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                {"Authorization": f"Bearer {self.deepseek_key}"},
                payload,
                self.opener,
                self.timeout,
            )
            return _content(data)
        except Exception as exc:
            log.warning("DeepSeek official endpoint failed, falling back to SiliconFlow same-name: %s", exc)
            self.failovers.append(f"deepseek_official_to_siliconflow:{exc}")
            return self._siliconflow(DEEPSEEK_BACKUP_MODEL, messages, **kwargs)

    def _siliconflow(self, model: str, messages: list[dict], **kwargs: Any) -> str:
        data = _post_json(
            f"{SILICONFLOW_BASE_URL}/v1/chat/completions",
            {"Authorization": f"Bearer {self._sf_key()}"},
            self._payload(model, messages, **kwargs),
            self.opener,
            self.timeout,
        )
        return _content(data)

    @staticmethod
    def _payload(model: str, messages: list[dict], **kwargs: Any) -> dict:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        for key in ("response_format", "max_tokens", "temperature"):
            if kwargs.get(key) is not None:
                payload[key] = kwargs[key]
        return payload


def _coerce_chat_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return "".join(parts).strip()
    return ""


def _content(data: dict) -> str:
    """Prefer message.content; fall back to reasoning_content. Never return empty."""
    message = ((data.get("choices") or [{}])[0] or {}).get("message") or {}
    text = _coerce_chat_text(message.get("content")) or _coerce_chat_text(message.get("reasoning_content"))
    if not text:
        reason = ((data.get("choices") or [{}])[0] or {}).get("finish_reason") or ""
        suffix = f" finish_reason={reason}" if reason else ""
        raise ValueError(f"empty llm reply{suffix}")
    return text


def client_from_env(opener: HttpOpener | None = None) -> LlmClient | None:
    """None when no director key is configured, so callers can fall back to an offline draft."""
    import os

    from anime_factory.config import siliconflow_keys

    deepseek = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    sf = siliconflow_keys()
    if not deepseek and not sf:
        return None
    return LlmClient(deepseek_key=deepseek, siliconflow_keys=sf, opener=opener)

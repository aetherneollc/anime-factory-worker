"""DeepSeek (official first) + Qwen clients with role lock. Tests mock HTTP."""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
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
MASTER_VISION_KINDS = frozenset({"character_sheet", "character_view_derive"})
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

FRONT_VISION_FIELDS = ("full_body", "both_feet_visible", "single_subject", "looking_at_viewer")
SIDE_VISION_FIELDS = FRONT_VISION_FIELDS + ("strict_profile",)
BACK_VISION_FIELDS = (
    "rear_view",
    "face_visible",
    "head_rotated",
    "full_body",
    "both_feet_visible",
    "single_subject",
)


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


@dataclass
class MasterVisionResult:
    """Closed-boolean master QC. Not a 0–100 score."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    fields: dict[str, bool] = field(default_factory=dict)
    raw: str = ""

    @property
    def verdict(self) -> str:
        return "pass" if self.passed else "fail"


def _png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _vision_fields_for(view: str) -> tuple[str, ...]:
    key = str(view or "front").strip().lower()
    if key == "side":
        return SIDE_VISION_FIELDS
    if key == "back":
        return BACK_VISION_FIELDS
    return FRONT_VISION_FIELDS


def _vision_system_prompt(view: str) -> str:
    fields = ", ".join(_vision_fields_for(view))
    return (
        "You are a strict anime character-sheet inspector. "
        f"Inspect the single image and reply with JSON only — no markdown — using exactly these "
        f"boolean keys: {fields}. true means the statement holds; false means it does not."
    )


def _parse_vision_json(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty vision qc reply")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(raw)
        if not match:
            raise ValueError(f"vision qc reply is not JSON: {raw[:160]}")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("vision qc JSON must be an object")
    return data


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "1"}:
            return True
        if low in {"false", "no", "0"}:
            return False
    return None


def evaluate_master_vision_fields(view: str, fields: dict[str, Any]) -> MasterVisionResult:
    """AND the closed booleans. Transport/JSON errors are handled by the caller."""
    required = _vision_fields_for(view)
    parsed: dict[str, bool] = {}
    reasons: list[str] = []
    for key in required:
        coerced = _coerce_bool(fields.get(key))
        if coerced is None:
            reasons.append(f"vision_missing_{key}")
            continue
        parsed[key] = coerced
    if reasons:
        return MasterVisionResult(passed=False, reasons=reasons, fields=parsed)

    key = str(view or "front").strip().lower()
    if key == "back":
        ok = (
            parsed["rear_view"]
            and parsed["full_body"]
            and parsed["both_feet_visible"]
            and parsed["single_subject"]
            and not parsed["face_visible"]
            and not parsed["head_rotated"]
        )
        if parsed.get("face_visible"):
            reasons.append("face_visible")
        if parsed.get("head_rotated"):
            reasons.append("head_rotated")
        if not parsed.get("rear_view"):
            reasons.append("not_rear_view")
        if not parsed.get("full_body"):
            reasons.append("not_full_body")
        if not parsed.get("both_feet_visible"):
            reasons.append("feet_not_visible")
        if not parsed.get("single_subject"):
            reasons.append("not_single_subject")
        return MasterVisionResult(passed=ok and not reasons, reasons=reasons, fields=parsed)

    want_true = list(FRONT_VISION_FIELDS) if key != "side" else list(SIDE_VISION_FIELDS)
    for name in want_true:
        if not parsed.get(name):
            reasons.append(f"not_{name}")
    return MasterVisionResult(passed=not reasons, reasons=reasons, fields=parsed)


def master_vision_qc(
    png: bytes,
    *,
    view: str,
    kind: str = "character_sheet",
    client: LlmClient | None = None,
    opener: HttpOpener | None = None,
) -> MasterVisionResult:
    """SiliconFlow Qwen3.5-4B closed JSON on character masters only. temperature 0."""
    if str(kind or "") not in MASTER_VISION_KINDS:
        return MasterVisionResult(passed=True, reasons=[], fields={})
    resolved_view = str(view or "front").strip().lower() or "front"
    llm = client
    if llm is None:
        llm = client_from_env(opener=opener)
    if llm is None:
        # No SiliconFlow key on this process — fail the attempt (do not lock).
        return MasterVisionResult(passed=False, reasons=["vision_transport_error"], fields={})
    messages = [
        {"role": "system", "content": _vision_system_prompt(resolved_view)},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _png_data_url(bytes(png))},
                },
                {
                    "type": "text",
                    "text": (
                        f"Character sheet view={resolved_view}. "
                        "Return JSON only with the boolean keys listed in the system prompt."
                    ),
                },
            ],
        },
    ]
    try:
        raw = llm.chat(
            ROLE_QC,
            messages,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=512,
        )
        data = _parse_vision_json(raw)
        result = evaluate_master_vision_fields(resolved_view, data)
        result.raw = raw
        return result
    except Exception as exc:  # noqa: BLE001 — non-JSON / transport = failed attempt
        log.warning("master vision qc failed: %s", exc)
        return MasterVisionResult(
            passed=False,
            reasons=["vision_transport_error"],
            fields={},
            raw=str(exc),
        )

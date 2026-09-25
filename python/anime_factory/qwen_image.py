"""DashScope Qwen-Image client for character sheets.

Production default for character stills (see CHARACTER_STILL_BACKEND).
Uses commercial ``qwen-image-2.0-pro``: open-weight Qwen-Image-2.1 is
research-licensed and not listed on DashScope.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from base64 import b64decode
from typing import Callable
from urllib.request import Request

from anime_factory.models import (
    DASHSCOPE_MULTIMODAL_URL,
    QwenImageError,
    qwen_image_model,
)

HttpOpener = Callable[[Request], dict]


def dashscope_api_key() -> str:
    return (os.environ.get("DASHSCOPE_API_KEY") or "").strip()


def image_size_for_dashscope(image_size: str | None, *, default: str = "768*1344") -> str:
    """DashScope wants ``W*H``; our payloads use ``WxH``."""
    raw = str(image_size or "").strip().lower().replace(" ", "")
    if "x" in raw:
        return raw.replace("x", "*", 1)
    if "*" in raw:
        return raw
    return default


def _bytes_from_content_item(item: dict) -> bytes | None:
    if not isinstance(item, dict):
        return None
    raw = item.get("image") or item.get("b64_json") or item.get("image_base64")
    url = item.get("url") or item.get("image_url")
    if isinstance(raw, str) and raw.startswith("http"):
        url = raw
        raw = None
    if isinstance(raw, str) and raw.startswith("data:"):
        _, _, payload = raw.partition(",")
        if payload:
            return b64decode(payload)
    if isinstance(raw, str) and len(raw) > 64 and not raw.startswith("http"):
        try:
            return b64decode(raw)
        except Exception:  # noqa: BLE001
            pass
    if isinstance(url, str) and url.startswith("http"):
        with urllib.request.urlopen(url, timeout=120) as resp:
            return resp.read()
    return None


def extract_image_bytes(parsed: dict) -> bytes:
    choices = (((parsed.get("output") or {}).get("choices")) or [])
    if not choices:
        raise QwenImageError(f"qwen-image empty choices: keys={list(parsed)[:8]}")
    content = ((choices[0].get("message") or {}).get("content")) or []
    if isinstance(content, dict):
        content = [content]
    for item in content:
        blob = _bytes_from_content_item(item) if isinstance(item, dict) else None
        if blob:
            return blob
    raise QwenImageError(f"qwen-image returned no image bytes: {parsed!r}"[:400])


def generate_qwen_image(
    *,
    prompt: str,
    negative_prompt: str = "",
    image_size: str = "768x1344",
    seed: int | None = None,
    api_key: str | None = None,
    model: str | None = None,
    opener: HttpOpener | None = None,
    timeout_s: float = 180.0,
    max_retries: int = 6,
) -> bytes:
    """Text-to-image via DashScope multimodal-generation. Fail closed on empty art."""
    key = (api_key if api_key is not None else dashscope_api_key()).strip()
    if not key and opener is None:
        raise QwenImageError("DASHSCOPE_API_KEY missing")
    model_id = (model or qwen_image_model()).strip()
    size = image_size_for_dashscope(image_size)
    parameters: dict = {
        "n": 1,
        "size": size,
        "prompt_extend": False,
        "watermark": False,
    }
    if negative_prompt:
        parameters["negative_prompt"] = negative_prompt
    if seed is not None:
        parameters["seed"] = int(seed) % (2**31 - 1)
    body = {
        "model": model_id,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ]
        },
        "parameters": parameters,
    }
    data = json.dumps(body).encode("utf-8")
    req = Request(
        DASHSCOPE_MULTIMODAL_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {key or 'offline'}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    if opener is not None:
        parsed = opener(req)
        if isinstance(parsed, dict) and isinstance(parsed.get("bytes"), (bytes, bytearray)):
            return bytes(parsed["bytes"])
        if not isinstance(parsed, dict):
            raise QwenImageError("qwen-image opener returned non-dict")
        return extract_image_bytes(parsed)

    last_err: Exception | None = None
    for attempt in range(max(1, int(max_retries))):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            return extract_image_bytes(parsed)
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < max_retries:
                time.sleep(min(60.0, 2.0 * (2**attempt)))
                continue
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            raise QwenImageError(f"qwen-image HTTP {exc.code}: {detail or exc}") from exc
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt + 1 < max_retries:
                time.sleep(min(30.0, 1.5 * (2**attempt)))
                continue
            raise QwenImageError(f"qwen-image request failed: {exc}") from exc
    raise QwenImageError(f"qwen-image failed: {last_err}")

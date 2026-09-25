"""Tencent HunyuanImage (Hy-Image-3.0) client for character canaries.

Backends (first match wins):
  - TokenHub: ``TOKENHUB_API_KEY`` / ``HUNYUAN_API_KEY``
    POST https://tokenhub.tencentmaas.com/v1/wand/hunyuan-image/v3-generation
  - fal.ai: ``FAL_KEY``
    POST https://fal.run/fal-ai/hunyuan-image/v3/text-to-image

TokenHub constrains area ≤ 1024×1024 (768×1344 is OK; 1024×1536 is not).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from base64 import b64decode, b64encode
from typing import Callable, Sequence
from urllib.request import Request

HttpOpener = Callable[[Request], dict]

TOKENHUB_URL = "https://tokenhub.tencentmaas.com/v1/wand/hunyuan-image/v3-generation"
FAL_URL = "https://fal.run/fal-ai/hunyuan-image/v3/text-to-image"
HUNYUAN_MODEL = "hy-image-v3"


class HunyuanImageError(ValueError):
    """HunyuanImage request failed or returned no art."""


def hunyuan_api_key() -> str:
    return (
        os.environ.get("TOKENHUB_API_KEY")
        or os.environ.get("HUNYUAN_API_KEY")
        or os.environ.get("TENCENT_TOKENHUB_API_KEY")
        or ""
    ).strip()


def fal_api_key() -> str:
    return (os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY") or "").strip()


def parse_wxh(image_size: str | None, *, default: tuple[int, int] = (768, 1344)) -> tuple[int, int]:
    raw = str(image_size or "").strip().lower().replace(" ", "").replace("*", "x")
    if "x" in raw:
        a, b = raw.split("x", 1)
        try:
            return max(int(a), 64), max(int(b), 64)
        except ValueError:
            pass
    return default


def assert_tokenhub_area(width: int, height: int) -> None:
    area = int(width) * int(height)
    if area > 1024 * 1024:
        raise HunyuanImageError(
            f"TokenHub Hy-Image-3.0 area must be ≤ 1024×1024, got {width}x{height} ({area})"
        )


def _download(url: str, timeout: float = 120.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _bytes_from_data_uri_or_url(value: str) -> bytes:
    if value.startswith("data:"):
        _, _, payload = value.partition(",")
        return b64decode(payload)
    if value.startswith("http"):
        return _download(value)
    raise HunyuanImageError(f"unsupported image payload prefix: {value[:32]!r}")


def _png_data_uri(png: bytes) -> str:
    return "data:image/png;base64," + b64encode(png).decode("ascii")


def generate_hunyuan_image(
    *,
    prompt: str,
    negative_prompt: str = "",
    image_size: str = "768x1344",
    seed: int | None = None,
    revise: bool = False,
    reference_pngs: Sequence[bytes] | None = None,
    opener: HttpOpener | None = None,
    timeout_s: float = 180.0,
    max_retries: int = 5,
) -> bytes:
    """Text-to-image (optional reference images) via TokenHub or fal.ai."""
    tokenhub = hunyuan_api_key()
    fal = fal_api_key()
    if opener is None and not tokenhub and not fal:
        raise HunyuanImageError(
            "missing Hunyuan credentials: set TOKENHUB_API_KEY (or HUNYUAN_API_KEY) or FAL_KEY"
        )
    width, height = parse_wxh(image_size)
    refs = [bytes(p) for p in (reference_pngs or []) if p][:3]

    if opener is not None or tokenhub:
        if opener is None:
            assert_tokenhub_area(width, height)
        body: dict = {
            "model": HUNYUAN_MODEL,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "revise": bool(revise),
        }
        if refs:
            body["images"] = [_png_data_uri(p) for p in refs]
        if negative_prompt:
            # TokenHub schema may ignore unknown fields; keep for backends that accept it.
            body["negative_prompt"] = negative_prompt
        if seed is not None and int(seed) > 0:
            body["seed"] = int(seed) % (2**32 - 1) or 1
        return _tokenhub_generate(
            body,
            api_key=tokenhub or "offline",
            opener=opener,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )

    return _fal_generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
        api_key=fal,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )


def _tokenhub_generate(
    body: dict,
    *,
    api_key: str,
    opener: HttpOpener | None,
    timeout_s: float,
    max_retries: int,
) -> bytes:
    data = json.dumps(body).encode("utf-8")
    req = Request(
        TOKENHUB_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    if opener is not None:
        parsed = opener(req)
        if isinstance(parsed, dict) and isinstance(parsed.get("bytes"), (bytes, bytearray)):
            return bytes(parsed["bytes"])
        if not isinstance(parsed, dict):
            raise HunyuanImageError("hunyuan opener returned non-dict")
        return _extract_tokenhub_bytes(parsed)

    last_err: Exception | None = None
    for attempt in range(max(1, int(max_retries))):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            return _extract_tokenhub_bytes(parsed)
        except urllib.error.HTTPError as exc:
            last_err = exc
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < max_retries:
                time.sleep(min(60.0, 2.0 * (2**attempt)))
                continue
            raise HunyuanImageError(f"TokenHub HTTP {exc.code}: {detail or exc}") from exc
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt + 1 < max_retries:
                time.sleep(min(30.0, 1.5 * (2**attempt)))
                continue
            raise HunyuanImageError(f"TokenHub request failed: {exc}") from exc
    raise HunyuanImageError(f"TokenHub failed: {last_err}")


def _extract_tokenhub_bytes(parsed: dict) -> bytes:
    # Documented shapes vary; accept common URL / b64 fields.
    for key in ("image", "url", "image_url"):
        val = parsed.get(key)
        if isinstance(val, str) and val:
            return _bytes_from_data_uri_or_url(val)
    images = parsed.get("images") or parsed.get("data") or []
    if isinstance(images, list):
        for item in images:
            if isinstance(item, str) and item:
                return _bytes_from_data_uri_or_url(item)
            if isinstance(item, dict):
                for key in ("url", "image", "image_url", "b64_json"):
                    val = item.get(key)
                    if isinstance(val, str) and val:
                        return _bytes_from_data_uri_or_url(val)
    output = parsed.get("output")
    if isinstance(output, dict):
        return _extract_tokenhub_bytes(output)
    raise HunyuanImageError(f"TokenHub returned no image: {str(parsed)[:400]}")


def _fal_generate(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int | None,
    api_key: str,
    timeout_s: float,
    max_retries: int,
) -> bytes:
    body: dict = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "image_size": {"width": width, "height": height},
        "num_images": 1,
        "enable_prompt_expansion": False,
        "output_format": "png",
    }
    if seed is not None:
        body["seed"] = int(seed)
    data = json.dumps(body).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(max(1, int(max_retries))):
        req = Request(
            FAL_URL,
            data=data,
            headers={
                "Authorization": f"Key {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            images = parsed.get("images") or []
            if not images:
                raise HunyuanImageError(f"fal returned no images: {str(parsed)[:300]}")
            url = images[0].get("url") if isinstance(images[0], dict) else None
            if not url:
                raise HunyuanImageError(f"fal image missing url: {images[0]!r}")
            return _download(url)
        except urllib.error.HTTPError as exc:
            last_err = exc
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < max_retries:
                time.sleep(min(60.0, 2.0 * (2**attempt)))
                continue
            raise HunyuanImageError(f"fal HTTP {exc.code}: {detail or exc}") from exc
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt + 1 < max_retries:
                time.sleep(min(30.0, 1.5 * (2**attempt)))
                continue
            raise HunyuanImageError(f"fal request failed: {exc}") from exc
    raise HunyuanImageError(f"fal failed: {last_err}")

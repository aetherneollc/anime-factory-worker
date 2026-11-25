"""fal.ai model ids and payload adapters.

Env defaults (overridden by a non-empty control-plane ``fal_config`` unless
``FAL_OVERRIDE_CONTROL_PLANE=1``, in which case the env value wins):

- ``FAL_IMAGE_T2I_MODEL``
- ``FAL_IMAGE_EDIT_MODEL``
- ``FAL_VIDEO_I2V_MODEL``
- ``FAL_VIDEO_R2V_MODEL``
- ``FAL_MAX_EPISODE_USD``
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

from anime_factory.fal_client import (
    FalApiError,
    FalClient,
    estimate_image_usd,
    estimate_video_usd,
    fal_dry_run,
    get_fal_client,
    media_url_from_result,
    synthetic_image_bytes,
    synthetic_video_bytes,
)

DEFAULT_FAL_IMAGE_T2I_MODEL = "alibaba/qwen-image-2.1/text-to-image"
DEFAULT_FAL_IMAGE_EDIT_MODEL = "fal-ai/qwen-image-edit-2511"
DEFAULT_FAL_VIDEO_I2V_MODEL = "fal-ai/minimax/hailuo-02/standard/image-to-video"
DEFAULT_FAL_VIDEO_R2V_MODEL = "fal-ai/kling-video/o1/reference-to-video"

_CONFIG_FIELDS = {
    "image_t2i_model": ("FAL_IMAGE_T2I_MODEL", DEFAULT_FAL_IMAGE_T2I_MODEL),
    "image_edit_model": ("FAL_IMAGE_EDIT_MODEL", DEFAULT_FAL_IMAGE_EDIT_MODEL),
    "video_i2v_model": ("FAL_VIDEO_I2V_MODEL", DEFAULT_FAL_VIDEO_I2V_MODEL),
    "video_r2v_model": ("FAL_VIDEO_R2V_MODEL", DEFAULT_FAL_VIDEO_R2V_MODEL),
    "max_episode_usd": ("FAL_MAX_EPISODE_USD", "20.0"),
}

_ACTIVE_FAL_CONFIG: dict[str, Any] | None = None
_IMAGE_MARKER = re.compile(r"<<<image_(\d+)>>>", re.I)
_R2V_MODES = frozenset({"r2v", "ref2va", "reference", "reference-to-video"})
_FL2V_MODES = frozenset({"fl2v", "fl2va_first_last", "i2v", "fl2va_first"})


def set_active_fal_config(fal_config: dict[str, Any] | None) -> None:
    global _ACTIVE_FAL_CONFIG
    if isinstance(fal_config, dict) and fal_config:
        _ACTIVE_FAL_CONFIG = dict(fal_config)
    else:
        _ACTIVE_FAL_CONFIG = None


def active_fal_config() -> dict[str, Any] | None:
    return None if _ACTIVE_FAL_CONFIG is None else dict(_ACTIVE_FAL_CONFIG)


def control_plane_overridden_by_env() -> bool:
    return str(os.environ.get("FAL_OVERRIDE_CONTROL_PLANE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def resolve_fal_value(key: str, *, fal_config: dict[str, Any] | None = None) -> str:
    """Control-plane value wins when non-empty, unless the env override flag is set."""
    env_name, default = _CONFIG_FIELDS[key]
    env_val = str(os.environ.get(env_name) or "").strip()
    source = fal_config if fal_config is not None else _ACTIVE_FAL_CONFIG
    cfg_val = ""
    if isinstance(source, dict):
        cfg_val = str(source.get(key) or "").strip()
    if control_plane_overridden_by_env():
        return env_val or default
    if cfg_val:
        return cfg_val
    return env_val or default


def resolve_episode_budget_usd(fal_config: dict[str, Any] | None = None) -> float:
    raw = resolve_fal_value("max_episode_usd", fal_config=fal_config)
    try:
        return float(raw)
    except ValueError:
        return 20.0


def begin_fal_episode(fal_config: dict[str, Any] | None = None) -> float:
    """Apply one episode's fal_config and reset the spend tracker."""
    from anime_factory.fal_client import reset_episode_budget

    set_active_fal_config(fal_config)
    cap = resolve_episode_budget_usd(fal_config)
    reset_episode_budget(cap)
    return cap


def _seed_value(seed: Any) -> int | None:
    if isinstance(seed, bool) or seed is None:
        return None
    if isinstance(seed, int):
        return int(seed)
    text = str(seed).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return None


def _with_seed(payload: dict[str, Any], seed: Any) -> dict[str, Any]:
    value = _seed_value(seed)
    if value is not None:
        payload["seed"] = value
    return payload


def build_t2i_payload(
    prompt: str,
    width: int,
    height: int,
    *,
    negative_prompt: str = "",
    seed: Any = None,
) -> dict[str, Any]:
    return _with_seed(
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "image_size": {"width": int(width), "height": int(height)},
            "num_images": 1,
            "output_format": "png",
        },
        seed,
    )


def build_edit_payload(
    prompt: str,
    image_urls: Sequence[str],
    width: int,
    height: int,
    *,
    negative_prompt: str = "",
    seed: Any = None,
) -> dict[str, Any]:
    return _with_seed(
        {
            "prompt": prompt,
            "image_urls": [str(url) for url in image_urls if str(url).strip()],
            "negative_prompt": negative_prompt or "",
            "image_size": {"width": int(width), "height": int(height)},
            "num_images": 1,
            "output_format": "png",
        },
        seed,
    )


def _clamp_int(duration_s: float, lo: int, hi: int) -> int:
    try:
        value = int(round(float(duration_s)))
    except (TypeError, ValueError):
        value = lo
    return max(lo, min(hi, value))


def _is_hailuo(model: str) -> bool:
    text = model.lower()
    return "minimax/hailuo" in text or "hailuo" in text


def _is_kling_i2v(model: str) -> bool:
    text = model.lower()
    if "reference-to-video" in text or "reference_to_video" in text:
        return False
    return "kling-video/v3" in text or "kling-video/v2" in text


def _is_kling_r2v(model: str) -> bool:
    """Kling O1 reference-to-video, and Kling v3 element endpoints."""
    text = model.lower()
    if "kling-video/o1/reference-to-video" in text:
        return True
    if "kling" in text and "reference-to-video" in text:
        return True
    return "kling-video/v3" in text and "element" in text


def build_i2v_payload(
    *,
    model: str,
    prompt: str,
    image_url: str,
    end_image_url: str | None = None,
    duration_s: float = 6.0,
    negative_prompt: str = "",
    seed: Any = None,
) -> dict[str, Any]:
    model_id = str(model or "")
    if _is_hailuo(model_id):
        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_url": image_url,
            "duration": "6" if float(duration_s) <= 6.5 else "10",
            "resolution": "768P",
            "prompt_optimizer": False,
        }
        if end_image_url:
            payload["end_image_url"] = end_image_url
        return _with_seed(payload, seed)
    if _is_kling_i2v(model_id):
        payload = {
            "prompt": prompt,
            "start_image_url": image_url,
            "duration": str(_clamp_int(duration_s, 3, 10)),
            "generate_audio": False,
            "negative_prompt": negative_prompt or "",
        }
        if end_image_url:
            payload["end_image_url"] = end_image_url
        return _with_seed(payload, seed)
    payload = {"prompt": prompt, "image_url": image_url, "duration": _clamp_int(duration_s, 1, 15)}
    if end_image_url:
        payload["end_image_url"] = end_image_url
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    return _with_seed(payload, seed)


def translate_reference_markers(prompt: str, n_elements: int, n_images: int) -> str:
    """Map ``<<<image_N>>>`` onto Kling ``@ElementN`` / ``@ImageN`` (1-based, elements first)."""

    def repl(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index <= n_elements and index >= 1:
            return f"@Element{index}"
        image_index = index - n_elements
        if image_index >= 1 and image_index <= n_images:
            return f"@Image{image_index}"
        return match.group(0)

    return _IMAGE_MARKER.sub(repl, prompt or "")


def _nonempty_url(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _url_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Path)):
        text = _nonempty_url(value)
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        urls: list[str] = []
        for item in value:
            urls.extend(_url_sequence(item))
        return urls
    return []


def kling_element(item: Any) -> dict[str, Any] | None:
    """One Kling element. A single frontal URL is repeated as the required angle.

    Also accepts ``(frontal, [extra_angles...])`` and
    ``{"frontal_image_url", "reference_image_urls"}``.
    """
    frontal = ""
    extras: list[str] = []
    if isinstance(item, dict):
        frontal = _nonempty_url(item.get("frontal_image_url") or item.get("frontal"))
        raw_extras = item.get("reference_image_urls")
        if raw_extras is None:
            raw_extras = item.get("references") or []
        extras = _url_sequence(raw_extras)
    elif isinstance(item, (list, tuple)):
        if len(item) == 2 and isinstance(item[1], (list, tuple)):
            frontal = _nonempty_url(item[0])
            extras = _url_sequence(item[1])
        else:
            urls = _url_sequence(item)
            if not urls:
                return None
            frontal = urls[0]
            extras = urls[1:]
    else:
        frontal = _nonempty_url(item)
    if not frontal and extras:
        frontal = extras.pop(0)
    if not frontal:
        return None
    if not extras:
        extras = [frontal]
    return {"frontal_image_url": frontal, "reference_image_urls": extras}


def _flat_ref_urls(item: Any) -> list[str]:
    """Generic R2V image list. A lone URL stays one input, not a duplicated angle."""
    if isinstance(item, dict):
        frontal = _nonempty_url(item.get("frontal_image_url") or item.get("frontal"))
        extras = _url_sequence(item.get("reference_image_urls") or item.get("references") or [])
        urls = ([frontal] if frontal else []) + [url for url in extras if url != frontal]
        return urls or extras
    if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
        frontal = _nonempty_url(item[0])
        extras = [url for url in _url_sequence(item[1]) if url != frontal]
        return ([frontal] if frontal else []) + extras
    if isinstance(item, (list, tuple)):
        return _url_sequence(item)
    text = _nonempty_url(item)
    return [text] if text else []


def build_r2v_payload(
    *,
    model: str,
    prompt: str,
    char_urls: Sequence[Any],
    scene_urls: Sequence[Any],
    duration_s: float = 6.0,
    negative_prompt: str = "",
    seed: Any = None,
) -> dict[str, Any]:
    model_id = str(model or "")
    scenes = [url for item in scene_urls for url in _flat_ref_urls(item)]
    if _is_kling_r2v(model_id):
        elements: list[dict[str, Any]] = []
        for item in char_urls:
            element = kling_element(item)
            if element is not None:
                elements.append(element)
        # Kling counts each element and each start/scene image toward a cap of 7.
        elements = elements[:7]
        scenes = scenes[: max(0, 7 - len(elements))]
        text = translate_reference_markers(prompt, len(elements), len(scenes))
        payload: dict[str, Any] = {
            "prompt": text,
            "aspect_ratio": "16:9",
            "duration": str(_clamp_int(duration_s, 3, 10)),
        }
        if elements:
            payload["elements"] = elements
        if scenes:
            payload["image_urls"] = scenes
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        return _with_seed(payload, seed)
    chars = [url for item in char_urls for url in _flat_ref_urls(item)]
    images = [*chars, *scenes][:7]
    payload = {
        "prompt": prompt,
        "duration": _clamp_int(duration_s, 1, 15),
        "aspect_ratio": "16:9",
    }
    if images:
        payload["image_urls"] = images
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    return _with_seed(payload, seed)


def _as_bytes(item: str | bytes | Path) -> bytes:
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    path = Path(str(item))
    if path.is_file():
        return path.read_bytes()
    raise FalApiError(f"reference is not bytes or a file: {item!r}")


def _upload_one(client: FalClient, item: Any, filename: str) -> str:
    text = item if isinstance(item, str) else ""
    if text.startswith(("http://", "https://", "data:")):
        return text
    return client.upload_bytes(_as_bytes(item), filename=filename)


def _upload_character_ref(client: FalClient, item: Any, index: int) -> Any:
    """Upload one character pack, keeping frontal/angle structure for Kling."""
    if isinstance(item, dict):
        frontal_src = item.get("frontal_image_url") or item.get("frontal")
        raw_extras = item.get("reference_image_urls")
        if raw_extras is None:
            raw_extras = item.get("references") or []
        if isinstance(raw_extras, (str, bytes, Path)):
            raw_extras = [raw_extras]
        frontal = _upload_one(client, frontal_src, f"char-{index}-front.png") if frontal_src else ""
        extras = [
            _upload_one(client, extra, f"char-{index}-angle-{number}.png")
            for number, extra in enumerate(raw_extras, start=1)
        ]
        packed: dict[str, Any] = {"reference_image_urls": extras}
        if frontal:
            packed["frontal_image_url"] = frontal
        return packed
    if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
        frontal = _upload_one(client, item[0], f"char-{index}-front.png")
        extras = [
            _upload_one(client, extra, f"char-{index}-angle-{number}.png")
            for number, extra in enumerate(item[1], start=1)
        ]
        return frontal, extras
    return _upload_one(client, item, f"char-{index}.png")


def _client(dry_run: bool | None) -> FalClient:
    return get_fal_client(dry_run=dry_run)


def _image_bytes_from_result(client: FalClient, result: dict[str, Any]) -> bytes:
    return client.download(media_url_from_result(result))


def generate_fal_image_t2i(
    prompt: str,
    width: int,
    height: int,
    *,
    negative_prompt: str = "",
    seed: Any = None,
    model: str | None = None,
    dry_run: bool | None = None,
    fal_config: dict[str, Any] | None = None,
) -> bytes:
    if fal_dry_run(dry_run):
        return synthetic_image_bytes(width, height, tag=f"fal-t2i:{seed}:{prompt[:48]}")
    model_id = model or resolve_fal_value("image_t2i_model", fal_config=fal_config)
    client = _client(False)
    client.reserve(estimate_image_usd())
    payload = build_t2i_payload(prompt, width, height, negative_prompt=negative_prompt, seed=seed)
    result = client.submit(model_id, payload, timeout=float(os.environ.get("FAL_IMAGE_TIMEOUT_S") or 180))
    return _image_bytes_from_result(client, result)


def edit_fal_image(
    prompt: str,
    reference_pngs: Sequence[str | bytes | Path],
    width: int,
    height: int,
    *,
    negative_prompt: str = "",
    seed: Any = None,
    master_path: str = "",
    view: str | None = None,
    model: str | None = None,
    dry_run: bool | None = None,
    fal_config: dict[str, Any] | None = None,
) -> bytes:
    """Reference edit. Side view refuses to run unless ``master_path`` is ``master.png``."""
    from anime_factory.qwen_image_21 import assert_side_master_reference

    if str(view or "").strip().lower() == "side":
        assert_side_master_reference(master_path)
    refs = list(reference_pngs or [])
    if not refs:
        raise FalApiError("edit_fal_image requires at least one reference")
    if fal_dry_run(dry_run):
        return synthetic_image_bytes(width, height, tag=f"fal-edit:n={len(refs)}:{seed}:{prompt[:48]}")
    model_id = model or resolve_fal_value("image_edit_model", fal_config=fal_config)
    client = _client(False)
    client.reserve(estimate_image_usd())
    urls = [
        client.upload_bytes(_as_bytes(ref), content_type="image/png", filename=f"ref-{index}.png")
        for index, ref in enumerate(refs, start=1)
    ]
    payload = build_edit_payload(prompt, urls, width, height, negative_prompt=negative_prompt, seed=seed)
    result = client.submit(model_id, payload, timeout=float(os.environ.get("FAL_IMAGE_TIMEOUT_S") or 180))
    return _image_bytes_from_result(client, result)


def _video_mode(mode: str) -> str:
    raw = str(mode or "").strip().lower().replace(" ", "")
    if raw in _R2V_MODES or "ref" in raw:
        return "r2v"
    return "i2v"


def generate_fal_video(
    mode: str,
    prompt: str,
    *,
    duration_s: float = 6.0,
    first_frame_png: bytes | str | Path | None = None,
    last_frame_png: bytes | str | Path | None = None,
    char_ref_pngs: Sequence[str | bytes | Path] | None = None,
    scene_ref_pngs: Sequence[str | bytes | Path] | None = None,
    negative_prompt: str = "",
    seed: Any = None,
    out_path: str | Path | None = None,
    fal_config: dict[str, Any] | None = None,
    dry_run: bool | None = None,
) -> Path:
    dest = Path(out_path) if out_path else Path(os.environ.get("TMPDIR") or "/tmp") / "fal-video.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if fal_dry_run(dry_run):
        dest.write_bytes(synthetic_video_bytes(duration_s))
        return dest
    kind = _video_mode(mode)
    client = _client(False)
    client.reserve(estimate_video_usd(duration_s))
    if kind == "r2v":
        model_id = resolve_fal_value("video_r2v_model", fal_config=fal_config)
        char_urls = [
            _upload_character_ref(client, ref, index)
            for index, ref in enumerate(char_ref_pngs or [], start=1)
        ]
        scene_items: list[Any] = list(scene_ref_pngs or [])
        if first_frame_png:
            scene_items.append(first_frame_png)
        scene_urls = [
            client.upload_bytes(_as_bytes(ref), filename=f"scene-{index}.png")
            for index, ref in enumerate(scene_items, start=1)
        ]
        payload = build_r2v_payload(
            model=model_id,
            prompt=prompt,
            char_urls=char_urls,
            scene_urls=scene_urls,
            duration_s=duration_s,
            negative_prompt=negative_prompt,
            seed=seed,
        )
    else:
        model_id = resolve_fal_value("video_i2v_model", fal_config=fal_config)
        if not first_frame_png:
            raise FalApiError(f"fal {mode or 'i2v'} requires a first frame")
        image_url = client.upload_bytes(_as_bytes(first_frame_png), filename="first.png")
        end_url = None
        if last_frame_png:
            end_url = client.upload_bytes(_as_bytes(last_frame_png), filename="last.png")
        payload = build_i2v_payload(
            model=model_id,
            prompt=prompt,
            image_url=image_url,
            end_image_url=end_url,
            duration_s=duration_s,
            negative_prompt=negative_prompt,
            seed=seed,
        )
    result = client.submit(model_id, payload, timeout=float(os.environ.get("FAL_VIDEO_TIMEOUT_S") or 900))
    dest.write_bytes(client.download(media_url_from_result(result)))
    return dest

"""GPU 生图 on the leased card: anime SDXL (Animagine XL 4.0) via Comfy.

Flux schnell was guidance-distilled, so `cfg=1` bypassed CFG entirely and the whole
`FIXED_NEGATIVE` list was dead code. SDXL at cfg 5 / 28 steps / euler_ancestral applies it.

Character consistency is real here: the parent locked sheet is uploaded to Comfy
and bound to `IPAdapterAdvanced`. A `costume_derive` request that carries a parent
image but hits a workflow with no IP-Adapter node **fails** instead of silently
drawing an unconditioned character.

SiliconFlow remains available on the control-plane / pre-GPU path (no local weights).
On Vast this module must actually draw — never silent-fallback to the API.
Do not keep H3 and SDXL resident: unload the still UNet before H3 sampling.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

from anime_factory.models import (
    ANIMAGINE_WORKFLOW,
    FIXED_NEGATIVE,
    IMAGE_CFG,
    IMAGE_CKPT,
    KOLORS_CHATGLM_FILE,
    KOLORS_VAE_FILE,
    IMAGE_SAMPLER,
    IMAGE_SCHEDULER,
    IMAGE_STEPS,
    STILL_HEIGHT,
    STILL_WIDTH,
    STYLE_PREFIX,
    scrub_copycat,
    still_model_id,
    still_unet_file,
    still_workflow_name,
    style_prefix_for_kind,
)

STILL_WORKFLOW = ANIMAGINE_WORKFLOW
TEXT_ENCODE_TYPES = frozenset(
    {
        "CLIPTextEncode",
        "CLIPTextEncodeSDXL",
        "CLIPTextEncodeFlux",
        "MZ_ChatGLM3_V2",
        "MZ_ChatGLM3_Advance_V2",
    }
)
# Kinds whose whole point is to inherit a parent's identity. No reference node → fail.
REFERENCE_REQUIRED_KINDS = frozenset({"character_view_derive", "costume_derive"})
IPADAPTER_CLASS_TYPES = frozenset(
    {
        "IPAdapterAdvanced",
        "IPAdapter",
        "IPAdapterApply",
        "MZ_IPAdapterAdvancedKolors",
    }
)
REFERENCE_LOADER_TYPES = frozenset(
    {
        "IPAdapterModelLoader",
        "CLIPVisionLoader",
        "MZ_IPAdapterModelLoaderKolors",
        "MZ_KolorsCLIPVisionLoader",
    }
)
REFERENCE_ROLE = "reference_image"
VIEW_DERIVE_IPADAPTER_WEIGHT = 0.75
VIEW_DERIVE_IPADAPTER_END_AT = 0.9
KEYFRAME_PLATE_IPADAPTER_WEIGHT = 0.35
KEYFRAME_PLATE_IPADAPTER_END_AT = 0.9


def parse_size(image_size: str | None) -> tuple[int, int]:
    raw = (image_size or "").lower().replace(" ", "")
    if "x" in raw:
        a, b = raw.split("x", 1)
        try:
            return max(int(a), 64), max(int(b), 64)
        except ValueError:
            pass
    return STILL_WIDTH, STILL_HEIGHT


def still_payload_to_prompt(payload: dict) -> dict[str, Any]:
    prompt = scrub_copycat(payload.get("prompt") or "")
    kind = str(payload.get("_kind") or payload.get("kind") or "")
    prefix = style_prefix_for_kind(kind)
    # `_styled` is set by whoever prepended the kind-specific prefix.
    if prompt and not payload.get("_styled") and not prompt.startswith(prefix):
        prompt = f"{prefix}, {prompt}"
    negative = payload.get("negative_prompt") or FIXED_NEGATIVE
    width, height = parse_size(payload.get("image_size"))
    seed = payload.get("seed")
    try:
        seed_i = int(seed) if seed is not None else int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    except (TypeError, ValueError):
        seed_i = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    steps = int(payload.get("num_inference_steps") or IMAGE_STEPS)
    if steps < IMAGE_STEPS:
        steps = IMAGE_STEPS
    try:
        cfg = float(payload.get("guidance_scale") if payload.get("guidance_scale") is not None else IMAGE_CFG)
    except (TypeError, ValueError):
        cfg = IMAGE_CFG
    return {
        "prompt": prompt,
        "negative": negative,
        "width": width,
        "height": height,
        "seed": seed_i,
        "steps": steps,
        "cfg": cfg,
        "sampler": str(payload.get("sampler_name") or IMAGE_SAMPLER),
        "scheduler": str(payload.get("scheduler") or IMAGE_SCHEDULER),
        "ckpt": str(payload.get("ckpt_name") or still_unet_file()),
        "vae": str(payload.get("vae_name") or KOLORS_VAE_FILE),
        "chatglm": str(payload.get("chatglm_checkpoint") or KOLORS_CHATGLM_FILE),
        "model": payload.get("model") or still_model_id(),
        "reference_image": payload.get("reference_image") or "",
        "kind": kind,
        "parent_kind": str(payload.get("_parent_kind") or ""),
    }


def _reference_node_ids(prompt: dict) -> list[str]:
    return [
        nid
        for nid, node in prompt.items()
        if isinstance(node, dict) and node.get("class_type") in IPADAPTER_CLASS_TYPES
    ]


def _drop_reference_chain(prompt: dict) -> None:
    """No parent image: unhook IPAdapter/LoadImage and sample straight off the checkpoint."""
    ipadapter_ids = _reference_node_ids(prompt)
    if not ipadapter_ids:
        return
    passthrough: dict[str, Any] = {}
    for nid in ipadapter_ids:
        model_in = (prompt[nid].get("inputs") or {}).get("model")
        if isinstance(model_in, list):
            passthrough[nid] = model_in
    drop = set(ipadapter_ids)
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        role = (node.get("_meta") or {}).get("role")
        if node.get("class_type") in REFERENCE_LOADER_TYPES or role == REFERENCE_ROLE:
            drop.add(nid)
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key, value in list(inputs.items()):
            if isinstance(value, list) and value and value[0] in passthrough:
                inputs[key] = passthrough[value[0]]
    for nid in drop:
        prompt.pop(nid, None)


def fill_still_workflow(template: dict, spec: dict[str, Any]) -> dict:
    graph = json.loads(json.dumps(template))
    prompt = graph.get("prompt", graph)
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.setdefault("inputs", {})
        ctype = node.get("class_type") or ""
        role = (node.get("_meta") or {}).get("role")
        if ctype in TEXT_ENCODE_TYPES:
            if role == "negative" or inputs.get("text") == "NEGATIVE":
                inputs["text"] = spec["negative"]
            else:
                inputs["text"] = spec["prompt"]
        if ctype == "CheckpointLoaderSimple" and "ckpt_name" in inputs:
            inputs["ckpt_name"] = spec.get("ckpt", IMAGE_CKPT)
        if ctype == "MZ_KolorsUNETLoaderV2" and "unet_name" in inputs:
            inputs["unet_name"] = spec.get("ckpt", still_unet_file())
        if ctype == "VAELoader" and "vae_name" in inputs:
            inputs["vae_name"] = spec.get("vae", KOLORS_VAE_FILE)
        if ctype == "MZ_ChatGLM3Loader" and "chatglm3_checkpoint" in inputs:
            inputs["chatglm3_checkpoint"] = spec.get("chatglm", KOLORS_CHATGLM_FILE)
        if ctype in {"EmptyLatentImage", "EmptySD3LatentImage"} or "width" in inputs:
            if "width" in inputs:
                inputs["width"] = spec["width"]
            if "height" in inputs:
                inputs["height"] = spec["height"]
            if "target_width" in inputs:
                inputs["target_width"] = spec["width"]
            if "target_height" in inputs:
                inputs["target_height"] = spec["height"]
        if ctype in IPADAPTER_CLASS_TYPES and ctype != "IPAdapterModelLoader":
            kind = str(spec.get("kind") or "")
            parent_kind = str(spec.get("parent_kind") or "")
            if kind == "character_view_derive":
                inputs["weight"] = VIEW_DERIVE_IPADAPTER_WEIGHT
                inputs["end_at"] = VIEW_DERIVE_IPADAPTER_END_AT
            elif kind == "keyframe" and parent_kind in {"scene_plate", "scene_derive", ""}:
                # Locked plate bound at low weight; separate from character-view 0.75.
                inputs["weight"] = KEYFRAME_PLATE_IPADAPTER_WEIGHT
                inputs["end_at"] = KEYFRAME_PLATE_IPADAPTER_END_AT
            elif kind == "keyframe":
                inputs["weight"] = KEYFRAME_PLATE_IPADAPTER_WEIGHT
                inputs["end_at"] = KEYFRAME_PLATE_IPADAPTER_END_AT
        if ctype in {"LoadImage", "LoadImageOutput"} and role == REFERENCE_ROLE:
            inputs["image"] = spec.get("reference_image") or ""
        if "seed" in inputs:
            inputs["seed"] = spec["seed"]
        if ctype in {"KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced", "RandomNoise"}:
            if "steps" in inputs:
                inputs["steps"] = spec["steps"]
            if "cfg" in inputs:
                inputs["cfg"] = spec.get("cfg", IMAGE_CFG)
            if "sampler_name" in inputs:
                inputs["sampler_name"] = spec.get("sampler", IMAGE_SAMPLER)
            if "scheduler" in inputs:
                inputs["scheduler"] = spec.get("scheduler", IMAGE_SCHEDULER)
            if "seed" in inputs:
                inputs["seed"] = spec["seed"]
        elif "steps" in inputs:
            inputs["steps"] = spec["steps"]
    if not spec.get("reference_image"):
        _drop_reference_chain(prompt)
    return graph if "prompt" in graph else {"prompt": prompt}


def _parent_png_bytes(payload: dict) -> bytes | None:
    raw = payload.get("_parent_png")
    if isinstance(raw, (bytes, bytearray)) and raw:
        return bytes(raw)
    return None


def _reference_name(payload: dict, blob: bytes) -> str:
    label = str(payload.get("_kind") or "ref").replace("/", "_")
    parent = str(payload.get("_parent_id") or payload.get("_label") or "").replace("/", "_")
    digest = hashlib.sha256(blob).hexdigest()[:12]
    stem = "_".join(part for part in (label, parent) if part) or "ref"
    return f"af_{stem}_{digest}.png"


def bind_reference_image(payload: dict, template: dict, router) -> str:
    """Upload the parent locked sheet to Comfy and return its input filename.

    Fail closed for kinds whose identity comes from a parent: before this the
    bytes were dropped and the card drew a brand-new character.
    """
    blob = _parent_png_bytes(payload)
    kind = str(payload.get("_kind") or "")
    prompt = template.get("prompt", template)
    has_node = bool(_reference_node_ids(prompt)) if isinstance(prompt, dict) else False
    if blob is None:
        if kind in REFERENCE_REQUIRED_KINDS:
            raise RuntimeError(
                f"{kind} still has no parent sheet bytes; refusing to draw an unconditioned character"
            )
        return ""
    if not has_node:
        raise RuntimeError(
            f"{kind or 'derive'} still carries a parent reference but {still_workflow_name()} has no "
            "IPAdapter node; refusing to silently drop character identity"
        )
    name = _reference_name(payload, blob)
    result = router.upload_image(name, blob)
    if isinstance(result, dict):
        name = str(result.get("name") or name)
    return name


def generate_via_comfy(
    payload: dict,
    *,
    router,
    load_workflow: Callable[[str], dict] | None = None,
    poll_s: float = 2.0,
    timeout_s: float = 300.0,
) -> bytes:
    from gpu_worker.h3 import load_workflow as _load

    from gpu_worker.comfy import extract_execution_error, format_execution_error

    loader = load_workflow or _load
    template = loader(still_workflow_name())
    spec = still_payload_to_prompt(payload)
    spec["reference_image"] = bind_reference_image(payload, template, router)
    graph = fill_still_workflow(template, spec)
    prompt_id = router.prompt(graph.get("prompt", graph), client_id="anime-stills")
    deadline = time.time() + timeout_s
    history: dict = {}
    while time.time() < deadline:
        history = router.history(prompt_id) or {}
        exec_err = extract_execution_error(history, prompt_id)
        if exec_err:
            raise RuntimeError(format_execution_error(exec_err, "still"))
        body = history.get(prompt_id) if prompt_id in history else history
        outputs = (body or {}).get("outputs") if isinstance(body, dict) else None
        if outputs:
            break
        time.sleep(poll_s)
    else:
        raise TimeoutError(f"still comfy timed out prompt_id={prompt_id}")
    body = history.get(prompt_id) if prompt_id in history else history
    outputs = (body or {}).get("outputs") or {}
    filename = None
    subfolder = ""
    for node_out in outputs.values():
        images = (node_out or {}).get("images") or []
        if images:
            filename = images[0].get("filename")
            subfolder = images[0].get("subfolder") or ""
            break
    if not filename:
        raise RuntimeError(f"still comfy produced no image: keys={list(outputs)[:8]}")
    blob = router.view(filename, subfolder=subfolder, type_="output")
    if not blob:
        raise RuntimeError("still comfy /view empty")
    return blob


def unload_still_models(router) -> dict:
    """Drop still weights (Animagine or Kolors+ChatGLM) before H3 on the 32GB card."""
    if router is None:
        return {"ok": False, "error": "no_router"}
    free = getattr(router, "free", None)
    if free is None:
        return {"ok": False, "error": "no_free"}
    try:
        return free(unload_models=True, free_memory=True) or {"ok": True}
    except Exception as exc:  # noqa: BLE001 — anim can still try; Comfy may LRU-evict
        return {"ok": False, "error": str(exc)}


def generate_still(payload: dict, router=None) -> bytes:
    from anime_factory.design import assert_no_reference_images, assert_still_blob

    if payload.get("_kind") in {"scene_plate", "scene_derive"}:
        assert_no_reference_images({k: v for k, v in payload.items() if not str(k).startswith("_")})
        if payload.get("_parent_png"):
            raise RuntimeError("scene plate must not bind a parent image (reference-image bleed)")
    if router is None:
        raise RuntimeError("GPU 生图 needs Comfy anime SDXL on this card (no SiliconFlow fallback on Vast)")
    width, height = parse_size(payload.get("image_size"))
    blob = generate_via_comfy(payload, router=router)
    return assert_still_blob(
        blob,
        width=width,
        height=height,
        label=str(payload.get("_label") or payload.get("_kind") or "gpu_still"),
    )

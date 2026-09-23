"""Anime SDXL stills: IP-Adapter identity binding and fail-closed generation.

Before this, a `costume_derive` uploaded no reference at all — the parent sheet
bytes were assembled and then dropped, so the card drew a brand-new character
and the "locked identity" was a filename.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anime_factory.design import StillBlobError, assert_still_blob, synthetic_still_png
from anime_factory.models import IMAGE_CKPT, LUMINOUS_CINEMATIC_ANIME_PRESET, MIN_STILL_BYTES, STYLE_PREFIX
from gpu_worker.h3 import load_workflow
from gpu_worker.stills import (
    STILL_WORKFLOW,
    bind_reference_image,
    fill_still_workflow,
    generate_still,
    still_payload_to_prompt,
)


@pytest.fixture(autouse=True)
def _animagine_still_backend(monkeypatch):
    """This module locks the Animagine workflow. Kolors has its own tests."""
    monkeypatch.setenv("STILL_BACKEND", "animagine")


class FakeRouter:
    def __init__(self, blob: bytes | None = None):
        self.uploads: list[tuple[str, int]] = []
        self.graphs: list[dict] = []
        self.blob = blob if blob is not None else synthetic_still_png(1344, 768, tag="router")

    def upload_image(self, name, blob):
        self.uploads.append((name, len(blob)))
        return {"name": name}

    def prompt(self, graph, client_id=None):
        self.graphs.append(graph)
        return "pid-1"

    def history(self, prompt_id):
        return {prompt_id: {"outputs": {"7": {"images": [{"filename": "anime_still_0001.png", "subfolder": ""}]}}}}

    def view(self, filename, subfolder="", type_="output"):
        return self.blob


def _template():
    return load_workflow(STILL_WORKFLOW)


def test_gpu_worker_dockerfile_pins_full_shas_and_fetch_checkout():
    dockerfile = (Path(__file__).resolve().parents[1] / "deploy/gpu-worker/Dockerfile").read_text(encoding="utf-8")
    comfy_ref = next(line for line in dockerfile.splitlines() if line.startswith("ARG COMFYUI_REF="))
    ip_ref = next(line for line in dockerfile.splitlines() if line.startswith("ARG IPADAPTER_PLUS_REF="))
    for line in (comfy_ref, ip_ref):
        sha = line.split("=", 1)[1].strip()
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
    assert "git -C /opt/ComfyUI fetch --depth 1 origin" in dockerfile
    assert "git -C /opt/ComfyUI checkout --detach FETCH_HEAD" in dockerfile
    assert "git -C /opt/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus fetch --depth 1 origin" in dockerfile


def test_ipadapter_weight_and_end_at_are_pinned():
    ipadapter = next(
        n for n in load_workflow(STILL_WORKFLOW)["prompt"].values() if n.get("class_type") == "IPAdapterAdvanced"
    )
    assert ipadapter["inputs"]["weight"] == 0.6
    assert ipadapter["inputs"]["end_at"] == 0.7


def test_view_derive_raises_ipadapter_weight_and_end_at():
    spec = still_payload_to_prompt(
        {
            "prompt": "1boy, navy jacket, from side",
            "_kind": "character_view_derive",
            "image_size": "832x1216",
            "reference_image": "parent.png",
        }
    )
    prompt = fill_still_workflow(_template(), spec)["prompt"]
    ipadapter = next(n for n in prompt.values() if n.get("class_type") == "IPAdapterAdvanced")
    assert ipadapter["inputs"]["weight"] == 0.75
    assert ipadapter["inputs"]["end_at"] == 0.9


def test_reference_image_is_uploaded_and_bound_to_ipadapter():
    parent = synthetic_still_png(832, 1216, tag="parent-sheet")
    router = FakeRouter()
    payload = {
        "prompt": "rain-soaked coat, same face",
        "_kind": "costume_derive",
        "_parent_id": "char_ke_sheet",
        "_parent_png": parent,
        "image_size": "832x1216",
    }
    template = _template()
    name = bind_reference_image(payload, template, router)
    assert name.endswith(".png")
    assert router.uploads == [(name, len(parent))]
    spec = still_payload_to_prompt(payload)
    spec["reference_image"] = name
    prompt = fill_still_workflow(template, spec)["prompt"]
    loader = next(n for n in prompt.values() if n.get("class_type") == "LoadImage")
    assert loader["inputs"]["image"] == name
    ipadapter = next(n for n in prompt.values() if n.get("class_type") == "IPAdapterAdvanced")
    sampler = next(n for n in prompt.values() if n.get("class_type") == "KSampler")
    # The sampler must take the IP-Adapter-patched model, not the bare checkpoint.
    assert sampler["inputs"]["model"][0] == next(
        nid for nid, n in prompt.items() if n.get("class_type") == "IPAdapterAdvanced"
    )
    assert ipadapter["inputs"]["image"][0] == next(
        nid for nid, n in prompt.items() if n.get("class_type") == "LoadImage"
    )


def test_no_reference_prunes_the_ipadapter_chain():
    spec = still_payload_to_prompt({"prompt": "empty night street, anime location background", "image_size": "1344x768"})
    prompt = fill_still_workflow(_template(), spec)["prompt"]
    classes = {n.get("class_type") for n in prompt.values()}
    assert "IPAdapterAdvanced" not in classes
    assert "IPAdapterModelLoader" not in classes
    assert "CLIPVisionLoader" not in classes
    assert "LoadImage" not in classes
    sampler = next(n for n in prompt.values() if n.get("class_type") == "KSampler")
    ckpt_id = next(nid for nid, n in prompt.items() if n.get("class_type") == "CheckpointLoaderSimple")
    assert sampler["inputs"]["model"] == [ckpt_id, 0]
    assert prompt[ckpt_id]["inputs"]["ckpt_name"] == IMAGE_CKPT


def test_derive_without_reference_fails_closed():
    router = FakeRouter()
    with pytest.raises(RuntimeError) as exc:
        bind_reference_image({"_kind": "costume_derive", "prompt": "same face"}, _template(), router)
    assert "unconditioned" in str(exc.value)
    assert router.uploads == []


def test_character_view_derive_without_reference_fails_closed():
    router = FakeRouter()
    with pytest.raises(RuntimeError) as exc:
        bind_reference_image({"_kind": "character_view_derive", "prompt": "side profile"}, _template(), router)
    assert "unconditioned" in str(exc.value)
    assert router.uploads == []


def test_reference_without_an_ipadapter_node_fails_closed():
    """A workflow with no reference node must not silently drop identity."""
    template = _template()
    prompt = template["prompt"]
    for nid in [nid for nid, n in prompt.items() if "IPAdapter" in (n.get("class_type") or "")]:
        prompt.pop(nid)
    with pytest.raises(RuntimeError) as exc:
        bind_reference_image(
            {"_kind": "costume_derive", "_parent_png": synthetic_still_png(832, 1216, tag="p")},
            template,
            FakeRouter(),
        )
    assert "IPAdapter" in str(exc.value)


def test_scene_plate_never_binds_a_parent_image():
    with pytest.raises(RuntimeError) as exc:
        generate_still(
            {"_kind": "scene_plate", "prompt": "night street plate", "_parent_png": b"x" * 20_000},
            router=FakeRouter(),
        )
    assert "reference-image bleed" in str(exc.value)


def test_generate_still_validates_the_returned_blob():
    payload = {"prompt": "night street, anime location background", "image_size": "1344x768", "_styled": True}
    ok = generate_still(payload, router=FakeRouter())
    assert len(ok) >= MIN_STILL_BYTES
    with pytest.raises(StillBlobError):
        generate_still(payload, router=FakeRouter(blob=b"PNG"))
    with pytest.raises(StillBlobError) as exc:
        generate_still(payload, router=FakeRouter(blob=synthetic_still_png(1024, 1024, tag="square")))
    assert "1024x1024" in str(exc.value)


def test_styled_marker_stops_double_prefixing():
    styled = still_payload_to_prompt({"prompt": f"{STYLE_PREFIX}, a night counter", "_styled": True})
    assert styled["prompt"].count("original anime production still") == 1
    # A story-specific bible prefix does not start with STYLE_PREFIX, and the old
    # `prompt.startswith(STYLE_PREFIX[:20])` guess double-prefixed exactly there.
    bible = still_payload_to_prompt({"prompt": "salt-bleached harbour palette, a night counter", "_styled": True})
    assert not bible["prompt"].startswith(STYLE_PREFIX)
    unstyled = still_payload_to_prompt({"prompt": "a night counter"})
    assert unstyled["prompt"].startswith(STYLE_PREFIX)


def test_luminous_preset_reaches_unstyled_gpu_location_and_keyframe_payloads(monkeypatch):
    monkeypatch.setenv("STYLE_PRESET", LUMINOUS_CINEMATIC_ANIME_PRESET)
    scene = still_payload_to_prompt({"prompt": "rainy alley", "_kind": "scene_plate"})
    keyframe = still_payload_to_prompt({"prompt": "rainy alley first frame", "_kind": "keyframe"})
    character = still_payload_to_prompt(
        {"prompt": "1boy, navy jacket, from side", "_kind": "character_view_derive", "image_size": "832x1216"}
    )
    assert "clear luminous atmosphere" in scene["prompt"]
    assert "clear luminous atmosphere" in keyframe["prompt"]
    assert "rainy alley" in scene["prompt"]
    assert "rainy alley first frame" in keyframe["prompt"]
    assert "warm ivory studio background" in character["prompt"]
    assert "clear luminous atmosphere" not in character["prompt"]


def test_still_master_uses_animagine_sdxl_bucket():
    """Location/keyframe masters use the Animagine XL 1344×768 bucket before H3 crop."""
    from anime_factory.models import STILL_HEIGHT, STILL_WIDTH

    assert (STILL_WIDTH, STILL_HEIGHT) == (1344, 768)
    assert STILL_WIDTH % 32 == 0 and STILL_HEIGHT % 32 == 0


def test_still_spec_defaults_are_real_cfg_and_still_size():
    spec = still_payload_to_prompt({"prompt": "a night counter"})
    assert (spec["steps"], spec["cfg"]) == (28, 5.0)
    assert (spec["width"], spec["height"]) == (1344, 768)
    assert (spec["sampler"], spec["scheduler"]) == ("euler_ancestral", "normal")
    # 8 steps was a Flux schnell number; it must not drag SDXL back down.
    assert still_payload_to_prompt({"prompt": "x", "num_inference_steps": 8})["steps"] == 28


def test_router_prompt_is_serialized_by_max_concurrent(monkeypatch):
    """COMFYUI_MAX_CONCURRENT used to be config nobody read.

    POSTMORTEM_CONCURRENT_COMFYUI_CROSS_CONTAMINATION: two jobs on one backend
    both reported success and came back matched to the wrong prompts.
    """
    from gpu_worker import router as router_mod

    monkeypatch.setenv("COMFYUI_MAX_CONCURRENT", "1")
    router_mod.reset_prompt_gate()
    assert router_mod.max_concurrent() == 1
    gate = router_mod.prompt_gate()
    assert gate.acquire(timeout=0.1)
    try:
        assert not gate.acquire(timeout=0.05), "a second concurrent /prompt must wait"
    finally:
        gate.release()
    assert gate.acquire(timeout=0.1)
    gate.release()
    router_mod.reset_prompt_gate()


def _handler_for(path: str, body: bytes = b"{}"):
    """A RouterHandler with just enough plumbing for do_POST, no socket."""
    from gpu_worker.router import RouterHandler

    handler = RouterHandler.__new__(RouterHandler)
    handler.path = path
    handler.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
    handler.rfile = _Body(body)
    handler.sent = []
    handler._send = lambda status, data, content_type="application/json": handler.sent.append((status, data))
    return handler


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, n):
        return self._data[:n]


def test_router_returns_503_when_busy(monkeypatch):
    from gpu_worker import router as router_mod

    monkeypatch.setenv("COMFYUI_MAX_CONCURRENT", "1")
    monkeypatch.setattr(router_mod, "PROMPT_WAIT_S", 0.05)
    router_mod.reset_prompt_gate()
    forwarded: list = []
    monkeypatch.setattr(router_mod, "_forward", lambda *a, **k: (forwarded.append(a), (200, b"{}", "application/json"))[1])
    held = router_mod.prompt_gate()
    assert held.acquire(timeout=0.1)
    busy = _handler_for("/prompt")
    try:
        busy.do_POST()
    finally:
        held.release()
    status, data = busy.sent[0]
    assert status == 503
    assert json.loads(data)["error"] == "comfy_router_busy"
    assert forwarded == []

    # Once the slot frees up the same request goes through.
    free = _handler_for("/prompt")
    free.do_POST()
    assert free.sent[0][0] == 200
    assert len(forwarded) == 1
    router_mod.reset_prompt_gate()


def test_router_does_not_gate_uploads(monkeypatch):
    """Uploading a reference image must not wait behind a running sample."""
    from gpu_worker import router as router_mod

    monkeypatch.setenv("COMFYUI_MAX_CONCURRENT", "1")
    monkeypatch.setattr(router_mod, "PROMPT_WAIT_S", 0.05)
    router_mod.reset_prompt_gate()
    monkeypatch.setattr(router_mod, "_forward", lambda *a, **k: (200, b"{}", "application/json"))
    held = router_mod.prompt_gate()
    assert held.acquire(timeout=0.1)
    upload = _handler_for("/upload/image", body=b"multipart")
    try:
        upload.do_POST()
    finally:
        held.release()
        router_mod.reset_prompt_gate()
    assert upload.sent[0][0] == 200


def test_assert_still_blob_accepts_offline_placeholder_only_offline():
    art = synthetic_still_png(1344, 768, tag="dry-run", placeholder=True)
    assert assert_still_blob(art, width=1344, height=768, allow_placeholder=True) == art
    with pytest.raises(StillBlobError):
        assert_still_blob(art, width=1344, height=768)


def test_a_live_run_redraws_a_dry_run_placeholder(tmp_path):
    """A dry run writes tagged placeholders; a GPU run must not accept them as art."""
    from anime_factory.db import migrate, open_db
    from anime_factory.design import KolorsClient
    from anime_factory.keyframe import ensure_keyframe, keyframe_file_ok

    dest = tmp_path / "episodes" / "EP001" / "keyframes" / "s001" / "f1.png"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(synthetic_still_png(1344, 768, tag="dry-run", placeholder=True))
    assert keyframe_file_ok(dest)
    assert not keyframe_file_ok(dest, allow_placeholder=False)

    conn = open_db(tmp_path / "s.sqlite")
    migrate(conn)
    art = synthetic_still_png(1344, 768, tag="gpu")
    client = KolorsClient(["k"], live=True, gpu_generate=lambda p: art)
    from anime_factory.visual_qc import reset_clip_scorer, set_clip_scorer
    from tests.clip_fakes import passing_scorer

    token = set_clip_scorer(passing_scorer())
    try:
        ensure_keyframe(
            conn,
            "story-x",
            "EP001",
            {"id": "s001", "prompt": "MCU at a night counter, living skin"},
            {},
            {"items": []},
            {},
            client,
            None,
            "fiction",
            tmp_path,
        )
    finally:
        reset_clip_scorer(token)
    assert dest.read_bytes() == art

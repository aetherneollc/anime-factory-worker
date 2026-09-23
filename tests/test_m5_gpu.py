from datetime import timedelta

from anime_factory.instrument import Counters
from gpu_worker.h3 import prune_empty_last_frame
from gpu_worker.registry import GpuRegistry
from gpu_worker.scheduler import ensure_capacity
from gpu_worker.vast_client import VastClient, VastSafetyError


def test_register_heartbeat_state_machine():
    reg = GpuRegistry()
    w = reg.register("inst-1", gpu_type="RTX_5090")
    assert w.status == "ready"
    reg.heartbeat("inst-1", "busy")
    assert reg.workers["inst-1"].status == "busy"
    reg.heartbeat("inst-1", "idle")
    assert reg.workers["inst-1"].status == "idle"


def test_no_tasks_gpu_workers_count_zero():
    client = VastClient(dry_run=True)
    reg = GpuRegistry(vast=client)
    reg.register("inst-1")
    assert reg.scale_for_tasks() == 0
    assert len(reg.workers) == 0


def test_empty_last_frame_pruned():
    wf = {
        "prompt": {
            "1": {"class_type": "LoadImage", "_meta": {"role": "first_frame"}, "inputs": {"image": "f1.png"}},
            "2": {"class_type": "LoadImage", "_meta": {"role": "last_frame"}, "inputs": {"image": "input/"}},
            "3": {"class_type": "H3", "inputs": {"last_frame": ["2", 0], "first_frame": ["1", 0]}},
        }
    }
    pruned = prune_empty_last_frame(wf)
    prompt = pruned["prompt"]
    assert "2" not in prompt
    assert "last_frame" not in prompt["3"]["inputs"]
    # no LoadImage of a directory remains
    for node in prompt.values():
        if node.get("class_type") == "LoadImage":
            assert not str(node.get("inputs", {}).get("image", "")).endswith("/")


def test_h3_length_8s_is_192():
    from gpu_worker.h3 import h3_length_frames

    assert h3_length_frames(8) == 192
    assert h3_length_frames(8) % 17 == 5
    assert h3_length_frames(5.0) == 124 or h3_length_frames(5.0) % 17 == 5


def test_prepare_workflow_uses_native_h3_nodes():
    from gpu_worker.h3 import prepare_workflow

    wf = prepare_workflow({"id": "E01-01", "duration": 8, "first_frame_path": "E01-01_f1.png"})
    prompt = wf["prompt"]
    types = {n["class_type"] for n in prompt.values() if isinstance(n, dict)}
    assert "MiniMaxH3ImageToVideo" in types
    assert "MiniMaxH3SigmaShift" in types
    assert "MiniMaxH3" not in types
    cond = next(n for n in prompt.values() if n.get("class_type") == "MiniMaxH3ImageToVideo")
    assert cond["inputs"]["length"] == 192
    assert cond["inputs"]["first_frame"] == ["5", 0]
    load = next(n for n in prompt.values() if n.get("class_type") == "LoadImage")
    assert load["inputs"]["image"] == "E01-01_f1.png"


def test_character_shot_stays_ref2va_with_keyframe():
    from gpu_worker.h3 import prepare_workflow, select_mode

    seg = {
        "id": "E01-01",
        "duration": 8,
        "first_frame_path": "E01-01_f1.png",
        "h3_mode": "ref2va",
        "character_id": "ke",
        "refs": ["assets/characters/ke/sheet_front.png"],
    }
    assert select_mode(seg) == "ref2va"
    types = {n["class_type"] for n in prepare_workflow(seg)["prompt"].values()}
    assert "MiniMaxH3ReferenceToVideo" in types
    assert "MiniMaxH3ImageToVideo" not in types


def test_ref2va_uses_dotted_autogrow_keys():
    from gpu_worker.h3 import native_h3_graph

    graph = native_h3_graph(
        {
            "id": "E01-01",
            "duration": 8,
            "first_frame_path": "r.png",
            "refs": ["sheet_front.png", "sheet_side.png"],
        },
        "ref2va",
    )
    cond = graph["6"]
    assert cond["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert "ref_images" not in cond["inputs"]
    assert "ref_image_1" not in cond["inputs"]
    assert "ref_image_2" not in cond["inputs"]
    assert cond["inputs"]["ref_images.ref_image_0"] == ["r1", 0]
    assert cond["inputs"]["ref_images.ref_image_1"] == ["r2", 0]
    assert graph["r1"]["inputs"]["image"] == "sheet_front.png"
    assert graph["r2"]["inputs"]["image"] == "sheet_side.png"


def test_ref2va_without_first_frame_has_no_dummy_f1():
    from gpu_worker.h3 import native_h3_graph

    graph = native_h3_graph(
        {
            "id": "E01-01",
            "duration": 8,
            "refs": ["sheet_front.png"],
        },
        "ref2va",
    )
    images = [
        node["inputs"]["image"]
        for node in graph.values()
        if isinstance(node, dict) and node.get("class_type") == "LoadImage"
    ]
    assert images == ["sheet_front.png"]
    assert "f1.png" not in images
    assert "5" not in graph


def test_h3_spatial_size_primary_canvas(monkeypatch):
    from gpu_worker.h3 import h3_spatial_size, prepare_workflow

    monkeypatch.delenv("H3_WIDTH", raising=False)
    monkeypatch.delenv("H3_HEIGHT", raising=False)
    w, h = h3_spatial_size({"width": 1344, "height": 768})
    assert (w, h) == (1024, 576)
    wf = prepare_workflow({"id": "E01-01", "duration": 8, "first_frame_path": "f.png", "width": 1344, "height": 768})
    cond = next(n for n in wf["prompt"].values() if n.get("class_type") == "MiniMaxH3ImageToVideo")
    assert cond["inputs"]["width"] == w
    assert cond["inputs"]["height"] == h
    types = {n.get("class_type") for n in wf["prompt"].values() if isinstance(n, dict)}
    assert "MiniMaxLowVRAMAttention" in types
    assert "MiniMaxChunkFeedForward" in types


def test_i2v_graph_skips_audio_vae():
    from gpu_worker.h3 import prepare_workflow

    types = {n["class_type"] for n in prepare_workflow({"id": "E01-01", "duration": 8, "first_frame_path": "f.png"})["prompt"].values()}
    assert "VAEDecodeAudio" not in types
    create = next(n for n in prepare_workflow({"id": "E01-01", "duration": 8, "first_frame_path": "f.png"})["prompt"].values() if n.get("class_type") == "CreateVideo")
    assert "audio" not in create["inputs"]


def test_ref2va_graph_skips_h3_speech():
    from gpu_worker.h3 import native_h3_graph

    graph = native_h3_graph(
        {
            "id": "E01-01",
            "duration": 8,
            "refs": ["sheet_front.png", "plate_store.png"],
            "character_id": "ke",
        },
        "ref2va",
    )
    types = {n["class_type"] for n in graph.values() if isinstance(n, dict)}
    assert "VAEDecodeAudio" not in types
    assert "audio" not in graph["14"]["inputs"]
    assert "audio_vae" not in graph["6"]["inputs"]
    # Bind order: character locks (board sheet + character_id token) then scene plate.
    assert graph["r1"]["inputs"]["image"] == "sheet_front.png"
    assert graph["r2"]["inputs"]["image"] == "char_ke_sheet.png"
    assert graph["r3"]["inputs"]["image"] == "plate_store.png"


def test_oom_fallback_canvas_aligns_32(monkeypatch):
    from gpu_worker.h3 import h3_spatial_size

    monkeypatch.delenv("H3_WIDTH", raising=False)
    monkeypatch.delenv("H3_HEIGHT", raising=False)
    w, h = h3_spatial_size({"width": 1344, "height": 768}, oom_fallback=True)
    assert w % 32 == 0 and h % 32 == 0
    assert (w, h) == (864, 480)


def test_upload_image_is_multipart():
    from gpu_worker.comfy import ComfyRouter

    seen = {}

    def opener(req):
        seen["ctype"] = req.get_header("Content-type")
        seen["data"] = req.data
        return {"name": "x.png"}

    router = ComfyRouter("http://127.0.0.1:8199", opener=opener)
    router.upload_image("x.png", b"PNGDATA")
    assert seen["ctype"] and "multipart/form-data" in seen["ctype"]
    assert b"PNGDATA" in seen["data"]
    assert b"x.png" in seen["data"]
    assert b"application/json" not in (seen["ctype"] or "").encode()


def test_vast_dry_run_does_not_put_asks():
    Counters.reset()
    puts = []

    def opener(req):
        if req.get_method() == "PUT" and "/asks/" in req.full_url:
            puts.append(req.full_url)
        return {"offers": [{"id": "9"}]}

    client = VastClient("fake", opener=opener, dry_run=True)
    reg = GpuRegistry(vast=client)
    out = ensure_capacity(reg, client, need=1)
    assert out["put_asks"] is False
    assert out["action"] == "dry_run_skip_lease"
    assert puts == []
    assert Counters.vast_asks_put == 0
    assert not any(m == "PUT" and "/asks/" in u for m, u in client.calls)


def test_instance_lookup_uses_v0():
    client = VastClient(dry_run=True)
    url = client.instance_lookup_url("12345")
    assert "/api/v0/instances/12345/" in url
    assert "/api/v1/" not in url
    client.get_instance("12345")
    assert any("/api/v0/instances/12345/" in u for _, u in client.calls)


def test_vast_search_sends_dict_filters_mb():
    seen = {}

    def opener(req):
        import json as _json

        seen["body"] = _json.loads(req.data.decode())
        return {"offers": [{"id": "9", "gpu_name": "RTX 4090", "gpu_ram": 24576, "dph_total": 0.4}]}

    client = VastClient("fake", opener=opener, dry_run=False)
    client.search_offers()
    assert seen["body"]["gpu_ram"]["gte"] == 32000
    assert "q" not in seen["body"]


def test_destroy_only_registered_ids():
    client = VastClient(dry_run=True)
    try:
        client.destroy("not-in-table", registered_ids={"inst-1"})
        raise AssertionError("should refuse")
    except VastSafetyError:
        pass
    client.destroy("inst-1", registered_ids={"inst-1"})


def test_empty_establishing_is_fl2va_not_ref2va():
    from gpu_worker.h3 import select_mode

    seg = {
        "id": "s001",
        "duration": 5,
        "h3_mode": "fl2va_first",
        "plate_id": "plate_office",
        "refs": ["plate_office"],
    }
    assert select_mode(seg) == "fl2va_first"
    char = {
        "id": "s002",
        "duration": 8,
        "character_id": "hero",
        "h3_mode": "ref2va",
        "refs": ["char_hero_sheet"],
    }
    assert select_mode(char) == "ref2va"


def test_run_anim_skip_existing_uploads_to_r2(tmp_path, monkeypatch):
    from gpu_worker import session

    calls = []
    monkeypatch.setattr(session, "put_file", lambda key, path, ctype="": calls.append((key, path.name, ctype)) or {"ok": True})
    monkeypatch.setattr(session, "mark_completed_passing", lambda *a, **k: None)
    monkeypatch.setattr(session, "checkpoint_and_upload", lambda *a, **k: None)
    monkeypatch.setattr(session, "_board_shots", lambda root: [{"id": "E01-01", "duration": 8}])
    dest = tmp_path / "shots" / "E01-01" / "v001.mp4"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"x" * 5000)
    # Orphan mp4 without a QC-pass row must not auto-skip; only a passing
    # generation resume uploads and counts as skipped_existing.
    monkeypatch.setattr(
        session,
        "select_passing_generation",
        lambda _conn, root, sid: (root / "shots" / sid / "v001.mp4", 1, "pass"),
    )
    monkeypatch.setattr(session, "_ensure_last_frame", lambda *_a, **_k: None)
    out = session.run_anim("story-x", tmp_path, object(), router=None)
    assert out["skipped_existing"] == ["E01-01"]
    assert out["shots_done"] == 1
    assert calls and calls[0][0] == "stories/story-x/shots/E01-01/v001.mp4"
    assert calls[0][2] == "video/mp4"

"""Historical Vast boot failures encoded as fixtures (not markdown)."""

from pathlib import Path

from gpu_worker.boot import (
    DESTROY_REASONS,
    GPU_ALLOWED_STAGES,
    PRE_LEASE_STAGES,
    VOICE_UPLOAD_PATH,
    WHY_MULTI_CARD,
    WRONG_VOICE_UPLOAD_PATH,
    DestroyRefused,
    ImageCapabilityError,
    LeaseSession,
    OneLeaseError,
    PreLeaseError,
    ReadyProbe,
    assert_gpu_session_only,
    assert_lease_capabilities,
    assert_pre_lease,
    bash_health_script_is_unsafe,
    cap_wait_no_sku_flip,
    decide_probes,
    destroy_if_allowed,
    evaluate_readiness,
    health_wait_loop,
    may_destroy,
    production_health_wait_seconds,
    request_lease,
    retry_install_same_instance,
    vast_payload_not_ready_reason,
    voice_upload_url_ok,
)
from gpu_worker.images import (
    AGENT_IMAGE,
    LEGACY_AGENT_ONLY_IMAGE,
    image_can_lease,
    image_capabilities,
    should_probe_comfy,
)
from gpu_worker.offers import pick_one_offer, search_payload
from gpu_worker.vast_client import VastClient
from gpu_worker.weights import (
    H3_AND_KOLORS_FILES,
    ensure_weights,
    extra_model_paths_yaml,
    hf_local_dir_for,
    materialize_runtime_weights,
)


def _passed(extra: dict | None = None) -> dict[str, str]:
    flags = {s: "passed" for s in PRE_LEASE_STAGES}
    if extra:
        flags.update(extra)
    return flags


def test_tts_before_lease_refused():
    """Failure 1: CosyVoice 404 while a box was already leased."""
    flags = _passed({"tts": "failed"})
    try:
        assert_pre_lease(flags)
        raise AssertionError("must refuse")
    except PreLeaseError as exc:
        assert "tts" in str(exc)
    flags = _passed({"timing": "pending"})
    try:
        assert_pre_lease(flags)
        raise AssertionError("must refuse")
    except PreLeaseError:
        pass
    assert_pre_lease(_passed())
    try:
        assert_gpu_session_only("tts")
        raise AssertionError("tts is not a GPU stage")
    except PreLeaseError:
        pass
    for stage in GPU_ALLOWED_STAGES:
        assert_gpu_session_only(stage)

    client = VastClient(dry_run=True)
    session = LeaseSession()
    out = request_lease(client, "offer-1", session, _passed())
    assert out["put_asks"] is False
    session.last_error = "tts_404"
    try:
        destroy_if_allowed(client, session, "success", {"x"}, error="tts_404")
        raise AssertionError("TTS 404 must not destroy")
    except DestroyRefused:
        pass
    assert not may_destroy("success", "tts_404")
    assert not may_destroy("explicit_abort", "voice_upload_404")
    assert "uploads/audio/voice" in VOICE_UPLOAD_PATH
    assert not voice_upload_url_ok("https://api.siliconflow.cn/v1/audio/voice/upload")
    assert voice_upload_url_ok("https://api.siliconflow.cn" + VOICE_UPLOAD_PATH)
    assert WRONG_VOICE_UPLOAD_PATH not in VOICE_UPLOAD_PATH


def test_running_but_no_ssh_not_ready():
    """Failure 2: KR 5090 class — cur_state=running, actual_status None, empty ports."""
    payload = {"cur_state": "running", "actual_status": None, "ports": {}}
    assert vast_payload_not_ready_reason(payload) == "actual_status_none_empty_ports"
    payload2 = {"cur_state": "running", "actual_status": "running", "ports": {}, "ssh_host": None}
    assert vast_payload_not_ready_reason(payload2) == "running_but_no_ssh"
    probes = ReadyProbe(ssh_echo_ok=False, nvidia_smi_ok=False, comfy_system_stats_ok=False)
    ok, reason = evaluate_readiness(payload, probes, AGENT_IMAGE)
    assert ok is False
    assert reason == "actual_status_none_empty_ports"
    ready_probes = ReadyProbe(ssh_echo_ok=True, nvidia_smi_ok=True, comfy_required=False)
    healthy = {
        "cur_state": "running",
        "actual_status": "running",
        "ssh_host": "1.2.3.4",
        "ssh_port": 22,
        "ports": {"22": 22},
    }
    ok, reason = evaluate_readiness(healthy, ready_probes, LEGACY_AGENT_ONLY_IMAGE)
    assert ok is True
    hub_probes = ReadyProbe(ssh_echo_ok=True, nvidia_smi_ok=True, comfy_system_stats_ok=True)
    ok, reason = evaluate_readiness(healthy, hub_probes, AGENT_IMAGE)
    assert ok is True
    hub_wait = ReadyProbe(ssh_echo_ok=True, nvidia_smi_ok=True, comfy_system_stats_ok=False)
    ok, reason = evaluate_readiness(healthy, hub_wait, AGENT_IMAGE)
    assert ok is False
    assert "comfy_system_stats" in (reason or "")
    session = LeaseSession(sku="RTX_5090")
    cap_wait_no_sku_flip(session)
    assert session.state == "wait_capped"
    # Do not flip SKU — same session, same offer.
    assert session.sku == "RTX_5090"


def test_set_e_health_kill_uses_python_loop():
    """Failure 3: bash set -e + missing seq killed the box after ~12s."""
    killer = "#!/bin/bash\nset -e\nfor i in $(seq 1 30); do nvidia-smi; sleep 1; done\n"
    assert bash_health_script_is_unsafe(killer)
    safe_python = "health_wait_loop(probe, timeout_s=900, production=True)"
    assert not bash_health_script_is_unsafe(safe_python)
    ticks = {"n": 0}

    def probe():
        ticks["n"] += 1
        return ReadyProbe(
            ssh_echo_ok=True,
            nvidia_smi_ok=True,
            comfy_required=False,
            comfy_system_stats_ok=False,
        )

    slept = []
    last = health_wait_loop(probe, timeout_s=0.05, interval_s=0.01, sleep=slept.append, clock=None)
    assert last.ready
    assert ticks["n"] >= 1
    assert production_health_wait_seconds() >= 10 * 60
    assert production_health_wait_seconds() <= 20 * 60
    session = LeaseSession(instance_id="inst-1")
    session.last_error = "health_script_bug"
    client = VastClient(dry_run=True)
    try:
        destroy_if_allowed(client, session, "explicit_abort", {"inst-1"}, error="set_e_health_kill")
        raise AssertionError("health-script bugs must not destroy")
    except DestroyRefused:
        pass


def test_agent_image_without_comfy_skips_8199():
    """Failure 4: legacy agent-only image has no Comfy; do not probe :8199 or lease it."""
    caps = image_capabilities("docker.io/aetherneo/anime-factory-gpu-agent")
    assert caps["comfy"] is False
    assert caps["h3"] is False
    assert caps["kolors"] is False
    assert caps["agent"] is True
    assert image_can_lease(caps) is False
    assert should_probe_comfy(LEGACY_AGENT_ONLY_IMAGE) is False
    probes = decide_probes(LEGACY_AGENT_ONLY_IMAGE)
    assert probes.comfy_required is False
    probes.ssh_echo_ok = True
    probes.nvidia_smi_ok = True
    probes.comfy_system_stats_ok = False
    assert probes.ready is True
    comfy_probes = decide_probes("vast-template/comfy-h3")
    assert comfy_probes.comfy_required is True
    comfy_probes.ssh_echo_ok = True
    comfy_probes.nvidia_smi_ok = True
    comfy_probes.comfy_system_stats_ok = False
    assert comfy_probes.ready is False
    try:
        assert_lease_capabilities(caps)
        raise AssertionError("must refuse comfy:false")
    except ImageCapabilityError:
        pass


def test_hub_image_comfy_h3_kolors_can_lease():
    caps = image_capabilities("docker.io/aetherneo/anime-factory-gpu:main")
    assert caps["comfy"] is True
    assert caps["h3"] is True
    assert caps["kolors"] is True
    assert caps["image_gen"] is True
    assert image_can_lease(caps) is True
    assert image_can_lease(image_capabilities("ghcr.io/example-org/anime-factory-gpu:sha-a2a7ac9")) is True
    assert should_probe_comfy(AGENT_IMAGE) is True
    probes = decide_probes(AGENT_IMAGE)
    assert probes.comfy_required is True
    probes.ssh_echo_ok = True
    probes.nvidia_smi_ok = True
    probes.comfy_system_stats_ok = False
    assert probes.ready is False
    probes.comfy_system_stats_ok = True
    assert probes.ready is True
    assert_lease_capabilities(caps)


def test_refuse_lease_when_image_gen_false():
    try:
        assert_lease_capabilities({"comfy": True, "h3": True, "kolors": False, "image_gen": False})
        raise AssertionError("must refuse")
    except ImageCapabilityError:
        pass
    try:
        assert_lease_capabilities({"comfy": False, "h3": True, "kolors": True})
        raise AssertionError("must refuse")
    except ImageCapabilityError:
        pass
    try:
        assert_lease_capabilities({"comfy": True, "h3": False, "kolors": True})
        raise AssertionError("must refuse")
    except ImageCapabilityError:
        pass
    assert_lease_capabilities({"comfy": True, "h3": True, "image_gen": True})


def test_search_filters_and_default_5090_policy_pick():
    """Search uses gpu_ram MB dict filters; default gpu_model_policy=5090 ignores cheaper 4090."""
    payload = search_payload()
    assert payload["gpu_ram"]["gte"] == 32000
    assert "q" not in payload
    chosen = pick_one_offer(
        [
            {"id": "cheap-8g", "gpu_name": "RTX 3060", "gpu_ram": 8000, "dph_total": 0.1},
            {
                "id": "4090-a",
                "gpu_name": "RTX 4090",
                "gpu_ram": 49152,
                "dph_total": 0.6,
                "duration": 4,
                "inet_down": 400,
                "reliability": 0.99,
                "geolocation": "US",
            },
            {
                "id": "5090-a",
                "gpu_name": "RTX 5090",
                "gpu_ram": 32640,
                "dph_total": 0.9,
                "duration": 4,
                "inet_down": 400,
                "reliability": 0.99,
                "geolocation": "US",
            },
        ]
    )
    assert chosen and str(chosen["id"]) == "5090-a"


def test_one_lease_retry_same_instance_no_replace():
    puts: list[str] = []

    def opener(req):
        puts.append(f"{req.get_method()} {req.full_url}")
        return {"id": "should-not-lease"}

    session = LeaseSession()
    session.instance_id = "49950858"
    session.state = "waiting_comfy"
    try:
        request_lease(VastClient("fake", opener=opener, dry_run=False), "offer-2", session, _passed())
        raise AssertionError("must not lease a second box")
    except OneLeaseError as exc:
        assert "one lease" in str(exc).lower() or "same" in str(exc).lower()
    assert puts == []
    retry_install_same_instance(session)
    assert session.instance_id == "49950858"
    assert session.state == "install_retry"
    assert not may_destroy("success", "install_failure")
    assert may_destroy("idle_ttl", None)
    assert may_destroy("explicit_abort", None)
    assert may_destroy("success", None)
    assert DESTROY_REASONS == {"success", "idle_ttl", "explicit_abort"}
    assert "4 cards" in WHY_MULTI_CARD or "N cards" in WHY_MULTI_CARD


def test_weights_from_hf_not_r2(tmp_path):
    """Boot must pull UNET/CLIP/VAE from HuggingFace onto local disk — never R2."""
    import inspect

    from gpu_worker import weights as weights_mod

    src = inspect.getsource(weights_mod)
    assert "r2_client" not in src
    assert "download_file" not in inspect.getsource(weights_mod.materialize_runtime_weights)
    assert "upload_file" not in inspect.getsource(weights_mod.materialize_runtime_weights)
    assert "weights/" not in inspect.getsource(weights_mod.materialize_runtime_weights)
    assert "_hf_file" in src
    assert "snapshot_download" not in src
    assert all("r2" not in item for item in H3_AND_KOLORS_FILES)
    assert all(item.get("snapshot") != "1" for item in H3_AND_KOLORS_FILES)

    hf_calls: list[str] = []

    def hf(name: str) -> bytes:
        hf_calls.append(name)
        return b"WEIGHT:" + name.encode()

    first = ensure_weights(["h3.safetensors"], tmp_path, hf)
    assert first["misses"]
    assert first["source"] == "huggingface"
    assert first["files"]["h3.safetensors"]["source"] == "huggingface"
    assert (tmp_path / "h3.safetensors").read_bytes() == b"WEIGHT:h3.safetensors"
    second = ensure_weights(["h3.safetensors"], tmp_path, hf)
    assert second["hits"]
    assert second["files"]["h3.safetensors"]["source"] == "local"
    assert hf_calls == ["h3.safetensors"]

    r2_calls: list[str] = []

    def boom_r2(*_a, **_k):
        r2_calls.append("r2")
        raise AssertionError("weights must not touch R2")

    def fake_hf(repo, filename, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if filename == weights_mod.VISUAL_QC_CLIP_WEIGHT_HF:
            dest.write_bytes(b"W" * weights_mod.VISUAL_QC_CLIP_MIN_BYTES)
        elif filename == weights_mod.VISUAL_QC_CLIP_CONFIG_HF:
            dest.write_bytes(b"{" + b"x" * weights_mod.VISUAL_QC_CLIP_CONFIG_MIN_BYTES + b"}")
        else:
            dest.write_bytes(b"HF:" + filename.encode())

    from unittest.mock import patch

    with (
        patch.object(weights_mod, "_hf_file", fake_hf),
        patch("anime_factory.r2_client.download_file", boom_r2),
        patch("anime_factory.r2_client.upload_file", boom_r2),
    ):
        out = materialize_runtime_weights(tmp_path)

    assert r2_calls == []
    assert out["source"] == "huggingface"
    assert "cached_to_r2" not in out
    unet = tmp_path / "models/diffusion_models/minimax_h3_fl2va_pruned_nvfp4.safetensors"
    clip = tmp_path / "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    vae = tmp_path / "models/vae/minimax_h3_video_vae_fp16.safetensors"
    still = tmp_path / weights_mod.KOLORS_UNET_DEST
    ipadapter = tmp_path / weights_mod.KOLORS_IPADAPTER_DEST
    visual_qc = tmp_path / weights_mod.VISUAL_QC_CLIP_WEIGHT_DEST
    assert unet.is_file() and clip.is_file() and vae.is_file() and still.is_file() and ipadapter.is_file()
    assert visual_qc.is_file() and visual_qc.stat().st_size >= weights_mod.VISUAL_QC_CLIP_MIN_BYTES
    weights_mod.validate_visual_qc_clip_weights(tmp_path)
    yaml = extra_model_paths_yaml(tmp_path)
    assert "LLM:" in yaml
    assert "r2://" not in yaml
    assert "weights/" not in yaml
    assert str(tmp_path.resolve()) in yaml
    assert (tmp_path / "extra_model_paths.yaml").is_file()


def test_anime_sdxl_stills_workflow_fill_and_gpu_client(monkeypatch):
    """Stills run anime SDXL at real CFG; cfg=1 on Flux threw FIXED_NEGATIVE away."""
    monkeypatch.setenv("STILL_BACKEND", "animagine")
    from gpu_worker.h3 import load_workflow
    from gpu_worker.stills import STILL_WORKFLOW, fill_still_workflow, still_payload_to_prompt
    from anime_factory.design import KolorsClient, synthetic_still_png

    spec = still_payload_to_prompt(
        {"prompt": "lighthouse dusk", "negative_prompt": "text", "image_size": "1344x768", "seed": 11}
    )
    graph = fill_still_workflow(load_workflow(STILL_WORKFLOW), spec)
    prompt = graph["prompt"]
    texts = [n["inputs"].get("text") for n in prompt.values() if n.get("class_type") == "CLIPTextEncode"]
    assert any("lighthouse" in (t or "") for t in texts)
    assert any(t == "text" for t in texts)
    sampler = next(n for n in prompt.values() if n.get("class_type") == "KSampler")
    assert sampler["inputs"]["steps"] == 28
    assert sampler["inputs"]["cfg"] == 5
    assert sampler["inputs"]["sampler_name"] == "euler_ancestral"
    assert sampler["inputs"]["scheduler"] == "normal"
    latent = next(n for n in prompt.values() if n.get("class_type") == "EmptyLatentImage")
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (1344, 768)
    ckpt = next(n for n in prompt.values() if n.get("class_type") == "CheckpointLoaderSimple")
    assert ckpt["inputs"]["ckpt_name"] == "animagine-xl-4.0.safetensors"
    a = still_payload_to_prompt({"prompt": "CU convenience store"})
    b = still_payload_to_prompt({"prompt": "wide rainy bike lane"})
    assert a["seed"] != 42 or b["seed"] != 42
    assert a["seed"] != b["seed"]
    drawn = []
    art = synthetic_still_png(1344, 768, tag="boot-lesson")

    def gpu_gen(payload):
        drawn.append(payload["prompt"])
        return art

    client = KolorsClient(["k"], live=True, gpu_generate=gpu_gen)
    blob = client.generate({"prompt": "a plate", "negative_prompt": "x"})
    assert blob == art
    assert drawn


def test_lease_body_onstart_starts_agent():
    """ssh runtype replaces the image ENTRYPOINT; onstart must start af-start."""
    body = VastClient("fake", dry_run=True).lease_body()
    assert body["onstart"] == "/usr/local/bin/af-start"
    assert "ssh" in str(body.get("runtype") or "")
    assert "jupyter" not in str(body.get("runtype") or "")
    assert body["image"].endswith("anime-factory-gpu:main")
    assert body["disk"] == 200


def test_hf_local_dir_does_not_nest_under_dest_parent():
    """local_dir=dest.parent doubled 21GB files into diffusion_models/diffusion_models/."""
    dest = Path("/opt/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_nvfp4.safetensors")
    assert hf_local_dir_for(dest) == "/opt/ComfyUI/models"
    dest = Path("/opt/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    assert hf_local_dir_for(dest) == "/opt/ComfyUI/models"
    dest = Path("/opt/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors")
    assert hf_local_dir_for(dest) == "/opt/ComfyUI/models"


def test_ensure_story_db_rebuilds_segments_from_board(tmp_path):
    """Hosted path may not upload story.sqlite; GPU must rebuild from board.json."""
    from gpu_worker.session import ensure_story_db

    root = tmp_path / "story"
    board = root / "episodes" / "EP001" / "board.json"
    board.parent.mkdir(parents=True)
    board.write_text(
        '{"shots": [{"id": "E01-01", "duration": 8}, {"id": "E01-02", "duration": 8}]}',
        encoding="utf-8",
    )
    conn = ensure_story_db("story-x", root)
    rows = conn.execute("SELECT id, duration, status FROM segments ORDER BY seq").fetchall()
    assert [r["id"] for r in rows] == ["E01-01", "E01-02"]
    assert rows[0]["duration"] == 8
    assert rows[0]["status"] == "prepared"
    assert (root / "story.sqlite").is_file()


def test_blackwell_torch_index_is_cu130(monkeypatch):
    """H3 Blackwell production stack uses cu130 wheels."""
    from gpu_worker import stack

    monkeypatch.delenv("TORCH_INDEX_URL", raising=False)
    monkeypatch.delenv("AF_GPU_PROFILE", raising=False)
    assert stack.torch_index_url("sm_120") == stack.TORCH_CU130
    monkeypatch.setenv("AF_GPU_PROFILE", "longlive-nvfp4-sm120")
    assert stack.torch_index_url("sm_120") == stack.TORCH_CU128
    assert stack.torch_index_url("sm_90") == stack.TORCH_CU124
    monkeypatch.setenv("TORCH_INDEX_URL", "https://example.invalid/custom")
    assert stack.torch_index_url("sm_120") == "https://example.invalid/custom"


def test_blackwell_nvidia_smi_fallback_picks_cu130(monkeypatch):
    """Broken torch probe still routes H3 Blackwell to cu130."""
    from gpu_worker import stack

    monkeypatch.delenv("TORCH_INDEX_URL", raising=False)
    monkeypatch.delenv("AF_GPU_PROFILE", raising=False)
    monkeypatch.setattr(stack, "cuda_sm", lambda: None)
    monkeypatch.setattr(stack, "_nvidia_smi_sm", lambda: "sm_120")
    assert stack.torch_index_url("sm_120") == stack.TORCH_CU130
    monkeypatch.setattr(stack, "cuda_sm", lambda: stack._nvidia_smi_sm())
    assert stack.torch_index_url() == stack.TORCH_CU130


def test_kitchen_torch26_rewrites_list_int(tmp_path):
    """Torch 2.6 infer_schema rejects builtin list[int] used by comfy_kitchen."""
    from gpu_worker import stack

    src = tmp_path / "na.py"
    src.write_text(
        "from __future__ import annotations\n\ndef _op(kernel_size: list[int], is_causal: list[bool]):\n    return kernel_size, is_causal\n",
        encoding="utf-8",
    )
    hits = stack.patch_comfy_kitchen_for_torch26(roots=[src])
    assert hits == [str(src)]
    text = src.read_text(encoding="utf-8")
    assert text.index("from __future__") < text.index("import typing")
    assert "typing.List[int]" in text
    assert "typing.List[bool]" in text
    assert "list[int]" not in text


def test_production_stack_forbids_live_torch_upgrade(monkeypatch):
    """Baked production images must not pip install --upgrade torch on mismatch."""
    from gpu_worker import stack
    from gpu_worker.preflight import PreflightFailure

    monkeypatch.setenv("AF_PRODUCTION_STACK", "1")
    monkeypatch.setattr(stack, "torch_supports_device", lambda: False)
    try:
        stack.ensure_torch()
        raise AssertionError("must refuse live torch upgrade")
    except PreflightFailure as exc:
        assert exc.failure_class == "capability_mismatch"
        assert exc.code == "torch_abi_mismatch"


def test_sitecustomize_does_not_import_torch_at_startup():
    """Importing torch in sitecustomize breaks CUDAAllocatorConfig vs Comfy env."""
    from gpu_worker.stack import _SITECUSTOMIZE_INFER_SCHEMA

    preamble = _SITECUSTOMIZE_INFER_SCHEMA.split("def _install", 1)[0]
    assert "\nimport torch" not in preamble
    assert not any(line.strip() == "import torch" for line in preamble.splitlines())
    assert "_AfTorchPatchFinder" in _SITECUSTOMIZE_INFER_SCHEMA


def test_comfy_launch_args_disable_cuda_malloc(monkeypatch):
    """expandable_segments + cudaMallocAsync (cu130 default) SIGSEGVs; Comfy must opt out."""
    from gpu_worker import stack

    monkeypatch.delenv("AF_GPU_PROFILE", raising=False)
    args = stack.comfy_launch_args()
    assert "--disable-cuda-malloc" in args
    assert "--disable-pinned-memory" in args


def test_probe_comfy_torch_import_chdirs_to_comfy_dir(tmp_path, monkeypatch):
    """Vast 51523076: probe chdir'd to main.py (NotADirectoryError) after weights."""
    from gpu_worker import stack

    seen = {}

    class _Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(args, **_k):
        seen["args"] = list(args)
        return _Proc()

    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    (comfy / "main.py").write_text("#\n", encoding="utf-8")
    monkeypatch.setattr(stack, "COMFY_DIR", comfy)
    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    out = stack.probe_comfy_torch_import(comfy / "main.py")
    assert out["ok"] is True
    assert seen["args"][-1] == str(comfy)
    assert seen["args"][-1] != str(comfy / "main.py")


def test_probe_comfy_torch_import_maps_sigsegv_to_preflight(monkeypatch):
    from gpu_worker import stack
    from gpu_worker.preflight import PreflightFailure

    class _Proc:
        returncode = -11
        stdout = ""
        stderr = ""

    monkeypatch.setattr(stack.subprocess, "run", lambda *_a, **_k: _Proc())
    try:
        stack.probe_comfy_torch_import()
        raise AssertionError("must fail closed on SIGSEGV")
    except PreflightFailure as exc:
        assert exc.failure_class == "comfy_startup_failed"
        assert exc.code == "torch_import_sigsegv"
        assert "cudaMallocAsync" in exc.message


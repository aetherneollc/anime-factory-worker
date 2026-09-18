"""GPU canary watchdog — dry-run safety tests (never live PUT/DELETE)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))
if str(_REPO / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "python"))

import gpu_canary as gc
from gpu_worker.vast_client import VastClient, VastSafetyError

DIGEST = "sha256:" + ("c" * 64)
HEX64 = "c" * 64
IMAGE_REF = f"docker.io/aetherneo/anime-factory-gpu@{DIGEST.split(':', 1)[1]}"
IMAGE_AT = f"docker.io/aetherneo/anime-factory-gpu@sha256:{HEX64}"
GHCR_H3_AT = f"ghcr.io/aetherneollc/anime-factory-worker-h3@sha256:{HEX64}"
GHCR_LONGLIVE_AT = f"ghcr.io/aetherneollc/anime-factory-worker-longlive@sha256:{HEX64}"


def _eligible(**partial):
    return {
        "id": "5090-us",
        "gpu_name": "RTX 5090",
        "gpu_ram": 32768,
        "duration": 4,
        "reliability": 0.995,
        "inet_down": 400,
        "inet_up": 200,
        "disk_space": 250,
        "geolocation": "US",
        "dph_total": 0.639,
        **partial,
    }


def _cfg(**overrides) -> gc.CanaryConfig:
    base = {
        "live": False,
        "image_ref": IMAGE_AT,
        "story_id": gc.DEFAULT_STORY_ID,
    }
    base.update(overrides)
    return gc.CanaryConfig(**base)


def test_validate_image_rejects_tag():
    with pytest.raises(gc.CanaryImageError, match="tag"):
        gc.validate_lease_image_ref("docker.io/aetherneo/anime-factory-gpu:main")
    with pytest.raises(gc.CanaryImageError):
        gc.validate_lease_image_ref("docker.io/aetherneo/anime-factory-gpu@sha256:abc")


def test_validate_image_accepts_digest_pin():
    ref, digest = gc.validate_lease_image_ref(IMAGE_AT)
    assert digest == DIGEST
    assert "@sha256:" in ref


def test_validate_image_accepts_ghcr_h3_digest():
    ref, digest = gc.validate_lease_image_ref(GHCR_H3_AT, backend="h3")
    assert digest == DIGEST
    assert ref == GHCR_H3_AT


def test_validate_image_accepts_ghcr_longlive_digest():
    ref, digest = gc.validate_lease_image_ref(GHCR_LONGLIVE_AT, backend="longlive")
    assert digest == DIGEST
    assert ref == GHCR_LONGLIVE_AT


def test_validate_image_rejects_backend_mismatch():
    with pytest.raises(gc.CanaryImageError, match="mismatch"):
        gc.validate_lease_image_ref(GHCR_LONGLIVE_AT, backend="h3")
    with pytest.raises(gc.CanaryImageError, match="mismatch"):
        gc.validate_lease_image_ref(GHCR_H3_AT, backend="longlive")
    with pytest.raises(gc.CanaryImageError, match="mismatch"):
        gc.validate_lease_image_ref(IMAGE_AT, backend="longlive")


def test_validate_image_rejects_longlive_tag():
    with pytest.raises(gc.CanaryImageError, match="tag"):
        gc.validate_lease_image_ref(
            "ghcr.io/aetherneollc/anime-factory-worker-longlive:main",
            backend="longlive",
        )


def test_clamp_max_dph_never_above_one(monkeypatch):
    monkeypatch.setenv("VAST_MAX_DPH_USD", "5.0")
    assert gc.clamp_canary_max_dph() == 1.0
    assert gc.clamp_canary_max_dph(2.5) == 1.0
    assert gc.clamp_canary_max_dph(0.5) == 0.5


def test_dry_run_never_puts(monkeypatch):
    puts: list[str] = []

    def opener(req):
        if req.get_method() == "PUT" and "/asks/" in req.full_url:
            puts.append(req.full_url)
        if req.get_method() == "POST" and "/bundles/" in req.full_url:
            return {"offers": [_eligible()]}
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(_cfg(), client=client)
    out = wd.run()
    assert out["action"] == "dry_run_would_lease"
    assert out["put_asks"] is False
    assert puts == []
    assert client.calls == [("POST", client.calls[0][1])]
    assert sum(1 for method, _ in client.calls if method == "PUT") == 0


def test_default_client_allows_real_search_not_dry_run_shortcut():
    wd = gc.CanaryWatchdog(_cfg())
    assert wd.client.dry_run is False


def test_dry_run_returns_before_lease_selected():
    lease_calls: list[str] = []

    class TrackingWatchdog(gc.CanaryWatchdog):
        def lease_selected(self, offer, *, image_ref, lease_env):
            lease_calls.append(str(offer.get("id")))
            return super().lease_selected(offer, image_ref=image_ref, lease_env=lease_env)

    client = VastClient("fake", opener=lambda req: {"offers": [_eligible()]}, dry_run=False)
    wd = TrackingWatchdog(_cfg(), client=client)
    out = wd.run()
    assert out["action"] == "dry_run_would_lease"
    assert lease_calls == []


def test_rejects_non_5090_offer(monkeypatch):
    def opener(req):
        return {
            "offers": [
                _eligible(id="4090", gpu_name="RTX 4090", gpu_ram=49152, dph_total=0.2),
            ]
        }

    client = VastClient("fake", opener=opener, dry_run=True)
    wd = gc.CanaryWatchdog(_cfg(), client=client)
    out = wd.run()
    assert out["action"] == "wait_no_offer"
    assert out["put_asks"] is False


def test_rejects_over_one_dph_offer():
    def opener(req):
        return {"offers": [_eligible(id="pricey", dph_total=1.5, geolocation="Norway")]}

    client = VastClient("fake", opener=opener, dry_run=True)
    wd = gc.CanaryWatchdog(_cfg(max_dph=1.0), client=client)
    out = wd.run()
    assert out["action"] == "wait_no_offer"


def test_no_offers_wait_without_lease():
    client = VastClient("fake", opener=lambda req: {"offers": []}, dry_run=True)
    wd = gc.CanaryWatchdog(_cfg(), client=client)
    out = wd.run()
    assert out["action"] == "wait_no_offer"
    assert out["offer_count"] == 0
    assert "PUT" not in str(client.calls)


def test_destroy_refuses_unregistered_id():
    client = VastClient("fake", dry_run=True)
    with pytest.raises(VastSafetyError, match="unregistered"):
        client.destroy("999", set())


def test_destroy_allowlist_only_registered():
    deleted: list[str] = []

    def opener(req):
        if req.get_method() == "DELETE":
            deleted.append(req.full_url)
        return {"ok": True}

    client = VastClient("fake", opener=opener, dry_run=False)
    allow = {"inst-canary-1"}
    client.destroy("inst-canary-1", allow)
    assert deleted
    with pytest.raises(VastSafetyError):
        client.destroy("other-id", allow)


def test_secret_redaction_in_output():
    env = gc.build_canary_env(image_digest=DIGEST, story_id=gc.DEFAULT_STORY_ID)
    env["R2_SECRET_ACCESS_KEY"] = "supersecretvalue"
    env["HF_TOKEN"] = "hf_abcdefghijklmnopqrstuvwxyz"
    redacted = gc.redact_mapping(env)
    assert "supersecretvalue" not in json.dumps(redacted)
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in json.dumps(redacted)
    assert redacted["R2_SECRET_ACCESS_KEY"].startswith("su")
    assert "***" in redacted["HF_TOKEN"]


def test_live_gate_requires_confirm_and_key(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k-test")
    wd = gc.CanaryWatchdog(_cfg(live=True, confirm="wrong"))
    with pytest.raises(gc.CanaryConfigError, match="confirm"):
        wd.validate_config()
    wd2 = gc.CanaryWatchdog(_cfg(live=True, confirm=gc.LIVE_CONFIRM_PHRASE, api_key=""))
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    with pytest.raises(gc.CanaryConfigError, match="VAST_API_KEY"):
        wd2.validate_config()


def test_finally_destroy_retries_on_failure(monkeypatch):
    attempts = {"n": 0}

    def opener(req):
        if req.get_method() == "DELETE":
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise OSError("network blip")
            return {"ok": True}
        if "/api/v1/instances" in req.full_url:
            return {"instances": []}
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(_cfg(), client=client)
    wd._register_instance("inst-x")
    out = wd.destroy_registered()
    assert out["ok"] is True
    assert out["attempts"] == 3
    assert out.get("confirmed_v1") is True


def test_canary_env_sets_required_keys():
    env = gc.build_canary_env(image_digest=DIGEST, story_id=gc.DEFAULT_STORY_ID)
    assert env["AF_STORY_ID"] == gc.DEFAULT_STORY_ID
    assert env["AF_EPISODE"] == "EP001"
    assert env["AF_ONCE"] == "1"
    assert env["AF_VIDEO_BACKEND"] == "h3"
    assert env["AF_IMAGE_CAPABILITY"] == "h3"
    assert env["AF_GPU_PROFILE"] == gc.CANARY_H3_PROFILE
    assert "AF_VIDEO_BACKEND_LOCKED" not in env
    assert env["VAST_ALLOW_REPLACE"] == "0"
    assert env["AF_EXPECTED_IMAGE_DIGEST"] == DIGEST
    assert env["AF_REPORTED_IMAGE_DIGEST"] == DIGEST


def test_canary_env_longlive_sets_profile_and_lock():
    env = gc.build_canary_env(image_digest=DIGEST, story_id=gc.DEFAULT_STORY_ID, backend="longlive")
    assert env["AF_VIDEO_BACKEND"] == "longlive"
    assert env["AF_VIDEO_BACKEND_LOCKED"] == "longlive"
    assert env["AF_IMAGE_CAPABILITY"] == "longlive"
    assert env["AF_GPU_PROFILE"] == gc.CANARY_LONGLIVE_PROFILE


def test_canary_env_omits_production_keys_even_when_parent_set(monkeypatch):
    monkeypatch.setenv("CONTROL_PLANE_URL", "https://control-plane.example.com")
    monkeypatch.setenv("STUDIO_USER", "studio")
    monkeypatch.setenv("STUDIO_PASSWORD", "secret-pass")
    monkeypatch.setenv("STUDIO_AGENT_KEY", "agent-key")
    monkeypatch.setenv("CLOUDFLARE_TUNNEL_TOKEN", "tunnel-token")
    monkeypatch.setenv("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu128")
    env = gc.build_canary_env(image_digest=DIGEST, story_id=gc.DEFAULT_STORY_ID)
    for key in gc.ISOLATED_ENV_FORBIDDEN_KEYS:
        assert key not in env


def test_clamp_max_spend_never_above_one(monkeypatch):
    monkeypatch.setenv("GPU_CANARY_MAX_SPEND_USD", "15")
    assert gc.clamp_canary_max_spend() == 1.0
    assert gc.clamp_canary_max_spend(5.0) == 1.0
    assert gc.clamp_canary_max_spend(0.5) == 0.5


def test_clamp_max_wall_never_above_ninety_minutes(monkeypatch):
    monkeypatch.setenv("GPU_CANARY_MAX_WALL_MINUTES", "240")
    assert gc.clamp_canary_max_wall_s() == gc.CANARY_MAX_WALL_S
    assert gc.clamp_canary_max_wall_s(120 * 60) == gc.CANARY_MAX_WALL_S
    assert gc.clamp_canary_max_wall_s(30 * 60) == 30 * 60


def test_offer_scoring_uses_higher_budget_than_watchdog():
    def opener(req):
        return {"offers": [_eligible(dph_total=0.639)]}

    client = VastClient("fake", opener=opener, dry_run=True)
    wd = gc.CanaryWatchdog(_cfg(max_spend_usd=1.0), client=client)
    out = wd.run()
    assert out["action"] == "dry_run_would_lease"
    selected = out["selected"]
    assert selected["offer_score_budget_usd"] == gc.CANARY_OFFER_SCORE_BUDGET_USD
    assert selected["watchdog_max_spend_usd"] == 1.0
    assert selected["watchdog_max_wall_s"] == gc.CANARY_MAX_WALL_S
    assert selected["expected_total_usd"] > 1.0


def test_lease_env_includes_vast_watchdog_keys_redacted():
    offer = _eligible(dph_total=0.82)
    base = gc.build_canary_env(image_digest=DIGEST, story_id=gc.DEFAULT_STORY_ID)
    merged = gc.inject_lease_watchdog_env(
        base,
        offer=offer,
        api_key="sk-live-vast-secret-key",
        max_spend_usd=1.0,
        max_wall_s=gc.CANARY_MAX_WALL_S,
    )
    assert merged["VAST_API_KEY"] == "sk-live-vast-secret-key"
    assert merged["VAST_LEASE_HOURLY_USD"] == "0.82"
    assert merged["VAST_MAX_LEASE_MINUTES"] == "90"
    assert merged["VAST_MAX_LEASE_USD"] == "1.0"
    assert merged["VAST_WATCH_USD"] == "1.0"
    redacted = gc.redact_mapping(merged)
    dumped = json.dumps(redacted)
    assert "sk-live-vast-secret-key" not in dumped
    assert "VAST_API_KEY" in dumped


def test_dry_run_output_never_leaks_vast_api_key():
    secret = "sk-live-vast-secret-key"
    client = VastClient("fake", opener=lambda req: {"offers": [_eligible()]}, dry_run=True)
    wd = gc.CanaryWatchdog(_cfg(api_key=secret), client=client)
    result = wd.run()
    dumped = json.dumps(gc.redact_for_log(result))
    assert secret not in dumped
    assert result["lease_env"]["VAST_API_KEY"].startswith("sk")
    assert "***" in result["lease_env"]["VAST_API_KEY"]


def _mock_r2_inventory(keys: dict[str, int]):
    def list_prefix(prefix: str, max_keys: int = 2000):
        out = []
        for key, size in keys.items():
            if key.startswith(prefix):
                out.append({"key": key, "size": size})
        return out[:max_keys]

    return list_prefix


def test_r2_generation_shot_prefix_is_story_scoped():
    sid = gc.DEFAULT_STORY_ID
    assert gc.r2_generation_shot_prefix(sid, "s001") == f"stories/{sid}/shots/s001"
    assert f"stories/{sid}/shots/s001/generation-001.mp4".startswith(
        gc.r2_generation_shot_prefix(sid, "s001")
    )


def test_r2_canary_complete_requires_all_artifacts(monkeypatch):
    sid = gc.DEFAULT_STORY_ID
    episode = "EP001"
    gen_key = f"stories/{sid}/shots/s001/generation-001.mp4"
    inventory = _completion_inventory(sid, episode, gen_key)
    monkeypatch.setattr(
        "anime_factory.r2_client.list_prefix",
        _mock_r2_inventory(inventory),
    )
    assert gc.r2_canary_complete(sid, episode) is True


def test_r2_canary_complete_accepts_generation_002(monkeypatch):
    sid = gc.DEFAULT_STORY_ID
    episode = "EP001"
    gen_key = f"stories/{sid}/shots/s001/generation-002.mp4"
    inventory = _completion_inventory(sid, episode, gen_key)
    monkeypatch.setattr(
        "anime_factory.r2_client.list_prefix",
        _mock_r2_inventory(inventory),
    )
    assert gc.r2_canary_complete(sid, episode) is True


def test_r2_canary_complete_rejects_episode_scoped_generation_path(monkeypatch):
    sid = gc.DEFAULT_STORY_ID
    episode = "EP001"
    inventory = _completion_inventory(sid, episode, f"stories/{sid}/episodes/{episode}/shots/s001/generation-001.mp4")
    wrong_key = f"stories/{sid}/episodes/{episode}/shots/s001/generation-001.mp4"
    inventory.pop(f"stories/{sid}/shots/s001/generation-001.mp4", None)

    monkeypatch.setattr(
        "anime_factory.r2_client.list_prefix",
        _mock_r2_inventory(inventory),
    )
    assert gc.r2_canary_complete(sid, episode) is False


def test_r2_canary_complete_fails_if_any_required_missing(monkeypatch):
    sid = gc.DEFAULT_STORY_ID
    episode = "EP001"
    gen_key = f"stories/{sid}/shots/s001/generation-001.mp4"
    inventory = _completion_inventory(sid, episode, gen_key)
    missing = gc.r2_canary_output_keys(sid, episode)[0]
    inventory.pop(missing)

    monkeypatch.setattr(
        "anime_factory.r2_client.list_prefix",
        _mock_r2_inventory(inventory),
    )
    assert gc.r2_canary_complete(sid, episode) is False


def test_r2_canary_complete_ignores_final_mp4(monkeypatch):
    sid = gc.DEFAULT_STORY_ID
    episode = "EP001"
    gen_key = f"stories/{sid}/shots/s001/generation-001.mp4"
    inventory = _completion_inventory(sid, episode, gen_key)
    final_key = f"stories/{sid}/episodes/{episode}/final/episode.mp4"
    inventory[final_key] = 9000
    inventory.pop(gen_key)

    monkeypatch.setattr(
        "anime_factory.r2_client.list_prefix",
        _mock_r2_inventory(inventory),
    )
    assert gc.r2_canary_complete(sid, episode) is False


def test_cli_main_dry_run_exit_zero(capsys):
    code = gc.main(
        [
            "--image",
            IMAGE_AT,
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True


def test_main_calls_load_dotenv(monkeypatch, capsys):
    called: list[str | None] = []

    def fake_load(path=None):
        called.append(path)
        return None

    monkeypatch.setattr(gc, "load_dotenv", fake_load)
    code = gc.main(["--image", IMAGE_AT])
    assert code == 0
    assert called == [None]
    capsys.readouterr()


def _input_inventory(sid: str, episode: str = "EP001", backend: str = "h3") -> dict[str, int]:
    return {key: 100 for key in gc.r2_canary_input_keys(sid, episode, backend)}


def _completion_inventory(
    sid: str,
    episode: str,
    gen_key: str,
    backend: str = "h3",
) -> dict[str, int]:
    inventory = _input_inventory(sid, episode, backend)
    inventory.update({key: 100 for key in gc.r2_canary_output_keys(sid, episode)})
    inventory[gen_key] = 5000
    return inventory


def test_r2_canary_input_keys_exact_set():
    sid = gc.DEFAULT_STORY_ID
    expected = {
        f"stories/{sid}/bible/period.md",
        f"stories/{sid}/bible/world.md",
        f"stories/{sid}/canon/wiki/characters/hero.md",
        f"stories/{sid}/canon/wiki/locations/loc_dorm.md",
        f"stories/{sid}/episodes/EP001/audio/tts_manifest.json",
        f"stories/{sid}/episodes/EP001/board.json",
        f"stories/{sid}/episodes/EP001/script.json",
    }
    assert set(gc.r2_canary_input_keys(sid, "EP001")) == expected
    assert len(gc.r2_canary_input_keys(sid, "EP001")) == 7


def test_r2_canary_input_keys_longlive_includes_factory_json():
    sid = gc.DEFAULT_STORY_ID
    keys = gc.r2_canary_input_keys(sid, "EP001", backend="longlive")
    assert f"stories/{sid}/factory.json" in keys
    assert len(keys) == 8


def test_r2_canary_complete_longlive_accepts_legacy_v_mp4(monkeypatch):
    sid = gc.DEFAULT_STORY_ID
    episode = "EP001"
    gen_key = f"stories/{sid}/shots/s001/v001.mp4"
    inventory = _completion_inventory(sid, episode, gen_key, backend="longlive")
    monkeypatch.setattr(
        "anime_factory.r2_client.list_prefix",
        _mock_r2_inventory(inventory),
    )
    assert gc.r2_canary_complete(sid, episode, backend="longlive") is True


def test_r2_canary_complete_longlive_rejects_h3_only_generation_without_factory(monkeypatch):
    sid = gc.DEFAULT_STORY_ID
    episode = "EP001"
    gen_key = f"stories/{sid}/shots/s001/generation-001.mp4"
    inventory = _completion_inventory(sid, episode, gen_key, backend="longlive")
    inventory.pop(f"stories/{sid}/factory.json", None)
    monkeypatch.setattr(
        "anime_factory.r2_client.list_prefix",
        _mock_r2_inventory(inventory),
    )
    assert gc.r2_canary_complete(sid, episode, backend="longlive") is False


def test_dry_run_longlive_backend_never_puts(monkeypatch):
    puts: list[str] = []

    def opener(req):
        if req.get_method() == "PUT" and "/asks/" in req.full_url:
            puts.append(req.full_url)
        if req.get_method() == "POST" and "/bundles/" in req.full_url:
            return {"offers": [_eligible()]}
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(
        _cfg(image_ref=GHCR_LONGLIVE_AT, backend="longlive"),
        client=client,
    )
    out = wd.run()
    assert out["action"] == "dry_run_would_lease"
    assert out["put_asks"] is False
    assert puts == []
    assert out["lease_env"]["AF_VIDEO_BACKEND"] == "longlive"
    assert out["lease_env"]["AF_GPU_PROFILE"] == gc.CANARY_LONGLIVE_PROFILE


def test_r2_canary_output_keys_exact_set():
    sid = gc.DEFAULT_STORY_ID
    expected = {
        f"stories/{sid}/assets/index.json",
        f"stories/{sid}/assets/characters/hero/sheet_front.png",
        f"stories/{sid}/assets/characters/hero/sheet_side.png",
        f"stories/{sid}/assets/characters/hero/sheet_back.png",
        f"stories/{sid}/assets/characters/hero/sheet_turnaround.png",
        f"stories/{sid}/assets/scenes/loc_dorm/plate_base.png",
        f"stories/{sid}/episodes/EP001/keyframes/s001/last.png",
        f"stories/{sid}/story.sqlite",
    }
    assert set(gc.r2_canary_output_keys(sid, "EP001")) == expected


def test_preflight_ready_complete_false_with_only_inputs(monkeypatch):
    monkeypatch.setenv("R2_ENDPOINT", "https://r2.example")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(gc, "_r2_probe_boto3", lambda: None)
    sid = gc.DEFAULT_STORY_ID
    monkeypatch.setattr(
        "anime_factory.r2_client.list_prefix",
        _mock_r2_inventory(_input_inventory(sid)),
    )
    preflight = gc.r2_live_preflight(sid)
    assert preflight["ok"] is True
    assert preflight["reason"] == "ready"
    assert gc.r2_canary_complete(sid) is False


def test_r2_preflight_rejects_missing_creds(monkeypatch):
    monkeypatch.delenv("R2_ENDPOINT", raising=False)
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    out = gc.r2_live_preflight(gc.DEFAULT_STORY_ID)
    assert out["ok"] is False
    assert out["reason"] == "r2_creds_missing"


def test_r2_preflight_rejects_existing_outputs(monkeypatch):
    monkeypatch.setenv("R2_ENDPOINT", "https://r2.example")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(gc, "_r2_probe_boto3", lambda: None)
    sid = gc.DEFAULT_STORY_ID
    inputs = {key: 100 for key in gc.r2_canary_input_keys(sid, "EP001")}
    gen_key = f"stories/{sid}/shots/s001/generation-001.mp4"
    inventory = dict(inputs)
    inventory[gen_key] = 9000
    monkeypatch.setattr("anime_factory.r2_client.list_prefix", _mock_r2_inventory(inventory))
    out = gc.r2_live_preflight(sid)
    assert out["ok"] is False
    assert out["reason"] == "r2_outputs_present"


def test_live_preflight_blocks_put(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k-test")
    puts: list[str] = []

    def opener(req):
        if req.get_method() == "PUT":
            puts.append(req.full_url)
        if req.get_method() == "POST" and "/bundles/" in req.full_url:
            return {"offers": [_eligible()]}
        if "/api/v1/instances" in req.full_url:
            return {"instances": []}
        return {}

    monkeypatch.setattr(
        gc,
        "r2_live_preflight",
        lambda story_id, episode=gc.DEFAULT_EPISODE, backend="h3": {
            "ok": False,
            "reason": "r2_outputs_present",
        },
    )
    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(
        _cfg(live=True, confirm=gc.LIVE_CONFIRM_PHRASE, api_key="k-test"),
        client=client,
    )
    out = wd.run()
    assert out["action"] == "preflight_failed"
    assert out["put_asks"] is False
    assert puts == []


def test_live_rejects_ssh_completion(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k-test")
    wd = gc.CanaryWatchdog(
        _cfg(live=True, confirm=gc.LIVE_CONFIRM_PHRASE, ssh_completion=True, api_key="k-test"),
    )
    with pytest.raises(gc.CanaryConfigError, match="ssh-completion"):
        wd.validate_config()


def test_live_rejects_no_r2_completion(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k-test")
    wd = gc.CanaryWatchdog(
        _cfg(live=True, confirm=gc.LIVE_CONFIRM_PHRASE, r2_completion=False, api_key="k-test"),
    )
    with pytest.raises(gc.CanaryConfigError, match="r2-completion"):
        wd.validate_config()


def test_parse_lease_instance_id_nested_and_scalar():
    assert gc.parse_lease_instance_id({"new_contract": {"id": 99123}}) == "99123"
    assert gc.parse_lease_instance_id({"instance_id": "445566"}) == "445566"
    assert gc.parse_lease_instance_id({"contract_id": {"instance_id": "7788"}}) == "7788"
    assert gc.parse_lease_instance_id({"success": True}) is None


def test_lease_never_falls_back_to_offer_id(monkeypatch):
    def opener(req):
        if req.get_method() == "PUT":
            return {"success": True}
        if "/api/v1/instances" in req.full_url:
            return {"instances": []}
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(_cfg(live=True, confirm=gc.LIVE_CONFIRM_PHRASE, api_key="k-test"), client=client)
    out = wd.lease_selected(_eligible(id="offer-999"), image_ref=IMAGE_AT, lease_env={})
    assert out["action"] == "cleanup_unknown"
    assert out["offer_id"] == "offer-999"
    assert wd._instance_id is None


def test_destroy_rejects_false_ok():
    def opener(req):
        if req.get_method() == "DELETE":
            return {"ok": False, "msg": "failure"}
        if "/api/v1/instances" in req.full_url:
            return {"instances": [{"id": "inst-x", "label": "x"}]}
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(_cfg(), client=client)
    wd._register_instance("inst-x")
    out = wd.destroy_registered()
    assert out["ok"] is False
    assert out["errors"]


def test_poll_progress_redacted(capsys):
    secret = "sk-live-vast-secret-key"

    def opener(req):
        if req.get_method() == "GET" and "/instances/inst-p" in req.full_url:
            return {"cur_state": "running", "ssh_host": "1.2.3.4"}
        if "/api/v1/instances" in req.full_url:
            return {
                "instances": [
                    {"id": "inst-p", "actual_status": "running", "cur_state": "running"},
                ]
            }
        return {"instances": []}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(
        _cfg(api_key=secret, poll_interval_s=0.01, max_wall_s=1.0),
        client=client,
    )
    wd._started_at = wd.clock()
    wd._register_instance("inst-p")
    wd.r2_checker = lambda sid, ep: True
    wd.poll_until_done(_eligible(dph_total=0.6))
    err = capsys.readouterr().err
    assert secret not in err
    assert "canary_progress" in err
    assert "inst-p" in err


def test_poll_instance_gone_after_v1_missing_streak():
    poll_n = {"n": 0}

    def opener(req):
        if req.get_method() == "GET" and "/instances/inst-p" in req.full_url:
            return {"instances": None}
        if "/api/v1/instances" in req.full_url:
            poll_n["n"] += 1
            return {"instances": []}
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(
        _cfg(poll_interval_s=0.01, max_wall_s=3600.0),
        client=client,
        sleep=lambda _: None,
    )
    wd._started_at = wd.clock()
    wd._register_instance("inst-p")
    wd.r2_checker = lambda sid, ep: False
    out = wd.poll_until_done(_eligible(dph_total=0.6))
    assert out["status"] == "failed"
    assert out["reason"] == "instance_gone"
    assert out["v1_missing"] >= gc.V1_GONE_STREAK
    assert poll_n["n"] >= gc.V1_GONE_STREAK


def test_poll_progress_uses_v1_actual_status(capsys):
    def opener(req):
        if req.get_method() == "GET" and "/instances/inst-p" in req.full_url:
            return {"instances": None}
        if "/api/v1/instances" in req.full_url:
            return {
                "instances": [
                    {"id": "inst-p", "actual_status": "loading", "cur_state": "running"},
                ]
            }
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(
        _cfg(poll_interval_s=0.01, max_wall_s=1.0),
        client=client,
        sleep=lambda _: None,
    )
    wd._started_at = wd.clock()
    wd._register_instance("inst-p")
    wd.r2_checker = lambda sid, ep: True
    out = wd.poll_until_done(_eligible(dph_total=0.6))
    assert out["status"] == "success"
    err = capsys.readouterr().err
    assert '"state": "loading"' in err


def test_destroy_404_already_gone_from_v1():
    from urllib.error import HTTPError

    def opener(req):
        if req.get_method() == "DELETE":
            raise HTTPError("http://vast/instances/inst-x/", 404, "Not Found", hdrs=None, fp=None)
        if "/api/v1/instances" in req.full_url:
            return {"instances": []}
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(_cfg(), client=client, sleep=lambda _: None)
    wd._register_instance("inst-x")
    out = wd.destroy_registered()
    assert out["ok"] is True
    assert out.get("already_gone") is True
    assert out.get("confirmed_v1") is True


def test_destroy_404_still_listed_retries():
    from urllib.error import HTTPError

    def opener(req):
        if req.get_method() == "DELETE":
            raise HTTPError("http://vast/instances/inst-x/", 404, "Not Found", hdrs=None, fp=None)
        if "/api/v1/instances" in req.full_url:
            return {"instances": [{"id": "inst-x", "actual_status": "running"}]}
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(_cfg(), client=client, sleep=lambda _: None)
    wd._register_instance("inst-x")
    out = wd.destroy_registered()
    assert out["ok"] is False
    assert out["attempts"] == gc.DELETE_MAX_RETRIES


def test_main_loads_dotenv_before_reading_env(monkeypatch, tmp_path, capsys):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "VAST_API_KEY=dotenv-vast-secret\n"
        "R2_BUCKET=canary-test-bucket\n"
        "HF_TOKEN=hf_dotenv_secret_token\n",
        encoding="utf-8",
    )

    def fake_load(path=None):
        from anime_factory.config import load_dotenv as real_load

        return real_load(dotenv)

    monkeypatch.setattr(gc, "load_dotenv", fake_load)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("R2_BUCKET", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    seen: dict[str, str] = {}

    def fake_run(self):
        seen["api_key"] = self.config.api_key
        env = gc.build_canary_env(image_digest=DIGEST, story_id=self.config.story_id)
        seen["r2_bucket"] = env["R2_BUCKET"]
        seen["hf_token"] = env.get("HF_TOKEN", "")
        return {"dry_run": True, "action": "wait_no_offer", "put_asks": False}

    monkeypatch.setattr(gc.CanaryWatchdog, "run", fake_run)
    code = gc.main(["--image", IMAGE_AT])
    assert code == 0
    out = capsys.readouterr().out
    assert seen["api_key"] == "dotenv-vast-secret"
    assert seen["r2_bucket"] == "canary-test-bucket"
    assert seen["hf_token"] == "hf_dotenv_secret_token"
    assert "dotenv-vast-secret" not in out
    assert "hf_dotenv_secret_token" not in out


def test_instance_gone_does_not_re_lease(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k-test")
    monkeypatch.setenv("VAST_ALLOW_REPLACE", "0")
    monkeypatch.setattr(gc, "r2_live_preflight", lambda *_a, **_k: {"ok": True, "reason": "ready"})
    leases: list[str] = []

    class NoRetryWatchdog(gc.CanaryWatchdog):
        def lease_selected(self, offer, *, image_ref, lease_env):
            oid = str(offer.get("id"))
            leases.append(oid)
            self._register_instance(f"inst-{oid}")
            return {"action": "leased", "offer_id": oid, "instance_id": f"inst-{oid}"}

        def poll_until_done(self, offer):
            return {"status": "failed", "reason": "instance_gone"}

        def destroy_registered(self):
            self._instance_id = None
            return {"ok": True, "already_gone": True}

    offers = [
        _eligible(id="offer-a", dph_total=0.4),
        _eligible(id="offer-b", dph_total=0.55, inet_down=500),
    ]

    def opener(req):
        if req.get_method() == "POST":
            return {"offers": offers}
        return {"instances": []}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = NoRetryWatchdog(
        _cfg(live=True, confirm=gc.LIVE_CONFIRM_PHRASE, api_key="k-test"),
        client=client,
    )
    out = wd.run()
    assert leases == ["offer-a"]
    assert out["final_status"] == "failed"
    assert out["poll"]["reason"] == "instance_gone"
    assert "gone_retries" not in out


class _FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


def test_poll_wall_uses_lease_start_not_watchdog_start():
    clock = _FakeClock()

    def opener(req):
        if req.get_method() == "GET" and "/instances/inst-p" in req.full_url:
            return {"instances": None}
        if "/api/v1/instances" in req.full_url:
            return {
                "instances": [
                    {
                        "id": "inst-p",
                        "actual_status": "running",
                        "cur_state": "running",
                        "status_msg": "success, running jupyter",
                        "machine_id": 137732,
                    }
                ]
            }
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(
        _cfg(poll_interval_s=60.0, max_wall_s=10 * 60),
        client=client,
        sleep=clock.advance,
        clock=clock,
    )
    wd._started_at = 0.0
    clock.advance(80 * 60)
    wd._register_instance("inst-p")
    wd.r2_checker = lambda sid, ep: clock.t >= 80 * 60 + 5 * 60
    out = wd.poll_until_done(_eligible(dph_total=0.48, machine_id=137732))
    assert out["status"] == "success"
    assert out["reason"] == "r2_canary"
    assert clock.t < 90 * 60


def test_poll_stuck_loading_is_instance_gone():
    clock = _FakeClock()

    def opener(req):
        if req.get_method() == "GET" and "/instances/inst-p" in req.full_url:
            return {"instances": None}
        if "/api/v1/instances" in req.full_url:
            return {
                "instances": [
                    {
                        "id": "inst-p",
                        "actual_status": "loading",
                        "cur_state": "running",
                        "machine_id": 31275,
                        "status_msg": "docker.io/aetherneo/anime-factory-gpu@sha256:abc: Pulling from aetherneo/anime-factory-gpu\n",
                        "cpu_util": 0.0,
                        "disk_usage": -1,
                        "gpu_util": None,
                    }
                ]
            }
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(
        _cfg(poll_interval_s=60.0, max_wall_s=3600.0),
        client=client,
        sleep=clock.advance,
        clock=clock,
    )
    wd._started_at = 0.0
    wd._register_instance("inst-p")
    wd.r2_checker = lambda sid, ep: False
    out = wd.poll_until_done(_eligible(dph_total=0.48, machine_id=31275))
    assert out["status"] == "failed"
    assert out["reason"] == "instance_gone"
    assert out["pull_stuck"] == "status_msg_stalled"
    assert out["stalled_s"] >= gc.CANARY_PULL_STALL_S
    assert "31275" in wd._exclude_machine_ids


def test_poll_layer_progress_does_not_kill_at_stall():
    clock = _FakeClock()
    n = {"i": 0}

    def opener(req):
        if req.get_method() == "GET" and "/instances/inst-p" in req.full_url:
            return {"instances": None}
        if "/api/v1/instances" in req.full_url:
            n["i"] += 1
            msg = "Pulling from aetherneo/anime-factory-gpu"
            if n["i"] >= 8:
                msg = "9204fd319802: Verifying Checksum\n9204fd319802: Download complete"
            return {
                "instances": [
                    {
                        "id": "inst-p",
                        "actual_status": "loading",
                        "status_msg": msg,
                        "machine_id": 150348,
                    }
                ]
            }
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(
        _cfg(poll_interval_s=60.0, max_wall_s=3600.0),
        client=client,
        sleep=clock.advance,
        clock=clock,
    )
    wd._started_at = 0.0
    wd._register_instance("inst-p")
    wd.r2_checker = lambda sid, ep: clock.t >= 13 * 60
    out = wd.poll_until_done(_eligible(dph_total=0.48, machine_id=150348))
    assert out["status"] == "success"
    assert out["reason"] == "r2_canary"
    assert clock.t >= 12 * 60


def test_poll_tls_status_msg_fails_immediately():
    clock = _FakeClock()

    def opener(req):
        if req.get_method() == "GET" and "/instances/inst-p" in req.full_url:
            return {"instances": None}
        if "/api/v1/instances" in req.full_url:
            return {
                "instances": [
                    {
                        "id": "inst-p",
                        "actual_status": "loading",
                        "status_msg": "Error pulling: tls handshake timeout",
                        "machine_id": "m-1",
                    }
                ]
            }
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(
        _cfg(poll_interval_s=10.0, max_wall_s=3600.0),
        client=client,
        sleep=clock.advance,
        clock=clock,
    )
    wd._started_at = 0.0
    wd._register_instance("inst-p")
    wd.r2_checker = lambda sid, ep: False
    out = wd.poll_until_done(_eligible(dph_total=0.48))
    assert out["status"] == "failed"
    assert out["reason"] == "instance_gone"
    assert out["pull_stuck"] == "tls"


def test_poll_progress_includes_status_msg(capsys):
    def opener(req):
        if req.get_method() == "GET" and "/instances/inst-p" in req.full_url:
            return {"instances": None}
        if "/api/v1/instances" in req.full_url:
            return {
                "instances": [
                    {
                        "id": "inst-p",
                        "actual_status": "loading",
                        "cur_state": "running",
                        "status_msg": "Pulling from aetherneo/anime-factory-gpu",
                    }
                ]
            }
        return {}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = gc.CanaryWatchdog(
        _cfg(poll_interval_s=0.01, max_wall_s=1.0),
        client=client,
        sleep=lambda _: None,
    )
    wd._started_at = wd.clock()
    wd._register_instance("inst-p")
    wd.r2_checker = lambda sid, ep: True
    out = wd.poll_until_done(_eligible(dph_total=0.6))
    assert out["status"] == "success"
    err = capsys.readouterr().err
    assert '"state": "loading"' in err
    assert "Pulling from" in err


def test_destroy_failure_does_not_re_lease(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k-test")
    monkeypatch.setenv("VAST_ALLOW_REPLACE", "0")
    monkeypatch.setattr(gc, "r2_live_preflight", lambda *_a, **_k: {"ok": True, "reason": "ready"})
    leases: list[str] = []

    class NoRetryWatchdog(gc.CanaryWatchdog):
        def lease_selected(self, offer, *, image_ref, lease_env):
            oid = str(offer.get("id"))
            leases.append(oid)
            self._register_instance(f"inst-{oid}")
            return {"action": "leased", "offer_id": oid, "instance_id": f"inst-{oid}"}

        def poll_until_done(self, offer):
            return {"status": "failed", "reason": "instance_gone"}

        def destroy_registered(self):
            return {"ok": False, "errors": ["still_listed"]}

    offers = [_eligible(id="offer-a", dph_total=0.4), _eligible(id="offer-b", dph_total=0.55)]

    def opener(req):
        if req.get_method() == "POST":
            return {"offers": offers}
        return {"instances": []}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = NoRetryWatchdog(
        _cfg(live=True, confirm=gc.LIVE_CONFIRM_PHRASE, api_key="k-test"),
        client=client,
    )
    out = wd.run()
    assert leases == ["offer-a"]
    assert out["final_status"] == "failed"
    assert out["poll"]["reason"] == "instance_gone"
    assert out.get("reason") != "destroy_failed_before_retry"


def test_sigint_does_not_put_another_ask(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k-test")
    monkeypatch.setenv("VAST_ALLOW_REPLACE", "0")
    monkeypatch.setattr(gc, "r2_live_preflight", lambda *_a, **_k: {"ok": True, "reason": "ready"})
    leases: list[str] = []

    class RetryWatchdog(gc.CanaryWatchdog):
        def lease_selected(self, offer, *, image_ref, lease_env):
            oid = str(offer.get("id"))
            leases.append(oid)
            self._register_instance(f"inst-{oid}")
            return {"action": "leased", "offer_id": oid, "instance_id": f"inst-{oid}"}

        def poll_until_done(self, offer):
            self.request_abort()
            return {"status": "failed", "reason": "instance_gone"}

        def destroy_registered(self):
            self._instance_id = None
            return {"ok": True, "already_gone": True}

    offers = [_eligible(id="offer-a", dph_total=0.4), _eligible(id="offer-b", dph_total=0.55)]

    def opener(req):
        if req.get_method() == "POST":
            return {"offers": offers}
        return {"instances": []}

    client = VastClient("fake", opener=opener, dry_run=False)
    wd = RetryWatchdog(
        _cfg(live=True, confirm=gc.LIVE_CONFIRM_PHRASE, api_key="k-test"),
        client=client,
    )
    out = wd.run()
    assert leases == ["offer-a"]
    assert out.get("reason") == "sigint" or out.get("final_status") == "aborted"

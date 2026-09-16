"""VastClient helpers — offline only."""

from __future__ import annotations

from gpu_worker.vast_client import VAST_V1_API_BASE, VastClient


def test_list_instances_v1_uses_v1_base_and_opener():
    seen: list[str] = []

    def opener(req):
        seen.append(req.full_url)
        return {"instances": [{"id": 123, "label": "anime-factory-gpu"}]}

    client = VastClient("fake-key", opener=opener, dry_run=False)
    payload = client.list_instances_v1(limit=64)
    assert payload["instances"][0]["id"] == 123
    assert seen == [f"{VAST_V1_API_BASE}/instances?limit=64"]
    assert client.calls[0][0] == "GET"


def test_list_instances_v1_dry_run_without_opener():
    client = VastClient("fake-key", dry_run=True)
    payload = client.list_instances_v1()
    assert payload.get("dry_run") is True
    assert payload.get("instances") == []


def test_quote_vast_env_value_shell_safe():
    from gpu_worker.vast_client import quote_vast_env_value

    assert quote_vast_env_value("plain") == "plain"
    assert quote_vast_env_value("a b") == "'a b'"
    assert quote_vast_env_value("x'y") == "'x'\\''y'"


def test_lease_body_quotes_env_values_with_spaces():
    body = VastClient("fake", dry_run=True).lease_body(env={"FOO": "bar baz", "SKIP": "ok"})
    assert "-e FOO='bar baz'" in body["env"]
    assert "-e SKIP=ok" in body["env"]

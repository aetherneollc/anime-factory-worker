from gpu_worker.poll import parse_hitchhikers, parse_pre_gpu, parse_work_payload, pick_job
from gpu_worker.session import pre_gpu_artifacts_ready


def test_parse_waiting_gpu_jobs():
    jobs = parse_work_payload(
        {
            "jobs": [
                {"story_id": "s1", "status": "waiting_gpu", "stage": "anim", "episode_code": "EP001"},
                {"story_id": "s2", "status": "running", "stage": "anim"},
            ]
        }
    )
    assert len(jobs) == 1
    assert pick_job(jobs)["story_id"] == "s1"
    assert pick_job(jobs, "missing")["story_id"] == "s1"
    assert pick_job([], "s1") is None


def test_stories_fallback_when_jobs_empty():
    jobs = parse_work_payload({"jobs": [], "stories": [{"story_id": "story-7343368bf7c8-a1e0eb", "episode_code": "EP001"}]})
    assert jobs[0]["story_id"] == "story-7343368bf7c8-a1e0eb"
    assert jobs[0]["status"] == "waiting_gpu"


def test_stories_fallback_skips_archived_and_until_archive():
    jobs = parse_work_payload(
        {
            "jobs": [],
            "stories": [
                {"story_id": "archived-1", "episode_code": "EP001", "status": "archived"},
                {"story_id": "done-1", "episode_code": "EP001", "until_stage": "archive"},
                {"story_id": "live-1", "episode_code": "EP001", "status": "producing"},
            ],
        }
    )
    assert [j["story_id"] for j in jobs] == ["live-1"]


def test_parse_hitchhikers_and_pre_gpu():
    payload = {
        "batch": {"batch_id": "b-1", "story_id": "series", "episodes": [{"episode_code": "EP001"}]},
        "hitchhikers": [
            {"story_id": "oneshot-1", "episode_code": "EP001", "kind": "oneshot"},
            {"episode_code": "EP002"},
        ],
        "pre_gpu": [{"story_id": "draft-1", "episode_code": "EP001", "stage": "bible"}],
    }
    hitch = parse_hitchhikers(payload)
    assert hitch == [{"story_id": "oneshot-1", "episode_code": "EP001", "kind": "oneshot"}]
    assert parse_pre_gpu(payload)[0]["stage"] == "bible"
    assert parse_hitchhikers(
        {
            "hitchhikers": [
                {"story_id": "done", "episode_code": "EP001", "status": "archived"},
                {"story_id": "live", "episode_code": "EP001", "kind": "short"},
            ]
        }
    ) == [{"story_id": "live", "episode_code": "EP001", "kind": "short"}]


def test_pre_gpu_artifacts_ready(tmp_path):
    ep = tmp_path / "episodes" / "EP001"
    ep.mkdir(parents=True)
    assert pre_gpu_artifacts_ready(tmp_path, "EP001") is False
    (ep / "board.json").write_text('{"shots":[]}', encoding="utf-8")
    assert pre_gpu_artifacts_ready(tmp_path, "EP001") is False
    audio = ep / "audio" / "lines"
    audio.mkdir(parents=True)
    (audio / "s001.zh.wav").write_bytes(b"RIFF")
    # Empty board + wav is still not ready — skip-produce would drop bible/script.
    assert pre_gpu_artifacts_ready(tmp_path, "EP001") is False
    (ep / "board.json").write_text(
        '{"shots":[{"id":"s001","line":{"zh":"灯还在。"}}]}',
        encoding="utf-8",
    )
    # Board + spoken line + dialogue wav is enough to lease/生图; H3 weights are not a gate.
    assert pre_gpu_artifacts_ready(tmp_path, "EP001") is True


def test_run_pre_gpu_if_needed_pulls_r2_before_produce(tmp_path, monkeypatch):
    from gpu_worker import session

    pulled = []
    produced = []
    monkeypatch.setattr(session, "pull_story", lambda *args, **kwargs: pulled.append((args[0], str(args[1]))) or ["board.json"])
    monkeypatch.setattr(session, "pre_gpu_artifacts_ready", lambda root, ep: True)
    monkeypatch.setattr(
        session,
        "produce_episode",
        lambda *args, **kwargs: produced.append(kwargs) or {"blocked": True},
    )
    out = session.run_pre_gpu_if_needed("story-1", tmp_path, "EP001")
    assert pulled == [("story-1", str(tmp_path))]
    assert produced == []
    assert out == {"skipped": True, "reason": "pre_gpu_ready"}


def test_gpu_dockerfile_copies_language_registry():
    from pathlib import Path

    docker = Path(__file__).resolve().parents[1] / "deploy" / "gpu-worker" / "Dockerfile"
    text = docker.read_text(encoding="utf-8")
    assert "COPY config/ /app/config/" in text
    langs = Path(__file__).resolve().parents[1] / "config" / "languages.json"
    assert langs.is_file()

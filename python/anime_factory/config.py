"""Runtime config. 待定 items are env-backed with pending comments — do not treat defaults as decisions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    style_preset: str
    qc_model: str
    llm_model_simple: str
    # 待定 (ANIME_FACTORY.md §12.7 / appendix B): 5.5s (110 shots) vs 8.0s (75 shots, -32% GPU).
    # Default 5.5 is a placeholder for tests/dev, NOT a locked creative decision.
    # 8.0 is the cost-saving alternative.
    target_shot_seconds: float
    max_concurrent_stories: int
    cold_storage_backend: str
    archive_part_max_gb: float
    # 待定: whether to archive media packs. Default false (data pack is enough to rebuild).
    archive_media_pack: bool
    continuity_max_rewrites: int
    default_world_mode: str
    map_render: bool
    embedding_model: str
    reranker_model: str
    canon_context_max_tokens: int
    app_langs: tuple[str, ...]
    tts_model: str
    translation_max_refits: int
    tts_speed_tolerance: float
    # 待定 (§13.4): missing-shot compose. Default disallow.
    compose_allow_missing_shots: bool
    vast_dry_run: bool
    vast_allow_replace: int
    vast_min_vram_gb: int
    vast_search_query: str
    vast_lease_count: int
    vast_idle_minutes: int
    vast_hold_for_compose: bool
    comfyui_base_url: str
    comfyui_dual_track: bool
    comfyui_max_concurrent: int | None
    github_backup_repo: str
    r2_vast_cred_ttl_seconds: int
    live_tts: bool
    live_llm: bool
    live_kolors: bool
    live_vast: bool


def load_settings() -> Settings:
    langs = tuple(
        part.strip()
        for part in _env("APP_LANGS", "zh,en,ja").split(",")
        if part.strip()
    )
    return Settings(
        style_preset=_env("STYLE_PRESET", "cinematic"),
        qc_model=_env("QC_MODEL", "Qwen/Qwen3.5-4B"),
        llm_model_simple=_env("LLM_MODEL_SIMPLE", "Qwen/Qwen3-8B"),
        target_shot_seconds=_env_float("TARGET_SHOT_SECONDS", 5.5),
        max_concurrent_stories=_env_int("MAX_CONCURRENT_STORIES", 2),
        cold_storage_backend=_env("COLD_STORAGE_BACKEND", "github"),
        archive_part_max_gb=_env_float("ARCHIVE_PART_MAX_GB", 1.9),
        archive_media_pack=_env_bool("ARCHIVE_MEDIA_PACK", False),
        continuity_max_rewrites=_env_int("CONTINUITY_MAX_REWRITES", 3),
        default_world_mode=_env("DEFAULT_WORLD_MODE", "fiction"),
        map_render=_env_bool("MAP_RENDER", False),
        embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-m3"),
        reranker_model=_env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        canon_context_max_tokens=_env_int("CANON_CONTEXT_MAX_TOKENS", 100000),
        app_langs=langs or ("zh", "en", "ja"),
        tts_model=_env("TTS_MODEL", "FunAudioLLM/CosyVoice2-0.5B"),
        translation_max_refits=_env_int("TRANSLATION_MAX_REFITS", 3),
        tts_speed_tolerance=_env_float("TTS_SPEED_TOLERANCE", 0.08),
        compose_allow_missing_shots=_env_bool("COMPOSE_ALLOW_MISSING_SHOTS", False),
        vast_dry_run=_env_bool("VAST_DRY_RUN", True),
        vast_allow_replace=_env_int("VAST_ALLOW_REPLACE", 0),
        vast_min_vram_gb=_env_int("VAST_MIN_VRAM_GB", 24),
        vast_search_query=_env(
            "VAST_SEARCH_QUERY",
            "gpu_ram>=24 geolocation!=China inet_down>=200 disk_space>=150 num_gpus=1",
        ),
        vast_lease_count=_env_int("VAST_LEASE_COUNT", 1),
        vast_idle_minutes=_env_int("VAST_IDLE_MINUTES", 15),
        vast_hold_for_compose=_env_bool("VAST_HOLD_FOR_COMPOSE", True),
        comfyui_base_url=_env("COMFYUI_BASE_URL", "http://127.0.0.1:8199"),
        comfyui_dual_track=_env_bool("COMFYUI_DUAL_TRACK", False),
        comfyui_max_concurrent=(
            _env_int("COMFYUI_MAX_CONCURRENT", 0) or None
        ),
        github_backup_repo=_env(
            "GITHUB_BACKUP_REPO",
            "",
        ),
        r2_vast_cred_ttl_seconds=_env_int("R2_VAST_CRED_TTL_SECONDS", 14400),
        live_tts=_env_bool("ANIME_FACTORY_LIVE_TTS", False),
        live_llm=_env_bool("ANIME_FACTORY_LIVE_LLM", False),
        live_kolors=_env_bool("ANIME_FACTORY_LIVE_KOLORS", False),
        live_vast=_env_bool("ANIME_FACTORY_LIVE_VAST", False),
    )


def load_dotenv(path: str | Path | None = None) -> Path | None:
    """Load `.env` into os.environ without overriding already-set keys. Never prints values."""
    from pathlib import Path as _Path

    env_path = _Path(path) if path is not None else _Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value
    return env_path


def siliconflow_keys() -> list[str]:
    raw = os.environ.get("SILICONFLOW_API_KEYS") or os.environ.get("SILICONFLOW_API_KEY") or ""
    return [part.strip() for part in raw.split(",") if part.strip()]

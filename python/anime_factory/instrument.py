"""Call counters so tests can prove gates never hit embeddings or live GPU lease APIs."""

from __future__ import annotations


class Counters:
    embedding_api: int = 0
    embedding_search: int = 0
    vast_asks_put: int = 0
    vast_instance_get: int = 0
    vast_instance_delete: int = 0
    kolors_requests: int = 0
    tts_requests: int = 0
    llm_requests: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.embedding_api = 0
        cls.embedding_search = 0
        cls.vast_asks_put = 0
        cls.vast_instance_get = 0
        cls.vast_instance_delete = 0
        cls.kolors_requests = 0
        cls.tts_requests = 0
        cls.llm_requests = 0

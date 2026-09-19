.PHONY: test pytest docker-check

PYTEST := $(shell if [ -x .venv/bin/pytest ]; then echo .venv/bin/pytest; else echo python3 -m pytest; fi)
# --check must not pull GHCR moss / kv-dequant (private / unpublished).
# Production builds pass the real MOSS_IMAGE and KV_DEQUANT_IMAGE.
MOSS_CHECK_IMAGE := nvidia/cuda:12.8.1-runtime-ubuntu24.04
KV_DEQUANT_CHECK_IMAGE := nvidia/cuda:12.8.1-runtime-ubuntu24.04

pytest:
	PYTHONPATH=python $(PYTEST) tests python/tests python/gpu_worker/tests -q

test: pytest

docker-check:
	docker buildx build --check --file deploy/gpu-worker/Dockerfile.moss --progress=plain .
	docker buildx build --check --file deploy/gpu-worker/Dockerfile.kv-dequant --progress=plain .
	docker buildx build --check --file deploy/gpu-worker/Dockerfile --build-arg MOSS_IMAGE=$(MOSS_CHECK_IMAGE) --progress=plain .
	docker buildx build --check --file deploy/gpu-worker/Dockerfile.longlive --build-arg MOSS_IMAGE=$(MOSS_CHECK_IMAGE) --build-arg KV_DEQUANT_IMAGE=$(KV_DEQUANT_CHECK_IMAGE) --progress=plain .

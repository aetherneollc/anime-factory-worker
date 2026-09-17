.PHONY: test pytest docker-check

PYTEST := $(shell if [ -x .venv/bin/pytest ]; then echo .venv/bin/pytest; else echo python3 -m pytest; fi)
# --check must not pull GHCR moss (private / unpublished). Production builds pass MOSS_IMAGE.
MOSS_CHECK_IMAGE := nvidia/cuda:12.8.1-runtime-ubuntu24.04

pytest:
	PYTHONPATH=python $(PYTEST) tests python/gpu_worker/tests -q

test: pytest

docker-check:
	docker buildx build --check --file deploy/gpu-worker/Dockerfile.moss --progress=plain .
	docker buildx build --check --file deploy/gpu-worker/Dockerfile --build-arg MOSS_IMAGE=$(MOSS_CHECK_IMAGE) --progress=plain .
	docker buildx build --check --file deploy/gpu-worker/Dockerfile.longlive --build-arg MOSS_IMAGE=$(MOSS_CHECK_IMAGE) --progress=plain .

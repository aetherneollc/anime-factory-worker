.PHONY: test pytest docker-check

PYTEST := $(shell if [ -x .venv/bin/pytest ]; then echo .venv/bin/pytest; else echo python3 -m pytest; fi)

pytest:
	PYTHONPATH=python $(PYTEST) tests python/gpu_worker/tests -q

test: pytest

docker-check:
	docker buildx build --check --file deploy/gpu-worker/Dockerfile --progress=plain .

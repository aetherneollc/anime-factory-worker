#!/bin/sh
# Vast onstart / image entrypoint. No `set -e`. No `seq`.
# Comfy and router bind 127.0.0.1 only. cloudflared dials out if token is set.
# Weights: HuggingFace → instance disk. HF_TOKEN optional. Never R2 weight cache.
# /opt/venv holds the image Python (torch 2.8.0+cu128 official wheels).

export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/venv}"
if [ -x /opt/venv/bin/python ]; then
  export PATH="/opt/venv/bin:${PATH}"
fi
export PYTHONPATH="${PYTHONPATH:-/app/python}"
export COMFYUI_DIR="${COMFYUI_DIR:-/opt/ComfyUI}"
export COMFYUI_BASE_URL="${COMFYUI_BASE_URL:-http://127.0.0.1:8199}"
export AF_START_COMFY="${AF_START_COMFY:-1}"
export AF_PRODUCTION_STACK="${AF_PRODUCTION_STACK:-1}"
export AF_COMFY_REQS_BAKED="${AF_COMFY_REQS_BAKED:-1}"
export ANIME_FACTORY_GPU_STILLS="${ANIME_FACTORY_GPU_STILLS:-1}"
# Preflight validate_container_contract requires /work before weight pull.
mkdir -p "${AF_WORK_DIR:-/work}"
exec python -m gpu_worker

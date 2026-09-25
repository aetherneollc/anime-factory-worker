#!/bin/sh
# HunyuanImage-2.1 worker entry. No Comfy / H3 stack in this image.
# Weights stay off-image; first generate pulls Hub onto instance disk.

export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/venv}"
if [ -x /opt/venv/bin/python ]; then
  export PATH="/opt/venv/bin:${PATH}"
fi
export PYTHONPATH="/opt/HunyuanImage-2.1:/opt/anime-factory-worker/python${PYTHONPATH:+:$PYTHONPATH}"
export AF_START_COMFY=0
export AF_PRODUCTION_STACK=0
export IMAGE_BACKEND="${IMAGE_BACKEND:-hunyuan21}"
export CONTROL_BACKEND="${CONTROL_BACKEND:-kolors_ipadapter}"
export VIDEO_BACKEND="${VIDEO_BACKEND:-hunyuan15}"
export AF_VIDEO_BACKEND="${AF_VIDEO_BACKEND:-hunyuan15}"
export AF_GPU_PROFILE="${AF_GPU_PROFILE:-hy-cu128-sm120}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_XET=1
mkdir -p "${AF_WORK_DIR:-/work}" "${AF_WEIGHTS_DIR:-/workspace/weights}"
exec python -m gpu_worker

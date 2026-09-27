#!/bin/sh
# SkyReels V3 R2V + FLUX.2 Klein 4B worker entry. No Comfy / H3 stack.
# Weights stay off-image; first generate pulls Hub onto instance disk.
# Klein and R2V run one at a time. --low_vram stays on unless SKYREELS_OFFLOAD=0.

export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/venv}"
if [ -x /opt/venv/bin/python ]; then
  export PATH="/opt/venv/bin:${PATH}"
fi
export PYTHONPATH="/opt/SkyReels-V3:/opt/anime-factory-worker/python${PYTHONPATH:+:$PYTHONPATH}"
export SKYREELS_ROOT="${SKYREELS_ROOT:-/opt/SkyReels-V3}"
export AF_START_COMFY=0
export AF_PRODUCTION_STACK=0
export IMAGE_BACKEND="${IMAGE_BACKEND:-flux2_klein4b}"
export CONTROL_BACKEND="${CONTROL_BACKEND:-flux2_klein_ref}"
export VIDEO_BACKEND="${VIDEO_BACKEND:-skyreels_v3_r2v}"
export AF_VIDEO_BACKEND="${AF_VIDEO_BACKEND:-skyreels_v3_r2v}"
export AF_IMAGE_CAPABILITY="${AF_IMAGE_CAPABILITY:-skyreels_v3_r2v}"
export AF_GPU_PROFILE="${AF_GPU_PROFILE:-sr3-cu128-sm120}"
export SKYREELS_LOW_VRAM="${SKYREELS_LOW_VRAM:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_XET=1
mkdir -p "${AF_WORK_DIR:-/work}" "${AF_WEIGHTS_DIR:-/workspace/weights}" "${HF_HOME:-/workspace/huggingface}"

# Comfy boot used to be the only caller of start_tunnel(). This image forces
# AF_START_COMFY=0, so cloudflared must be launched here or the control plane
# records tunnel_connected=0 for the whole lease.
if [ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
  cf=""
  if [ -x /usr/local/bin/cloudflared ]; then
    cf=/usr/local/bin/cloudflared
  elif command -v cloudflared >/dev/null 2>&1; then
    cf=$(command -v cloudflared)
  fi
  if [ -n "$cf" ]; then
    "$cf" tunnel --no-autoupdate --metrics 127.0.0.1:49312 run --token "$CLOUDFLARE_TUNNEL_TOKEN" >>/tmp/cloudflared.log 2>&1 &
  fi
fi
exec python -m gpu_worker

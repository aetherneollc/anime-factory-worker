#!/bin/sh
# fal.ai CPU worker. No Comfy, no CUDA, no Vast self-destroy.
# Provide FAL_KEY in the environment. Do not bake it into the image.

export PYTHONPATH="/opt/anime-factory-worker/python${PYTHONPATH:+:$PYTHONPATH}"
export AF_WORKER_MODE=fal
export AF_START_COMFY=0
export IMAGE_BACKEND="${IMAGE_BACKEND:-fal}"
export VIDEO_BACKEND="${VIDEO_BACKEND:-fal}"
export AF_VIDEO_BACKEND="${AF_VIDEO_BACKEND:-fal}"
export AF_IMAGE_CAPABILITY="${AF_IMAGE_CAPABILITY:-fal}"
export AF_GPU_PROFILE="${AF_GPU_PROFILE:-fal-api-cpu}"
export AF_VISUAL_QC_DEVICE="${AF_VISUAL_QC_DEVICE:-cpu}"
export CHARACTER_STILL_BACKEND="${CHARACTER_STILL_BACKEND:-fal}"
export SFX_BACKEND="${SFX_BACKEND:-freesound}"
mkdir -p "${AF_WORK_DIR:-/work}"
exec python -m gpu_worker

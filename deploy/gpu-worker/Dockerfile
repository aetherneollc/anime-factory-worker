# Anime Factory GPU worker: ComfyUI + MiniMax H3 + anime SDXL 生图 (IP-Adapter) + agent + ffmpeg + cloudflared.
# Weights are NOT baked — first boot pulls HuggingFace onto instance-local disk (never R2).
# H3-only production image: official torch cu130 wheels; no CUDA extension compile path.
# Labels must stay comfy/h3/kolors/image-gen true so the scheduler will lease.
# MOSS runtime is the shared moss image, not rebuilt here.

ARG MOSS_IMAGE=ghcr.io/aetherneollc/anime-factory-worker-moss:main
FROM ${MOSS_IMAGE} AS moss-sfx

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ARG TARGETARCH=amd64
# Immutable source revisions. Update only after the image canary passes.
ARG COMFYUI_REF=7ba217d6b5ac7ef984b157f7e5dfc93465b9937d
ARG KJ_NODES_REF=d3cfe21625e5170126ce06fbfcfe1d88108688c3
ARG MOSS_TORCH=2.9.0
ARG MOSS_TORCHAUDIO=2.9.0
ARG MOSS_TORCHVISION=0.24.0

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_ONLY_BINARY=:all: \
    AF_IMAGE_CAPABILITY=h3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         python3 python3-venv ca-certificates curl ffmpeg git git-lfs libsndfile1 \
         fontconfig fonts-noto-cjk \
    && case "${TARGETARCH}" in amd64|x86_64) cf_arch=amd64 ;; arm64|aarch64) cf_arch=arm64 ;; *) cf_arch=amd64 ;; esac \
    && curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${cf_arch}" \
         -o /usr/local/bin/cloudflared \
    && chmod +x /usr/local/bin/cloudflared \
    && git lfs install \
    && python3 -m venv /opt/venv \
    && pip install --no-cache-dir --only-binary=:all: --upgrade pip \
    && pip install --no-cache-dir --only-binary=:all: --index-url https://download.pytorch.org/whl/cu130 \
         torch==2.13.0+cu130 torchvision==0.28.0+cu130 torchaudio==2.11.0+cu130 \
    && git init /opt/ComfyUI \
    && git -C /opt/ComfyUI remote add origin https://github.com/comfyanonymous/ComfyUI.git \
    && git -C /opt/ComfyUI fetch --depth 1 origin "${COMFYUI_REF}" \
    && git -C /opt/ComfyUI checkout --detach FETCH_HEAD \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/* /var/tmp/*

COPY --from=moss-sfx /opt/moss-sfx /opt/moss-sfx

RUN COMFYUI_REF="${COMFYUI_REF}" python3 - <<'PY'
import json
import os
import subprocess
from pathlib import Path

comfy = Path("/opt/ComfyUI")

def head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return ""

manifest = {
    "comfyui_ref": os.environ["COMFYUI_REF"],
    "comfyui_sha": head(comfy),
}
(comfy / ".af-source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

# Character consistency needs IPAdapterAdvanced, which core ComfyUI does not ship.
# Baked into the image so first boot does not clone on the leased card; the weights
# themselves still come from HuggingFace at boot (weights.STILL_FILES), including
# OpenCLIP ViT-B-32/openai for visual_qc.py (models/visual_qc/, not ViT-H clip_vision).
ARG IPADAPTER_PLUS_REF=a0f451a5113cf9becb0847b92884cb10cbdec0ef
RUN git init /opt/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus \
    && git -C /opt/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus remote add origin \
         https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
    && git -C /opt/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus fetch --depth 1 origin "${IPADAPTER_PLUS_REF}" \
    && git -C /opt/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus checkout --detach FETCH_HEAD \
    && mkdir -p /opt/ComfyUI/models/ipadapter /opt/ComfyUI/models/clip_vision /opt/ComfyUI/models/visual_qc

RUN IPADAPTER_PLUS_REF="${IPADAPTER_PLUS_REF}" python3 - <<'PY'
import json
import os
import subprocess
from pathlib import Path

comfy = Path("/opt/ComfyUI")
ip = comfy / "custom_nodes" / "ComfyUI_IPAdapter_plus"
manifest_path = comfy / ".af-source-manifest.json"
manifest = {}
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["ipadapter_ref"] = os.environ["IPADAPTER_PLUS_REF"]
try:
    manifest["ipadapter_sha"] = subprocess.check_output(
        ["git", "-C", str(ip), "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    manifest["ipadapter_sha"] = ""
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

# KJNodes low-VRAM MiniMax nodes (pinned; no on-box compile).
RUN git init /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes \
    && git -C /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes remote add origin \
         https://github.com/kijai/ComfyUI-KJNodes.git \
    && git -C /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes fetch --depth 1 origin "${KJ_NODES_REF}" \
    && git -C /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes checkout --detach FETCH_HEAD

RUN KJ_NODES_REF="${KJ_NODES_REF}" python3 - <<'PY'
import json
import os
import subprocess
from pathlib import Path

comfy = Path("/opt/ComfyUI")
kj = comfy / "custom_nodes" / "ComfyUI-KJNodes"
manifest_path = comfy / ".af-source-manifest.json"
manifest = {}
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["kjnodes_ref"] = os.environ["KJ_NODES_REF"]
try:
    manifest["kjnodes_sha"] = subprocess.check_output(
        ["git", "-C", str(kj), "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    manifest["kjnodes_sha"] = ""
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

WORKDIR /app

COPY python/ /app/python/
COPY schema/ /app/schema/
COPY workflows/ /app/workflows/
COPY config/ /app/config/
COPY deploy/gpu-worker/start.sh /usr/local/bin/af-start
COPY deploy/gpu-worker/moss_sfx_runner.py /opt/moss-sfx/moss_sfx_runner.py
RUN chmod +x /usr/local/bin/af-start \
    /opt/moss-sfx/moss_sfx_runner.py \
    && mkdir -p /work \
    && pip install --no-cache-dir --only-binary=:all: \
         "Pillow>=10,<12" huggingface_hub boto3 "open_clip_torch==2.32.0" \
    && python3 - <<'PY'
from pathlib import Path
import subprocess
import sys

req = Path("/opt/ComfyUI/requirements.txt")
skip = {"torch", "torchvision", "torchaudio", "open_clip_torch", "open-clip-torch"}
kept: list[str] = []
for line in req.read_text(encoding="utf-8").splitlines():
    pkg = line.strip().split("#", 1)[0].strip().split("[", 1)[0].split("==")[0].split(">=")[0].strip().lower()
    if pkg and pkg not in skip:
        kept.append(line)
filt = req.parent / ".af-comfy-requirements.filtered.txt"
filt.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
if kept:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--only-binary=:all:",
            "-r",
            str(filt),
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu130",
        ]
    )
PY

# Prove the isolated MOSS stack did not replace H3's torch, and that the final
# image contains neither nvcc nor a compiler.
RUN MOSS_TORCH="${MOSS_TORCH}" MOSS_TORCHAUDIO="${MOSS_TORCHAUDIO}" \
    MOSS_TORCHVISION="${MOSS_TORCHVISION}" python3 - <<'PY'
import os
import shutil
import subprocess

import torch
import torchaudio
import torchvision

worker = (torch.__version__, torchaudio.__version__, torchvision.__version__, torch.version.cuda)
if worker != ("2.13.0+cu130", "2.11.0+cu130", "0.28.0+cu130", "13.0"):
    raise SystemExit(f"h3_worker_stack_contaminated: {worker}")
probe = subprocess.check_output(
    [
        "/opt/moss-sfx/bin/python3.12",
        "-c",
        "import sys,torch,torchaudio,torchvision;"
        "print(sys.version_info.major,sys.version_info.minor,torch.__version__,"
        "torchaudio.__version__,torchvision.__version__,torch.version.cuda)",
    ],
    text=True,
).strip()
want = (
    f"3 12 {os.environ['MOSS_TORCH']}+cu128 {os.environ['MOSS_TORCHAUDIO']}+cu128 "
    f"{os.environ['MOSS_TORCHVISION']}+cu128 12.8"
)
if probe != want:
    raise SystemExit(f"moss_sfx_runtime_mismatch: {probe!r} != {want!r}")
if shutil.which("nvcc"):
    raise SystemExit("nvcc present in final runtime")
for command in ("gcc", "g++", "cc"):
    if shutil.which(command):
        raise SystemExit(f"build tool present in final runtime: {command}")
print("isolated stacks ok:", worker, probe)
PY

ENV PYTHONPATH=/app/python \
    PYTHONUNBUFFERED=1 \
    VAST_DRY_RUN=1 \
    VAST_ALLOW_REPLACE=0 \
    ANIME_FACTORY_LIVE_VAST=0 \
    AF_START_COMFY=1 \
    AF_COMFY_REQS_BAKED=1 \
    AF_PRODUCTION_STACK=1 \
    ANIME_FACTORY_GPU_STILLS=1 \
    COMFYUI_DIR=/opt/ComfyUI \
    COMFYUI_BASE_URL=http://127.0.0.1:8199 \
    COMFY_BACKENDS=http://127.0.0.1:8188 \
    COMFY_ROUTER_HOST=127.0.0.1 \
    COMFY_ROUTER_PORT=8199 \
    OPEN_CLIP_TORCH_VERSION=2.32.0 \
    AF_VISUAL_QC_CLIP_DIR=/opt/ComfyUI/models/visual_qc \
    AF_VISUAL_QC_CLIP_WEIGHTS=/opt/ComfyUI/models/visual_qc/open_clip_model.safetensors \
    AF_VISUAL_QC_CLIP_CONFIG=/opt/ComfyUI/models/visual_qc/open_clip_config.json \
    AF_OPEN_CLIP_MODEL=ViT-B-32 \
    AF_OPEN_CLIP_PRETRAINED=openai \
    MOSS_SFX_PYTHON=/opt/moss-sfx/bin/python3.12 \
    MOSS_SFX_SCRIPT=/opt/moss-sfx/moss_sfx_runner.py \
    MOSS_SFX_WEIGHTS_DIR=/work/.model-cache/moss-sfx \
    MOSS_SFX_DEVICE=cuda:0 \
    TORCHDYNAMO_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LABEL org.aetherneo.anime-factory.comfy="true" \
      org.aetherneo.anime-factory.h3="true" \
      org.aetherneo.anime-factory.video-backend="h3" \
      org.aetherneo.anime-factory.kolors="true" \
      org.aetherneo.anime-factory.image-gen="true" \
      org.aetherneo.anime-factory.agent="true"

WORKDIR /app

ENTRYPOINT ["/usr/local/bin/af-start"]

# Anime Factory GPU worker — HunyuanImage-2.1 (anime-factory-worker-hy).
#
# Standalone image: does not depend on the Moss or H3/Comfy images.
# cu128 runtime for Blackwell (sm_120 / 5090). Model weights are NOT baked —
# first generate pulls Hugging Face onto instance-local disk.
# Official HunyuanImage-2.1 code is cloned; flash-attn is replaced with SDPA
# because 2.7.3 has no sm_120 kernels.
#
# Env defaults:
#   IMAGE_BACKEND=hunyuan21
#   CONTROL_BACKEND=kolors_ipadapter
#   VIDEO_BACKEND=hunyuan15
#   AF_GPU_PROFILE=hy-cu128-sm120
#   AF_START_COMFY=0

FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

ARG TARGETARCH=amd64
ARG HY21_REPO=https://github.com/Tencent-Hunyuan/HunyuanImage-2.1.git

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    IMAGE_BACKEND=hunyuan21 \
    CONTROL_BACKEND=kolors_ipadapter \
    VIDEO_BACKEND=hunyuan15 \
    AF_VIDEO_BACKEND=hunyuan15 \
    AF_IMAGE_CAPABILITY=hunyuan21 \
    AF_GPU_PROFILE=hy-cu128-sm120 \
    AF_START_COMFY=0 \
    AF_PRODUCTION_STACK=0 \
    HUNYUAN21_DRY_RUN=0 \
    PYTHONPATH=/opt/HunyuanImage-2.1:/opt/anime-factory-worker/python \
    HF_HOME=/workspace/huggingface \
    AF_WEIGHTS_DIR=/workspace/weights \
    HF_HUB_DISABLE_XET=1

LABEL org.opencontainers.image.title="anime-factory-worker-hy" \
      af.capability.image-gen="true" \
      af.capability.hunyuan21="true" \
      af.capability.hunyuan15="scaffold" \
      af.capability.h3="false" \
      af.capability.comfy="false"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         ca-certificates curl ffmpeg git git-lfs libsndfile1 \
         fontconfig fonts-noto-cjk python3-venv wget \
    && case "${TARGETARCH}" in amd64|x86_64) cf_arch=amd64 ;; arm64|aarch64) cf_arch=arm64 ;; *) cf_arch=amd64 ;; esac \
    && curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${cf_arch}" \
         -o /usr/local/bin/cloudflared \
    && chmod +x /usr/local/bin/cloudflared \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/venv \
    || { rm -rf /opt/venv && python -m venv --system-site-packages --without-pip /opt/venv; } \
    && /opt/venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/python -c "import torch; print('venv torch', torch.__version__, torch.version.cuda)"

# Upstream code only. Do not pip-install requirements.txt (it re-pins torch).
RUN git clone --depth 1 "${HY21_REPO}" /opt/HunyuanImage-2.1 \
    && /opt/venv/bin/python -m pip install --no-cache-dir \
         "Pillow>=10,<12" \
         "einops==0.8.0" \
         "loguru==0.7.3" \
         "omegaconf>=2.3.0" \
         "safetensors==0.4.5" \
         "diffusers>=0.32.0" \
         "transformers==4.56.0" \
         "accelerate" \
         "sentencepiece" \
         "protobuf" \
         "timm" \
         "peft" \
         "httpx==0.27.2" \
         "huggingface_hub[cli]==0.34.0" \
         "numpy==1.26.4" \
    && rm -rf /root/.cache/pip

COPY deploy/gpu-worker/patch_hy21_sdpa.py /opt/patch_hy21_sdpa.py
RUN python /opt/patch_hy21_sdpa.py \
    && python -c "import hyimage.models.hunyuan.modules.flash_attn_no_pad as m; assert 'SDPA_FALLBACK' in open(m.__file__).read()"

WORKDIR /opt/anime-factory-worker
COPY python/ /opt/anime-factory-worker/python/
COPY schema/ /opt/anime-factory-worker/schema/
COPY config/ /opt/anime-factory-worker/config/

COPY deploy/gpu-worker/start-hy.sh /usr/local/bin/af-start
RUN chmod +x /usr/local/bin/af-start \
    && mkdir -p /work /workspace/weights /workspace/huggingface \
    && python -c "import anime_factory.hunyuan21, anime_factory.stills_phase, torch; print('hy worker import ok', torch.__version__)"

WORKDIR /workspace
CMD ["/usr/local/bin/af-start"]

#!/usr/bin/env bash
# Reproducible isolated MOSS-SoundEffect v2 runtime under /opt/moss-sfx.
# Invoked from the moss-sfx Docker build stage in both worker images.
set -euo pipefail

MOSS_TTS_REF="${MOSS_TTS_REF:?MOSS_TTS_REF required}"
MOSS_TORCH="${MOSS_TORCH:?MOSS_TORCH required}"
MOSS_TORCHAUDIO="${MOSS_TORCHAUDIO:?MOSS_TORCHAUDIO required}"
MOSS_TORCHVISION="${MOSS_TORCHVISION:?MOSS_TORCHVISION required}"

MOSS_TORCH_BASE="${MOSS_TORCH%%+*}"
MOSS_TORCHAUDIO_BASE="${MOSS_TORCHAUDIO%%+*}"
MOSS_TORCHVISION_BASE="${MOSS_TORCHVISION%%+*}"

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

apt-get update
apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates git libsndfile1

git init /tmp/MOSS-TTS
git -C /tmp/MOSS-TTS remote add origin https://github.com/OpenMOSS/MOSS-TTS.git
git -C /tmp/MOSS-TTS fetch --depth 1 origin "${MOSS_TTS_REF}"
git -C /tmp/MOSS-TTS checkout --detach FETCH_HEAD
test "$(git -C /tmp/MOSS-TTS rev-parse HEAD)" = "${MOSS_TTS_REF}"

mkdir -p /opt/moss-sfx/source/moss_soundeffect_v2
cp -a \
    /tmp/MOSS-TTS/moss_soundeffect_v2/__init__.py \
    /tmp/MOSS-TTS/moss_soundeffect_v2/README.md \
    /tmp/MOSS-TTS/moss_soundeffect_v2/pipeline_moss_soundeffect.py \
    /tmp/MOSS-TTS/moss_soundeffect_v2/diffsynth \
    /tmp/MOSS-TTS/moss_soundeffect_v2/pyproject.toml \
    /opt/moss-sfx/source/moss_soundeffect_v2/

python3 -m venv /opt/moss-sfx
if [ ! -e /opt/moss-sfx/bin/python3.12 ] && [ -e /opt/moss-sfx/bin/python3 ]; then
    ln -sf python3 /opt/moss-sfx/bin/python3.12
fi
/opt/moss-sfx/bin/pip install --only-binary=:all: --upgrade pip setuptools wheel

/opt/moss-sfx/bin/pip install --only-binary=:all: \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==${MOSS_TORCH_BASE}+cu128" \
    "torchaudio==${MOSS_TORCHAUDIO_BASE}+cu128" \
    "torchvision==${MOSS_TORCHVISION_BASE}+cu128"

/opt/moss-sfx/bin/pip install --only-binary=:all: \
    "numpy==1.26.4" \
    "einops==0.8.2" \
    "pillow==12.2.0" \
    "tqdm==4.67.3" \
    "safetensors==0.7.0" \
    "transformers==4.57.1" \
    "diffusers==0.37.1" \
    "ftfy==6.3.1" \
    "regex==2026.4.4" \
    "soundfile==0.13.1" \
    "imageio==2.37.3" \
    "typing-extensions>=4.10" \
    "descript-audiotools==0.7.2" \
    "huggingface_hub>=0.26.0"

cd /opt/moss-sfx/source/moss_soundeffect_v2
/opt/moss-sfx/bin/pip install --no-deps -e .

MOSS_TTS_REF="${MOSS_TTS_REF}" \
MOSS_TORCH="${MOSS_TORCH_BASE}" \
MOSS_TORCHAUDIO="${MOSS_TORCHAUDIO_BASE}" \
MOSS_TORCHVISION="${MOSS_TORCHVISION_BASE}" \
/opt/moss-sfx/bin/python - <<'PY'
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import moss_soundeffect_v2  # noqa: F401
import torch
import torchaudio
import torchvision

expected = {
    "torch": os.environ["MOSS_TORCH"] + "+cu128",
    "torchaudio": os.environ["MOSS_TORCHAUDIO"] + "+cu128",
    "torchvision": os.environ["MOSS_TORCHVISION"] + "+cu128",
}
actual = {
    "torch": torch.__version__,
    "torchaudio": torchaudio.__version__,
    "torchvision": torchvision.__version__,
}
problems = []
if sys.version_info[:2] != (3, 12):
    problems.append(f"python {sys.version}")
for name, want in expected.items():
    if actual[name] != want:
        problems.append(f"{name} {actual[name]} != {want}")
if torch.version.cuda != "12.8":
    problems.append(f"torch cuda {torch.version.cuda} != 12.8")
if version("moss-soundeffect-v2") != "0.1.0":
    problems.append(f"moss-soundeffect-v2 {version('moss-soundeffect-v2')} != 0.1.0")
if problems:
    raise SystemExit("moss_sfx_build_fail_closed: " + "; ".join(problems))
Path("/opt/moss-sfx/.af-source-manifest.json").write_text(
    json.dumps({"moss_tts_ref": os.environ["MOSS_TTS_REF"], **actual}, indent=2) + "\n",
    encoding="utf-8",
)
print("moss runtime ok:", sys.version.split()[0], actual, torch.version.cuda)
PY

rm -rf /tmp/MOSS-TTS /root/.cache /tmp/* /var/tmp/*
apt-get clean
rm -rf /var/lib/apt/lists/*

#!/usr/bin/env python3
"""GPU canary: HunyuanImage-2.1 on Vast 5090 vs Kolors full-body failure mode.

Leases one card, runs official Diffusers pipeline (FP8, distilled first),
pulls PNGs, scores with figure_is_full_body (no bottom-align), destroys the box.

Usage:
  VAST_API_KEY=... PYTHONPATH=python \\
    .venv/bin/python tools/hunyuan21_gpu_canary.py [--live] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from anime_factory.config import load_dotenv  # noqa: E402
from gpu_worker.offers import pick_one_offer, search_payload  # noqa: E402
from gpu_worker.vast_client import VastClient  # noqa: E402

REMOTE_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/workspace/hunyuan21_out
mkdir -p "$OUT"
cd /workspace

if [[ ! -d HunyuanImage-2.1 ]]; then
  git clone --depth 1 https://github.com/Tencent-Hunyuan/HunyuanImage-2.1.git
fi
cd HunyuanImage-2.1

python - <<'PY'
import subprocess, sys
# Prefer existing torch; only install missing Python deps.
missing=[]
for p in ("transformers","accelerate","safetensors","einops","sentencepiece","pillow","huggingface_hub","diffusers","loguru","timm"):
    try:
        __import__(p if p!="pillow" else "PIL")
    except Exception:
        missing.append("Pillow" if p=="pillow" else p)
if missing:
    subprocess.check_call([sys.executable,"-m","pip","install","-q",*missing])
print("deps ok", missing or "all present")
PY

# Blackwell (sm_120 / 5090) needs torch cu128+; Ada/Ampere images already OK.
python - <<'PY'
import subprocess, sys
try:
    import torch
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    need = (cap[0] >= 12) or ("cu124" in torch.__version__ and cap[0] >= 12)
    # Force upgrade when capability is Blackwell or torch lacks sm_120.
    force = False
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
        except Exception:
            force = True
    if cap[0] >= 12 or force:
        print(f"upgrading torch for cap={cap} ver={torch.__version__}", flush=True)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--upgrade",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu128",
        ])
        import importlib; importlib.reload(torch)
        print("torch now", torch.__version__, torch.version.cuda, flush=True)
except Exception as exc:
    print("torch check skipped:", exc, flush=True)
PY

# Official pin; compile may fail on Blackwell — SDPA patch below covers that.
python -m pip install -q "flash-attn==2.7.3" --no-build-isolation || \
  python -m pip install -q flash-attn --no-build-isolation || true

# Install package if present
if [[ -f requirements.txt ]]; then
  # Avoid re-pinning torch from requirements over cu128.
  grep -vE '^(torch|torchvision|torchaudio)' requirements.txt > /tmp/hy21_reqs.txt || true
  python -m pip install -q -r /tmp/hy21_reqs.txt || true
fi
python -m pip install -q -e . || true

# SDPA fallback when flash_attn missing / no sm_120 kernels (5090).
python - <<'PY'
from pathlib import Path
p = Path("hyimage/models/hunyuan/modules/flash_attn_no_pad.py")
if not p.is_file():
    raise SystemExit("flash_attn_no_pad.py missing")
text = p.read_text()
if "SDPA_FALLBACK" in text:
    print("sdpa patch already present")
else:
    p.write_text(r'''import torch
from einops import rearrange

# SDPA_FALLBACK — used when flash_attn is unavailable (e.g. RTX 5090 / sm_120).
_HAS_FLASH = False
try:
    from flash_attn_interface import flash_attn_varlen_func  # noqa: F401
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    _HAS_FLASH = True
    print("Using FlashAttention v3.")
except ImportError:
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        from flash_attn import flash_attn_varlen_qkvpacked_func
        from flash_attn.bert_padding import pad_input, unpad_input
        _HAS_FLASH = True
        print("FlashAttention v3 not found, falling back to v2.")
    except ImportError:
        print("FlashAttention not found, using PyTorch SDPA fallback.")


def get_cu_seqlens(text_mask: torch.Tensor, img_len: int):
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len
    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=text_mask.device)
    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2
    return cu_seqlens, max_len


def flash_attn_v3(q, k, v, cu_seqlens, max_s, causal=False, deterministic=False):
    if not _HAS_FLASH:
        # q,k,v: (B, S, H, D) — unused cu_seqlens path on SDPA
        qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, is_causal=causal)
        return out.transpose(1, 2)
    batch_size, seqlen = q.shape[:2]
    q = q.reshape(-1, *q.shape[2:])
    k = k.reshape(-1, *k.shape[2:])
    v = v.reshape(-1, *v.shape[2:])
    output = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_s, max_s, causal=causal, deterministic=deterministic
    )
    return output.view(batch_size, seqlen, *output.shape[-2:])


def flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    if not _HAS_FLASH:
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
        )
        return out.transpose(1, 2)
    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)[:4]
    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_seqlens, max_s, dropout_p,
        softmax_scale=softmax_scale, causal=causal, deterministic=deterministic,
    )
    if isinstance(output_unpad, tuple):
        output_unpad = output_unpad[0]
    output = pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen)
    return rearrange(output, "b s (h d) -> b s h d", h=nheads)
''')
    print("wrote SDPA fallback patch")
PY

# httpx pin avoids hf download TypeError with newer httpx2 decoders
python -m pip install -q 'httpx==0.27.2' 'huggingface_hub[cli]==0.34.0' modelscope || true

# Official checkpoints (distilled fp8 + VAE). Skip PromptEnhancer/refiner/full DiT.
if [[ ! -f ./ckpts/dit/hunyuanimage2.1-distilled_fp8.safetensors || ! -f ./ckpts/vae/vae_2_1/config.json ]]; then
  echo "downloading distilled fp8 + vae …"
  hf download tencent/HunyuanImage-2.1 --local-dir ./ckpts \
    --include "dit/hunyuanimage2.1-distilled_fp8*" \
    --include "dit/hunyuanimage2.1-distilled.safetensors" \
    --include "vae/**" \
    --include "config.json" --include "assets/**" \
    --include "LICENSE" --include "NOTICE" --include "README*" \
    --include "checkpoints-download.md" --include ".gitattributes"
fi
if [[ ! -f ./ckpts/text_encoder/llm/config.json ]]; then
  echo "downloading Qwen2.5-VL-7B …"
  hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/text_encoder/llm
fi
if [[ ! -d ./ckpts/text_encoder/byt5-small ]]; then
  hf download google/byt5-small --local-dir ./ckpts/text_encoder/byt5-small
fi
if [[ ! -d ./ckpts/text_encoder/Glyph-SDXL-v2/checkpoints || ! -f ./ckpts/text_encoder/Glyph-SDXL-v2/checkpoints/byt5_model.pt ]]; then
  echo "downloading Glyph byt5_model.pt …"
  G=./ckpts/text_encoder/Glyph-SDXL-v2
  mkdir -p "$G/checkpoints" "$G/assets"
  BASE="https://www.modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/resolve/master"
  for f in assets/color_idx.json assets/multilingual_10-lang_idx.json checkpoints/byt5_model.pt; do
    mkdir -p "$G/$(dirname "$f")"
    [[ -f "$G/$f" ]] || wget -q -O "$G/$f" "$BASE/$f" || true
  done
fi

python - <<'PY'
import os, json, time
from pathlib import Path
from io import BytesIO

OUT = Path("/workspace/hunyuan21_out")
OUT.mkdir(parents=True, exist_ok=True)

# Prefer distilled for wall-clock; env can force base.
model_name = os.environ.get("HY21_MODEL", "hunyuanimage-v2.1-distilled")
use_refiner = os.environ.get("HY21_REFINER", "0") == "1"
limit = int(os.environ.get("HY21_LIMIT", "6"))

chars = {
  "lin-xiao": "1boy, male focus, young adult 19 y/o, short messy black hair, sharp brown eyes, black leather jacket, red shirt, black pants, black boots, lean athletic build",
  "a-kai": "1boy, male focus, young adult 20 y/o, slicked-back dark brown hair, narrow amber eyes, yellow leather jacket, black shirt, black pants, black shoes, tall athletic build",
}
framing = (
  "original anime character reference sheet, solo, cel shaded, clean lineart, "
  "warm ivory studio background, full body shot, long shot, zoomed out, head to toe, "
  "the whole figure from the top of the head to both shoes fits inside the frame, "
  "both feet and shoes fully visible, standing with empty space below the shoes, "
  "front view, looking at viewer, 全身立绘，从头顶到两只鞋完整入画，双脚和鞋子都完整可见，脚下留白"
)

# Official 9:16 2K bucket (1K causes artifacts per README).
W, H = 1536, 2560

print(f"loading {model_name} fp8 refiner={use_refiner}", flush=True)
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline
pipe = HunyuanImagePipeline.from_pretrained(model_name=model_name, use_fp8=True)
pipe = pipe.to("cuda")

rows=[]
n=0
for cid, identity in chars.items():
  for attempt in range(1, 4):
    if n >= limit:
      break
    n += 1
    prompt = f"{framing}, {identity}"
    seed = 3000 + n * 97 + attempt * 13
    t0=time.time()
    print(f"[{n}] {cid} a{attempt} seed={seed}", flush=True)
    try:
      image = pipe(
        prompt=prompt,
        width=W,
        height=H,
        use_reprompt=False,
        use_refiner=use_refiner,
        num_inference_steps=8 if "distilled" in model_name else 28,
        guidance_scale=3.25 if "distilled" in model_name else 3.5,
        shift=4 if "distilled" in model_name else 5,
        seed=seed,
      )
      path = OUT / f"{cid}_{W}x{H}_a{attempt}.png"
      image.save(path)
      rows.append({
        "character_id": cid,
        "attempt": attempt,
        "seed": seed,
        "path": str(path),
        "bytes": path.stat().st_size,
        "width": W,
        "height": H,
        "model": model_name,
        "latency_s": round(time.time()-t0, 2),
        "error": "",
      })
      print(f"  saved {path} {path.stat().st_size}", flush=True)
    except Exception as exc:
      rows.append({
        "character_id": cid,
        "attempt": attempt,
        "seed": seed,
        "path": "",
        "bytes": 0,
        "width": W,
        "height": H,
        "model": model_name,
        "latency_s": round(time.time()-t0, 2),
        "error": f"{type(exc).__name__}:{exc}",
      })
      print(f"  FAIL {exc}", flush=True)
  if n >= limit:
    break

(OUT/"rows.json").write_text(json.dumps(rows, indent=2))
(OUT/"DONE").write_text("ok\n")
print("DONE", len(rows), flush=True)
PY
'''


def _load_vast_key() -> str:
    load_dotenv(ROOT / ".env")
    key = (os.environ.get("VAST_API_KEY") or "").strip()
    if not key:
        raise SystemExit("VAST_API_KEY missing")
    return key


def _ssh_target(inst: dict) -> tuple[str, int, str]:
    """Return (host, port, user) from Vast instance payload."""
    host = str(inst.get("ssh_host") or inst.get("public_ipaddr") or "").strip()
    port = inst.get("ssh_port")
    ports = inst.get("ports") or {}
    if not port and isinstance(ports, dict):
        for key in ("22/tcp", "22"):
            mapping = ports.get(key)
            if isinstance(mapping, list) and mapping:
                port = mapping[0].get("HostPort") or mapping[0].get("host_port")
                host = host or mapping[0].get("HostIp") or host
                break
    if not host or not port:
        raise RuntimeError(f"no ssh endpoint yet: host={host!r} port={port!r} keys={list(inst)[:20]}")
    user = str(inst.get("ssh_user") or "root").strip() or "root"
    return host, int(port), user


def _ssh(host: str, port: int, user: str, cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15",
            "-p", str(port),
            f"{user}@{host}",
            cmd,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _scp_from(host: str, port: int, user: str, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(port),
            f"{user}@{host}:{remote}",
            str(local),
        ],
        check=True,
        timeout=600,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Actually PUT /asks/")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--max-dph", type=float, default=0.85)
    ap.add_argument("--out", default=f"/tmp/hunyuan21_gpu_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--offer-id", default="", help="Pin a Vast offer id")
    ap.add_argument("--model", default="hunyuanimage-v2.1-distilled")
    ap.add_argument("--refiner", action="store_true")
    ap.add_argument("--max-wall-min", type=int, default=75)
    args = ap.parse_args()

    key = _load_vast_key()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = VastClient(api_key=key, dry_run=not args.live)

    body = client.search_offers(filters=search_payload())
    offers = body.get("offers") or []
    if args.offer_id:
        chosen = next((o for o in offers if str(o.get("id")) == str(args.offer_id)), None)
        if not chosen:
            # direct id lease without search hit
            chosen = {"id": args.offer_id, "gpu_name": "pinned", "dph_total": 0, "gpu_ram": 32000}
    else:
        chosen = pick_one_offer(offers, gpu_model_policy="5090", max_dph=args.max_dph, max_usd=80)
        if not chosen:
            # fall back: any 24GB+ under dph
            cheap = []
            for o in offers:
                ram = int(o.get("gpu_ram") or 0)
                dph = float(o.get("dph_total") or o.get("dph") or 99)
                if ram >= 24000 and dph <= args.max_dph:
                    cheap.append(o)
            cheap.sort(key=lambda o: float(o.get("dph_total") or o.get("dph") or 99))
            chosen = cheap[0] if cheap else None
    if not chosen:
        print("no offer", file=sys.stderr)
        return 2

    offer_id = str(chosen.get("id"))
    print(json.dumps({
        "offer_id": offer_id,
        "gpu": chosen.get("gpu_name") or chosen.get("gpu"),
        "ram": chosen.get("gpu_ram"),
        "dph": chosen.get("dph_total") or chosen.get("dph"),
        "geo": chosen.get("geolocation"),
        "live": args.live,
        "model": args.model,
    }, indent=2))

    # PyTorch CUDA image — smaller pull than our H3 agent image.
    image = os.environ.get("HY21_VAST_IMAGE") or "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
    lease_body = client.lease_body(
        image=image,
        disk_gb=150,
        env={
            "HY21_MODEL": args.model,
            "HY21_REFINER": "1" if args.refiner else "0",
            "HY21_LIMIT": str(args.limit),
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
        },
        extra={
            "label": "hunyuan21-fullbody-canary",
            "onstart": "mkdir -p /workspace && sleep infinity",
            "runtype": "args ssh",
        },
    )
    leased = client.lease(offer_id, lease_body)
    if leased.get("dry_run") or leased.get("skipped"):
        print("dry-run skip", json.dumps(leased)[:400])
        return 0

    instance_id = str(leased.get("new_contract") or leased.get("id") or leased.get("instance_id") or "")
    if not instance_id:
        print("lease returned no instance", leased, file=sys.stderr)
        return 1
    registered = {instance_id}
    print("leased", instance_id)

    deadline = time.time() + args.max_wall_min * 60
    host = port = user = None
    try:
        while time.time() < deadline:
            payload = client.get_instance(instance_id)
            inst = payload.get("instances") if isinstance(payload.get("instances"), dict) else payload
            if isinstance(payload.get("instances"), list) and payload["instances"]:
                inst = payload["instances"][0]
            # Vast sometimes nests under instances[id]
            if isinstance(payload.get("instances"), dict) and instance_id in payload["instances"]:
                inst = payload["instances"][instance_id]
            inst = inst if isinstance(inst, dict) else {}
            status = inst.get("actual_status") or inst.get("cur_state") or inst.get("status")
            print(f"status={status} keys_hint={list(inst)[:8]}", flush=True)
            try:
                host, port, user = _ssh_target(inst)
                probe = _ssh(host, port, user, "nvidia-smi -L && echo SSH_OK", timeout=30)
                if probe.returncode == 0 and "SSH_OK" in probe.stdout:
                    print(probe.stdout)
                    break
                print("ssh probe fail", probe.stderr[:200], flush=True)
            except Exception as exc:
                print(f"wait ssh: {exc}", flush=True)
            time.sleep(15)
        else:
            raise RuntimeError("timed out waiting for SSH")

        assert host and port and user
        # Upload remote script
        local_script = out / "remote_run.sh"
        local_script.write_text(REMOTE_SCRIPT)
        subprocess.run(
            [
                "scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-P", str(port), str(local_script), f"{user}@{host}:/workspace/remote_run.sh",
            ],
            check=True,
            timeout=60,
        )
        print("starting remote HunyuanImage-2.1 …", flush=True)
        # Background must detach stdin or SSH waits until the job ends.
        run = _ssh(
            host,
            port,
            user,
            "bash -lc 'nohup env HY21_MODEL=%s HY21_REFINER=%s HY21_LIMIT=%s "
            "bash /workspace/remote_run.sh </dev/null >/workspace/hunyuan21_run.log 2>&1 & "
            "echo STARTED $!; disown || true'"
            % (args.model, "1" if args.refiner else "0", args.limit),
            timeout=45,
        )
        print(run.stdout, run.stderr, flush=True)
        if "STARTED" not in (run.stdout or ""):
            raise RuntimeError(f"failed to start remote job: rc={run.returncode} {run.stderr or run.stdout}")

        # Poll DONE
        while time.time() < deadline:
            check = _ssh(
                host,
                port,
                user,
                "if test -f /workspace/hunyuan21_out/DONE; then echo DONE; else echo WAIT; tail -n 8 /workspace/hunyuan21_run.log; fi",
                timeout=60,
            )
            print(check.stdout[-800:], flush=True)
            if check.stdout.strip().startswith("DONE"):
                break
            time.sleep(30)
        else:
            # pull log anyway
            _ssh(host, port, user, "tail -n 80 /workspace/hunyuan21_run.log", timeout=60)

        # Fetch artifacts
        img_dir = out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        listing = _ssh(host, port, user, "ls -1 /workspace/hunyuan21_out/", timeout=60)
        print("remote out:", listing.stdout)
        for line in listing.stdout.splitlines():
            name = line.strip()
            if not name:
                continue
            dest = (out / "images" / name) if name.endswith(".png") else (out / name)
            try:
                _scp_from(host, port, user, f"/workspace/hunyuan21_out/{name}", dest)
            except Exception as exc:
                print(f"scp {name}: {exc}")
        # also log
        try:
            _scp_from(host, port, user, "/workspace/hunyuan21_run.log", out / "remote_run.log")
        except Exception:
            pass

    finally:
        try:
            destroyed = client.destroy(instance_id, registered)
            print("destroyed", destroyed)
        except Exception as exc:
            print(f"destroy failed: {exc}", file=sys.stderr)

    # Local QC
    from PIL import Image
    from anime_factory.visual_qc import figure_is_full_body, structure_check

    rows = []
    rows_path = out / "rows.json"
    if rows_path.is_file():
        rows = json.loads(rows_path.read_text())
    results = []
    for png in sorted((out / "images").glob("*.png")) if (out / "images").exists() else []:
        blob = png.read_bytes()
        img = Image.open(BytesIO(blob)).convert("RGB")
        ok, metrics = figure_is_full_body(img)
        struct = structure_check(blob, kind="character_sheet", allow_placeholder=True)
        results.append({
            "path": str(png),
            "geometric_full_body": bool(ok),
            "span": float(metrics["span"]),
            "top": float(metrics["top"]),
            "bottom": float(metrics["bottom"]),
            "structure_verdict": struct.verdict,
            "structure_reasons": list(struct.reasons or []),
            "width": img.size[0],
            "height": img.size[1],
            "bytes": len(blob),
        })
        print(f"QC {png.name} geo={ok} span={metrics['span']:.3f} bottom={metrics['bottom']:.3f}")

    n = len(results)
    n_geo = sum(1 for r in results if r["geometric_full_body"])
    summary = {
        "model": args.model,
        "n": n,
        "geometric_full_body_rate": (n_geo / n) if n else 0.0,
        "go": (n_geo / n) >= 0.5 if n else False,
        "note": "Kolors bust-loop comparator; official 1536x2560 9:16; no bottom-align",
        "results": results,
        "remote_rows": rows,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("model", "n", "geometric_full_body_rate", "go", "note")}, indent=2))
    print(f"VERDICT={'GO_KEEP_HUNYUAN21' if summary['go'] else 'NO_GO'} out={out}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())

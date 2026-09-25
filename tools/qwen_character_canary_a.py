#!/usr/bin/env python3
"""Asset Factory ImageProvider canary — Phase A T2I full-body benchmark.

Uses DashScope ``qwen-image-2.0-pro`` (``qwen-image-2.1`` is not on DashScope yet).
Does NOT bottom-align / reframe before QC (avoids Kolors canary false positives).

Contract (Go/No-Go):
  valid_asset = geometric_full_body AND (vision full_body+both_feet if available)

Usage:
  DASHSCOPE_API_KEY=... PYTHONPATH=python \\
    .venv/bin/python tools/qwen_character_canary_a.py [--out DIR] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from PIL import Image  # noqa: E402

from anime_factory.visual_qc import (  # noqa: E402
    FIGURE_BOTTOM_MIN,
    FIGURE_SPAN_MIN,
    FIGURE_TOP_MAX,
    figure_is_full_body,
    structure_check,
)

DASHSCOPE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)
# Closest available cloud model — 2.1 is open-weight / Comfy only for now.
MODEL = "qwen-image-2.0-pro"

CHARACTERS = {
    "lin-xiao": {
        "name": "林骁",
        "identity": (
            "1boy, male focus, young adult 19 y/o, short messy black hair, "
            "sharp brown eyes, black leather jacket, red shirt, lean athletic build"
        ),
    },
    "a-kai": {
        "name": "阿凯",
        "identity": (
            "1boy, male focus, young adult 20 y/o, slicked-back dark brown hair, "
            "narrow amber eyes, yellow leather jacket, black shirt, black pants, "
            "tall athletic build"
        ),
    },
}

SIZES = ("768*1344", "1024*1536")
ATTEMPTS = 5

# Same framing intent as production character sheets (English; Qwen is not ChatGLM-pooled).
FRAMING = (
    "original anime character reference sheet, solo, cel shaded, clean lineart, "
    "warm ivory studio background, full body shot, long shot, zoomed out, head to toe, "
    "the whole figure from the top of the head to both shoes fits inside the frame, "
    "both feet and shoes fully visible, standing with empty space below the shoes, "
    "front view, looking at viewer"
)
NEGATIVE = (
    "close-up, portrait, headshot, bust, cowboy shot, upper body, half body, "
    "cropped legs, cropped feet, missing shoes, zoomed in, photorealistic, 3d render, "
    "extra limbs, watermark, text"
)
CROP_LOCK_ZH = "全身立绘，从头顶到两只鞋完整入画，双脚和鞋子都完整可见，脚下留白"


@dataclass
class Row:
    character_id: str
    size: str
    attempt: int
    seed: int
    path: str
    model: str
    bytes: int
    width: int
    height: int
    figure_span: float
    figure_top: float
    figure_bottom: float
    geometric_full_body: bool
    structure_verdict: str
    structure_reasons: str
    vision_full_body: bool | None
    vision_both_feet: bool | None
    vision_reasons: str
    valid_asset: bool
    error: str
    latency_s: float


def build_prompt(identity: str) -> str:
    return f"{FRAMING}, {identity}, {CROP_LOCK_ZH}"


def dashscope_generate(*, api_key: str, prompt: str, size: str, seed: int) -> bytes:
    body = {
        "model": MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ]
        },
        "parameters": {
            "n": 1,
            "size": size,
            "prompt_extend": False,
            "watermark": False,
            "negative_prompt": NEGATIVE,
            "seed": int(seed) % (2**31 - 1),
        },
    }
    data = json.dumps(body).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(8):
        req = urllib.request.Request(
            DASHSCOPE_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < 7:
                wait = min(60.0, 3.0 * (2**attempt))
                print(f"  retry after HTTP {exc.code}, sleep {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            raise
    else:
        raise last_err or RuntimeError("dashscope failed")
    choices = (((payload.get("output") or {}).get("choices")) or [])
    if not choices:
        raise RuntimeError(f"empty choices: {payload}")
    content = ((choices[0].get("message") or {}).get("content")) or []
    url = None
    for part in content:
        if isinstance(part, dict) and part.get("image"):
            url = part["image"]
            break
    if not url:
        raise RuntimeError(f"no image url: {payload}")
    with urllib.request.urlopen(url, timeout=120) as img_resp:
        return img_resp.read()


def vision_qc(blob: bytes) -> tuple[bool | None, bool | None, str]:
    """Optional SiliconFlow Qwen vision using production master fields."""
    keys = os.environ.get("SILICONFLOW_API_KEYS") or os.environ.get("SILICONFLOW_API_KEY") or ""
    if not keys.strip():
        return None, None, "vision_skipped_no_key"
    try:
        from anime_factory.llm import master_vision_qc
    except Exception as exc:
        return None, None, f"vision_import_failed:{exc}"
    try:
        result = master_vision_qc(bytes(blob), view="front", kind="character_sheet")
        fields = dict(getattr(result, "fields", None) or {})
        reasons = list(getattr(result, "reasons", None) or [])
        if "vision_transport_error" in reasons or any("transport" in r for r in reasons):
            return None, None, ",".join(reasons)
        full = fields.get("full_body")
        feet = fields.get("both_feet_visible")
        if full is None:
            full = bool(getattr(result, "passed", False)) and "not_full_body" not in reasons
        if feet is None:
            joined = ",".join(reasons)
            feet = "both_feet" not in joined and "feet" not in joined
        return bool(full), bool(feet), ",".join(reasons)
    except Exception as exc:
        return None, None, f"vision_error:{exc}"


def score_blob(blob: bytes) -> dict:
    img = Image.open(BytesIO(blob)).convert("RGB")
    ok, metrics = figure_is_full_body(img)
    struct = structure_check(blob, kind="character_sheet", allow_placeholder=True)
    v_full, v_feet, v_reasons = vision_qc(blob)
    geometric = bool(ok)
    # valid_asset: geometric required; vision if present must also pass
    if v_full is None:
        valid = geometric
    else:
        valid = geometric and bool(v_full) and (v_feet is not False)
    return {
        "width": img.size[0],
        "height": img.size[1],
        "figure_span": float(metrics["span"]),
        "figure_top": float(metrics["top"]),
        "figure_bottom": float(metrics["bottom"]),
        "geometric_full_body": geometric,
        "structure_verdict": struct.verdict,
        "structure_reasons": ",".join(struct.reasons or []),
        "vision_full_body": v_full,
        "vision_both_feet": v_feet,
        "vision_reasons": v_reasons,
        "valid_asset": valid,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"/tmp/qwen_canary_a_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--limit", type=int, default=0, help="Cap total gens (0=all 20)")
    ap.add_argument("--characters", default="lin-xiao,a-kai")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY") or ""
    if not api_key:
        # load from SoundStoryWriter .env
        env_path = Path("/code/SoundStoryWriter/.env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("DASHSCOPE_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        print("DASHSCOPE_API_KEY missing", file=sys.stderr)
        return 2

    # Optional SF keys for vision
    if not os.environ.get("SILICONFLOW_API_KEY"):
        env_path = Path("/code/SoundStoryWriter/.env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("SILICONFLOW_API_KEY=") or line.startswith("SILICONFLOW_API_KEYS="):
                    os.environ[line.split("=", 1)[0]] = line.split("=", 1)[1].strip().strip('"').strip("'")

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    char_ids = [c.strip() for c in args.characters.split(",") if c.strip()]
    jobs = []
    for cid in char_ids:
        for size in SIZES:
            for attempt in range(1, ATTEMPTS + 1):
                jobs.append((cid, size, attempt))
    if args.limit > 0:
        jobs = jobs[: args.limit]

    print(f"model={MODEL} jobs={len(jobs)} out={out}")
    print(f"FIGURE thresholds span>={FIGURE_SPAN_MIN} top<={FIGURE_TOP_MAX} bottom>={FIGURE_BOTTOM_MIN}")
    print("NOTE: DashScope has no qwen-image-2.1; using qwen-image-2.0-pro as family canary.")

    for i, (cid, size, attempt) in enumerate(jobs, 1):
        identity = CHARACTERS[cid]["identity"]
        prompt = build_prompt(identity)
        seed = 1000 + i * 97 + attempt * 13
        rel = f"images/{cid}_{size.replace('*','x')}_a{attempt}.png"
        path = out / rel
        t0 = time.time()
        err = ""
        blob = b""
        try:
            blob = dashscope_generate(api_key=api_key, prompt=prompt, size=size, seed=seed)
            path.write_bytes(blob)
            scored = score_blob(blob)
        except Exception as exc:
            scored = {
                "width": 0,
                "height": 0,
                "figure_span": 0.0,
                "figure_top": 1.0,
                "figure_bottom": 0.0,
                "geometric_full_body": False,
                "structure_verdict": "error",
                "structure_reasons": "",
                "vision_full_body": None,
                "vision_both_feet": None,
                "vision_reasons": "",
                "valid_asset": False,
            }
            err = f"{type(exc).__name__}:{exc}"
            print(f"[{i}/{len(jobs)}] FAIL {cid} {size} a{attempt}: {err}")
        else:
            print(
                f"[{i}/{len(jobs)}] {cid} {size} a{attempt} "
                f"geo={scored['geometric_full_body']} valid={scored['valid_asset']} "
                f"span={scored['figure_span']:.3f} bottom={scored['figure_bottom']:.3f} "
                f"reasons={scored['structure_reasons'] or '-'}"
            )
        row = Row(
            character_id=cid,
            size=size,
            attempt=attempt,
            seed=seed,
            path=str(path) if blob else "",
            model=MODEL,
            bytes=len(blob),
            latency_s=round(time.time() - t0, 2),
            error=err,
            **{k: scored[k] for k in (
                "width", "height", "figure_span", "figure_top", "figure_bottom",
                "geometric_full_body", "structure_verdict", "structure_reasons",
                "vision_full_body", "vision_both_feet", "vision_reasons", "valid_asset",
            )},
        )
        rows.append(row)
        time.sleep(args.sleep)

    # Summary
    total = len(rows)
    valid = sum(1 for r in rows if r.valid_asset)
    geo = sum(1 for r in rows if r.geometric_full_body)
    by_char: dict[str, list[Row]] = {}
    by_size: dict[str, list[Row]] = {}
    for r in rows:
        by_char.setdefault(r.character_id, []).append(r)
        by_size.setdefault(r.size, []).append(r)

    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "note": "qwen-image-2.1 not on DashScope; Phase A uses qwen-image-2.0-pro",
        "thresholds": {
            "FIGURE_SPAN_MIN": FIGURE_SPAN_MIN,
            "FIGURE_TOP_MAX": FIGURE_TOP_MAX,
            "FIGURE_BOTTOM_MIN": FIGURE_BOTTOM_MIN,
        },
        "total": total,
        "geometric_full_body_rate": geo / total if total else 0.0,
        "valid_asset_rate": valid / total if total else 0.0,
        "valid_asset_count": valid,
        "by_character": {
            cid: {
                "n": len(rs),
                "valid": sum(1 for r in rs if r.valid_asset),
                "geo": sum(1 for r in rs if r.geometric_full_body),
                "rate": sum(1 for r in rs if r.valid_asset) / len(rs),
            }
            for cid, rs in by_char.items()
        },
        "by_size": {
            size: {
                "n": len(rs),
                "valid": sum(1 for r in rs if r.valid_asset),
                "geo": sum(1 for r in rs if r.geometric_full_body),
                "rate": sum(1 for r in rs if r.valid_asset) / len(rs),
            }
            for size, rs in by_size.items()
        },
        "go_no_go": {
            "phase_a_generation": (
                "GO_EDIT_PHASE"
                if (valid / total if total else 0) >= 0.40
                else "NO_GO_WEAK_T2I"
            ),
            "rule": "valid_asset_rate >= 0.40 → proceed to Phase B (edit repair)",
        },
    }

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out / "rows.json").write_text(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2))
    with (out / "rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()) if rows else ["character_id"])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

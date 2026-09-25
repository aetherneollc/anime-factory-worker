#!/usr/bin/env python3
"""Asset Factory ImageProvider canary — HunyuanImage Phase A (full-body).

Uses TokenHub ``hy-image-v3`` (Hy-Image-3.0) or fal.ai fallback.
Does NOT bottom-align / reframe before QC.

Credentials (first match):
  TOKENHUB_API_KEY / HUNYUAN_API_KEY  → Tencent TokenHub
  FAL_KEY                              → fal-ai/hunyuan-image/v3/text-to-image

Size note: TokenHub area ≤ 1024×1024, so 768×1344 OK; 1024×1536 is skipped.

Usage:
  TOKENHUB_API_KEY=... PYTHONPATH=python \\
    .venv/bin/python tools/hunyuan_character_canary_a.py [--out DIR] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from PIL import Image  # noqa: E402

from anime_factory.hunyuan_image import (  # noqa: E402
    HUNYUAN_MODEL,
    fal_api_key,
    generate_hunyuan_image,
    hunyuan_api_key,
)
from anime_factory.visual_qc import (  # noqa: E402
    FIGURE_BOTTOM_MIN,
    FIGURE_SPAN_MIN,
    FIGURE_TOP_MAX,
    figure_is_full_body,
    structure_check,
)

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

# TokenHub-legal tall portraits (area ≤ 1_048_576).
SIZES = ("768x1024", "768x1344")
ATTEMPTS = 5

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
    backend: str
    bytes: int
    width: int
    height: int
    figure_span: float
    figure_top: float
    figure_bottom: float
    geometric_full_body: bool
    structure_verdict: str
    structure_reasons: str
    valid_asset: bool
    error: str
    latency_s: float


def build_prompt(identity: str) -> str:
    return f"{FRAMING}, {identity}, {CROP_LOCK_ZH}"


def load_dotenv_keys() -> None:
    for env_path in (
        ROOT / ".env",
        Path("/code/SoundStoryWriter2/.env"),
        Path("/code/SoundStoryWriter/.env"),
    ):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key in os.environ:
                continue
            value = value.strip().strip('"').strip("'")
            if key in {
                "TOKENHUB_API_KEY",
                "HUNYUAN_API_KEY",
                "TENCENT_TOKENHUB_API_KEY",
                "FAL_KEY",
                "FAL_API_KEY",
                "SILICONFLOW_API_KEY",
                "SILICONFLOW_API_KEYS",
            }:
                os.environ[key] = value


def score_blob(blob: bytes) -> dict:
    img = Image.open(BytesIO(blob)).convert("RGB")
    ok, metrics = figure_is_full_body(img)
    struct = structure_check(blob, kind="character_sheet", allow_placeholder=True)
    geometric = bool(ok)
    return {
        "width": img.size[0],
        "height": img.size[1],
        "figure_span": float(metrics["span"]),
        "figure_top": float(metrics["top"]),
        "figure_bottom": float(metrics["bottom"]),
        "geometric_full_body": geometric,
        "structure_verdict": struct.verdict,
        "structure_reasons": ",".join(struct.reasons or []),
        "valid_asset": geometric,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=f"/tmp/hunyuan_canary_a_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    ap.add_argument("--limit", type=int, default=0, help="Cap total gens (0=all 20)")
    ap.add_argument("--characters", default="lin-xiao,a-kai")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    load_dotenv_keys()
    backend = "tokenhub" if hunyuan_api_key() else ("fal" if fal_api_key() else "")
    if not backend:
        print(
            "BLOCKED: set TOKENHUB_API_KEY (Tencent TokenHub) or FAL_KEY. "
            "No Hunyuan credentials in env.",
            file=sys.stderr,
        )
        return 2

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    char_ids = [c.strip() for c in args.characters.split(",") if c.strip()]
    jobs = [
        (cid, size, attempt)
        for cid in char_ids
        for size in SIZES
        for attempt in range(1, ATTEMPTS + 1)
    ]
    if args.limit > 0:
        jobs = jobs[: args.limit]

    print(f"model={HUNYUAN_MODEL} backend={backend} jobs={len(jobs)} out={out}")
    print(f"FIGURE thresholds span>={FIGURE_SPAN_MIN} top<={FIGURE_TOP_MAX} bottom>={FIGURE_BOTTOM_MIN}")

    for i, (cid, size, attempt) in enumerate(jobs, 1):
        prompt = build_prompt(CHARACTERS[cid]["identity"])
        seed = 2000 + i * 97 + attempt * 13
        rel = f"images/{cid}_{size}_a{attempt}.png"
        path = out / rel
        t0 = time.time()
        err = ""
        blob = b""
        try:
            blob = generate_hunyuan_image(
                prompt=prompt,
                negative_prompt=NEGATIVE,
                image_size=size,
                seed=seed,
                revise=False,
            )
            path.write_bytes(blob)
            scored = score_blob(blob)
        except Exception as exc:  # noqa: BLE001
            scored = {
                "width": 0,
                "height": 0,
                "figure_span": 0.0,
                "figure_top": 1.0,
                "figure_bottom": 0.0,
                "geometric_full_body": False,
                "structure_verdict": "error",
                "structure_reasons": "",
                "valid_asset": False,
            }
            err = f"{type(exc).__name__}:{exc}"
            print(f"[{i}/{len(jobs)}] FAIL {cid} {size} a{attempt}: {err}", flush=True)
        else:
            print(
                f"[{i}/{len(jobs)}] {cid} {size} a{attempt} "
                f"geo={scored['geometric_full_body']} valid={scored['valid_asset']} "
                f"span={scored['figure_span']:.3f} bottom={scored['figure_bottom']:.3f}",
                flush=True,
            )
        rows.append(
            Row(
                character_id=cid,
                size=size,
                attempt=attempt,
                seed=seed,
                path=str(path) if blob else "",
                model=HUNYUAN_MODEL,
                backend=backend,
                bytes=len(blob),
                latency_s=round(time.time() - t0, 2),
                error=err,
                **{k: scored[k] for k in (
                    "width", "height", "figure_span", "figure_top", "figure_bottom",
                    "geometric_full_body", "structure_verdict", "structure_reasons",
                    "valid_asset",
                )},
            )
        )
        time.sleep(args.sleep)

    n = len(rows)
    n_geo = sum(1 for r in rows if r.geometric_full_body)
    n_valid = sum(1 for r in rows if r.valid_asset)
    by_char: dict[str, dict] = {}
    for r in rows:
        bucket = by_char.setdefault(r.character_id, {"n": 0, "geo": 0, "valid": 0})
        bucket["n"] += 1
        bucket["geo"] += int(r.geometric_full_body)
        bucket["valid"] += int(r.valid_asset)

    summary = {
        "model": HUNYUAN_MODEL,
        "backend": backend,
        "n": n,
        "geometric_full_body_rate": (n_geo / n) if n else 0.0,
        "valid_asset_rate": (n_valid / n) if n else 0.0,
        "by_character": by_char,
        "go_edit_phase": (n_valid / n) >= 0.40 if n else False,
        "note": "TokenHub area≤1024²; sizes 768x1024 + 768x1344",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with (out / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(asdict(r) for r in rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    verdict = "GO_EDIT_PHASE" if summary["go_edit_phase"] else "NO_GO"
    print(f"VERDICT={verdict} valid_asset_rate={summary['valid_asset_rate']:.2f}")
    return 0 if n and not all(r.error for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

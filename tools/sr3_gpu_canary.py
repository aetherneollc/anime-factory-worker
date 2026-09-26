#!/usr/bin/env python3
"""5090 canary for anime-factory-worker-sr3. Script only.

On one instance, in order:
  1. FLUX.2 Klein 4B writes three full-body masters, then one reference edit.
  2. SkyReels V3 R2V writes a 5s 720P clip from two of those masters.

The instance is destroyed in ``finally`` on success, failure, and interrupt.
Default is Vast dry-run (no PUT /asks/). Live mode needs ``--live``.
Do not point this at the non-commercial Klein 9B checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from anime_factory.config import load_dotenv  # noqa: E402
from anime_factory.flux2_klein import KLEIN_4B_MODEL_ID as KLEIN_MODEL_ID  # noqa: E402
from anime_factory.skyreels_r2v import SKYREELS_MODEL_ID as R2V_MODEL_ID  # noqa: E402
from gpu_worker.offers import pick_one_offer, search_payload  # noqa: E402
from gpu_worker.vast_client import VastClient  # noqa: E402

SR3_IMAGE = os.environ.get(
    "SR3_VAST_IMAGE",
    "ghcr.io/aetherneollc/anime-factory-worker-sr3:main",
)

REMOTE_PY = r'''
import os
from pathlib import Path
os.environ["FLUX2_KLEIN_DRY_RUN"] = "0"
os.environ["SKYREELS_DRY_RUN"] = "0"
os.environ["SKYREELS_LOW_VRAM"] = "1"
out = Path("/workspace/sr3_out")
out.mkdir(parents=True, exist_ok=True)
from anime_factory.flux2_klein import edit_klein_refs, generate_klein_t2i, master_prompt
idents = [
    "1girl, short black hair, school uniform, brown shoes",
    "1boy, silver hair, long coat, black boots",
    "1girl, twin tails, red jacket, white sneakers",
]
masters = []
for i, ident in enumerate(idents):
    png = generate_klein_t2i(master_prompt(ident), 768, 1344, seed=1000 + i)
    path = out / f"master_{i}.png"
    path.write_bytes(png)
    masters.append(path)
    print("master", path, len(png), flush=True)
edited = edit_klein_refs(
    "same character, three-quarter view, same outfit, studio background",
    [masters[0].read_bytes()],
    768,
    1344,
    seed=1100,
)
(out / "edit_from_master.png").write_bytes(edited)
print("edit", len(edited), flush=True)
from anime_factory.flux2_klein import release_klein_vram
release_klein_vram()
from anime_factory.skyreels_r2v import generate_reference_video
video = generate_reference_video(
    ref_imgs=[str(masters[0]), str(masters[1])],
    prompt="the two characters walk side by side through a rainy alley, anime",
    duration_s=5,
    aspect="16:9",
    resolution="720P",
    seed=42,
    out_path=str(out / "r2v_5s.mp4"),
    dry_run=False,
)
print("video", video["path"], "fps", video["fps"], flush=True)
(out / "DONE").write_text("ok\n", encoding="utf-8")
'''


def _ssh(host: str, port: int, user: str, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-p", str(port),
            f"{user}@{host}",
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ssh_target(inst: dict) -> tuple[str, int, str]:
    host = str(inst.get("ssh_host") or inst.get("public_ipaddr") or "")
    port = int(inst.get("ssh_port") or 22)
    user = str(inst.get("ssh_user") or "root")
    if not host:
        raise RuntimeError("instance has no ssh host yet")
    return host, port, user


def main() -> int:
    ap = argparse.ArgumentParser(description="sr3 Klein + R2V canary; always destroys")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--max-dph", type=float, default=0.85)
    ap.add_argument("--max-wall-min", type=int, default=90)
    ap.add_argument("--offer-id", default="")
    ap.add_argument(
        "--out",
        default=f"/tmp/sr3_canary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    args = ap.parse_args()
    load_dotenv()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = VastClient(api_key=os.environ.get("VAST_API_KEY") or "", dry_run=not args.live)
    body = client.search_offers(filters=search_payload())
    offers = list(body.get("offers") or []) if isinstance(body, dict) else []
    if args.offer_id:
        chosen = next((o for o in offers if str(o.get("id")) == str(args.offer_id)), {"id": args.offer_id})
    else:
        chosen = pick_one_offer(offers, gpu_model_policy="5090", max_dph=args.max_dph, max_usd=80)
    if not chosen:
        print("no offer", file=sys.stderr)
        return 2
    offer_id = str(chosen.get("id"))
    lease_body = client.lease_body(
        image=SR3_IMAGE,
        disk_gb=200,
        env={
            "IMAGE_BACKEND": "flux2_klein4b",
            "VIDEO_BACKEND": "skyreels_v3_r2v",
            "AF_VIDEO_BACKEND": "skyreels_v3_r2v",
            "AF_GPU_PROFILE": "sr3-cu128-sm120",
            "AF_START_COMFY": "0",
            "SKYREELS_LOW_VRAM": "1",
            "FLUX2_KLEIN_MODEL": KLEIN_MODEL_ID,
        },
        extra={"label": "sr3-klein-r2v-canary", "onstart": "mkdir -p /workspace && sleep infinity"},
    )
    print(json.dumps({"offer_id": offer_id, "image": SR3_IMAGE, "klein": KLEIN_MODEL_ID, "r2v": R2V_MODEL_ID, "live": args.live}))
    leased = client.lease(offer_id, lease_body)
    if leased.get("dry_run") or leased.get("skipped"):
        (out / "plan.json").write_text(
            json.dumps({"lease": leased, "remote_checks": ["klein_3_masters_plus_edit", "r2v_2ref_5s"], "destroy": True}, indent=2) + "\n",
            encoding="utf-8",
        )
        print("dry-run; no instance. destroy not required")
        return 0

    instance_id = str(leased.get("new_contract") or leased.get("id") or leased.get("instance_id") or "")
    registered = {instance_id} if instance_id else set()
    if not instance_id:
        print("lease returned no instance", leased, file=sys.stderr)
        return 1

    def _on_signal(signum, _frame):
        print(f"signal {signum}; destroying {instance_id}", file=sys.stderr)
        try:
            client.destroy(instance_id, registered)
        finally:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    deadline = time.time() + args.max_wall_min * 60
    try:
        host = port = user = None
        while time.time() < deadline:
            payload = client.get_instance(instance_id)
            inst = payload.get("instances") if isinstance(payload.get("instances"), dict) else payload
            inst = inst if isinstance(inst, dict) else {}
            try:
                host, port, user = _ssh_target(inst)
                probe = _ssh(host, port, user, "nvidia-smi -L && echo SSH_OK", timeout=30)
                if probe.returncode == 0 and "SSH_OK" in (probe.stdout or ""):
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"wait ssh: {exc}", flush=True)
            time.sleep(15)
        else:
            raise RuntimeError("timed out waiting for SSH")
        remote = out / "remote_check.py"
        remote.write_text(REMOTE_PY, encoding="utf-8")
        subprocess.run(
            [
                "scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-P", str(port), str(remote), f"{user}@{host}:/workspace/remote_check.py",
            ],
            check=True,
            timeout=60,
        )
        run = _ssh(
            host,
            port,
            user,
            "bash -lc 'nohup python /workspace/remote_check.py </dev/null >/workspace/sr3_run.log 2>&1 & echo STARTED $!'",
            timeout=45,
        )
        if "STARTED" not in (run.stdout or ""):
            raise RuntimeError(f"remote start failed: {run.stderr or run.stdout}")
        while time.time() < deadline:
            check = _ssh(
                host,
                port,
                user,
                "if test -f /workspace/sr3_out/DONE; then echo DONE; else echo WAIT; tail -n 12 /workspace/sr3_run.log; fi",
                timeout=60,
            )
            print((check.stdout or "")[-800:], flush=True)
            if (check.stdout or "").strip().startswith("DONE"):
                break
            time.sleep(30)
        else:
            raise RuntimeError("remote Klein/R2V check did not finish")
        listing = _ssh(host, port, user, "ls -1 /workspace/sr3_out/", timeout=60)
        print(listing.stdout)
        for name in (listing.stdout or "").splitlines():
            name = name.strip()
            if not name:
                continue
            subprocess.run(
                [
                    "scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                    "-P", str(port), f"{user}@{host}:/workspace/sr3_out/{name}", str(out / name),
                ],
                check=False,
                timeout=600,
            )
        return 0
    finally:
        if instance_id:
            try:
                print("destroyed", client.destroy(instance_id, registered))
            except Exception as exc:  # noqa: BLE001
                print(f"destroy failed: {exc}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
